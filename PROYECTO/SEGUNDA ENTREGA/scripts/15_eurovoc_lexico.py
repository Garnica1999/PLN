"""15_eurovoc_lexico.py
Fecha: 2026-09-11
Descripcion: Construye el mapeo de las 16 clases efectivas del M4 a
    categorias de EuroVoc en espanol y el lexico de terminos para la variante
    E de la Fase W(b). Para cada concepto raiz mapeado se descargan sus
    skos:prefLabel/skos:altLabel en espanol (incluido el cierre de
    descendientes via skos:broader+) desde el endpoint SPARQL oficial de
    Publications Office y se lematizan con el MISMO pipeline de 05
    (es_core_news_md, tok.is_alpha, lemma_.lower().strip(), union por espacio).

    Salidas NUEVAS (gane o pierda la senal EuroVoc, el mapeo queda registrado):
        config/eurovoc_bloques.json     (mapeo clases -> categorias + macro-grupos)
        config/eurovoc_terminos_es.json (lexico lematizado por clase)

Estado del arte aplicado: Fase W(b) de PLAN_v2_ciclo_mejora ("EuroVoc-es como
    features auxiliares"): MultiEURLEX (Chalkidis et al. 2021) usa etiquetas
    EuroVoc multietiqueta en 23 lenguas; E1 (nota_ontologias.md) anticipo
    cobertura PARCIAL (derecho UE, no CP colombiana), por lo que la senal se
    adopta SOLO si gana el benchmark de validacion de la Fase U.

Uso:
    /home/carlos/venv/bin/python scripts/15_eurovoc_lexico.py
"""

import json
import sys
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import spacy

# ===== SECCION 1: configuracion =====
CONFIG = {
    "endpoint": "https://publications.europa.eu/webapi/rdf/sparql",
    "esquema_uri": "http://eurovoc.europa.eu/100141",
    "modelo_spacy": "es_core_news_md",
    "ruta_etiquetas": "data/etiquetas_areas.parquet",
    "ruta_bloques": "config/eurovoc_bloques.json",
    "ruta_terminos": "config/eurovoc_terminos_es.json",
    "timeout_seg": 120,
    "reintentos": 3,
    "espera_reintento_seg": 5,
    "max_tokens_termino": 6,  # cota del n-grama de CountVectorizer en 06
}

# Las 16 clases efectivas del M4 (los 19 bloques de la CP con <3 docs se
# agruparon en otras_areas en v1; se valida en runtime contra el parquet).
CLASES_M4 = [
    "derechos_colectivos_ambiente",
    "derechos_fundamentales",
    "derechos_sociales_economicos_culturales",
    "elecciones",
    "habitantes_territorio",
    "organismos_control",
    "organizacion_estado",
    "organizacion_territorial",
    "participacion_democratica",
    "principios_fundamentales",
    "proteccion_aplicacion_derechos",
    "rama_ejecutiva",
    "rama_judicial",
    "rama_legislativa",
    "regimen_economico",
    "reforma_constitucion",
]

# Dominios (campos) de EuroVoc verificados en el endpoint el 2026-09-11.
DOMINIOS_EUROVOC = {
    "100143": "08 RELACIONES INTERNACIONALES",
    "100144": "10 UNION EUROPEA",
    "100145": "12 DERECHO",
    "100146": "16 ECONOMIA",
    "100147": "20 INTERCAMBIOS ECONOMICOS Y COMERCIALES",
    "100148": "24 ASUNTOS FINANCIEROS",
    "100149": "28 ASUNTOS SOCIALES",
    "100150": "32 EDUCACION Y COMUNICACION",
    "100151": "36 CIENCIA",
    "100152": "40 EMPRESA Y COMPETENCIA",
    "100153": "44 TRABAJO Y EMPLEO",
    "100154": "48 TRANSPORTES",
    "100155": "52 MEDIO AMBIENTE",
    "100156": "56 AGRICULTURA, SILVICULTURA Y PESCA",
    "100157": "60 SECTOR AGROALIMENTARIO",
    "100158": "64 PRODUCCION, TECNOLOGIA E INVESTIGACION",
    "100159": "66 ENERGIA",
    "100160": "68 INDUSTRIA",
    "100161": "72 GEOGRAFIA",
    "100184": "1236 DERECHOS Y LIBERTADES",
}

# Mapeo clase M4 -> conceptos EuroVoc raiz (IDs verificados en el endpoint).
# La justificacion es documental: cercania semantica del bloque de la CP al
# descriptor EuroVoc; EuroVoc cubre derecho de la UE, no el CP colombiano.
MAPPING_CLASES = {
    "derechos_fundamentales": {
        "raices": ["538", "584"],
        "justificacion": (
            "Titulo II de la CP (derechos de aplicacion inmediata y derechos "
            "humanos del bloque de constitucionalidad): EuroVoc 538 'derechos "
            "fundamentales' (dominio 12 DERECHO, microtesauro 1236) y 584 "
            "'derechos humanos'."
        ),
    },
    "derechos_sociales_economicos_culturales": {
        "raices": ["3899", "3901"],
        "justificacion": (
            "Titulo II, capitulos 2 y 3 de la CP (derechos economicos, "
            "sociales y culturales): EuroVoc 3899 'derechos sociales' y 3901 "
            "'derechos economicos' (dominio 28 ASUNTOS SOCIALES). Sin "
            "equivalente exacto a 'DESC' como descriptor unico."
        ),
    },
    "derechos_colectivos_ambiente": {
        "raices": ["2470", "2825"],
        "justificacion": (
            "Capitulo 3 de la CP (derechos colectivos y ambiente): EuroVoc "
            "2470 'politica de medio ambiente' y 2825 'proteccion del medio "
            "ambiente' (dominio 52 MEDIO AMBIENTE)."
        ),
    },
    "proteccion_aplicacion_derechos": {
        "raices": ["5400", "5371", "151"],
        "justificacion": (
            "Acciones constitucionales (tutela, cumplimiento, habeas corpus): "
            "EuroVoc 5400 'acceso a la justicia', 5371 'accion judicial' y 151 "
            "'control de constitucionalidad' (dominio 12 DERECHO)."
        ),
    },
    "principios_fundamentales": {
        "raices": ["5622", "148"],
        "justificacion": (
            "Titulo I de la CP (principios y fines del Estado): EuroVoc 5622 "
            "'Estado de Derecho' y 148 'Constitucion' (dominio 12 DERECHO). "
            "No existen descriptores EuroVoc para 'Estado social de derecho' "
            "ni 'soberania popular'."
        ),
    },
    "rama_legislativa": {
        "raices": ["2242", "2702", "41"],
        "justificacion": (
            "Titulo V, capitulo 1 (Congreso): EuroVoc 2242 'Parlamento', 2702 "
            "'procedimiento legislativo' y 41 'competencias del Parlamento' "
            "(dominio 04 POLITICA)."
        ),
    },
    "rama_ejecutiva": {
        "raices": ["3062", "77"],
        "justificacion": (
            "Titulo V, capitulos 2 y 3 (Presidente, Gobierno): EuroVoc 3062 "
            "'Ejecutivo' y 77 'administracion publica' (dominio 04 POLITICA)."
        ),
    },
    "rama_judicial": {
        "raices": ["5923", "2701", "2351"],
        "justificacion": (
            "Titulo V, capitulo 4 (jurisdiccion): EuroVoc 5923 'sistema "
            "judicial', 2701 'procedimiento judicial' y 2351 'profesion "
            "juridica' (dominio 12 DERECHO)."
        ),
    },
    "organizacion_estado": {
        "raices": ["883", "2166", "4304"],
        "justificacion": (
            "Estructura del Estado (Titulo V y organos autonomos): EuroVoc 883 "
            "'Estado', 2166 'organizacion administrativa' y 4304 'estructura "
            "institucional' (dominio 04 POLITICA)."
        ),
    },
    "organismos_control": {
        "raices": ["2579", "1448"],
        "justificacion": (
            "Contraloria, Procuraduria, Defensoria y control fiscal: EuroVoc "
            "2579 'poder de control' y 1448 'instancia de control' (dominio 04 "
            "POLITICA). Cobertura solo conceptual: EuroVoc no tiene "
            "equivalentes de 'contraloria' o 'ministerio publico'."
        ),
    },
    "organizacion_territorial": {
        "raices": ["6034", "2515"],
        "justificacion": (
            "Titulo XI (entidades territoriales): EuroVoc 6034 'entidad "
            "territorial' y 2515 'politica regional' (dominios 04 POLITICA y "
            "72 GEOGRAFIA)."
        ),
    },
    "habitantes_territorio": {
        "raices": ["385", "52", "1909"],
        "justificacion": (
            "Titulo III (nacionalidad, poblacion del territorio): EuroVoc 385 "
            "'demografia', 52 'composicion de la poblacion' y 1909 'migracion' "
            "(dominios 28 ASUNTOS SOCIALES y 72 GEOGRAFIA)."
        ),
    },
    "participacion_democratica": {
        "raices": ["3900", "4750", "2258"],
        "justificacion": (
            "Titulo IV y mecanismos de participacion: EuroVoc 3900 'derechos "
            "politicos', 4750 'votacion' y 2258 'partidos politicos' "
            "(dominio 04 POLITICA)."
        ),
    },
    "elecciones": {
        "raices": ["695", "4367", "558", "2186"],
        "justificacion": (
            "Organizacion electoral (CNE, circunscripciones): EuroVoc 695 "
            "'elecciones', 4367 'sistema electoral', 558 'Derecho electoral' "
            "y 2186 'organizacion electoral' (dominio 04 POLITICA)."
        ),
    },
    "regimen_economico": {
        "raices": ["3013", "2497", "2448", "424"],
        "justificacion": (
            "Titulo XII (regimen economico y hacienda publica): EuroVoc 3013 "
            "'regimen economico', 2497 'politica economica', 2448 'politica "
            "presupuestaria' y 424 'deuda publica' (dominios 16 ECONOMIA y 24 "
            "ASUNTOS FINANCIEROS)."
        ),
    },
    "reforma_constitucion": {
        "raices": ["3699", "148"],
        "justificacion": (
            "Titulo XIII (reforma de la Constitucion): EuroVoc 3699 'revision "
            "de la Constitucion' (altLabel 'reforma constitucional') y 148 "
            "'Constitucion' (dominio 12 DERECHO)."
        ),
    },
}

# Macro-grupos de la variante D: agrupacion de clases por categorias EuroVoc
# compartidas (~6-8 grupos, particion exacta de las 16 clases).
MACRO_GRUPOS_EUROVOC = {
    "derechos_persona": {
        "clases": ["derechos_fundamentales", "proteccion_aplicacion_derechos"],
        "categorias_eurovoc": [
            "12 DERECHO",
            "1236 DERECHOS Y LIBERTADES",
            "538 derechos fundamentales",
            "584 derechos humanos",
            "5400 acceso a la justicia",
            "151 control de constitucionalidad",
        ],
        "justificacion": (
            "Derechos de la persona y sus garantias judiciales comparten el "
            "dominio 12 DERECHO y el microtesauro 1236."
        ),
    },
    "derechos_sociales_colectivos": {
        "clases": ["derechos_sociales_economicos_culturales", "derechos_colectivos_ambiente"],
        "categorias_eurovoc": [
            "28 ASUNTOS SOCIALES",
            "52 MEDIO AMBIENTE",
            "3899 derechos sociales",
            "3901 derechos economicos",
            "2470 politica de medio ambiente",
        ],
        "justificacion": (
            "DESC y derechos colectivos/ambiente comparten la logica de "
            "derechos de segunda y tercera generacion (dominios 28 y 52)."
        ),
    },
    "principios_constitucionales": {
        "clases": ["principios_fundamentales"],
        "categorias_eurovoc": ["5622 Estado de Derecho", "148 Constitucion"],
        "justificacion": (
            "Principios y fines del Estado se anclan al descriptor 'Estado de "
            "Derecho' y a 'Constitucion' (dominio 12 DERECHO)."
        ),
    },
    "ramas_del_poder": {
        "clases": ["rama_legislativa", "rama_ejecutiva", "rama_judicial"],
        "categorias_eurovoc": [
            "04 POLITICA",
            "2242 Parlamento",
            "3062 Ejecutivo",
            "5923 sistema judicial",
        ],
        "justificacion": (
            "Las tres ramas del poder publico se corresponden con los "
            "descriptores institucionales de EuroVoc (Parlamento, Ejecutivo, "
            "sistema judicial)."
        ),
    },
    "estado_y_control": {
        "clases": ["organizacion_estado", "organismos_control"],
        "categorias_eurovoc": [
            "04 POLITICA",
            "883 Estado",
            "2166 organizacion administrativa",
            "2579 poder de control",
        ],
        "justificacion": (
            "Organizacion del Estado y organismos de control comparten los "
            "descriptores de estructura institucional y control publico."
        ),
    },
    "territorio_y_participacion": {
        "clases": [
            "organizacion_territorial",
            "habitantes_territorio",
            "participacion_democratica",
            "elecciones",
        ],
        "categorias_eurovoc": [
            "72 GEOGRAFIA",
            "04 POLITICA",
            "6034 entidad territorial",
            "385 demografia",
            "3900 derechos politicos",
            "695 elecciones",
        ],
        "justificacion": (
            "Territorio, poblacion y mecanismos de participacion electoral "
            "comparten los dominios 72 GEOGRAFIA y 04 POLITICA."
        ),
    },
    "regimen_economico": {
        "clases": ["regimen_economico"],
        "categorias_eurovoc": [
            "16 ECONOMIA",
            "24 ASUNTOS FINANCIEROS",
            "3013 regimen economico",
            "2497 politica economica",
        ],
        "justificacion": (
            "Regimen economico y hacienda publica se anclan a los dominios "
            "16 ECONOMIA y 24 ASUNTOS FINANCIEROS."
        ),
    },
    "reforma_constitucional": {
        "clases": ["reforma_constitucion"],
        "categorias_eurovoc": ["3699 revision de la Constitucion", "148 Constitucion"],
        "justificacion": (
            "Reforma constitucional tiene descriptor directo en EuroVoc (3699, "
            "altLabel 'reforma constitucional')."
        ),
    },
}

PREFIJOS_SPARQL = "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>"


def consultar_sparql(consulta: str) -> list[dict]:
    """Ejecuta una consulta SELECT contra el endpoint SPARQL de EuroVoc.

    Args:
        consulta: Consulta SPARQL completa (con prefijos).

    Returns:
        Lista de bindings del resultset.

    Raises:
        RuntimeError: Si la consulta falla tras agotar los reintentos.
    """
    url = CONFIG["endpoint"] + "?query=" + urllib.parse.quote(consulta)
    url += "&format=" + urllib.parse.quote("application/sparql-results+json")
    peticion = urllib.request.Request(url, headers={"User-Agent": "Juris-PLN-E2/2.0"})
    ultimo_error = ""
    for intento in range(CONFIG["reintentos"]):
        try:
            with urllib.request.urlopen(peticion, timeout=CONFIG["timeout_seg"]) as respuesta:
                datos = json.loads(respuesta.read().decode("utf-8"))
            return datos["results"]["bindings"]
        except Exception as error:  # noqa: BLE001 - se registra y se reintenta
            ultimo_error = f"{type(error).__name__}: {error}"
            time.sleep(CONFIG["espera_reintento_seg"] * (intento + 1))
    raise RuntimeError(f"SPARQL fallo tras {CONFIG['reintentos']} intentos: {ultimo_error}")


def cargar_clases_efectivas() -> list[str]:
    """Lee las clases efectivas del M4 desde el parquet de etiquetas.

    Aplica la misma regla de agrupacion de 06 (excluye sin_area_definida y
    agrupa en otras_areas las clases con <3 documentos).

    Returns:
        Lista ordenada de clases efectivas.

    Raises:
        ValueError: Si las clases efectivas no coinciden con CLASES_M4.
    """
    tabla = pq.read_table(CONFIG["ruta_etiquetas"], columns=["area_primaria"])
    df = tabla.to_pandas()
    df = df[df["area_primaria"] != "sin_area_definida"]
    conteo = df["area_primaria"].value_counts()
    df.loc[df["area_primaria"].isin(conteo[conteo < 3].index), "area_primaria"] = "otras_areas"
    clases = sorted(df["area_primaria"].unique())
    if clases != sorted(CLASES_M4):
        raise ValueError(f"clases del parquet distintas de CLASES_M4: {clases}")
    return clases


def validar_macro_grupos() -> None:
    """Valida que los macro-grupos EuroVoc sean una particion de CLASES_M4.

    Raises:
        ValueError: Si hay clases repetidas o la cobertura no es exacta.
    """
    cubiertas = [clase for grupo in MACRO_GRUPOS_EUROVOC.values() for clase in grupo["clases"]]
    if len(cubiertas) != len(set(cubiertas)):
        raise ValueError("hay clases repetidas en MACRO_GRUPOS_EUROVOC")
    if set(cubiertas) != set(CLASES_M4):
        raise ValueError(
            f"MACRO_GRUPOS_EUROVOC no cubre CLASES_M4: faltan="
            f"{sorted(set(CLASES_M4) - set(cubiertas))} sobran="
            f"{sorted(set(cubiertas) - set(CLASES_M4))}"
        )


def descargar_etiquetas_raiz(ids: list[str]) -> dict[str, str]:
    """Descarga los skos:prefLabel en espanol de los conceptos raiz.

    Args:
        ids: IDs numericos de concepto EuroVoc.

    Returns:
        Dict id -> prefLabel en espanol.

    Raises:
        ValueError: Si algun ID no tiene prefLabel espanol.
    """
    valores = " ".join(f"<http://eurovoc.europa.eu/{i}>" for i in ids)
    consulta = (
        f"{PREFIJOS_SPARQL} SELECT ?c ?l WHERE {{ VALUES ?c {{ {valores} }} "
        "?c skos:prefLabel ?l . FILTER(lang(?l)='es') } ORDER BY ?c"
    )
    etiquetas: dict[str, str] = {}
    for fila in consultar_sparql(consulta):
        etiquetas[fila["c"]["value"].rsplit("/", 1)[-1]] = fila["l"]["value"]
    faltantes = [i for i in ids if i not in etiquetas]
    if faltantes:
        raise ValueError(f"conceptos sin prefLabel espanol: {faltantes}")
    return etiquetas


def descargar_cierre(ids: list[str]) -> dict[str, list[str]]:
    """Descarga las etiquetas espanolas del cierre de descendientes de cada raiz.

    Usa skos:broader+ (propiedad transitiva) y recolecta skos:prefLabel y
    skos:altLabel en espanol. La raiz no se incluye: se agrega aparte desde
    descargar_etiquetas_raiz.

    Args:
        ids: IDs numericos de concepto EuroVoc.

    Returns:
        Dict id -> lista de etiquetas en espanol (sin duplicados).
    """
    valores = " ".join(f"<http://eurovoc.europa.eu/{i}>" for i in ids)
    consulta = (
        f"{PREFIJOS_SPARQL} SELECT ?raiz ?d ?l WHERE {{ VALUES ?raiz {{ {valores} }} "
        "?d skos:broader+ ?raiz . { ?d skos:prefLabel ?l } UNION { ?d skos:altLabel ?l } "
        "FILTER(lang(?l)='es') }"
    )
    cierres: dict[str, list[str]] = {i: [] for i in ids}
    for fila in consultar_sparql(consulta):
        raiz = fila["raiz"]["value"].rsplit("/", 1)[-1]
        cierres[raiz].append(fila["l"]["value"])
    return {i: list(dict.fromkeys(etiquetas)) for i, etiquetas in cierres.items()}


def lematizar_terminos(nlp, terminos: list[str]) -> tuple[list[str], int]:
    """Lematiza terminos con la regla exacta de 05 (es_core_news_md).

    Regla: por cada token con tok.is_alpha se toma tok.lemma_.lower().strip()
    y se conserva si no es vacio; los tokens se unen por espacio.

    Args:
        nlp: Pipeline de spaCy cargado.
        terminos: Etiquetas crudas en espanol.

    Returns:
        Tupla (terminos lematizados unicos, n descartados sin lemas).
    """
    lematizados: list[str] = []
    descartados = 0
    for doc in nlp.pipe(terminos, batch_size=128):
        lemas = [tok.lemma_.lower().strip() for tok in doc if tok.is_alpha]
        lemas = [lema for lema in lemas if lema]
        if not lemas:
            descartados += 1
            continue
        lematizados.append(" ".join(lemas))
    return list(dict.fromkeys(lematizados)), descartados


def construir_lexico(
    nlp,
    etiquetas_raiz: dict[str, str],
    cierres: dict[str, list[str]],
) -> dict[str, dict]:
    """Construye el lexico lematizado por clase M4.

    Args:
        nlp: Pipeline de spaCy cargado.
        etiquetas_raiz: Dict id -> prefLabel espanol de la raiz.
        cierres: Dict id -> etiquetas del cierre de descendientes.

    Returns:
        Dict clase -> {terminos, terminos_por_raiz, n_terminos, max_tokens,
        n_descartados_por_longitud, n_descartados_sin_lemas}.
    """
    lexico: dict[str, dict] = {}
    for clase, datos in MAPPING_CLASES.items():
        por_raiz: dict[str, list[str]] = {}
        n_descartados = 0
        for raiz in datos["raices"]:
            etiquetas = [etiquetas_raiz[raiz]] + list(cierres.get(raiz, []))
            etiquetas = list(dict.fromkeys(etiquetas))
            lematizados, descartados = lematizar_terminos(nlp, etiquetas)
            por_raiz[raiz] = lematizados
            n_descartados += descartados
        union = list(dict.fromkeys(t for raiz in datos["raices"] for t in por_raiz[raiz]))
        por_longitud = [t for t in union if len(t.split()) > CONFIG["max_tokens_termino"]]
        terminos = [t for t in union if len(t.split()) <= CONFIG["max_tokens_termino"]]
        lexico[clase] = {
            "terminos": terminos,
            "terminos_por_raiz": {
                raiz: [t for t in por_raiz[raiz] if len(t.split()) <= CONFIG["max_tokens_termino"]]
                for raiz in datos["raices"]
            },
            "n_terminos": len(terminos),
            "max_tokens": max((len(t.split()) for t in terminos), default=0),
            "n_descartados_por_longitud": len(por_longitud),
            "n_descartados_sin_lemas": n_descartados,
        }
    return lexico


def guardar_json(datos: dict, ruta: str) -> None:
    """Guarda un dict como JSON legible con UTF-8.

    Args:
        datos: Objeto serializable.
        ruta: Ruta de destino.
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)


def main() -> None:
    """Descarga EuroVoc, construye mapeo + lexico y escribe los dos JSON."""
    t0 = time.time()
    validar_macro_grupos()
    cargar_clases_efectivas()
    raices = sorted({raiz for datos in MAPPING_CLASES.values() for raiz in datos["raices"]}, key=int)
    print(f"clases M4: {len(CLASES_M4)}; conceptos raiz EuroVoc: {len(raices)}")
    etiquetas_raiz = descargar_etiquetas_raiz(raices)
    print("prefLabels verificados:", ", ".join(f"{r}={etiquetas_raiz[r]}" for r in raices[:5]), "...")
    cierres = descargar_cierre(raices)
    print("cierres (n etiquetas crudas):", {r: len(cierres[r]) for r in raices})
    nlp = spacy.load(CONFIG["modelo_spacy"])
    nlp.max_length = 1_000_000
    lexico = construir_lexico(nlp, etiquetas_raiz, cierres)
    n_total = sum(d["n_terminos"] for d in lexico.values())
    print(f"lexico lematizado: {n_total} terminos totales en {len(lexico)} clases")

    mapeo = {
        "fecha": pd.Timestamp.now().isoformat(),
        "fase": "ciclo v2 - Fase W(b) EuroVoc-es",
        "fuente": {
            "endpoint_sparql": CONFIG["endpoint"],
            "esquema": f"{CONFIG['esquema_uri']} (EuroVoc)",
            "dominios_verificados": DOMINIOS_EUROVOC,
            "metodo": (
                "skos:prefLabel/skos:altLabel con FILTER(lang='es') y cierre de "
                "descendientes con skos:broader+ (consulta 2026-09-11); "
                "lematizacion identica a 05 (es_core_news_md, is_alpha, "
                "lemma_.lower().strip())"
            ),
            "n_conceptos_raiz": len(raices),
        },
        "viabilidad_terminos_es": {
            "viable": True,
            "n_terminos_lematizados": n_total,
            "detalle": (
                "EuroVoc publica SKOS en espanol en el endpoint oficial; el "
                "cierre de descendientes de los conceptos raiz del mapeo "
                "produjo terminos suficientes para features lexicas."
            ),
        },
        "cobertura": {
            "anticipada": (
                "E1 (nota_ontologias.md): EuroVoc cubre derecho de la UE, no la "
                "CP colombiana; varios bloques solo tienen correspondencia "
                "conceptual (p.ej. organismos_control, DESC, principios)."
            ),
            "mapeo_clase_categoria": {
                clase: {
                    "categorias": [
                        {
                            "id": raiz,
                            "uri": f"http://eurovoc.europa.eu/{raiz}",
                            "prefLabel_es": etiquetas_raiz[raiz],
                            "n_terminos_cierre": len(cierres.get(raiz, [])),
                        }
                        for raiz in datos["raices"]
                    ],
                    "justificacion": datos["justificacion"],
                }
                for clase, datos in MAPPING_CLASES.items()
            },
        },
        "macro_grupos": MACRO_GRUPOS_EUROVOC,
        "nota_uso": (
            "Variante D: los macro-grupos de la cascada jerarquica usan estos "
            "grupos. Variante E: features de conteo normalizado de los terminos "
            "de config/eurovoc_terminos_es.json concatenadas al TF-IDF v1. La "
            "senal SOLO se adopta si gana en validacion/test (PLAN_v2 Fase W)."
        ),
    }
    terminos = {
        "fecha": pd.Timestamp.now().isoformat(),
        "fase": "ciclo v2 - Fase W(b) EuroVoc-es",
        "fuente": CONFIG["endpoint"],
        "max_tokens_termino": CONFIG["max_tokens_termino"],
        "terminos_por_clase": {
            clase: {
                "terminos": datos["terminos"],
                "n_terminos": datos["n_terminos"],
                "max_tokens": datos["max_tokens"],
                "n_descartados_por_longitud": datos["n_descartados_por_longitud"],
                "n_descartados_sin_lemas": datos["n_descartados_sin_lemas"],
                "terminos_por_raiz": datos["terminos_por_raiz"],
            }
            for clase, datos in lexico.items()
        },
        "nota": (
            "Terminos lematizados con es_core_news_md (regla de 05); se "
            "descartan etiquetas de mas de max_tokens_termino tokens por costo "
            "de n-gramas en CountVectorizer."
        ),
    }
    guardar_json(mapeo, CONFIG["ruta_bloques"])
    guardar_json(terminos, CONFIG["ruta_terminos"])
    print(f"OK en {time.time() - t0:.1f} s: {CONFIG['ruta_bloques']} y {CONFIG['ruta_terminos']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
