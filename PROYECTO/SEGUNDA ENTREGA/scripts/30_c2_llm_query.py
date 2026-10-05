"""30_c2_llm_query.py
Fecha: 2026-10-04
Descripcion: C2 - query understanding con el LLM legal local alia-legal
    (ALIA-es-legal-administrative-7B-Instruct, GGUF Q4_K_M servido por
    ollama en http://127.0.0.1:11434). Hipotesis C2: la consulta de una
    sentencia completa (~68k caracteres) diluye la senal del retrieval;
    un LLM que extrae problema juridico + hechos + pretensiones y redacta
    una consulta corta y densa (<=300 palabras, terminos juridicos y
    constitucionales) mejora el recall de la 1a etapa y el nDCG@10 frente
    a la ruta v13 con la consulta completa. Modos: --extraer (llama a
    ollama por caso y escribe un JSONL resumible), --evaluar (ruta
    completa con la consulta del LLM frente a la ruta v13 recalculada con
    el texto completo, comparacion PAREADA por caso) y --todo. Con
    --smoke/--limite todas las salidas van a /tmp/opencode/c2_smoke/ y
    nunca se sobrescriben los artefactos reales. La extraccion es
    resumible (JSONL append por caso); si la consulta devuelta copia las
    instrucciones del prompt o es trivial se hace un turno correctivo y,
    si aun falla, el caso queda pendiente para resume.
Estado del arte aplicado: patron query understanding + retrieval + rerank
    de los pipelines de statute retrieval (LegalMALR arXiv:2601.17692;
    COLIEE arXiv:2607.11400). Se reutiliza por importlib la maquinaria de
    scripts/12_evaluar_vinculo_cp_v2.py: pasajes permanentes de la CP,
    chunking BM25 800/200 chars, hibrido pre-registrado 0.5*BM25 + 0.5*B2
    sobre top-50 candidatos, rerank B1 por chunks con max-pooling, prior
    de hubs alpha=1.5, salida k=10 y bootstrap no parametrico
    (1000 remuestreos, seed 42). La comparacion pareada vs v13 RECALCULA
    la ruta con el texto completo en la misma corrida (misma maquinaria y
    mismos casos) y se valida contra el per-case congelado de
    comparacion_b2 (contexto con_hubs) del benchmark v13.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import re
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch

CONFIG = {
    "version": "c2-query-llm-v1",
    "semilla": 42,
    "modelo_llm": "alia-legal",
    "host_ollama": "http://127.0.0.1:11434",
    "temperature": 0.0,
    "num_ctx": 16384,
    "num_predict": 700,
    "think": False,
    "max_chars_entrada": 24000,
    "max_palabras_consulta": 300,
    "timeout_conexion_seg": 10,
    "timeout_lectura_seg": 600,
    "reintentos": 2,
    "backoff_base_seg": 2,
    "smoke_limite": 5,
    "ruta_script_12": "scripts/12_evaluar_vinculo_cp_v2.py",
    "ruta_benchmark_v2": "resultados/vinculo_cp_benchmark_v2.json",
    "ruta_benchmark_v13": "resultados/vinculo_cp_benchmark_v13.json",
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "estrategia": "hibrido_b2_rerank_chunks_top50_b1",
    "alpha_hubs": 1.5,
    "k_candidatos": 50,
    "k_medidas": 5,
    "k_ndcg": 10,
    "bootstrap": 1000,
    "contexto_v13_congelado": "con_hubs",
    "tolerancia_consistencia": 1e-3,
    "dir_salida": "resultados/c2_llm",
    "ruta_json": "resultados/c2_llm/eval_query_llm.json",
    "ruta_figura": "figuras/c2_query_llm.png",
    "ruta_log": "experimentos/30_c2_llm_query.log",
    "dir_smoke": "/tmp/opencode/c2_smoke",
}

ROOT = Path(__file__).resolve().parent.parent

# Medidas exigidas por la spec C2 (las cinco del protocolo M7 mas R@5).
MEDIDAS = (
    "ndcg_at_10",
    "f1_at_5",
    "f1_at_10",
    "mrr",
    "p_at_5",
    "r_at_5",
)

# Campos que el LLM debe devolver en el JSON de extraccion.
CAMPOS_LLM = (
    "consulta_busqueda",
    "problema_juridico",
    "hechos",
    "pretensiones",
)

PROMPT_SISTEMA = (
    "Eres un asistente juridico experto en derecho constitucional "
    "colombiano. Analizas sentencias de la Corte Constitucional de "
    "Colombia y respondes UNICAMENTE con un objeto JSON valido, sin texto "
    "adicional ni bloques de codigo.\n"
    "El JSON debe tener exactamente estas claves:\n"
    '{"consulta_busqueda": "...", "problema_juridico": "...", '
    '"hechos": "...", "pretensiones": "..."}\n'
    "Reglas:\n"
    "- consulta_busqueda: consulta de busqueda de MAXIMO 300 palabras, "
    "formada solo por terminos juridicos y constitucionales (derechos "
    "fundamentales, principios, acciones constitucionales, instituciones y "
    "garantias) que permitan recuperar los articulos de la Constitucion "
    "Politica de Colombia aplicables al caso. No incluyas numeros de "
    "sentencias, nombres propios ni datos procesales.\n"
    "- problema_juridico: problema juridico constitucional que resuelve la "
    "sentencia (maximo 120 palabras).\n"
    "- hechos: hechos relevantes del caso (maximo 150 palabras).\n"
    "- pretensiones: lo que solicita el demandante (maximo 100 palabras).\n"
    "- Si un campo no se puede determinar, usa una cadena vacia.\n"
    "- No copies estas instrucciones ni el esquema en los valores de los "
    "campos; redacta contenido nuevo a partir del fragmento.\n"
    "- Devuelve solo el JSON, sin explicaciones."
)

PROMPT_CORRECCION = (
    "Tu respuesta anterior no es valida: repetiste instrucciones o dejo "
    "vacio el campo consulta_busqueda. Devuelve SOLO el objeto JSON con "
    "los cuatro campos; en consulta_busqueda redacta una consulta nueva "
    "con los terminos juridicos y constitucionales del caso, sin copiar "
    "estas instrucciones."
)

# Marcas de fuga de instrucciones al campo consulta_busqueda (el modelo
# ocasionalmente copia las reglas del prompt en lugar de redactar).
MARCAS_FUGA_INSTRUCCION = (
    "maximo 300 palabras",
    "consulta de busqueda de",
    "formada solo por terminos juridicos",
    "no incluyas numeros de sentencias",
    "no copies estas instrucciones",
)


class _DuplicadorSalida:
    """Duplica stdout y stderr hacia el archivo de log de la corrida.

    Los print() del pipeline (incluidos los de las funciones reutilizadas
    del script 12) y los tracebacks de error quedan asi registrados en el
    log. El flujo original se captura al construir el objeto, de modo que
    no hay recursion aunque se instale en ambos canales.
    """

    def __init__(self, archivo) -> None:
        """Inicializa el duplicador.

        Args:
            archivo: Flujo de texto abierto en modo escritura.
        """
        self._archivo = archivo
        self._original = sys.stdout

    def write(self, datos: str) -> None:
        """Escribe en consola y en el archivo de log.

        Args:
            datos: Texto a escribir.
        """
        self._original.write(datos)
        self._archivo.write(datos)

    def flush(self) -> None:
        """Fuerza el vaciado de ambos flujos."""
        self._original.flush()
        self._archivo.flush()


# =====================================================================
# SECCION 1: CONFIGURACION, INSUMOS Y UTILIDADES
# =====================================================================
def importar_evaluador_v12(ruta: Path) -> object:
    """Importa scripts/12_evaluar_vinculo_cp_v2.py por importlib.

    El nombre del archivo empieza por digito y no admite import estandar.
    Se reutilizan sin duplicar logica: resolver_rutas, configurar_semillas,
    leer_filas_por_id, preparar_pasajes_constitucion, chunks_query,
    construir_indice_bm25, scores_bm25_caso, ranking_desde_scores,
    ranking_rerank_chunks, cargar_o_construir_prior_hubs, ranking_con_prior,
    metricas_ranking, bootstrap_ic_media, bootstrap_ic_pareado,
    asignar_periodo, asignar_terciles_longitud, resolver_modelo_b1/b2,
    aplicar_modo_b2 y calcular_denso_b2.

    Args:
        ruta: Ruta absoluta del script 12.

    Returns:
        Modulo del evaluador v12 con CONFIG resuelto contra su ROOT.

    Raises:
        FileNotFoundError: Si la ruta del script 12 no existe.
    """
    if not ruta.is_file():
        raise FileNotFoundError(f"no existe el evaluador v12: {ruta}")
    spec = importlib.util.spec_from_file_location("evaluador_v12", ruta)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el evaluador v12: {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    modulo.resolver_rutas()
    return modulo


def exigir_archivo(ruta: Path, descripcion: str) -> None:
    """Verifica que un insumo exista (fail-fast antes de usar GPU).

    Args:
        ruta: Ruta del archivo requerido.
        descripcion: Nombre legible del insumo para el mensaje de error.

    Raises:
        FileNotFoundError: Si la ruta no es un archivo.
    """
    if not ruta.is_file():
        raise FileNotFoundError(f"falta {descripcion}: {ruta}")


def cargar_benchmark(ruta: Path, claves: tuple) -> dict:
    """Carga un benchmark congelado (solo lectura).

    Args:
        ruta: Ruta del JSON del benchmark.
        claves: Secciones requeridas en el JSON.

    Returns:
        Dict del benchmark.

    Raises:
        ValueError: Si falta alguna seccion requerida.
    """
    with ruta.open("r", encoding="utf-8") as manejador:
        datos = json.load(manejador)
    for clave in claves:
        if clave not in datos:
            raise ValueError(f"{ruta.name} sin la seccion requerida: {clave}")
    return datos


def normalizar_muestra(bench: dict) -> list[dict]:
    """Normaliza la muestra de un benchmark a ids/tipos/anios strings.

    Args:
        bench: Benchmark con la seccion muestra (id, tipo, ano, arts_cp).

    Returns:
        Lista de casos {id, tipo, ano, arts} con canones como strings.
    """
    return [
        {
            "id": str(caso["id"]),
            "tipo": str(caso.get("tipo", "")),
            "ano": int(caso["ano"]),
            "arts": [str(canon) for canon in caso.get("arts_cp", [])],
        }
        for caso in bench["muestra"]
    ]


def recortar_muestra(muestra: list[dict], limite: int | None) -> list[dict]:
    """Recorta la muestra a los primeros N casos (modo smoke/limite).

    Args:
        muestra: Muestra normalizada en orden del benchmark.
        limite: Numero maximo de casos; None conserva la muestra completa.

    Returns:
        Muestra recortada (lista nueva si hubo recorte).
    """
    if limite is None or limite >= len(muestra):
        return list(muestra)
    return list(muestra[:limite])


def leer_textos_por_id(mod: object, ruta: Path, ids: list[str]) -> dict:
    """Lee el texto crudo y el limpio del corpus para los ids dados.

    Args:
        mod: Modulo del evaluador v12 (leer_filas_por_id con pushdown).
        ruta: Ruta del parquet del corpus.
        ids: Ids de sentencia a leer.

    Returns:
        Dict {id: {"texto": crudo, "texto_limpio": E1}} de cada id. La
        ruta v13 usa texto_limpio como ventana de query (fidelidad al
        benchmark congelado); el LLM recibe el texto crudo.

    Raises:
        RuntimeError: Si algun id no aparece o su texto esta vacio.
    """
    tabla = mod.leer_filas_por_id(
        str(ruta), ["id", "texto", "texto_limpio"], ids
    )
    textos = {
        str(sid): {"texto": str(crudo), "texto_limpio": str(limpio)}
        for sid, crudo, limpio in zip(
            tabla["id"], tabla["texto"], tabla["texto_limpio"], strict=True
        )
    }
    faltantes = [
        sid
        for sid in ids
        if not textos.get(sid, {}).get("texto", "").strip()
        or not textos.get(sid, {}).get("texto_limpio", "").strip()
    ]
    if faltantes:
        raise RuntimeError(f"casos sin texto en el corpus: {faltantes[:5]}")
    return textos


def cargar_queries_jsonl(ruta: Path) -> dict:
    """Carga el JSONL de consultas extraidas indexado por id.

    Args:
        ruta: Ruta de resultados/c2_llm/queries.jsonl.

    Returns:
        Dict {id: fila}; vacio si el archivo no existe.
    """
    if not ruta.is_file():
        return {}
    filas: dict = {}
    with ruta.open("r", encoding="utf-8") as manejador:
        for linea in manejador:
            linea = linea.strip()
            if not linea:
                continue
            datos = json.loads(linea)
            filas[str(datos["id"])] = datos
    return filas


def agregar_query_jsonl(ruta: Path, fila: dict) -> None:
    """Anade una fila al JSONL de consultas (append por caso, resumible).

    Args:
        ruta: Ruta del JSONL.
        fila: Fila serializable de la extraccion.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8") as manejador:
        manejador.write(json.dumps(fila, ensure_ascii=False) + "\n")


def guardar_json(ruta: Path, contenido: dict, sobrescribir: bool) -> None:
    """Guarda un JSON UTF-8 con indentacion (con guard de no sobrescritura).

    Args:
        ruta: Ruta de salida.
        contenido: Dict serializable.
        sobrescribir: True para artefactos de progreso o modo smoke.

    Raises:
        FileExistsError: Si la ruta existe y no se permite sobrescribir.
    """
    if ruta.exists() and not sobrescribir:
        raise FileExistsError(f"no se sobrescribe {ruta}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as manejador:
        json.dump(contenido, manejador, ensure_ascii=False, indent=2)


def _md5(ruta: Path) -> str:
    """Calcula el md5 de un archivo (trazabilidad de artefactos).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o cadena vacia si el archivo no existe.
    """
    if not ruta.is_file():
        return ""
    return hashlib.md5(ruta.read_bytes()).hexdigest()


# =====================================================================
# SECCION 2: EXTRACCION LLM CON OLLAMA (RESUMIBLE)
# =====================================================================
def construir_prompt_usuario(texto: str) -> str:
    """Construye el mensaje de usuario con el fragmento de sentencia.

    Args:
        texto: Fragmento (ya recortado) de la sentencia.

    Returns:
        Prompt de usuario en espanol.
    """
    return (
        "Analiza el siguiente fragmento de una sentencia de la Corte "
        "Constitucional de Colombia y devuelve el objeto JSON.\n\n"
        f"FRAGMENTO ({len(texto)} caracteres):\n{texto}\n\nJSON:"
    )


def construir_payload(modelo: str, texto: str) -> dict:
    """Construye el cuerpo del POST /api/chat con opciones reproducibles.

    Se envia think=false porque la spec lo pide para modelos con modo
    pensamiento; el llamador tolerante reintenta sin el si el modelo no lo
    soporta. format="json" activa la gramatica JSON de ollama (el parser
    igualmente tolera cercas de codigo y respuestas parciales).

    Args:
        modelo: Nombre del modelo en ollama.
        texto: Fragmento de sentencia que ve el modelo.

    Returns:
        Dict serializable con model, messages, format, stream, think y
        options (temperature 0, seed 42, num_ctx 16384, num_predict 700).
    """
    return {
        "model": modelo,
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": construir_prompt_usuario(texto)},
        ],
        "format": "json",
        "stream": False,
        "think": CONFIG["think"],
        "options": {
            "temperature": CONFIG["temperature"],
            "seed": CONFIG["semilla"],
            "num_ctx": CONFIG["num_ctx"],
            "num_predict": CONFIG["num_predict"],
        },
    }


def verificar_modelo_disponible(host: str, modelo: str) -> None:
    """Comprueba que ollama responde y que el modelo pedido esta disponible.

    Args:
        host: URL base de ollama.
        modelo: Nombre del modelo a usar.

    Raises:
        RuntimeError: Si ollama no responde o el modelo no aparece en
            /api/tags.
    """
    try:
        respuesta = requests.get(
            f"{host}/api/tags",
            timeout=(
                CONFIG["timeout_conexion_seg"],
                CONFIG["timeout_lectura_seg"],
            ),
        )
        respuesta.raise_for_status()
        nombres = [
            str(item.get("name", ""))
            for item in respuesta.json().get("models", [])
        ]
    except Exception as exc:
        raise RuntimeError(
            f"ollama no responde en {host}: {type(exc).__name__}: {exc}"
        ) from exc
    if not any(
        nombre == modelo or nombre.split(":")[0] == modelo
        for nombre in nombres
    ):
        raise RuntimeError(
            f"modelo {modelo!r} no disponible en {host}; "
            f"disponibles: {sorted(nombres)}"
        )


def _post_ollama(host: str, payload: dict) -> tuple[str, dict]:
    """Hace un POST /api/chat y devuelve el contenido textual y el JSON.

    Args:
        host: URL base de ollama.
        payload: Cuerpo del POST.

    Returns:
        Tupla (contenido, respuesta_completa).

    Raises:
        requests.RequestException: Si falla el transporte.
        RuntimeError: Si el HTTP no es 200, ollama reporta error o la
            respuesta no trae contenido.
    """
    respuesta = requests.post(
        f"{host}/api/chat",
        json=payload,
        timeout=(
            CONFIG["timeout_conexion_seg"],
            CONFIG["timeout_lectura_seg"],
        ),
    )
    if respuesta.status_code != 200:
        raise RuntimeError(
            f"HTTP {respuesta.status_code}: {respuesta.text[:200]}"
        )
    datos = respuesta.json()
    if isinstance(datos, dict) and datos.get("error"):
        raise RuntimeError(f"ollama error: {datos['error']}")
    mensaje = datos.get("message") or {}
    contenido = str(mensaje.get("content") or "").strip()
    if not contenido and mensaje.get("thinking"):
        contenido = str(mensaje["thinking"]).strip()
    if not contenido:
        raise RuntimeError("respuesta sin message.content")
    return contenido, datos


def llamar_ollama_tolerante(host: str, payload: dict) -> tuple[str, dict]:
    """Llama a ollama con cadena de variantes tolerante a modelos no-qwen.

    Variantes en orden: (1) payload completo (con think=false y
    format="json"), (2) sin la clave think, (3) sin think ni format. Los
    errores de transporte se reintentan por variante con backoff; los
    errores HTTP/ollama pasan a la siguiente variante.

    Args:
        host: URL base de ollama.
        payload: Payload base de construir_payload.

    Returns:
        Tupla (contenido, respuesta_completa) del primer intento exitoso.

    Raises:
        RuntimeError: Si todas las variantes fallan.
    """
    sin_think = {
        clave: valor for clave, valor in payload.items() if clave != "think"
    }
    sin_think_ni_format = {
        clave: valor
        for clave, valor in sin_think.items()
        if clave != "format"
    }
    variantes = (
        ("con think=false", payload),
        ("sin think", sin_think),
        ("sin think ni format", sin_think_ni_format),
    )
    ultimo_error = "sin intentos"
    for etiqueta, cuerpo in variantes:
        for intento in range(CONFIG["reintentos"] + 1):
            try:
                return _post_ollama(host, cuerpo)
            except requests.RequestException as exc:
                ultimo_error = f"{etiqueta}: {type(exc).__name__}: {exc}"
                if intento < CONFIG["reintentos"]:
                    time.sleep(CONFIG["backoff_base_seg"] ** (intento + 1))
                    continue
                break
            except RuntimeError as exc:
                ultimo_error = f"{etiqueta}: {exc}"
                break
        print(f"[aviso] variante ollama fallida ({etiqueta}): {ultimo_error}")
    raise RuntimeError(f"ollama no respondio: {ultimo_error}")


def limpiar_cerca(texto: str) -> str:
    """Quita cercas de codigo markdown alrededor de un JSON.

    Args:
        texto: Respuesta cruda del modelo.

    Returns:
        Texto sin las cercas ```json ... ``` (si estaban presentes).
    """
    limpio = texto.strip()
    if limpio.startswith("```"):
        limpio = re.sub(r"^```[a-zA-Z]*\s*", "", limpio)
        limpio = re.sub(r"\s*```$", "", limpio)
    return limpio.strip()


def _intentar_json(texto: str) -> dict | None:
    """Intenta json.loads y devuelve None si no es un dict valido.

    Args:
        texto: Texto candidato.

    Returns:
        Dict decodificado o None.
    """
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError:
        return None
    return datos if isinstance(datos, dict) else None


def _extraer_bloque(texto: str) -> str | None:
    """Extrae el primer bloque {...} balanceado por conteo de llaves.

    Args:
        texto: Texto donde buscar.

    Returns:
        Substring del bloque o None si no hay llaves.
    """
    inicio = texto.find("{")
    if inicio < 0:
        return None
    profundidad = 0
    en_cadena = False
    escape = False
    for posicion in range(inicio, len(texto)):
        caracter = texto[posicion]
        if en_cadena:
            if escape:
                escape = False
            elif caracter == "\\":
                escape = True
            elif caracter == '"':
                en_cadena = False
            continue
        if caracter == '"':
            en_cadena = True
        elif caracter == "{":
            profundidad += 1
        elif caracter == "}":
            profundidad -= 1
            if profundidad == 0:
                return texto[inicio : posicion + 1]
    return texto[inicio:]


def _extraer_campo_regex(texto: str, clave: str) -> str:
    """Extrae el valor string de una clave por regex (ultimo recurso).

    Args:
        texto: Respuesta cruda (posiblemente truncada).
        clave: Nombre del campo.

    Returns:
        Valor desescapado o cadena vacia.
    """
    patron = rf'"{re.escape(clave)}"\s*:\s*"(.*?)"'
    coincidencia = re.search(patron, texto, re.DOTALL)
    if coincidencia is None:
        return ""
    crudo = coincidencia.group(1)
    try:
        return str(json.loads(f'"{crudo}"'))
    except json.JSONDecodeError:
        return crudo


def parsear_respuesta(crudo: str) -> dict:
    """Parsea la respuesta del LLM al esquema de cuatro campos.

    Cadena tolerante: json.loads directo, bloque {...} balanceado y, como
    ultimo recurso, regex por campo (cubre truncamientos de num_predict).

    Args:
        crudo: Respuesta textual del modelo.

    Returns:
        Dict con las claves de CAMPOS_LLM normalizadas a string.

    Raises:
        ValueError: Si no se pudo recuperar una consulta_busqueda no vacia.
    """
    texto = limpiar_cerca(crudo)
    datos = _intentar_json(texto)
    if datos is None:
        bloque = _extraer_bloque(texto)
        if bloque is not None:
            datos = _intentar_json(bloque)
    if datos is None:
        datos = {}
        for clave in CAMPOS_LLM:
            valor = _extraer_campo_regex(texto, clave)
            if valor:
                datos[clave] = valor
    normalizado = {
        clave: str(datos.get(clave) or "").strip() for clave in CAMPOS_LLM
    }
    if not normalizado["consulta_busqueda"]:
        raise ValueError(
            f"respuesta sin consulta_busqueda parseable: {crudo[:200]!r}"
        )
    return normalizado


def _sin_tildes(texto: str) -> str:
    """Convierte a minusculas y quita diacriticos (comparacion robusta).

    Args:
        texto: Texto original.

    Returns:
        Texto normalizado.
    """
    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", texto)
        if not unicodedata.combining(caracter)
    ).lower()


def consulta_es_util(consulta: str) -> bool:
    """Valida que la consulta_busqueda sea util (no fuga ni trivial).

    Rechaza consultas de menos de 3 palabras y las que copian las
    instrucciones del prompt (fuga observada en el smoke: el modelo repite
    las reglas en lugar de redactar la consulta).

    Args:
        consulta: Campo consulta_busqueda devuelto por el modelo.

    Returns:
        True si la consulta es utilizable.
    """
    if len(consulta.split()) < 3:
        return False
    normalizada = _sin_tildes(consulta)
    return not any(marca in normalizada for marca in MARCAS_FUGA_INSTRUCCION)


def construir_consulta_fallback(campos: dict) -> str:
    """Construye una consulta de respaldo con los campos extraidos.

    Si el modelo no logra redactar consulta_busqueda pero si extrae
    problema_juridico, hechos y pretensiones, la consulta se forma con
    esos campos (maximo CONFIG['max_palabras_consulta'] palabras),
    coherente con la premisa C2 de limitarse a la extraccion de
    hechos/problema cuando la redaccion de la consulta falla.

    Args:
        campos: Dict con los cuatro campos normalizados.

    Returns:
        Consulta de respaldo (cadena vacia si no hay contenido util).
    """
    partes = [
        campos["problema_juridico"],
        campos["hechos"],
        campos["pretensiones"],
    ]
    palabras = " ".join(partes).split()
    return " ".join(palabras[: CONFIG["max_palabras_consulta"]])


def extraer_caso(caso: dict, texto: str, host: str, modelo: str) -> dict:
    """Extrae los cuatro campos LLM de un caso y construye su fila JSONL.

    El texto se recorta a CONFIG['max_chars_entrada'] caracteres (la
    cabecera y los antecedentes concentran hechos, pretensiones y problema
    juridico; el recorte mantiene el prompt dentro de num_ctx=16384). Si la
    consulta devuelta no pasa el guard de utilidad (fuga de instrucciones o
    trivial), se hace UN turno correctivo reutilizando la conversacion; si
    aun falla, se usa como respaldo la concatenacion de problema_juridico,
    hechos y pretensiones (consulta_fuente="fallback_campos"). Solo si
    tampoco hay contenido util el caso falla y queda pendiente para resume.

    Args:
        caso: Caso normalizado del benchmark (id, tipo, ano).
        texto: Texto completo de la sentencia (crudo).
        host: URL base de ollama.
        modelo: Nombre del modelo en ollama.

    Returns:
        Fila serializable con id, consulta_busqueda, problema_juridico,
        hechos, pretensiones, modelo, fecha, tiempo_seg y trazabilidad.

    Raises:
        RuntimeError: Si ollama falla tras la cadena de variantes.
        ValueError: Si la respuesta no trae una consulta_busqueda util.
    """
    inicio = time.time()
    recorte = texto[: CONFIG["max_chars_entrada"]]
    payload = construir_payload(modelo, recorte)
    crudo, _ = llamar_ollama_tolerante(host, payload)
    campos = parsear_respuesta(crudo)
    correccion_usada = False
    if not consulta_es_util(campos["consulta_busqueda"]):
        correccion_usada = True
        payload_correccion = dict(payload)
        payload_correccion["messages"] = [
            payload["messages"][0],
            payload["messages"][1],
            {"role": "assistant", "content": crudo[:4000]},
            {"role": "user", "content": PROMPT_CORRECCION},
        ]
        crudo, _ = llamar_ollama_tolerante(host, payload_correccion)
        campos = parsear_respuesta(crudo)
    consulta = " ".join(campos["consulta_busqueda"].split())
    consulta_fuente = "consulta_busqueda"
    if not consulta_es_util(consulta):
        consulta = construir_consulta_fallback(campos)
        consulta_fuente = "fallback_campos"
        if not consulta_es_util(consulta):
            raise ValueError(
                "sin consulta_busqueda util ni campos de respaldo: "
                f"{campos['consulta_busqueda'][:120]!r}"
            )
    n_palabras = len(consulta.split())
    return {
        "id": caso["id"],
        "consulta_busqueda": consulta,
        "consulta_fuente": consulta_fuente,
        "problema_juridico": " ".join(campos["problema_juridico"].split()),
        "hechos": " ".join(campos["hechos"].split()),
        "pretensiones": " ".join(campos["pretensiones"].split()),
        "modelo": modelo,
        "fecha": datetime.now(timezone.utc).isoformat(),
        "tiempo_seg": round(time.time() - inicio, 1),
        "n_chars_entrada": len(recorte),
        "n_palabras_consulta": n_palabras,
        "excede_max_palabras": n_palabras > CONFIG["max_palabras_consulta"],
        "correccion_usada": correccion_usada,
    }


def ejecutar_extraccion(
    mod: object, limite: int | None, rutas: dict
) -> dict:
    """Extrae consultas LLM para la muestra del benchmark (resumible).

    Estrategia: lee el JSONL existente, salta los ids ya extraidos y
    procesa los pendientes anadiendo una linea por caso. Los fallos por
    caso se registran y se dejan pendientes (no se escriben) para que una
    re-ejecucion los reintente. El meta se reescribe con el estado
    acumulado.

    Args:
        mod: Modulo del evaluador v12 (leer_filas_por_id).
        limite: Limite efectivo de casos (None en el run real).
        rutas: Dict de rutas resuelto por resolver_rutas_salida.

    Returns:
        Resumen de la extraccion (conteos, fallos, tiempos y pendientes).
    """
    inicio = time.time()
    host = CONFIG["host_ollama"]
    modelo = CONFIG["modelo_llm"]
    bench = cargar_benchmark(ROOT / CONFIG["ruta_benchmark_v2"], ("muestra",))
    muestra = recortar_muestra(normalizar_muestra(bench), limite)
    verificar_modelo_disponible(host, modelo)
    print(
        f"Extrayendo consultas LLM: {len(muestra)} casos | modelo "
        f"{modelo} | host {host}"
    )
    if limite is not None:
        print(
            f"[smoke] --limite/--smoke: muestra recortada a {len(muestra)} "
            "casos (primeros por orden del benchmark; NO es el run final)"
        )
    previos = cargar_queries_jsonl(rutas["queries"])
    pendientes = [caso for caso in muestra if caso["id"] not in previos]
    print(
        f"JSONL: {len(previos)} consultas previas | pendientes en esta "
        f"corrida: {len(pendientes)}"
    )
    fallos: list[dict] = []
    tiempos_nuevos: dict = {}
    if pendientes:
        textos = leer_textos_por_id(
            mod, ROOT / CONFIG["ruta_corpus"], [c["id"] for c in pendientes]
        )
        for posicion, caso in enumerate(pendientes, start=1):
            sid = caso["id"]
            try:
                fila = extraer_caso(caso, textos[sid]["texto"], host, modelo)
            except Exception as exc:  # noqa: BLE001 - guard por caso
                fallos.append(
                    {"id": sid, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"[fallo] caso {sid}: {type(exc).__name__}: {exc}")
                continue
            agregar_query_jsonl(rutas["queries"], fila)
            previos[sid] = fila
            tiempos_nuevos[sid] = fila["tiempo_seg"]
            marcas = []
            if fila.get("correccion_usada"):
                marcas.append("correccion")
            if fila.get("consulta_fuente") == "fallback_campos":
                marcas.append("fallback_campos")
            sufijo = f" ({', '.join(marcas)})" if marcas else ""
            print(
                f"[{posicion}/{len(pendientes)}] caso {sid}: "
                f"{fila['n_palabras_consulta']} palabras de consulta | "
                f"{fila['tiempo_seg']} s{sufijo}"
            )
    ids_benchmark = {caso["id"] for caso in normalizar_muestra(bench)}
    ids_jsonl = set(cargar_queries_jsonl(rutas["queries"]))
    pendientes_globales = sorted(ids_benchmark - ids_jsonl)
    tiempos_totales = [
        float(fila.get("tiempo_seg", 0.0)) for fila in previos.values()
    ]
    resumen = {
        "n_casos_benchmark": len(ids_benchmark),
        "n_muestra": len(muestra),
        "n_previos": len(muestra) - len(pendientes),
        "n_extraidos_nuevos": len(tiempos_nuevos),
        "n_fallos_corrida": len(fallos),
        "fallos_corrida": fallos,
        "n_consultas_jsonl": len(ids_jsonl),
        "n_correcciones": sum(
            1 for fila in previos.values() if fila.get("correccion_usada")
        ),
        "n_fallback_campos": sum(
            1
            for fila in previos.values()
            if fila.get("consulta_fuente") == "fallback_campos"
        ),
        "pendientes_benchmark": pendientes_globales,
        "tiempos_nuevos_seg": tiempos_nuevos,
        "media_tiempo_caso_seg": (
            round(float(np.mean(tiempos_totales)), 1)
            if tiempos_totales
            else None
        ),
        "segundos_corrida": round(time.time() - inicio, 1),
    }
    meta = {
        "fecha": datetime.now(timezone.utc).isoformat(),
        "version": CONFIG["version"],
        "script": "scripts/30_c2_llm_query.py",
        "modelo": modelo,
        "host": host,
        "config": {
            "temperature": CONFIG["temperature"],
            "seed": CONFIG["semilla"],
            "num_ctx": CONFIG["num_ctx"],
            "num_predict": CONFIG["num_predict"],
            "think": CONFIG["think"],
            "max_chars_entrada": CONFIG["max_chars_entrada"],
            "max_palabras_consulta": CONFIG["max_palabras_consulta"],
        },
        "jsonl": str(rutas["queries"]),
        "md5_jsonl": _md5(rutas["queries"]),
        **resumen,
        "nota": (
            "Extraccion resumible: cada re-ejecucion salta los ids ya "
            "presentes en el JSONL y reintenta los pendientes; el meta se "
            "reescribe con el estado acumulado."
        ),
    }
    guardar_json(rutas["meta"], meta, sobrescribir=True)
    print(
        f"Extraccion: {resumen['n_extraidos_nuevos']} nuevos | "
        f"{resumen['n_fallos_corrida']} fallos | "
        f"{len(pendientes_globales)} pendientes del benchmark | "
        f"media {resumen['media_tiempo_caso_seg']} s/caso"
    )
    print(f"JSONL: {rutas['queries']}")
    print(f"Meta: {rutas['meta']}")
    return resumen


# =====================================================================
# SECCION 3: EVALUACION - CASOS, BM25, DENSO B2 Y RERANK
# =====================================================================
def construir_casos_ruta(
    mod: object,
    muestra: list[dict],
    textos: dict,
    consultas: dict,
    usar_consulta_llm: bool,
) -> list[dict]:
    """Construye los casos de una ruta (consulta LLM o texto v13).

    Args:
        mod: Modulo del evaluador v12 (chunks_query, CONFIG).
        muestra: Casos normalizados a evaluar.
        textos: Dict {id: {"texto", "texto_limpio"}} del corpus.
        consultas: Dict {id: fila JSONL} de la extraccion.
        usar_consulta_llm: True usa consulta_busqueda como ventana de
            query; False usa el texto_limpio completo (ruta v13).

    Returns:
        Lista de casos con id, tipo, ano, arts, texto, ventana,
        chunks_bm25 y rankings vacios.

    Raises:
        KeyError: Si falta la consulta de un caso en modo LLM.
    """
    casos = []
    for caso in muestra:
        sid = caso["id"]
        texto = textos[sid]["texto"]
        limpio = textos[sid]["texto_limpio"]
        if usar_consulta_llm:
            ventana = " ".join(consultas[sid]["consulta_busqueda"].split())
            origen = "consulta_llm"
        else:
            ventana = limpio[: mod.CONFIG["ventana_query_chars"]]
            origen = "texto_limpio_completo"
        casos.append(
            {
                "id": sid,
                "tipo": caso["tipo"],
                "ano": caso["ano"],
                "arts": list(caso["arts"]),
                "texto": texto,
                "texto_limpio": limpio,
                "ventana": ventana,
                "origen_ventana": origen,
                "n_palabras_texto": len(limpio.split()),
                "n_palabras_ventana": len(ventana.split()),
                "chunks_bm25": mod.chunks_query(ventana),
                "rankings": {},
            }
        )
    return casos


def calcular_bm25_ruta(
    mod: object,
    casos: list[dict],
    pasajes: object,
    canones: list,
    grupos_indices: list,
) -> tuple[object, float]:
    """Calcula scores BM25 y la matriz (chunk, pasaje) por caso.

    Args:
        mod: Modulo del evaluador v12.
        casos: Casos con chunks_bm25.
        pasajes: Pasajes del indice activo.
        canones: Canones alineados con los scores.
        grupos_indices: Arrays de posiciones de pasajes por articulo.

    Returns:
        Tupla (indice_bm25, segundos) con el indice reutilizable.
    """
    inicio = time.time()
    indice = mod.construir_indice_bm25(pasajes)
    for caso in casos:
        scores, matriz = mod.scores_bm25_caso(
            indice, grupos_indices, canones, caso["chunks_bm25"]
        )
        caso["scores_bm25"] = scores
        caso["matriz_bm25"] = matriz
        caso["rankings"]["bm25"] = mod.ranking_desde_scores(scores, canones)
    return indice, round(time.time() - inicio, 1)


def calcular_denso_ruta(
    mod: object,
    casos: list[dict],
    pasajes: object,
    canones: list,
    grupos_indices: list,
    dispositivo: str,
) -> tuple[dict, str]:
    """Calcula el denso B2 y el hibrido 0.5*BM25 + 0.5*B2 por caso.

    Delega en calcular_denso_b2 del script 12 (cache de pasajes por slug
    en data/cache_emb_cp/<slug>.joblib) y devuelve la clave del vector
    hibrido materializado en cada caso.

    Args:
        mod: Modulo del evaluador v12.
        casos: Casos con scores_bm25 y ventana.
        pasajes: Pasajes del indice activo.
        canones: Canones alineados con los scores.
        grupos_indices: Arrays de posiciones de pasajes por articulo.
        dispositivo: "cuda"/"cpu" del guard de lib/reranker_bge.py.

    Returns:
        Tupla (stats, clave_hibrido).
    """
    inicio = time.time()
    stats = mod.calcular_denso_b2(
        casos, pasajes, canones, grupos_indices, dispositivo
    )
    stats["segundos_total"] = round(time.time() - inicio, 1)
    clave_hibrido = f"scores_hibrido_{stats['slug']}"
    return stats, clave_hibrido


def rerankear_ruta(
    mod: object,
    casos: list[dict],
    pasajes: object,
    canones: list,
    grupos_indices: list,
    reranker: object,
    estrategia: str,
    k_candidatos: int,
    clave_hibrido: str,
) -> dict:
    """Rerankea el top-k del hibrido B2 con el cross-encoder B1 por chunks.

    Mismos pares (chunk, pasaje) y max-pooling que la ruta de producto:
    mejor pasaje BM25 del candidato + top-3 chunks de la query,
    agregacion por maximo, salida truncada a 10 (ranking_rerank_chunks).

    Args:
        mod: Modulo del evaluador v12.
        casos: Casos con matriz_bm25, chunks_bm25 y scores hibridos.
        pasajes: Pasajes del indice activo.
        canones: Canones alineados con los scores.
        grupos_indices: Arrays de posiciones de pasajes por articulo.
        reranker: RerankerBGE cargado (modelo B1).
        estrategia: Nombre de la estrategia de producto.
        k_candidatos: Candidatos de la 1a etapa (50 en v13).
        clave_hibrido: Clave del vector de scores hibrido B2 por caso.

    Returns:
        Dict de stats del rerank (segundos, pares por caso, max tokens).
    """
    clave_scores = f"scores_{estrategia}"
    max_tokens = []
    inicio = time.time()
    for caso in casos:
        ranking, scores, max_tok = mod.ranking_rerank_chunks(
            reranker,
            caso[clave_hibrido],
            caso["matriz_bm25"],
            caso["chunks_bm25"],
            canones,
            pasajes,
            grupos_indices,
            k_candidatos,
        )
        caso[clave_scores] = scores
        caso["rankings"][estrategia] = ranking
        max_tokens.append(max_tok)
    return {
        "segundos": round(time.time() - inicio, 1),
        "pares_por_caso": int(k_candidatos * mod.CONFIG["k_chunks_rerank"]),
        "max_tokens_par": int(max(max_tokens)) if max_tokens else 0,
    }


# =====================================================================
# SECCION 4: METRICAS, IC BOOTSTRAP Y COMPARACION PAREADA
# =====================================================================
def medidas_caso(mod: object, relevantes: list, ranking: list) -> dict:
    """Calcula las seis medidas C2 de un caso (P/R/F1@5, F1@10, MRR, nDCG).

    Args:
        mod: Modulo del evaluador v12.
        relevantes: Citas CP de referencia del caso (silver NER).
        ranking: Ranking de producto de la estrategia (<= 10).

    Returns:
        Dict redondeado a 6 decimales con las medidas de MEDIDAS.
    """
    m5 = mod.metricas_ranking(relevantes, ranking, CONFIG["k_medidas"])
    m10 = mod.metricas_ranking(relevantes, ranking, CONFIG["k_ndcg"])
    valores = {
        "ndcg_at_10": m5["ndcg_at_10"],
        "f1_at_5": m5["f1_at_k"],
        "f1_at_10": m10["f1_at_k"],
        "mrr": m5["mrr"],
        "p_at_5": m5["p_at_k"],
        "r_at_5": m5["r_at_k"],
    }
    return {medida: round(float(valores[medida]), 6) for medida in MEDIDAS}


def evaluar_ruta(
    mod: object,
    casos: list[dict],
    estrategia: str,
    prior: dict,
    canones: list,
    alpha: float,
) -> list[dict]:
    """Evalua una ruta con el ranking de producto (hubs alpha, k=10).

    Reconstruye el ranking final con ranking_con_prior (la funcion que usa
    el run v13 para aplicar el prior y truncar a 10) y calcula las seis
    medidas por caso.

    Args:
        mod: Modulo del evaluador v12.
        casos: Casos con scores de la estrategia.
        estrategia: Nombre de la estrategia de producto.
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.
        alpha: Peso del prior (1.5 en la ruta de producto).

    Returns:
        Lista de filas por caso con id, ano, tipo, n_relevantes y medidas.
    """
    clave_scores = f"scores_{estrategia}"
    filas = []
    for caso in casos:
        ranking, _ = mod.ranking_con_prior(
            caso[clave_scores], canones, estrategia, prior, alpha
        )
        fila = {
            "id": caso["id"],
            "ano": int(caso["ano"]),
            "tipo": caso["tipo"],
            "n_relevantes": len(caso["arts"]),
            "n_palabras_texto": int(caso["n_palabras_texto"]),
            "n_palabras_ventana": int(caso["n_palabras_ventana"]),
        }
        fila.update(medidas_caso(mod, caso["arts"], ranking))
        filas.append(fila)
    return filas


def agregar_medidas(
    mod: object, filas: list[dict], n_bootstrap: int, semilla: int
) -> dict:
    """Agrega filas por caso con IC 95% bootstrap por medida.

    Args:
        mod: Modulo del evaluador v12 (bootstrap_ic_media).
        filas: Filas por caso con las medidas de MEDIDAS.
        n_bootstrap: Numero de remuestreos bootstrap.
        semilla: Semilla del generador.

    Returns:
        Dict {"n": n, "medidas": {medida: {media, p2_5, p97_5, n}}}.
    """
    return {
        "n": len(filas),
        "medidas": {
            medida: mod.bootstrap_ic_media(
                np.asarray([fila[medida] for fila in filas], dtype=np.float64),
                n_bootstrap,
                semilla,
            )
            for medida in MEDIDAS
        },
    }


def desglosar_por(
    mod: object, filas: list[dict], campo: str, n_bootstrap: int, semilla: int
) -> dict:
    """Agrupa filas por un campo (periodo o tercil) y agrega cada grupo.

    Args:
        mod: Modulo del evaluador v12.
        filas: Filas por caso.
        campo: Campo de agrupacion.
        n_bootstrap: Numero de remuestreos bootstrap.
        semilla: Semilla del generador.

    Returns:
        Dict {valor_del_campo: agregado con IC}.
    """
    grupos: dict = {}
    for fila in filas:
        grupos.setdefault(fila[campo], []).append(fila)
    return {
        str(clave): agregar_medidas(mod, grupo, n_bootstrap, semilla)
        for clave, grupo in sorted(grupos.items())
    }


def pareado_global(
    mod: object, filas_llm: list[dict], filas_v13: list[dict]
) -> dict:
    """IC 95% bootstrap PAREADO de delta = LLM - v13 por medida.

    Args:
        mod: Modulo del evaluador v12 (bootstrap_ic_pareado).
        filas_llm: Filas por caso de la ruta con consulta LLM.
        filas_v13: Filas por caso de la ruta v13 (texto completo).

    Returns:
        Dict {medida: IC pareado con media_delta, p2_5, p97_5,
        prob_mejora y gana}.

    Raises:
        ValueError: Si las filas no estan alineadas por id.
    """
    ids_llm = [fila["id"] for fila in filas_llm]
    ids_v13 = [fila["id"] for fila in filas_v13]
    if ids_llm != ids_v13:
        raise ValueError("filas LLM y v13 desalineadas por id")
    return {
        medida: mod.bootstrap_ic_pareado(
            np.asarray([fila[medida] for fila in filas_llm], dtype=np.float64),
            np.asarray([fila[medida] for fila in filas_v13], dtype=np.float64),
            CONFIG["bootstrap"],
            CONFIG["semilla"],
        )
        for medida in MEDIDAS
    }


def pareado_por_grupo(
    mod: object, filas_llm: list[dict], filas_v13: list[dict], campo: str
) -> dict:
    """IC pareado por subgrupo (periodo o tercil de longitud) y medida.

    Args:
        mod: Modulo del evaluador v12.
        filas_llm: Filas por caso de la ruta LLM.
        filas_v13: Filas por caso de la ruta v13.
        campo: Campo de agrupacion.

    Returns:
        Dict {valor: {medida: IC pareado}}.
    """
    grupos: dict = {}
    for fila_llm, fila_v13 in zip(filas_llm, filas_v13, strict=True):
        grupos.setdefault(fila_llm[campo], []).append((fila_llm, fila_v13))
    return {
        str(clave): {
            medida: mod.bootstrap_ic_pareado(
                np.asarray(
                    [llm[medida] for llm, _ in pares], dtype=np.float64
                ),
                np.asarray(
                    [v13[medida] for _, v13 in pares], dtype=np.float64
                ),
                CONFIG["bootstrap"],
                CONFIG["semilla"],
            )
            for medida in MEDIDAS
        }
        for clave, pares in sorted(grupos.items())
    }


def consistencia_congelada(
    bench_v13: dict, filas_v13: list[dict], estrategia: str, contexto: str
) -> dict:
    """Valida la ruta v13 recalculada contra el per-case congelado v13.

    Compara nDCG@10 y F1@10 por caso recalculados con los del campo
    comparacion_b2.metricas_por_caso[contexto][estrategia] del benchmark
    v13 (la ruta de producto congelada). Sirve de control de que la
    recalculo en la misma corrida reproduce la ruta v13 publicada.

    Args:
        bench_v13: Benchmark v13 cargado.
        filas_v13: Filas por caso de la ruta v13 recalculada.
        estrategia: Nombre de la estrategia de producto.
        contexto: Contexto del prior ("con_hubs").

    Returns:
        Dict con diffs maximas/medias por medida, tolerancia y veredicto;
        disponible=False si el benchmark no trae la referencia.
    """
    comparacion = (bench_v13.get("comparacion_b2") or {}).get(
        "metricas_por_caso", {}
    )
    referencia = (comparacion.get(contexto) or {}).get(estrategia)
    if not referencia:
        return {"disponible": False, "motivo": "sin referencia congelada"}
    posicion = {
        str(caso["id"]): indice
        for indice, caso in enumerate(bench_v13["muestra"])
    }
    diffs: dict = {"ndcg_at_10": [], "f1_at_10": []}
    sin_referencia = []
    for fila in filas_v13:
        indice = posicion.get(fila["id"])
        if indice is None:
            sin_referencia.append(fila["id"])
            continue
        for medida, valores in diffs.items():
            valores.append(
                abs(fila[medida] - float(referencia[medida][indice]))
            )
    resumen = {
        medida: {
            "max_abs_diff": round(float(np.max(valores)), 9),
            "media_abs_diff": round(float(np.mean(valores)), 9),
        }
        for medida, valores in diffs.items()
        if valores
    }
    maximos = [datos["max_abs_diff"] for datos in resumen.values()]
    return {
        "disponible": True,
        "n_comparados": len(diffs["ndcg_at_10"]),
        "ids_sin_referencia": sin_referencia,
        "medidas": resumen,
        "tolerancia": CONFIG["tolerancia_consistencia"],
        "consistente": bool(
            maximos and max(maximos) <= CONFIG["tolerancia_consistencia"]
        ),
        "fuente": (
            "resultados/vinculo_cp_benchmark_v13.json | "
            f"comparacion_b2.metricas_por_caso.{contexto}.{estrategia}"
        ),
    }


# =====================================================================
# SECCION 5: FIGURA, SALIDAS Y ORQUESTACION DE LA EVALUACION
# =====================================================================
def graficar_evaluacion(
    agregado_llm: dict,
    agregado_v13: dict,
    pareado: dict,
    pareado_periodo: dict,
    pareado_tercil: dict,
    filas_llm: list[dict],
    filas_v13: list[dict],
    ruta: Path,
    n_casos: int,
) -> None:
    """Genera la figura C2: metricas, dispersion, y deltas pareados.

    Paneles: (1) barras de las seis medidas con IC 95% bootstrap para las
    dos rutas; (2) dispersion nDCG@10 por caso con diagonal y conteo
    victorias/empates/derrotas; (3) delta pareado de nDCG@10 por periodo;
    (4) delta pareado de nDCG@10 por tercil de longitud del caso.

    Args:
        agregado_llm: Agregado global de la ruta LLM.
        agregado_v13: Agregado global de la ruta v13.
        pareado: IC pareado global por medida.
        pareado_periodo: IC pareado por periodo y medida.
        pareado_tercil: IC pareado por tercil y medida.
        filas_llm: Filas por caso de la ruta LLM.
        filas_v13: Filas por caso de la ruta v13.
        ruta: Ruta de salida de la figura.
        n_casos: Numero de casos evaluados.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura, ejes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    # Panel 1: barras por medida con IC 95%.
    posiciones = np.arange(len(MEDIDAS))
    ancho = 0.38
    medias_llm = [agregado_llm["medidas"][m]["media"] for m in MEDIDAS]
    medias_v13 = [agregado_v13["medidas"][m]["media"] for m in MEDIDAS]
    error_llm = [
        [
            agregado_llm["medidas"][m]["media"]
            - agregado_llm["medidas"][m]["p2_5"]
            for m in MEDIDAS
        ],
        [
            agregado_llm["medidas"][m]["p97_5"]
            - agregado_llm["medidas"][m]["media"]
            for m in MEDIDAS
        ],
    ]
    error_v13 = [
        [
            agregado_v13["medidas"][m]["media"]
            - agregado_v13["medidas"][m]["p2_5"]
            for m in MEDIDAS
        ],
        [
            agregado_v13["medidas"][m]["p97_5"]
            - agregado_v13["medidas"][m]["media"]
            for m in MEDIDAS
        ],
    ]
    ejes[0, 0].bar(
        posiciones - ancho / 2,
        medias_llm,
        ancho,
        yerr=error_llm,
        capsize=3,
        color="#dd8452",
        label="LLM query",
    )
    ejes[0, 0].bar(
        posiciones + ancho / 2,
        medias_v13,
        ancho,
        yerr=error_v13,
        capsize=3,
        color="#4c72b0",
        label="v13 texto completo",
    )
    ejes[0, 0].set_xticks(posiciones, [m.replace("_", " ") for m in MEDIDAS])
    ejes[0, 0].set_ylim(0.0, 1.0)
    ejes[0, 0].set_ylabel("media (IC 95% bootstrap)")
    ejes[0, 0].set_title(f"Metricas globales (n={n_casos})")
    ejes[0, 0].legend(fontsize=9)
    ejes[0, 0].grid(axis="y", alpha=0.3)

    # Panel 2: dispersion nDCG@10 por caso.
    ndcg_llm = [fila["ndcg_at_10"] for fila in filas_llm]
    ndcg_v13 = [fila["ndcg_at_10"] for fila in filas_v13]
    ejes[0, 1].scatter(ndcg_v13, ndcg_llm, alpha=0.65, s=26, color="#55a868")
    ejes[0, 1].plot([0.0, 1.0], [0.0, 1.0], "--", color="gray", linewidth=1)
    victorias = sum(
        1
        for llm, v13 in zip(ndcg_llm, ndcg_v13, strict=True)
        if llm > v13 + 1e-9
    )
    derrotas = sum(
        1
        for llm, v13 in zip(ndcg_llm, ndcg_v13, strict=True)
        if llm < v13 - 1e-9
    )
    empates = n_casos - victorias - derrotas
    delta_global = pareado["ndcg_at_10"]
    ejes[0, 1].set_xlim(0.0, 1.0)
    ejes[0, 1].set_ylim(0.0, 1.0)
    ejes[0, 1].set_xlabel("nDCG@10 v13 (texto completo)")
    ejes[0, 1].set_ylabel("nDCG@10 LLM query")
    ejes[0, 1].set_title(
        "nDCG@10 por caso | delta "
        f"{delta_global['media_delta']:+.3f} "
        f"[{delta_global['p2_5']:+.3f}, {delta_global['p97_5']:+.3f}]"
    )
    ejes[0, 1].text(
        0.03,
        0.97,
        f"victorias LLM: {victorias}\nempates: {empates}\n"
        f"derrotas LLM: {derrotas}",
        transform=ejes[0, 1].transAxes,
        va="top",
        fontsize=9,
    )
    ejes[0, 1].grid(alpha=0.3)

    # Panel 3: delta pareado de nDCG@10 por periodo.
    periodos = list(pareado_periodo)
    deltas_periodo = [
        pareado_periodo[p]["ndcg_at_10"]["media_delta"] for p in periodos
    ]
    bajos_periodo = [
        pareado_periodo[p]["ndcg_at_10"]["media_delta"]
        - pareado_periodo[p]["ndcg_at_10"]["p2_5"]
        for p in periodos
    ]
    altos_periodo = [
        pareado_periodo[p]["ndcg_at_10"]["p97_5"]
        - pareado_periodo[p]["ndcg_at_10"]["media_delta"]
        for p in periodos
    ]
    ejes[1, 0].bar(
        range(len(periodos)),
        deltas_periodo,
        yerr=[bajos_periodo, altos_periodo],
        capsize=4,
        color="#c44e52",
        width=0.55,
    )
    ejes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    for posicion, (periodo, delta) in enumerate(
        zip(periodos, deltas_periodo, strict=True)
    ):
        tamano = pareado_periodo[periodo]["ndcg_at_10"]["n"]
        ejes[1, 0].text(
            posicion,
            delta,
            f"{delta:+.3f}\nn={tamano}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=9,
        )
    ejes[1, 0].set_xticks(range(len(periodos)), periodos)
    ejes[1, 0].set_ylabel("delta nDCG@10 (LLM - v13)")
    ejes[1, 0].set_title("Delta pareado por periodo (IC 95%)")
    ejes[1, 0].grid(axis="y", alpha=0.3)

    # Panel 4: delta pareado de nDCG@10 por tercil de longitud.
    terciles = list(pareado_tercil)
    etiquetas_tercil = {
        "0": "tercil 1 (cortos)",
        "1": "tercil 2 (medios)",
        "2": "tercil 3 (largos)",
    }
    deltas_tercil = [
        pareado_tercil[t]["ndcg_at_10"]["media_delta"] for t in terciles
    ]
    bajos_tercil = [
        pareado_tercil[t]["ndcg_at_10"]["media_delta"]
        - pareado_tercil[t]["ndcg_at_10"]["p2_5"]
        for t in terciles
    ]
    altos_tercil = [
        pareado_tercil[t]["ndcg_at_10"]["p97_5"]
        - pareado_tercil[t]["ndcg_at_10"]["media_delta"]
        for t in terciles
    ]
    ejes[1, 1].bar(
        range(len(terciles)),
        deltas_tercil,
        yerr=[bajos_tercil, altos_tercil],
        capsize=4,
        color="#8172b3",
        width=0.55,
    )
    ejes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    for posicion, (tercil, delta) in enumerate(
        zip(terciles, deltas_tercil, strict=True)
    ):
        tamano = pareado_tercil[tercil]["ndcg_at_10"]["n"]
        ejes[1, 1].text(
            posicion,
            delta,
            f"{delta:+.3f}\nn={tamano}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=9,
        )
    ejes[1, 1].set_xticks(
        range(len(terciles)),
        [etiquetas_tercil.get(t, f"tercil {t}") for t in terciles],
    )
    ejes[1, 1].set_ylabel("delta nDCG@10 (LLM - v13)")
    ejes[1, 1].set_title("Delta pareado por longitud del caso (IC 95%)")
    ejes[1, 1].grid(axis="y", alpha=0.3)

    figura.suptitle(
        "C2 query understanding con LLM legal local (alia-legal)\n"
        "Ruta hibrido_b2_rerank_chunks_top50_b1 + hubs alpha=1.5, k=10 | "
        f"n={n_casos} casos del benchmark M7"
    )
    figura.tight_layout()
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def resolver_rutas_salida(limite: int | None) -> dict:
    """Resuelve las rutas de salida segun el modo (real o smoke/limite).

    Args:
        limite: Valor efectivo de --limite (None en el run real).

    Returns:
        Dict con rutas queries/meta/json/figura/log y flag sobrescribir.
    """
    if limite is None:
        return {
            "queries": ROOT / CONFIG["dir_salida"] / "queries.jsonl",
            "meta": ROOT / CONFIG["dir_salida"] / "meta_extraccion.json",
            "json": ROOT / CONFIG["ruta_json"],
            "figura": ROOT / CONFIG["ruta_figura"],
            "log": ROOT / CONFIG["ruta_log"],
            "sobrescribir": False,
        }
    base = Path(CONFIG["dir_smoke"])
    return {
        "queries": base / "queries.jsonl",
        "meta": base / "meta_extraccion.json",
        "json": base / "eval_query_llm.json",
        "figura": base / "c2_query_llm.png",
        "log": base / "30_c2_llm_query.log",
        "sobrescribir": True,
    }


def comprobar_salidas(rutas: dict) -> None:
    """Impide sobrescribir los artefactos reales de la evaluacion.

    Solo aplica al JSON y a la figura de la evaluacion (el JSONL y el
    meta de extraccion son artefactos de progreso resumibles).

    Args:
        rutas: Dict de rutas resuelto por resolver_rutas_salida.

    Raises:
        FileExistsError: Si el JSON o la figura reales ya existen.
    """
    if rutas["sobrescribir"]:
        return
    for clave, descripcion in (("json", "JSON"), ("figura", "figura")):
        if rutas[clave].exists():
            raise FileExistsError(
                f"ya existe el {descripcion} {rutas[clave]}: no se "
                "sobrescribe; renombra o elimina el artefacto antes de "
                "re-ejecutar"
            )


def _resumen_extraccion_evaluada(consultas: dict) -> dict:
    """Resume las consultas LLM usadas en la evaluacion.

    Args:
        consultas: Dict {id: fila JSONL} de los casos evaluados.

    Returns:
        Dict con conteos, longitudes y tiempos de las consultas.
    """
    filas = list(consultas.values())
    palabras = [int(fila.get("n_palabras_consulta", 0)) for fila in filas]
    tiempos = [float(fila.get("tiempo_seg", 0.0)) for fila in filas]
    return {
        "n_consultas": len(filas),
        "modelo": filas[0].get("modelo") if filas else None,
        "media_palabras_consulta": (
            round(float(np.mean(palabras)), 1) if palabras else None
        ),
        "min_palabras_consulta": int(min(palabras)) if palabras else None,
        "max_palabras_consulta": int(max(palabras)) if palabras else None,
        "n_excede_max_palabras": sum(
            1 for fila in filas if fila.get("excede_max_palabras")
        ),
        "n_correcciones": sum(
            1 for fila in filas if fila.get("correccion_usada")
        ),
        "n_fallback_campos": sum(
            1
            for fila in filas
            if fila.get("consulta_fuente") == "fallback_campos"
        ),
        "media_tiempo_caso_seg": (
            round(float(np.mean(tiempos)), 1) if tiempos else None
        ),
        "total_tiempo_caso_seg": round(float(np.sum(tiempos)), 1),
    }


def ejecutar_evaluacion(mod: object, limite: int | None, rutas: dict) -> dict:
    """Evalua la ruta con consulta LLM frente a la ruta v13 pareada.

    Recalcula ambas rutas en la misma corrida con la maquinaria del
    script 12 (BM25 + B2 + rerank B1 + hubs alpha=1.5, k=10), calcula las
    seis medidas por caso con IC 95% bootstrap, el IC pareado
    LLM - v13 (global y por periodo/tercil de longitud) y valida la ruta
    v13 recalculada contra el per-case congelado del benchmark v13.

    Args:
        mod: Modulo del evaluador v12.
        limite: Limite efectivo de casos (None en el run real).
        rutas: Dict de rutas resuelto por resolver_rutas_salida.

    Returns:
        Dict de salida serializado en el JSON de evaluacion.

    Raises:
        FileNotFoundError: Si no existe el JSONL de consultas.
        RuntimeError: Si ningun caso de la muestra tiene consulta.
    """
    inicio_total = time.time()
    if not rutas["queries"].is_file():
        raise FileNotFoundError(
            f"no existe {rutas['queries']}: ejecuta primero --extraer"
        )
    comprobar_salidas(rutas)
    print("=" * 78)
    print("30_c2_llm_query.py | evaluacion C2: consulta LLM vs v13 pareada")
    print(
        f"fecha: {datetime.now(timezone.utc).isoformat()} | modo: "
        f"{'smoke/limite' if limite is not None else 'real'} | semilla: "
        f"{CONFIG['semilla']}"
    )
    print(f"salidas: {rutas['json']} | {rutas['figura']}")
    print("=" * 78)

    estrategia = CONFIG["estrategia"]
    modelo_b2 = mod.resolver_modelo_b2(None)
    modelo_b1 = mod.resolver_modelo_b1(None)
    mod.aplicar_modo_b2(None, modelo_b2, modelo_b1)
    print(f"Ruta de producto: {estrategia} | hubs alpha={CONFIG['alpha_hubs']}")
    print(f"Retriever B2: {modelo_b2}")
    print(f"Reranker B1: {modelo_b1}")

    print("[1/6] Insumos, muestra y consultas extraidas")
    bench_v2 = cargar_benchmark(ROOT / CONFIG["ruta_benchmark_v2"], ("muestra",))
    bench_v13 = cargar_benchmark(
        ROOT / CONFIG["ruta_benchmark_v13"],
        ("muestra", "comparacion_b2"),
    )
    muestra = recortar_muestra(normalizar_muestra(bench_v2), limite)
    consultas = cargar_queries_jsonl(rutas["queries"])
    muestra_eval = [caso for caso in muestra if caso["id"] in consultas]
    sin_consulta = [caso["id"] for caso in muestra if caso["id"] not in consultas]
    if not muestra_eval:
        raise RuntimeError(
            "ningun caso de la muestra tiene consulta extraida; ejecuta "
            "--extraer primero"
        )
    if sin_consulta:
        print(
            f"[aviso] {len(sin_consulta)} casos sin consulta quedan fuera "
            f"de la evaluacion: {sin_consulta[:5]}"
        )
    print(
        f"Muestra: {len(muestra)} casos | evaluados: {len(muestra_eval)} | "
        f"benchmark v13: {len(bench_v13['muestra'])} casos congelados"
    )

    textos = leer_textos_por_id(
        mod, ROOT / CONFIG["ruta_corpus"], [c["id"] for c in muestra_eval]
    )
    pasajes, canones, grupos_indices, canon_valido = (
        mod.preparar_pasajes_constitucion(False)
    )
    prior = mod.cargar_o_construir_prior_hubs()
    print(
        f"CP: {len(canones)} canones / {len(pasajes)} pasajes | prior de "
        f"hubs: {len(prior)} canones"
    )

    print("[2/6] Casos de las dos rutas y BM25")
    casos_llm = construir_casos_ruta(
        mod, muestra_eval, textos, consultas, usar_consulta_llm=True
    )
    casos_v13 = construir_casos_ruta(
        mod, muestra_eval, textos, consultas, usar_consulta_llm=False
    )
    media_chunks_llm = float(
        np.mean([len(caso["chunks_bm25"]) for caso in casos_llm])
    )
    media_chunks_v13 = float(
        np.mean([len(caso["chunks_bm25"]) for caso in casos_v13])
    )
    _, seg_bm25_llm = calcular_bm25_ruta(
        mod, casos_llm, pasajes, canones, grupos_indices
    )
    _, seg_bm25_v13 = calcular_bm25_ruta(
        mod, casos_v13, pasajes, canones, grupos_indices
    )
    print(
        f"BM25: LLM {seg_bm25_llm} s ({media_chunks_llm:.1f} chunks/caso) | "
        f"v13 {seg_bm25_v13} s ({media_chunks_v13:.1f} chunks/caso)"
    )

    print("[3/6] Modelos de la ruta (GPU con guard de VRAM)")
    from lib.reranker_bge import RerankerBGE, elegir_dispositivo

    dispositivo = elegir_dispositivo()
    print(
        f"Dispositivo: {dispositivo} | torch.cuda disponible: "
        f"{torch.cuda.is_available()}"
    )
    stats_b2_llm, clave_hibrido_llm = calcular_denso_ruta(
        mod, casos_llm, pasajes, canones, grupos_indices, dispositivo
    )
    print(
        f"B2 LLM: {stats_b2_llm['n_chunks_query']} chunks de query en "
        f"{stats_b2_llm['segundos_total']} s"
    )
    stats_b2_v13, clave_hibrido_v13 = calcular_denso_ruta(
        mod, casos_v13, pasajes, canones, grupos_indices, dispositivo
    )
    print(
        f"B2 v13: {stats_b2_v13['n_chunks_query']} chunks de query en "
        f"{stats_b2_v13['segundos_total']} s"
    )
    reranker = RerankerBGE(modelo_id=modelo_b1, dispositivo=dispositivo)
    try:
        stats_rr_llm = rerankear_ruta(
            mod,
            casos_llm,
            pasajes,
            canones,
            grupos_indices,
            reranker,
            estrategia,
            CONFIG["k_candidatos"],
            clave_hibrido_llm,
        )
        stats_rr_v13 = rerankear_ruta(
            mod,
            casos_v13,
            pasajes,
            canones,
            grupos_indices,
            reranker,
            estrategia,
            CONFIG["k_candidatos"],
            clave_hibrido_v13,
        )
    finally:
        del reranker
        if dispositivo == "cuda":
            torch.cuda.empty_cache()
    print(
        f"Rerank B1: LLM {stats_rr_llm['segundos']} s | v13 "
        f"{stats_rr_v13['segundos']} s | max tokens par: "
        f"{max(stats_rr_llm['max_tokens_par'], stats_rr_v13['max_tokens_par'])}"
    )

    print("[4/6] Evaluacion de las dos rutas (hubs alpha=1.5, k=10)")
    filas_llm = evaluar_ruta(
        mod, casos_llm, estrategia, prior, canones, CONFIG["alpha_hubs"]
    )
    filas_v13 = evaluar_ruta(
        mod, casos_v13, estrategia, prior, canones, CONFIG["alpha_hubs"]
    )
    terciles, palabras_texto = mod.asignar_terciles_longitud(casos_v13)
    for indice, (fila_llm, fila_v13) in enumerate(
        zip(filas_llm, filas_v13, strict=True)
    ):
        periodo = mod.asignar_periodo(int(fila_llm["ano"]))
        for fila in (fila_llm, fila_v13):
            fila["periodo"] = periodo
            fila["tercil_longitud"] = int(terciles[indice])
            fila["n_palabras_texto"] = int(palabras_texto[indice])
    agregado_llm = agregar_medidas(
        mod, filas_llm, CONFIG["bootstrap"], CONFIG["semilla"]
    )
    agregado_v13 = agregar_medidas(
        mod, filas_v13, CONFIG["bootstrap"], CONFIG["semilla"]
    )
    pareado = pareado_global(mod, filas_llm, filas_v13)
    pareado_periodo = pareado_por_grupo(mod, filas_llm, filas_v13, "periodo")
    pareado_tercil = pareado_por_grupo(
        mod, filas_llm, filas_v13, "tercil_longitud"
    )
    consistencia = consistencia_congelada(
        bench_v13,
        filas_v13,
        estrategia,
        CONFIG["contexto_v13_congelado"],
    )
    global_llm = agregado_llm["medidas"]
    global_v13 = agregado_v13["medidas"]
    print(
        f"LLM: nDCG@10={global_llm['ndcg_at_10']['media']:.3f} "
        f"[{global_llm['ndcg_at_10']['p2_5']:.3f}, "
        f"{global_llm['ndcg_at_10']['p97_5']:.3f}] | "
        f"F1@10={global_llm['f1_at_10']['media']:.3f} | "
        f"MRR={global_llm['mrr']['media']:.3f}"
    )
    print(
        f"v13: nDCG@10={global_v13['ndcg_at_10']['media']:.3f} "
        f"[{global_v13['ndcg_at_10']['p2_5']:.3f}, "
        f"{global_v13['ndcg_at_10']['p97_5']:.3f}] | "
        f"F1@10={global_v13['f1_at_10']['media']:.3f} | "
        f"MRR={global_v13['mrr']['media']:.3f}"
    )
    delta = pareado["ndcg_at_10"]
    print(
        f"Pareado nDCG@10 (LLM - v13): {delta['media_delta']:+.3f} "
        f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] | "
        f"prob_mejora={delta['prob_mejora']:.3f} | gana={delta['gana']}"
    )
    if consistencia["disponible"]:
        print(
            "Consistencia v13 recalculado vs congelado: max|d|="
            f"{max(d['max_abs_diff'] for d in consistencia['medidas'].values()):.2e} "
            f"| consistente={consistencia['consistente']}"
        )
    else:
        print("[aviso] sin referencia congelada de v13 para consistencia")

    print("[5/6] Figura")
    graficar_evaluacion(
        agregado_llm,
        agregado_v13,
        pareado,
        pareado_periodo,
        pareado_tercil,
        filas_llm,
        filas_v13,
        rutas["figura"],
        len(filas_llm),
    )
    print(f"Figura: {rutas['figura']}")

    print("[6/6] Guardado del JSON de evaluacion")
    por_caso = [
        {
            "id": fila_llm["id"],
            "ano": int(fila_llm["ano"]),
            "tipo": fila_llm["tipo"],
            "periodo": fila_llm["periodo"],
            "tercil_longitud": int(fila_llm["tercil_longitud"]),
            "n_relevantes": int(fila_llm["n_relevantes"]),
            "n_palabras_texto": int(fila_llm["n_palabras_texto"]),
            "n_palabras_consulta": int(fila_llm["n_palabras_ventana"]),
            "metricas_llm": {m: fila_llm[m] for m in MEDIDAS},
            "metricas_v13": {m: fila_v13[m] for m in MEDIDAS},
            "delta_ndcg_at_10": round(
                fila_llm["ndcg_at_10"] - fila_v13["ndcg_at_10"], 6
            ),
        }
        for fila_llm, fila_v13 in zip(filas_llm, filas_v13, strict=True)
    ]
    salida = {
        "fecha": datetime.now(timezone.utc).isoformat(),
        "version": CONFIG["version"],
        "script": "scripts/30_c2_llm_query.py",
        "descripcion": (
            "C2 query understanding: consulta corta y densa extraida por el "
            "LLM legal local alia-legal (problema juridico + hechos + "
            "pretensiones) frente a la ruta v13 con el texto completo, "
            "evaluada con la ruta de producto hibrido_b2_rerank_chunks_"
            "top50_b1 + hubs alpha=1.5 y k=10."
        ),
        "terminologia": (
            "La referencia de relevancia son citas CP NER silver del "
            "benchmark M7; 'gold' queda reservado para anotacion humana "
            "(regla del proyecto)."
        ),
        "modo": {"smoke_o_limite": limite is not None, "limite": limite},
        "config": {
            "semilla": CONFIG["semilla"],
            "estrategia": estrategia,
            "alpha_hubs": CONFIG["alpha_hubs"],
            "k_candidatos": CONFIG["k_candidatos"],
            "k_medidas": CONFIG["k_medidas"],
            "k_ndcg": CONFIG["k_ndcg"],
            "bootstrap": CONFIG["bootstrap"],
            "contexto_v13_congelado": CONFIG["contexto_v13_congelado"],
            "modelo_llm": CONFIG["modelo_llm"],
            "host_ollama": CONFIG["host_ollama"],
            "llm": {
                "temperature": CONFIG["temperature"],
                "seed": CONFIG["semilla"],
                "num_ctx": CONFIG["num_ctx"],
                "num_predict": CONFIG["num_predict"],
                "think": CONFIG["think"],
                "max_chars_entrada": CONFIG["max_chars_entrada"],
                "max_palabras_consulta": CONFIG["max_palabras_consulta"],
            },
            "chunk_bm25_chars": mod.CONFIG["chunk_bm25_chars"],
            "solape_bm25_chars": mod.CONFIG["solape_bm25_chars"],
            "ventana_query_chars_v13": mod.CONFIG["ventana_query_chars"],
            "k_chunks_rerank": mod.CONFIG["k_chunks_rerank"],
            "pesos_hibrido": dict(mod.CONFIG["pesos_hibrido"]),
        },
        "insumos": {
            "benchmark_v2": str(ROOT / CONFIG["ruta_benchmark_v2"]),
            "benchmark_v13": str(ROOT / CONFIG["ruta_benchmark_v13"]),
            "corpus": str(ROOT / CONFIG["ruta_corpus"]),
            "queries_jsonl": str(rutas["queries"]),
            "md5_queries_jsonl": _md5(rutas["queries"]),
            "canones_indexados": len(canones),
            "canones_validos_gold": len(canon_valido),
            "pasajes": len(pasajes),
            "prior_hubs_canones": len(prior),
        },
        "cobertura_consultas": {
            "n_muestra": len(muestra),
            "n_con_consulta": len(muestra_eval),
            "n_sin_consulta": len(sin_consulta),
            "ids_sin_consulta": sin_consulta,
        },
        "extraccion": _resumen_extraccion_evaluada(
            {caso["id"]: consultas[caso["id"]] for caso in muestra_eval}
        ),
        "evaluacion": {
            "llm": {
                "global": agregado_llm,
                "por_periodo": desglosar_por(
                    mod, filas_llm, "periodo", CONFIG["bootstrap"],
                    CONFIG["semilla"],
                ),
                "por_tercil_longitud": desglosar_por(
                    mod, filas_llm, "tercil_longitud", CONFIG["bootstrap"],
                    CONFIG["semilla"],
                ),
            },
            "v13": {
                "global": agregado_v13,
                "por_periodo": desglosar_por(
                    mod, filas_v13, "periodo", CONFIG["bootstrap"],
                    CONFIG["semilla"],
                ),
                "por_tercil_longitud": desglosar_por(
                    mod, filas_v13, "tercil_longitud", CONFIG["bootstrap"],
                    CONFIG["semilla"],
                ),
            },
            "pareado_llm_vs_v13": {
                "global": pareado,
                "por_periodo": pareado_periodo,
                "por_tercil_longitud": pareado_tercil,
            },
            "consistencia_v13_congelado": consistencia,
            "por_caso": por_caso,
        },
        "rerank_stats": {"llm": stats_rr_llm, "v13": stats_rr_v13},
        "b2_stats": {"llm": stats_b2_llm, "v13": stats_b2_v13},
        "tiempos_segundos": {
            "bm25_llm": seg_bm25_llm,
            "bm25_v13": seg_bm25_v13,
            "b2_llm": stats_b2_llm["segundos_total"],
            "b2_v13": stats_b2_v13["segundos_total"],
            "rerank_llm": stats_rr_llm["segundos"],
            "rerank_v13": stats_rr_v13["segundos"],
            "total": round(time.time() - inicio_total, 1),
        },
        "notas": [
            (
                "La ruta v13 se RECALCULA en esta misma corrida con la "
                "consulta de texto_limpio completo (E1, la columna que usa "
                "el run v13) y la misma maquinaria (BM25 + B2 + rerank B1 + "
                "hubs alpha=1.5, k=10), de modo que la comparacion es "
                "PAREADA por caso; se valida contra el per-case congelado "
                "de comparacion_b2 (contexto con_hubs) del benchmark v13 "
                "para nDCG@10 y F1@10."
            ),
            (
                "La consulta LLM se recorta al texto que produce el "
                "pipeline de retrieval (consulta_busqueda); los campos "
                "problema_juridico, hechos y pretensiones se conservan en "
                "el JSONL como evidencia de la extraccion. El LLM lee el "
                "texto crudo del corpus (mas legible) recortado a "
                "max_chars_entrada caracteres."
            ),
            (
                "El desglose por longitud usa terciles del texto_limpio "
                "de la sentencia (asignar_terciles_longitud del script 12)."
            ),
        ],
    }
    guardar_json(rutas["json"], salida, rutas["sobrescribir"])
    print(f"JSON: {rutas['json']}")
    print("-" * 78)
    print(
        f"Resumen C2 (hubs alpha={CONFIG['alpha_hubs']}, "
        f"n={len(filas_llm)}):"
    )
    print(
        f"  LLM: nDCG@10={global_llm['ndcg_at_10']['media']:.3f} | "
        f"F1@5={global_llm['f1_at_5']['media']:.3f} | "
        f"F1@10={global_llm['f1_at_10']['media']:.3f} | "
        f"MRR={global_llm['mrr']['media']:.3f} | "
        f"P@5={global_llm['p_at_5']['media']:.3f} | "
        f"R@5={global_llm['r_at_5']['media']:.3f}"
    )
    print(
        f"  v13: nDCG@10={global_v13['ndcg_at_10']['media']:.3f} | "
        f"F1@5={global_v13['f1_at_5']['media']:.3f} | "
        f"F1@10={global_v13['f1_at_10']['media']:.3f} | "
        f"MRR={global_v13['mrr']['media']:.3f} | "
        f"P@5={global_v13['p_at_5']['media']:.3f} | "
        f"R@5={global_v13['r_at_5']['media']:.3f}"
    )
    print(
        f"  pareado nDCG@10: {delta['media_delta']:+.3f} "
        f"[{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] "
        f"(gana={delta['gana']})"
    )
    print(
        f"  tiempo total evaluacion: "
        f"{salida['tiempos_segundos']['total']} s"
    )
    print("=" * 78)
    return salida


# =====================================================================
# SECCION 6: CLI Y ENTRADA PRINCIPAL
# =====================================================================
def parsear_args() -> argparse.Namespace:
    """Define y parsea la interfaz de linea de comandos.

    Returns:
        Namespace con extraer, evaluar, todo, limite y smoke.
    """
    parser = argparse.ArgumentParser(
        description=(
            "C2 query understanding con el LLM legal local alia-legal: "
            "extrae consulta/problema/hechos/pretensiones y evalua la ruta "
            "de producto frente a v13 (texto completo) de forma pareada."
        )
    )
    parser.add_argument(
        "--extraer",
        action="store_true",
        help="Extrae consultas LLM por caso (JSONL resumible en resultados/c2_llm/).",
    )
    parser.add_argument(
        "--evaluar",
        action="store_true",
        help="Evalua la ruta con las consultas extraidas (requiere --extraer previo).",
    )
    parser.add_argument(
        "--todo",
        action="store_true",
        help="Ejecuta extraccion y evaluacion en una sola corrida.",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help=(
            "Recorta la muestra a N casos y redirige todas las salidas a "
            f"{CONFIG['dir_smoke']}/ (nunca toca artefactos reales)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Atajo de --limite {CONFIG['smoke_limite']} con salidas en "
            f"{CONFIG['dir_smoke']}/."
        ),
    )
    args = parser.parse_args()
    if args.limite is not None and args.limite < 1:
        parser.error("--limite debe ser >= 1")
    return args


def _ejecutar(args: argparse.Namespace, rutas: dict, limite: int | None) -> None:
    """Orquesta los modos solicitados (secciones 1 a 6).

    Args:
        args: Namespace del CLI.
        rutas: Dict de rutas de salida.
        limite: Limite efectivo de casos (None en el run real).
    """
    inicio_total = time.time()
    mod = importar_evaluador_v12(ROOT / CONFIG["ruta_script_12"])
    mod.configurar_semillas(CONFIG["semilla"])
    if args.extraer or args.todo:
        ejecutar_extraccion(mod, limite, rutas)
    if args.evaluar or args.todo:
        ejecutar_evaluacion(mod, limite, rutas)
    print(f"Tiempo total del proceso: {round(time.time() - inicio_total, 1)} s")


def main() -> None:
    """Punto de entrada: CLI, log duplicado y manejo de errores."""
    args = parsear_args()
    if not (args.extraer or args.evaluar or args.todo):
        print(
            "Nada que ejecutar: usa --extraer, --evaluar o --todo "
            "(--help para opciones)."
        )
        return
    limite = (
        args.limite
        if args.limite is not None
        else (CONFIG["smoke_limite"] if args.smoke else None)
    )
    rutas = resolver_rutas_salida(limite)
    rutas["log"].parent.mkdir(parents=True, exist_ok=True)
    with (
        rutas["log"].open("w", encoding="utf-8") as archivo,
        contextlib.redirect_stdout(_DuplicadorSalida(archivo)),
        contextlib.redirect_stderr(_DuplicadorSalida(archivo)),
    ):
        try:
            _ejecutar(args, rutas, limite)
        except Exception as exc:
            # El traceback va al log via stderr redirigido; la marca evita
            # que el manejador externo lo duplique en consola.
            traceback.print_exc()
            exc._reportado_c2 = True
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - reporta el fallo y sale con 1
        if not getattr(exc, "_reportado_c2", False):
            traceback.print_exc()
        sys.exit(1)
