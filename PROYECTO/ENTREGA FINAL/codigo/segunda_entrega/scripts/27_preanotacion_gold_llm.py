"""27_preanotacion_gold_llm.py
Fecha: 2026-10-04
Descripcion: Pre-anotacion AUTOMATICA (LLM local via ollama) del mini-gold humano
B3 (resultados/gold_mini/muestra_gold.jsonl, 50 casos). Para cada caso construye
un prompt en espanol con el extracto del encabezado de la sentencia y el texto
(hasta ~1500 chars) de cada articulo candidato leido de
data/constitucion_articulos.parquet, y pide al modelo un JSON estricto con un
grado de relevancia por articulo (2 = fundamento, 1 = mencionado, 0 = no
relevante). Escribe un registro por par (caso, articulo) en
resultados/gold_mini/respuestas_{anotador}.jsonl (con el default --anotador llm
se conserva el historico respuestas_llm.jsonl) con el MISMO esquema del modo
--anotar de scripts/16_anotador_gold.py (campo anotador configurable por CLI)
mas los campos de trazabilidad automatico, modelo, fecha y tiempo_seg, de modo
que --resumen, --exportar-csv, --evaluar y --calcular-acuerdo del script 16 lo
lean como un anotador mas.
NUNCA debe presentarse ni reportarse como anotacion humana: es una senal previa
para acelerar la revision.

Diseno:
    - Entrada: muestra_gold.jsonl (indice, id, tipo, ano, periodo,
      tercil_dificultad, extracto) y candidatos.jsonl (candidatos con articulo,
      snippet y citado_ner). El extracto del encabezado se usa tal cual (ya
      viene recortado a ~1500 chars por el script 16).
    - Blinding estricto: el prompt incluye SOLO el extracto del encabezado y el
      texto de los articulos candidatos. NUNCA entran el id interno, el flag
      citado_ner, n_citas_ner, ndcg_b2_caso, tercil_dificultad ni resultados de
      modelos.
    - Texto de articulos: data/constitucion_articulos.parquet (numero, texto)
      con tope de CONFIG["chars_articulo"] (1500); si el articulo no estuviera
      en el catalogo se usa el snippet del pack de candidatos como respaldo.
    - Salida: resultados/gold_mini/respuestas_{anotador}.jsonl (default "llm":
      respuestas_llm.jsonl), un registro por par (id, articulo) con los campos
      exactos del script 16 (anotador, id, indice_candidato, articulo,
      nota_grado, fecha, tiempo_seg) mas automatico=true y modelo. Append
      incremental con flush; reanudable por par. La nota del modelo se valida
      pero no se persiste (esquema exacto).
    - Metadatos: resultados/gold_mini/preanotacion_llm_meta.json (modelo, fecha,
      anotador, conteos de casos y registros, fallos, tiempos, distribucion de
      grados y md5 de muestra/candidatos). Si el meta existente pertenece a otro
      anotador, la corrida escribe preanotacion_llm_meta_{anotador}.json para no
      pisarlo.
    - Fallos por caso (transporte, HTTP, JSON ilegible, grados faltantes o
      invalidos tras 2 reintentos con backoff): el caso NO se escribe, queda
      pendiente para --reanudar y se registra en el meta; la corrida continua.
    - Modelo qwen3.8 (modo pensamiento): el payload envia think=false para que
      el JSON no se pierda dentro de la cadena de razonamiento; format=json y
      temperature=0 con semilla fija para reproducibilidad. num_predict base
      800 (CONFIG) con escalado x2 por reintento y tope 3200: el smoke mostro
      que un caso con 19 candidatos trunca el JSON a 800 tokens
      (done_reason=length), de modo que el reintento amplia el presupuesto de
      salida en lugar de perder el caso.

Estado del arte aplicado:
    - Anotacion asistida por LLM local como senal previa (weak supervision sin
      API externa; temperatura 0 y semilla fija).
    - Relevancia graduada y nDCG (Jarvelin y Kekalainen, 2002) como marco del
      gold que se pre-anota.
    - Mismo protocolo de registro del script 16 para reutilizar --evaluar y
      --calcular-acuerdo con un anotador automatico claramente etiquetado.

Uso:
    /home/carlos/venv/bin/python scripts/27_preanotacion_gold_llm.py --smoke
    /home/carlos/venv/bin/python scripts/27_preanotacion_gold_llm.py
    /home/carlos/venv/bin/python scripts/27_preanotacion_gold_llm.py --limite 10
    /home/carlos/venv/bin/python scripts/27_preanotacion_gold_llm.py --anotador llm2
    /home/carlos/venv/bin/python scripts/27_preanotacion_gold_llm.py --modelo gemma4:26b-factual-128k
"""

import argparse
import importlib.util
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

CONFIG = {
    "semilla": 42,
    "anotador": "llm",
    "modelo_default": "qwen3.8:docs-32k-q3_k_s",
    "host_default": "http://127.0.0.1:11434",
    "n_smoke": 3,
    "chars_extracto": 1500,
    "chars_articulo": 1500,
    "chars_nota_max": 200,
    "timeout_lectura_seg": 300,
    "timeout_conexion_seg": 10,
    "timeout_tags_seg": 15,
    "reintentos": 2,
    "backoff_base_seg": 2,
    "num_ctx": 16384,
    "num_predict": 800,
    "num_predict_max": 3200,
    "think": False,
    "temperature": 0.0,
    "seed_llm": 42,
    "dir_smoke": "/tmp/opencode/gold_llm_smoke",
    "rutas_datos": {
        "muestra": "resultados/gold_mini/muestra_gold.jsonl",
        "candidatos": "resultados/gold_mini/candidatos.jsonl",
        "articulos": "data/constitucion_articulos.parquet",
    },
    "rutas_salida": {
        "respuestas": "resultados/gold_mini/respuestas_{anotador}.jsonl",
        "meta": "resultados/gold_mini/preanotacion_llm_meta.json",
    },
    "rutas_salida_smoke": {
        "respuestas": "/tmp/opencode/gold_llm_smoke/respuestas_{anotador}.jsonl",
        "meta": "/tmp/opencode/gold_llm_smoke/preanotacion_llm_meta.json",
    },
}

LOGGER_NOMBRE = "preanotacion_gold_llm_27"

# Cache de modulos reutilizados (sus nombres empiezan por digito).
_MODULOS: dict[str, object] = {}


# ===== SECCION 2: reutilizacion por importlib y utilidades =====


def _cargar_modulo(ruta: Path, nombre: str) -> object:
    """Importa un script por ruta (su nombre empieza por digito).

    Args:
        ruta: Ruta del archivo .py a cargar.
        nombre: Nombre interno del modulo en sys.modules.

    Returns:
        Modulo ya ejecutado.

    Raises:
        ImportError: Si el archivo no existe o no se puede cargar.
    """
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"no se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def modulo_16() -> object:
    """Importa scripts/16_anotador_gold.py por ruta (cacheado).

    Returns:
        Modulo 16 ya ejecutado (normalizacion de articulos del mini-gold).
    """
    if "16" not in _MODULOS:
        _MODULOS["16"] = _cargar_modulo(
            RAIZ / "scripts" / "16_anotador_gold.py", "anotador_gold_16"
        )
    return _MODULOS["16"]


def modulo_21() -> object:
    """Importa scripts/21_pack_validacion_m4.py por ruta (cacheado).

    Returns:
        Modulo 21 ya ejecutado (helpers de jsonl, md5 y extractos).
    """
    if "21" not in _MODULOS:
        _MODULOS["21"] = _cargar_modulo(
            RAIZ / "scripts" / "21_pack_validacion_m4.py", "pack_validacion_m4_21"
        )
    return _MODULOS["21"]


def modulo_22() -> object:
    """Importa scripts/22_preanotacion_llm_m4.py por ruta (cacheado).

    Returns:
        Modulo 22 ya ejecutado (extraccion del JSON devuelto por el LLM).
    """
    if "22" not in _MODULOS:
        _MODULOS["22"] = _cargar_modulo(
            RAIZ / "scripts" / "22_preanotacion_llm_m4.py", "preanotacion_llm_m4_22"
        )
    return _MODULOS["22"]


def normalizar_articulo(valor: object) -> str:
    """Normaliza un numero de articulo de la CP (delegado al modulo 16).

    Args:
        valor: Numero crudo (string o numerico).

    Returns:
        Numero normalizado ("86", "22A", "Transitorio 5", "Preambulo").
    """
    return modulo_16().normalizar_articulo(valor)


def cargar_jsonl(ruta: str | Path) -> list[dict]:
    """Carga un JSON Lines validando linea por linea (modulo 21).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Lista de registros.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si alguna linea no es JSON valido o el archivo esta vacio.
    """
    return modulo_21().cargar_jsonl(ruta)


def escribir_jsonl_append(ruta: str | Path, registro: dict) -> None:
    """Agrega un registro al final de un JSON Lines con flush (modulo 21).

    Args:
        ruta: Ruta del archivo (se crea si no existe).
        registro: Dict serializable.
    """
    modulo_21().escribir_jsonl_append(ruta, registro)


def md5_archivo(ruta: str | Path) -> str | None:
    """Calcula el md5 de un archivo (delegado al modulo 21).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    return modulo_21().md5_archivo(ruta)


def extracto_texto(texto: str, n_chars: int) -> str:
    """Extrae el inicio de un texto normalizando finales de linea (modulo 21).

    Args:
        texto: Texto completo.
        n_chars: Maximo de caracteres a conservar.

    Returns:
        Extracto del inicio (sin CR, con strip).
    """
    return modulo_21().extracto_texto(texto, n_chars)


def normalizar_anotador(nombre: str) -> str:
    """Normaliza el nombre del anotador para usarlo como archivo (modulo 21).

    Args:
        nombre: Nombre crudo.

    Returns:
        Nombre con solo letras, digitos, guion y guion bajo.

    Raises:
        ValueError: Si el nombre queda vacio.
    """
    return modulo_21().normalizar_anotador(nombre)


def extraer_json(contenido: str) -> dict | None:
    """Extrae el primer objeto JSON de un texto del modelo (modulo 22).

    Args:
        contenido: Texto crudo del modelo.

    Returns:
        Dict si se pudo parsear; None si el texto no contiene JSON valido.
    """
    return modulo_22().extraer_json(contenido)


def configurar_logger() -> logging.Logger:
    """Configura el logger de consola del script.

    Returns:
        Logger con formato fecha | nivel | mensaje, solo consola.
    """
    logger = logging.getLogger(LOGGER_NOMBRE)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    manejador = logging.StreamHandler()
    manejador.setFormatter(formato)
    logger.addHandler(manejador)
    logger.propagate = False
    return logger


def rutas_activas(smoke: bool, anotador: str | None = None) -> dict:
    """Construye el mapa de rutas de entrada/salida segun el modo smoke.

    Args:
        smoke: True para usar salidas exclusivamente en /tmp/opencode/gold_llm_smoke.
        anotador: Nombre del anotador para el archivo de respuestas (default:
            CONFIG["anotador"], que reproduce la ruta historica respuestas_llm.jsonl).

    Returns:
        Dict combinado de rutas de datos y de salidas.
    """
    rutas = {}
    rutas.update(CONFIG["rutas_datos"])
    rutas.update(CONFIG["rutas_salida_smoke"] if smoke else CONFIG["rutas_salida"])
    nombre = anotador or CONFIG["anotador"]
    rutas["respuestas"] = str(rutas["respuestas"]).format(anotador=nombre)
    return rutas


def truncar_nota(valor: object) -> str:
    """Normaliza la nota del modelo a texto plano de a lo sumo 200 caracteres.

    Args:
        valor: Valor crudo de la clave "nota".

    Returns:
        Nota en una linea, recortada a chars_nota_max.
    """
    texto = " ".join(str(valor or "").split())
    return texto[: CONFIG["chars_nota_max"]]


def cargar_respuestas_tolerante(ruta: str | Path) -> tuple[dict[tuple[str, str], dict], int, int]:
    """Carga respuestas (id, articulo) -> ultimo registro ignorando lineas corruptas.

    Solo cuentan como respondidos los registros con nota_grado en (0, 1, 2) e
    id/articulo presentes (mismo criterio del cargador del script 16).

    Args:
        ruta: Ruta del jsonl de respuestas (si no existe, dict vacio).

    Returns:
        Tupla (respuestas, lineas_corruptas, registros_invalidos).
    """
    archivo = Path(ruta)
    respuestas: dict[tuple[str, str], dict] = {}
    corruptas = 0
    invalidos = 0
    if not archivo.exists():
        return respuestas, 0, 0
    with open(archivo, encoding="utf-8") as manejador:
        for linea in manejador:
            linea = linea.strip()
            if not linea:
                continue
            try:
                registro = json.loads(linea)
            except json.JSONDecodeError:
                corruptas += 1
                continue
            if not isinstance(registro, dict):
                invalidos += 1
                continue
            grado = registro.get("nota_grado")
            if grado not in (0, 1, 2) or "id" not in registro or "articulo" not in registro:
                invalidos += 1
                continue
            respuestas[(str(registro["id"]), str(registro["articulo"]))] = registro
    return respuestas, corruptas, invalidos


# ===== SECCION 3: catalogo de articulos y prompts (blinding) =====


def cargar_catalogo_articulos(ruta: str | Path) -> dict[str, str]:
    """Carga el catalogo completo de articulos de la CP (497 filas, liviano).

    Args:
        ruta: Ruta del parquet con numero y texto de cada articulo.

    Returns:
        Dict numero normalizado -> texto del articulo.
    """
    filas = pq.read_table(ruta, columns=["numero", "texto"]).to_pylist()
    return {normalizar_articulo(fila["numero"]): (fila["texto"] or "") for fila in filas}


def texto_candidato(candidato: dict, catalogo: dict[str, str], max_chars: int) -> str:
    """Obtiene el texto de un articulo candidato con respaldo al snippet.

    Args:
        candidato: Candidato del pack (articulo, snippet).
        catalogo: Dict articulo -> texto completo de la CP.
        max_chars: Maximo de caracteres del texto a usar.

    Returns:
        Texto del catalogo recortado a max_chars; si no existe, el snippet.
    """
    texto = catalogo.get(candidato["articulo"], "")
    if not texto:
        texto = candidato.get("snippet", "")
    return extracto_texto(texto, max_chars)


def construir_prompt_sistema() -> str:
    """Construye el mensaje de sistema con rol, escala graduada y reglas.

    Returns:
        Prompt de sistema listo para ollama.
    """
    return """Eres un experto en derecho constitucional colombiano y en el analisis del vinculo entre sentencias de la Corte Constitucional y los articulos de la Constitucion Politica (CP).

Tarea: dada una sentencia (extracto del encabezado) y una lista de articulos candidatos de la CP, asigna a CADA articulo un grado de relevancia como fundamento normativo de lo decidido o de la ratio analizada por la sentencia.

Escala graduada:
- 2 = fundamento: el articulo es base normativa directa de lo decidido o de la ratio analizada; la sentencia lo cita o lo desarrolla como sustento central.
- 1 = mencionado: el articulo aparece en el texto o se relaciona de forma tangencial (transcripciones de normas, listados, contexto, citas de paso), pero no es base del analisis.
- 0 = no relevante: el articulo no tiene relacion con lo que la sentencia decide.

Reglas:
1. Decide UNICAMENTE con el extracto de la sentencia y el texto de los articulos entregados; no uses conocimiento externo ni supongas hechos que el texto no menciona.
2. Califica TODOS los articulos listados usando exactamente el identificador mostrado (por ejemplo "86"); no agregues articulos que no esten en la lista.
3. Responde SOLO un objeto JSON valido, sin texto adicional, con la clave "grados" y un objeto por articulo:
{"grados": [{"articulo": "86", "grado": 2, "nota": "<=200 caracteres"}]}
4. En "grado" usa 2, 1 o 0; en "nota" justifica la decision en una frase corta."""


def construir_prompt_usuario(caso: dict, candidatos: list[dict], catalogo: dict[str, str]) -> str:
    """Construye el mensaje de usuario con el extracto y los articulos candidatos.

    Blinding: NO se incluyen el id interno, citado_ner, n_citas_ner,
    ndcg_b2_caso, tercil_dificultad, periodo ni resultados de modelos; solo el
    extracto del encabezado y el texto de los articulos candidatos.

    Args:
        caso: Registro de muestra_gold.jsonl (con extracto).
        candidatos: Lista de candidatos del caso (articulo, snippet, citado_ner).
        catalogo: Dict articulo -> texto completo de la CP.

    Returns:
        Prompt de usuario listo para ollama.
    """
    max_chars = CONFIG["chars_articulo"]
    bloques = [
        f"### Articulo {candidato['articulo']}\n{texto_candidato(candidato, catalogo, max_chars)}"
        for candidato in candidatos
    ]
    extracto = extracto_texto(caso.get("extracto", ""), CONFIG["chars_extracto"])
    return (
        "Sentencia de la Corte Constitucional de Colombia (extracto del encabezado):\n"
        f"{extracto}\n\n"
        f"Articulos candidatos de la Constitucion Politica ({len(candidatos)} articulos; "
        f"se muestra el inicio de cada texto, hasta {max_chars} caracteres):\n\n"
        + "\n\n".join(bloques)
        + f"\n\nGradua los {len(candidatos)} articulos y responde SOLO el JSON pedido."
    )


# ===== SECCION 4: cliente ollama =====


def construir_payload(modelo: str, prompt_sistema: str, prompt_usuario: str, num_predict: int) -> dict:
    """Construye el cuerpo del POST /api/chat con opciones reproducibles.

    Se envia think=false porque los modelos qwen3.8 usan modo pensamiento: con
    un presupuesto corto la cadena de razonamiento consumiria los tokens y el
    JSON de respuesta nunca se emitiria (evidencia del script 22).

    Args:
        modelo: Nombre del modelo en ollama.
        prompt_sistema: Mensaje de sistema (rol + escala + reglas).
        prompt_usuario: Mensaje de usuario (extracto + articulos candidatos).
        num_predict: Presupuesto de tokens de salida (escalado en reintentos).

    Returns:
        Dict serializable con model, messages, format, think y options.
    """
    return {
        "model": modelo,
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario},
        ],
        "format": "json",
        "stream": False,
        "think": CONFIG["think"],
        "options": {
            "temperature": CONFIG["temperature"],
            "seed": CONFIG["seed_llm"],
            "num_ctx": CONFIG["num_ctx"],
            "num_predict": num_predict,
        },
    }


def verificar_modelo_disponible(host: str, modelo: str, logger: logging.Logger) -> None:
    """Comprueba que ollama responde y que el modelo pedido esta disponible.

    Args:
        host: URL base de ollama.
        modelo: Nombre del modelo a usar.
        logger: Logger activo.

    Raises:
        RuntimeError: Si ollama no responde o el modelo no aparece en /api/tags.
    """
    try:
        respuesta = requests.get(
            f"{host}/api/tags",
            timeout=(CONFIG["timeout_conexion_seg"], CONFIG["timeout_tags_seg"]),
        )
        respuesta.raise_for_status()
        nombres = [str(item.get("name", "")) for item in respuesta.json().get("models", [])]
    except Exception as exc:  # se re-lanza con contexto legible
        raise RuntimeError(f"ollama no responde en {host}: {type(exc).__name__}: {exc}") from exc
    if modelo not in nombres:
        raise RuntimeError(
            f"modelo {modelo!r} no disponible en {host}; disponibles: {sorted(nombres)}"
        )
    logger.info("ollama OK en %s | modelo %s disponible", host, modelo)


def consultar_ollama(host: str, payload: dict) -> tuple[str, str]:
    """Hace UN POST /api/chat y devuelve el contenido del mensaje.

    Los reintentos con backoff los maneja procesar_caso (a nivel de caso, para
    cubrir tambien respuestas ilegibles, truncadas o incompletas).

    Args:
        host: URL base de ollama.
        payload: Cuerpo del POST (ver construir_payload).

    Returns:
        Tupla (contenido, done_reason): texto del mensaje y razon de fin de
        generacion ("stop", "length", ...).

    Raises:
        RuntimeError: Si el transporte, el HTTP o el contenido fallan.
    """
    respuesta = requests.post(
        f"{host}/api/chat",
        json=payload,
        timeout=(CONFIG["timeout_conexion_seg"], CONFIG["timeout_lectura_seg"]),
    )
    if respuesta.status_code != 200:
        raise RuntimeError(f"HTTP {respuesta.status_code}: {respuesta.text[:200]}")
    datos = respuesta.json()
    if isinstance(datos, dict) and datos.get("error"):
        raise RuntimeError(f"ollama error: {datos['error']}")
    mensaje = datos.get("message") or {}
    contenido = str(mensaje.get("content") or "").strip()
    if not contenido and mensaje.get("thinking"):
        contenido = str(mensaje["thinking"]).strip()
    if not contenido:
        raise RuntimeError("respuesta sin message.content")
    return contenido, str(datos.get("done_reason") or "")


# ===== SECCION 5: parseo y validacion de grados =====


def normalizar_grado(valor: object) -> int | None:
    """Normaliza el grado devuelto por el modelo a 0, 1 o 2.

    Args:
        valor: Valor crudo de la clave "grado" (int, float integral o string).

    Returns:
        Grado entero 0/1/2; None si no es valido (incluye booleanos).
    """
    if isinstance(valor, bool):
        return None
    if isinstance(valor, str):
        texto = valor.strip()
        if texto.lstrip("-").isdigit():
            valor = int(texto)
        else:
            return None
    if isinstance(valor, float) and valor.is_integer():
        valor = int(valor)
    if valor in (0, 1, 2) and isinstance(valor, int):
        return int(valor)
    return None


def parsear_grados(contenido: str, candidatos: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """Valida el JSON {"grados": [...]} del modelo contra los candidatos.

    Args:
        contenido: Texto crudo devuelto por el modelo.
        candidatos: Candidatos del caso (identificadores validos).

    Returns:
        Tupla (grados, problemas): grados es dict articulo -> {grado, nota};
        problemas queda vacio solo si TODOS los candidatos tienen un grado
        0/1/2 valido (en otro caso el caso no se escribe y se reintenta).
    """
    datos = extraer_json(contenido)
    if datos is None:
        return {}, ["json ilegible"]
    lista = datos.get("grados")
    if not isinstance(lista, list):
        return {}, ["sin clave 'grados' o no es lista"]
    articulos_validos = {candidato["articulo"] for candidato in candidatos}
    grados: dict[str, dict] = {}
    invalidos: list[str] = []
    for item in lista:
        if not isinstance(item, dict):
            invalidos.append(f"item no dict: {str(item)[:60]}")
            continue
        articulo = normalizar_articulo(item.get("articulo"))
        if articulo not in articulos_validos:
            coincidencia = re.search(r"(\d+)\s*([A-Za-z]?)", str(item.get("articulo") or ""))
            if coincidencia:
                articulo = normalizar_articulo(f"{coincidencia.group(1)}{coincidencia.group(2)}")
        if articulo not in articulos_validos:
            invalidos.append(f"articulo fuera del pool: {item.get('articulo')!r}")
            continue
        grado = normalizar_grado(item.get("grado"))
        if grado is None:
            invalidos.append(f"grado invalido para {articulo}: {item.get('grado')!r}")
            continue
        if articulo not in grados:
            grados[articulo] = {"grado": grado, "nota": truncar_nota(item.get("nota"))}
    faltantes = [c["articulo"] for c in candidatos if c["articulo"] not in grados]
    problemas: list[str] = []
    if invalidos:
        problemas.append("; ".join(invalidos[:5]))
    if faltantes:
        problemas.append(f"sin grado: {faltantes[:10]}")
    return grados, problemas


# ===== SECCION 6: procesamiento de casos y registros =====


def procesar_caso(
    caso: dict,
    candidatos: list[dict],
    catalogo: dict[str, str],
    prompt_sistema: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[dict[str, dict], list[str], float]:
    """Consulta el LLM para un caso con reintentos y backoff exponencial.

    El presupuesto de salida arranca en CONFIG["num_predict"] (800) y se
    duplica en cada reintento hasta CONFIG["num_predict_max"] (3200): el smoke
    mostro que un caso con muchos candidatos trunca el JSON a 800 tokens
    (done_reason=length) y el reintento amplia el presupuesto en lugar de
    perder el caso.

    Args:
        caso: Registro de muestra_gold.jsonl.
        candidatos: Candidatos del caso (prompt y validacion).
        catalogo: Dict articulo -> texto completo de la CP.
        prompt_sistema: Prompt de sistema.
        args: Argumentos CLI (modelo, host).
        logger: Logger activo.

    Returns:
        Tupla (grados, problemas, tiempo_seg): grados articulo -> {grado, nota};
        problemas vacio si la respuesta cubre todos los candidatos con grados
        validos; en otro caso el caso queda pendiente y se registra el motivo.
    """
    prompt_usuario = construir_prompt_usuario(caso, candidatos, catalogo)
    t_caso = time.perf_counter()
    problemas: list[str] = []
    for intento in range(CONFIG["reintentos"] + 1):
        num_predict = min(CONFIG["num_predict"] * (2**intento), CONFIG["num_predict_max"])
        payload = construir_payload(args.modelo, prompt_sistema, prompt_usuario, num_predict)
        if intento > 0:
            espera = CONFIG["backoff_base_seg"] ** intento
            logger.warning(
                "caso %s: reintento %d/%d en %.0f s con num_predict=%d (%s)",
                caso["id"],
                intento,
                CONFIG["reintentos"],
                espera,
                num_predict,
                "; ".join(problemas),
            )
            time.sleep(espera)
        problemas = []
        try:
            contenido, razon = consultar_ollama(args.host, payload)
        except Exception as exc:  # noqa: BLE001 - un caso no debe abortar la corrida
            problemas.append(f"{type(exc).__name__}: {exc}")
            continue
        grados, problemas = parsear_grados(contenido, candidatos)
        if not problemas:
            return grados, [], time.perf_counter() - t_caso
        if razon:
            problemas = [f"{problema} [done_reason={razon}]" for problema in problemas]
    return {}, problemas, time.perf_counter() - t_caso


def construir_registro(
    caso: dict,
    candidato: dict,
    datos: dict,
    modelo: str,
    tiempo_seg: float,
    anotador: str,
) -> dict:
    """Construye el registro compatible con el modo --anotar del script 16.

    Campos exactos del script 16: anotador, id, indice_candidato, articulo,
    nota_grado, fecha, tiempo_seg. Se agregan automatico=true y modelo para que
    la preanotacion NUNCA se confunda con anotacion humana.

    Args:
        caso: Registro de muestra_gold.jsonl.
        candidato: Candidato calificado (indice, articulo).
        datos: Dict {grado, nota} del parseo.
        modelo: Nombre del modelo usado.
        tiempo_seg: Tiempo medio por candidato del caso.
        anotador: Nombre del anotador de la corrida (--anotador).

    Returns:
        Dict serializable listo para escribir_jsonl_append.
    """
    return {
        "anotador": anotador,
        "id": str(caso["id"]),
        "indice_candidato": int(candidato["indice"]),
        "articulo": candidato["articulo"],
        "nota_grado": int(datos["grado"]),
        "fecha": pd.Timestamp.now().isoformat(timespec="seconds"),
        "tiempo_seg": round(tiempo_seg, 1),
        "automatico": True,
        "modelo": modelo,
    }


def construir_meta(
    args: argparse.Namespace,
    rutas: dict,
    muestra: list[dict],
    total_candidatos: int,
    previos: dict[tuple[str, str], dict],
    casos_objetivo: list[dict],
    resumen: dict,
    tiempo_total: float,
    corruptas: int,
    invalidos: int,
) -> dict:
    """Construye el registro de metadatos de la corrida de preanotacion.

    Args:
        args: Argumentos CLI.
        rutas: Dict de rutas activas.
        muestra: Muestra completa (50 casos reales, 3 smoke).
        total_candidatos: Total de pares (caso, articulo) de la muestra.
        previos: Respuestas previas ((id, articulo) -> registro).
        casos_objetivo: Casos procesados en esta corrida.
        resumen: Acumulador de la corrida (casos_ok, fallos, escritos, grados).
        tiempo_total: Tiempo total de la corrida en segundos.
        corruptas: Lineas corruptas ignoradas al cargar respuestas previas.
        invalidos: Registros invalidos ignorados al cargar respuestas previas.

    Returns:
        Dict serializable de metadatos.
    """
    tiempos = resumen["tiempos_caso"]
    media = (sum(tiempos) / len(tiempos)) if tiempos else None
    ruta_respuestas = Path(rutas["respuestas"])
    escritos = resumen["registros_escritos"]
    return {
        "script": "scripts/27_preanotacion_gold_llm.py",
        "tipo_anotacion": "automatica_llm",
        "advertencia": "Preanotacion automatica con LLM local; NO es anotacion humana",
        "fecha": pd.Timestamp.now().isoformat(timespec="seconds"),
        "modelo": args.modelo,
        "host": args.host,
        "smoke": bool(args.smoke),
        "anotador": args.anotador,
        "automatico": True,
        "chars_articulo": CONFIG["chars_articulo"],
        "limite": CONFIG["n_smoke"] if args.smoke else args.limite,
        "reanudar": bool(args.reanudar),
        "n_casos_muestra": len(muestra),
        "n_candidatos_muestra": total_candidatos,
        "n_casos_objetivo": len(casos_objetivo),
        "n_casos_ok": resumen["casos_ok"],
        "n_casos_fallo": resumen["casos_fallo"],
        "n_registros_previos": len(previos),
        "n_registros_escritos": escritos,
        "n_registros_total": len(previos) + escritos,
        "fallos": resumen["fallos"][:50],
        "distribucion_grados": {str(k): v for k, v in sorted(resumen["grados"].items())},
        "lineas_corruptas_ignoradas": corruptas,
        "registros_invalidos_ignorados": invalidos,
        "tiempo_total_seg": round(tiempo_total, 1),
        "tiempo_medio_caso_seg": (round(media, 2) if media is not None else None),
        "casos_por_minuto": (
            round(resumen["casos_ok"] / (tiempo_total / 60.0), 2)
            if (resumen["casos_ok"] and tiempo_total > 0)
            else None
        ),
        "md5_muestra": md5_archivo(rutas["muestra"]),
        "md5_candidatos": md5_archivo(rutas["candidatos"]),
        "md5_respuestas": md5_archivo(ruta_respuestas) if ruta_respuestas.exists() else None,
        "ruta_respuestas": str(ruta_respuestas),
        "config_llm": {
            "format": "json",
            "think": CONFIG["think"],
            "temperature": CONFIG["temperature"],
            "seed": CONFIG["seed_llm"],
            "num_ctx": CONFIG["num_ctx"],
            "num_predict": CONFIG["num_predict"],
            "num_predict_max": CONFIG["num_predict_max"],
            "timeout_lectura_seg": CONFIG["timeout_lectura_seg"],
            "reintentos": CONFIG["reintentos"],
        },
        "blinding": (
            "el prompt usa solo el extracto del encabezado y el texto de los articulos "
            "candidatos; excluye id, citado_ner, n_citas_ner, ndcg_b2_caso, "
            "tercil_dificultad y cualquier resultado de modelos"
        ),
    }


def ruta_meta_para_anotador(ruta: str | Path, anotador: str, logger: logging.Logger) -> Path:
    """Elige el destino del meta sin pisar el de otro anotador.

    Si el archivo base existe y pertenece a un anotador distinto, devuelve
    preanotacion_llm_meta_{anotador}.json para conservar el historico; si no
    existe, esta corrupto o es del mismo anotador, devuelve la ruta base.

    Args:
        ruta: Ruta base del meta (CONFIG["rutas_salida"]).
        anotador: Anotador de la corrida actual (--anotador).
        logger: Logger activo.

    Returns:
        Ruta de escritura del meta de la corrida.
    """
    destino = Path(ruta)
    if not destino.exists():
        return destino
    try:
        previo = json.loads(destino.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return destino
    anotador_previo = str(previo.get("anotador", "")) if isinstance(previo, dict) else ""
    if not anotador_previo or anotador_previo == anotador:
        return destino
    alterno = destino.with_name(f"{destino.stem}_{anotador}{destino.suffix}")
    logger.warning(
        "meta existente del anotador %s: se conserva %s y esta corrida escribe %s",
        anotador_previo,
        destino,
        alterno,
    )
    return alterno


def escribir_meta(ruta: str | Path, meta: dict) -> None:
    """Escribe el JSON de metadatos de la corrida (sobrescribe el anterior).

    Args:
        ruta: Ruta de salida.
        meta: Dict serializable de metadatos.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as archivo:
        json.dump(meta, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")


# ===== SECCION 7: orquestacion de la corrida =====


def calcular_pendientes(
    muestra: list[dict],
    candidatos_por_id: dict[str, list[dict]],
    previos: dict[tuple[str, str], dict],
    logger: logging.Logger,
) -> dict[str, list[dict]]:
    """Calcula los pares (caso, articulo) pendientes por caso (reanudacion).

    Args:
        muestra: Registros de muestra_gold.jsonl.
        candidatos_por_id: Candidatos por id de caso.
        previos: Respuestas previas ((id, articulo) -> registro).
        logger: Logger activo.

    Returns:
        Dict id_caso -> lista de candidatos sin respuesta (solo casos con
        pendientes).
    """
    pendientes: dict[str, list[dict]] = {}
    sin_candidatos = 0
    for caso in muestra:
        id_caso = str(caso["id"])
        lista = candidatos_por_id.get(id_caso, [])
        if not lista:
            sin_candidatos += 1
            continue
        faltan = [c for c in lista if (id_caso, c["articulo"]) not in previos]
        if faltan:
            pendientes[id_caso] = faltan
    if sin_candidatos:
        logger.warning("%d casos de la muestra sin candidatos en candidatos.jsonl", sin_candidatos)
    return pendientes


def procesar_casos_objetivo(
    casos_objetivo: list[dict],
    candidatos_por_id: dict[str, list[dict]],
    pendientes_por_caso: dict[str, list[dict]],
    catalogo: dict[str, str],
    prompt_sistema: str,
    args: argparse.Namespace,
    logger: logging.Logger,
    ruta_respuestas: Path,
) -> dict:
    """Procesa los casos objetivo y agrega cada grado al jsonl (append con flush).

    Un caso con fallo tras los reintentos NO se escribe (queda pendiente para
    --reanudar) y se registra en el resumen; la corrida continua.

    Args:
        casos_objetivo: Casos a procesar en esta corrida.
        candidatos_por_id: Candidatos por id de caso.
        pendientes_por_caso: Pares pendientes por caso (reanudacion).
        catalogo: Dict articulo -> texto completo de la CP.
        prompt_sistema: Prompt de sistema.
        args: Argumentos CLI.
        logger: Logger activo.
        ruta_respuestas: Ruta del jsonl de respuestas (append incremental).

    Returns:
        Dict con casos_ok, casos_fallo, registros_escritos, fallos, tiempos_caso
        y grados (Counter).
    """
    resumen = {
        "casos_ok": 0,
        "casos_fallo": 0,
        "registros_escritos": 0,
        "fallos": [],
        "tiempos_caso": [],
        "grados": Counter(),
    }
    total = len(casos_objetivo)
    try:
        for posicion, caso in enumerate(casos_objetivo, 1):
            id_caso = str(caso["id"])
            candidatos = candidatos_por_id.get(id_caso, [])
            pendientes = pendientes_por_caso[id_caso]
            grados, problemas, t_caso = procesar_caso(
                caso, candidatos, catalogo, prompt_sistema, args, logger
            )
            if problemas:
                resumen["casos_fallo"] += 1
                resumen["fallos"].append(
                    {
                        "id": id_caso,
                        "error": "; ".join(problemas),
                        "intentos": CONFIG["reintentos"] + 1,
                    }
                )
                logger.warning(
                    "[%d/%d] caso %s sin grados tras %d intentos: %s",
                    posicion,
                    total,
                    id_caso,
                    CONFIG["reintentos"] + 1,
                    "; ".join(problemas),
                )
                continue
            tiempo_candidato = t_caso / len(pendientes) if pendientes else 0.0
            escritos = 0
            for candidato in pendientes:
                datos = grados.get(candidato["articulo"])
                if datos is None:
                    continue
                registro = construir_registro(
                    caso, candidato, datos, args.modelo, tiempo_candidato, args.anotador
                )
                escribir_jsonl_append(ruta_respuestas, registro)
                resumen["grados"][registro["nota_grado"]] += 1
                escritos += 1
            resumen["casos_ok"] += 1
            resumen["registros_escritos"] += escritos
            resumen["tiempos_caso"].append(round(t_caso, 1))
            logger.info(
                "[%d/%d] caso %s -> %d registros en %.1f s",
                posicion,
                total,
                id_caso,
                escritos,
                t_caso,
            )
    except KeyboardInterrupt:
        logger.warning("interrumpido por el usuario: se conserva lo escrito (use --reanudar)")
    return resumen


def ejecutar_preanotacion(args: argparse.Namespace, logger: logging.Logger, rutas: dict) -> None:
    """Orquesta una corrida de preanotacion (smoke o real, reanudable).

    Args:
        args: Argumentos CLI.
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        FileExistsError: Si el archivo de respuestas existe y no hay --reanudar.
        RuntimeError: Si ollama no responde o el modelo no esta disponible.
    """
    t0 = time.perf_counter()
    ruta_respuestas = Path(rutas["respuestas"])
    if ruta_respuestas.exists() and not args.reanudar:
        raise FileExistsError(
            "el archivo de respuestas ya existe y --no-reanudar no sobrescribe: "
            f"{ruta_respuestas} (use --reanudar para continuar)"
        )
    host = args.host.rstrip("/")
    if args.smoke and args.limite is not None and args.limite != CONFIG["n_smoke"]:
        logger.warning("--smoke fuerza limite=%d (--limite=%s ignorado)", CONFIG["n_smoke"], args.limite)
    verificar_modelo_disponible(host, args.modelo, logger)
    muestra = cargar_jsonl(rutas["muestra"])
    candidatos_por_id = {
        str(registro["id"]): registro["candidatos"]
        for registro in cargar_jsonl(rutas["candidatos"])
    }
    previos, corruptas, invalidos = cargar_respuestas_tolerante(ruta_respuestas)
    if corruptas:
        logger.warning("lineas corruptas ignoradas en %s: %d", ruta_respuestas, corruptas)
    if invalidos:
        logger.warning("registros invalidos ignorados en %s: %d", ruta_respuestas, invalidos)
    ids_muestra = {str(caso["id"]) for caso in muestra}
    huerfanos = sorted({id_caso for id_caso, _ in previos} - ids_muestra)
    if huerfanos:
        logger.warning("respuestas previas de ids fuera de la muestra: %d", len(huerfanos))
    pendientes_por_caso = calcular_pendientes(muestra, candidatos_por_id, previos, logger)
    casos_objetivo = [caso for caso in muestra if str(caso["id"]) in pendientes_por_caso]
    limite = CONFIG["n_smoke"] if args.smoke else args.limite
    if limite is not None:
        casos_objetivo = casos_objetivo[:limite]
    total_candidatos = sum(len(lista) for lista in candidatos_por_id.values())
    logger.info(
        "muestra=%d casos (%d candidatos) | grados previos=%d | casos pendientes=%d | objetivo=%d%s",
        len(muestra),
        total_candidatos,
        len(previos),
        len(pendientes_por_caso),
        len(casos_objetivo),
        " (smoke)" if args.smoke else "",
    )
    if casos_objetivo:
        catalogo = cargar_catalogo_articulos(rutas["articulos"])
        prompt_sistema = construir_prompt_sistema()
        logger.info("catalogo de articulos: %d articulos", len(catalogo))
        resumen = procesar_casos_objetivo(
            casos_objetivo,
            candidatos_por_id,
            pendientes_por_caso,
            catalogo,
            prompt_sistema,
            args,
            logger,
            ruta_respuestas,
        )
    else:
        resumen = {
            "casos_ok": 0,
            "casos_fallo": 0,
            "registros_escritos": 0,
            "fallos": [],
            "tiempos_caso": [],
            "grados": Counter(),
        }
        logger.info("no hay casos pendientes: nada que preanotar")
    tiempo_total = time.perf_counter() - t0
    meta = construir_meta(
        args,
        rutas,
        muestra,
        total_candidatos,
        previos,
        casos_objetivo,
        resumen,
        tiempo_total,
        corruptas,
        invalidos,
    )
    ruta_meta = ruta_meta_para_anotador(rutas["meta"], args.anotador, logger)
    escribir_meta(ruta_meta, meta)
    logger.info("meta guardada: %s", ruta_meta)
    logger.info(
        "corrida: casos_ok=%d fallos=%d registros=%d | grados=%s | total=%d/%d pares | %.1f s",
        resumen["casos_ok"],
        resumen["casos_fallo"],
        resumen["registros_escritos"],
        dict(sorted(resumen["grados"].items())),
        len(previos) + resumen["registros_escritos"],
        total_candidatos,
        tiempo_total,
    )


# ===== SECCION 8: CLI y main =====


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Args:
        argv: Lista de argumentos (None usa sys.argv).

    Returns:
        Namespace con modelo, host, anotador, limite, reanudar y smoke.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pre-anotacion AUTOMATICA (LLM local via ollama) del mini-gold B3; "
            "compatible con --evaluar y --calcular-acuerdo del script 16. No es "
            "anotacion humana."
        )
    )
    parser.add_argument("--modelo", default=CONFIG["modelo_default"], help="modelo de ollama")
    parser.add_argument("--host", default=CONFIG["host_default"], help="URL base de ollama")
    parser.add_argument(
        "--anotador",
        default=CONFIG["anotador"],
        help=(
            "nombre del anotador: define respuestas_{anotador}.jsonl y el campo "
            '"anotador" del registro (default llm, historico respuestas_llm.jsonl)'
        ),
    )
    parser.add_argument("--limite", type=int, default=None, help="maximo de casos a procesar")
    parser.add_argument(
        "--reanudar",
        action="store_true",
        default=True,
        dest="reanudar",
        help="continua desde las respuestas existentes (default)",
    )
    parser.add_argument(
        "--no-reanudar",
        action="store_false",
        dest="reanudar",
        help="aborta si el archivo de respuestas ya existe (no sobrescribe)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="3 casos y salida solo en /tmp/opencode/gold_llm_smoke",
    )
    args = parser.parse_args(argv)
    try:
        args.anotador = normalizar_anotador(args.anotador)
    except ValueError as exc:
        parser.error(f"--anotador invalido: {exc}")
    if args.limite is not None and args.limite < 1:
        parser.error("--limite debe ser >= 1")
    return args


def main(argv: list[str] | None = None) -> None:
    """Orquesta la corrida de preanotacion del mini-gold.

    Args:
        argv: Argumentos CLI (None usa sys.argv).
    """
    args = parse_args(argv)
    os.chdir(RAIZ)
    logger = configurar_logger()
    random.seed(CONFIG["semilla"])
    np.random.seed(CONFIG["semilla"])
    rutas = rutas_activas(args.smoke, args.anotador)
    logger.info(
        "=== INICIO preanotacion LLM mini-gold B3 | smoke=%s modelo=%s host=%s anotador=%s ===",
        args.smoke,
        args.modelo,
        args.host,
        args.anotador,
    )
    if args.smoke:
        Path(CONFIG["dir_smoke"]).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ejecutar_preanotacion(args, logger, rutas)
    logger.info("=== FIN preanotacion LLM mini-gold B3 en %.1f s ===", time.perf_counter() - t0)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
