"""agent.py
Fecha: 2026-10-31
Descripcion: Agente de la Entrega Final con tres capas de guardrails.
    (1) Router determinista de intencion (saludo/small talk, fuera de alcance,
    en alcance, ambiguo) que resuelve los casos triviales sin LLM ni
    herramientas. (2) Agente de tool calling sobre LangChain: el modelo no
    responde de memoria, debe llamar a las herramientas y, si no hay
    evidencia, devolver el mensaje de evidencia insuficiente. (3) Verificacion
    de grounding: toda cita (canon de sentencia o articulo de la CP) que
    aparezca en la respuesta debe existir en la salida de las herramientas;
    si no, se descarta la redaccion libre y se devuelve la evidencia cruda.
    Si el modelo no soporta tool calling (o el bind/invoke falla), se activa
    el fallback documentado: router + plan determinista de herramientas + LLM
    solo para redactar sobre la evidencia.
Estado del arte aplicado: tool calling con langchain_core/langchain_ollama
    (ChatOllama), control explicito del bucle de herramientas para registrar
    la traza, temperature=0 y verificacion post-hoc de citas.
"""

import json
import re
import time
from typing import Any

import httpx
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama

import core
from tools import MAPA_TOOLS, TOOLS

# ===== SECCION 1: configuracion =====
OLLAMA_URL = "http://127.0.0.1:11434"
MODELO_DEFECTO = "qwen3.8:diario-8k-q3_k_s"
NUM_CTX = 8192
NUM_PREDICT = 512
TIMEOUT_LLM_S = 180.0
TIMEOUT_OLLAMA_S = 5.0
MAX_PASOS = 4
MAX_CITAS_PROMPT = 8
MAX_ARTICULOS_PROMPT = 8

MSG_SIN_EVIDENCIA = (
    "No encontré evidencia suficiente en el corpus para responder."
)
MSG_SALUDO = (
    "Hola. Soy el asistente de Juris-PLN, un sistema de investigación sobre "
    "23.742 sentencias de la Corte Constitucional de Colombia (1992-2021). "
    "Puedo validar una providencia, identificar los artículos de la "
    "Constitución vinculados y explorar las sentencias que cita. ¿Sobre qué "
    "providencia quieres consultar?"
)
MSG_IDENTIDAD = (
    "Soy el asistente de Juris-PLN. Trabajo sobre el corpus de sentencias de "
    "la Corte Constitucional de Colombia (1992-2021) y puedo: validar si un "
    "texto es una providencia, identificar los artículos de la Constitución "
    "vinculados (por id, canon como T-293/97 o texto) y listar las sentencias "
    "citadas con su área y PageRank. No respondo temas ajenos a la "
    "jurisprudencia constitucional."
)
MSG_GRACIAS = (
    "Con gusto. Si necesitas otra consulta sobre jurisprudencia "
    "constitucional, aquí estoy."
)
MSG_DESPEDIDA = (
    "Hasta luego. Cuando quieras retomar una consulta sobre jurisprudencia "
    "constitucional, aquí estaré."
)
MSG_FUERA_ALCANCE = (
    "Ese tema queda fuera de mi alcance: soy un asistente de investigación "
    "jurídica especializado en jurisprudencia de la Corte Constitucional de "
    "Colombia (1992-2021). Puedo ayudarte a validar providencias, identificar "
    "los artículos de la Constitución vinculados y explorar las sentencias "
    "citadas. ¿Tienes una consulta sobre una providencia?"
)
MSG_ACLARACION = (
    "No estoy seguro de qué necesitas. Puedo: (1) validar si un texto es una "
    "providencia de la Corte Constitucional, (2) identificar los artículos de "
    "la Constitución vinculados a una sentencia (por id, canon como T-293/97 "
    "o texto) y (3) listar y contextualizar las sentencias que cita. "
    "Reformula tu consulta o indícame la sentencia."
)

PROMPT_SISTEMA = """Eres el asistente de investigación jurídica de Juris-PLN,
un sistema sobre 23.742 sentencias de la Corte Constitucional de Colombia
(1992-2021).

Reglas obligatorias:
1. No respondas de memoria: llama SIEMPRE a una herramienta para obtener
   evidencia antes de afirmar cualquier dato.
2. Solo puedes citar sentencias (por ejemplo T-293/97) y artículos de la
   Constitución (por ejemplo 13, 22A) que aparezcan en la salida de las
   herramientas. No inventes citas, ids ni números.
3. Si las herramientas no devuelven evidencia suficiente, responde
   exactamente: "No encontré evidencia suficiente en el corpus para responder."
4. Responde en español, en 120 palabras o menos, sin repetir JSON crudo.
5. Usa vincular_cp para los artículos de la Constitución vinculados (acepta
   id, canon o texto), contexto_citas para las sentencias citadas y
   explorar_articulo para el texto de un artículo. validar_sentencia solo si
   el usuario pega el texto completo de la providencia.
"""

PROMPT_REDACTOR = """Eres un redactor jurídico. Responde la pregunta del usuario
usando EXCLUSIVAMENTE la evidencia incluida en el mensaje. No agregues
conocimiento propio, no inventes citas ni números. Si la evidencia no permite
responder, dilo en una frase sin inventar. Responde en español, en 120
palabras o menos.
"""

# ===== SECCION 2: router determinista =====
RE_LEGAL = re.compile(
    r"(?:sentencia|providencia|corte\s+constitucional|jurisprudencia|"
    r"constituci[oó]n|tutela|h[aá]beas\s+corpus|inconstitucionalidad|"
    r"magistrad[oa]\s+ponente|exequible|inexequible|"
    r"derechos?\s+fundamental|bloque\s+de\s+constitucionalidad|"
    r"debido\s+proceso|igualdad|libertad|amparo|pensi[oó]n|despido|"
    r"laboral|penal|delito|sancio[nó]n|derechos?\s+(?:a|al|de)\s+\w+|"
    r"\bderechos?\b|"
    r"art[ií]culos?\s+\d|art\.\s*\d|"
    r"\b(?:T|C|SU)\s*-\s*\d{1,4}\b)",
    re.IGNORECASE,
)
RE_ASCENDENCIA = re.compile(
    r"^\W*(?:hola|muy\s+buen(?:os|as)|buen(?:os|as)\s+(?:d[ií]as|tardes|noches)|"
    r"buenas|qu[eé]\s+tal|c[oó]mo\s+est[aá]s|c[oó]mo\s+va(?:s)?|saludos)\b",
    re.IGNORECASE,
)
RE_GRACIAS = re.compile(
    r"\b(?:gracias|muchas\s+gracias|mil\s+gracias|te\s+agradezco)\b",
    re.IGNORECASE,
)
RE_DESPEDIDA = re.compile(
    r"\b(?:adi[oó]s|hasta\s+luego|chao|nos\s+vemos|hasta\s+pronto)\b",
    re.IGNORECASE,
)
RE_IDENTIDAD = re.compile(
    r"\b(?:qui[eé]n\s+eres|qu[eé]\s+eres|qu[eé]\s+puedes\s+hacer(?:me)?|"
    r"en\s+qu[eé]\s+puedes\s+ayudar(?:me|nos|le|les)?|"
    r"para\s+qu[eé]\s+sirves|c[oó]mo\s+funcionas)\b",
    re.IGNORECASE,
)
RE_FUERA_ALCANCE = re.compile(
    r"\b(?:mundial|f[uú]tbol|baloncesto|nba|champions|tenis|"
    r"f[oó]rmula\s+1|clima|pron[oó]stico\s+del\s+tiempo|receta|cocina|"
    r"chiste|bitcoin|criptomoneda|hor[oó]scopo|signo\s+zodiacal|"
    r"pel[ií]cula|netflix|serie\s+de\s+televisi[oó]n|"
    r"presidente\s+de\s+(?:estados\s+unidos|eeuu|rusia)|"
    r"capital\s+de\s+(?:francia|italia|alemania))\b",
    re.IGNORECASE,
)
RE_REFERENCIAL = re.compile(
    r"\b(?:sobre\s+eso|de\s+eso|eso|esto|aquello|lo\s+anterior|lo\s+mismo|"
    r"(?:el|un|este|ese)\s+documento|(?:el|este|ese)\s+archivo|"
    r"revisarl[oa]|expl[ií]came\s+mejor|no\s+entiendo|rep[ií]telo|"
    r"en\s+este\s+caso|cu[eé]ntame\s+algo)\b",
    re.IGNORECASE,
)
RE_LISTA_ARTICULOS = re.compile(
    r"\b(?:art[ií]culos?|arts?\.)\s+"
    r"((?:\d{1,3}[A-Za-z]?\s*(?:,|y|-|a|al)?\s*)+)",
    re.IGNORECASE,
)
RE_ID_EN_PREGUNTA = re.compile(
    r"\b(?:id|documento|sentencia)\s+(\d{3,6})\b", re.IGNORECASE
)
RE_BLOQUE_PENSAMIENTO = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL
)
RE_SALTOS = re.compile(r"\n{3,}")

DIRECTO = {
    "saludo": MSG_SALUDO,
    "identidad": MSG_IDENTIDAD,
    "agradecimiento": MSG_GRACIAS,
    "despedida": MSG_DESPEDIDA,
    "fuera_alcance": MSG_FUERA_ALCANCE,
    "ambiguo": MSG_ACLARACION,
}


def clasificar_intencion(texto: str) -> str:
    """Clasifica la intencion del mensaje de forma determinista.

    Orden: una consulta juridica gana sobre un saludo ("gracias, ¿y la
    sentencia T-293?"); despues los actos conversacionales (identidad,
    despedida, agradecimiento, saludo) y los temas explicitamente ajenos.
    Un mensaje sin marcadores juridicos ni conversacionales se trata como
    fuera de alcance, salvo que sea referencial o incompleto (ambiguo), que
    es el caso de "¿que dice sobre eso?" o "revisalo".

    Args:
        texto: Mensaje del usuario.

    Returns:
        Una de: en_alcance, saludo, identidad, agradecimiento, despedida,
        fuera_alcance o ambiguo.
    """
    if not isinstance(texto, str) or not texto.strip():
        return "ambiguo"
    if RE_LEGAL.search(texto):
        return "en_alcance"
    if RE_IDENTIDAD.search(texto):
        return "identidad"
    if RE_DESPEDIDA.search(texto):
        return "despedida"
    if RE_GRACIAS.search(texto):
        return "agradecimiento"
    if RE_ASCENDENCIA.search(texto):
        return "saludo"
    if RE_FUERA_ALCANCE.search(texto):
        return "fuera_alcance"
    if RE_REFERENCIAL.search(texto):
        return "ambiguo"
    return "fuera_alcance"


# ===== SECCION 3: ollama =====
def ollama_disponible() -> bool:
    """Comprueba si el servidor de ollama responde.

    Returns:
        True si /api/version devuelve 200.
    """
    try:
        respuesta = httpx.get(f"{OLLAMA_URL}/api/version", timeout=TIMEOUT_OLLAMA_S)
        return respuesta.status_code == 200
    except Exception:  # noqa: BLE001 - health no debe romper por ollama
        return False


def _soporta_tools(
    nombre: str, familia: str | None, capacidades: list[str]
) -> bool:
    """Estima si un modelo soporta tool calling (heuristico, por familia).

    Ollama no siempre declara la capacidad "tools" aunque el modelo emita
    tool calls: qwen3.8:diario-8k-q3_k_s responde tool_calls correctamente y
    solo anuncia completion/vision (verificado el 2026-10-31). Por eso manda
    la familia (qwen3 si, gemma/alia no) y la capacidad solo decide en
    familias desconocidas.

    Args:
        nombre: Nombre del modelo en ollama.
        familia: Familia declarada por ollama.
        capacidades: Capacidades declaradas por ollama.

    Returns:
        True si se espera tool calling.
    """
    familia_limpia = (familia or "").lower()
    nombre_limpio = (nombre or "").lower()
    if familia_limpia.startswith(("qwen3", "qwen35")):
        return True
    if "alia" in nombre_limpio or familia_limpia.startswith(("gemma", "llama")):
        return False
    return "tools" in (capacidades or [])


def listar_modelos() -> dict:
    """Lista los modelos de ollama con su soporte estimado de tool calling.

    Returns:
        Dict con disponible, defecto y modelos (sin los modelos que solo
        sirven para embeddings).
    """
    try:
        respuesta = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=TIMEOUT_OLLAMA_S)
        respuesta.raise_for_status()
        datos = respuesta.json()
    except Exception:  # noqa: BLE001 - la API responde igual sin ollama
        return {"disponible": False, "defecto": MODELO_DEFECTO, "modelos": []}
    modelos = []
    for entrada in datos.get("models", []):
        nombre = entrada.get("name") or entrada.get("model") or ""
        capacidades = entrada.get("capabilities") or []
        if capacidades == ["embedding"]:
            continue
        detalles = entrada.get("details") or {}
        tamano = entrada.get("size")
        modelos.append(
            {
                "nombre": nombre,
                "familia": detalles.get("family"),
                "parametros": detalles.get("parameter_size"),
                "tamano_gb": round(tamano / 1e9, 2) if tamano else None,
                "tool_calling": _soporta_tools(
                    nombre, detalles.get("family"), capacidades
                ),
                "defecto": nombre == MODELO_DEFECTO,
            }
        )
    modelos.sort(key=lambda fila: (not fila["defecto"], fila["nombre"]))
    return {"disponible": True, "defecto": MODELO_DEFECTO, "modelos": modelos}


def _crear_llm(modelo: str) -> ChatOllama:
    """Construye el cliente de ollama con timeout y contexto acotados.

    Args:
        modelo: Nombre del modelo en ollama.

    Returns:
        ChatOllama con temperature=0 y num_ctx de 8192 tokens.
    """
    return ChatOllama(
        model=modelo,
        base_url=OLLAMA_URL,
        temperature=0.0,
        seed=42,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        sync_client_kwargs={"timeout": TIMEOUT_LLM_S},
    )


# ===== SECCION 4: evidencia y grounding =====
def _evidencia_vacia() -> dict:
    """Crea el contenedor de evidencia de una conversacion."""
    return {
        "canones": set(),
        "articulos": set(),
        "ids": set(),
        "documentos": [],
        "articulos_detalle": {},
        "citas_detalle": [],
    }


def _canon_legible(datos: dict) -> str | None:
    """Normaliza el canon detectado de la salida de una herramienta."""
    canon = datos.get("canon_detectado")
    if not canon:
        return None
    return core.normalizar_canon(canon) or str(canon)


def _acumular_evidencia(
    evidencia: dict, nombre: str, datos: Any, argumentos: dict
) -> None:
    """Agrega la salida de una herramienta al conjunto de evidencia.

    Args:
        evidencia: Contenedor mutable de evidencia.
        nombre: Nombre de la herramienta ejecutada.
        datos: Salida ya deserializada (dict) o None si no era JSON.
        argumentos: Argumentos con los que se llamo la herramienta.
    """
    if not isinstance(datos, dict):
        return
    if "error" in datos:
        return
    canon = _canon_legible(datos)
    id_documento = datos.get("id")
    if canon or id_documento:
        existente = {
            (fila.get("canon"), fila.get("id")) for fila in evidencia["documentos"]
        }
        if (canon, id_documento) not in existente:
            evidencia["documentos"].append(
                {"canon": canon, "id": str(id_documento) if id_documento else None}
            )
    if canon:
        evidencia["canones"].add(canon)
    if id_documento:
        evidencia["ids"].add(str(id_documento))
    if nombre == "vincular_cp":
        for campo in ("articulos_extraidos", "articulos_recuperados"):
            for fila in datos.get(campo) or []:
                numero = str(fila.get("numero") or "").upper()
                if not numero:
                    continue
                evidencia["articulos"].add(numero)
                evidencia["articulos_detalle"].setdefault(
                    numero,
                    {
                        "numero": numero,
                        "score": fila.get("score"),
                        "bloque": fila.get("bloque"),
                    },
                )
    elif nombre == "contexto_citas":
        for fila in datos.get("citas") or []:
            canon_cita = core.normalizar_canon(fila.get("canon") or "")
            if canon_cita:
                evidencia["canones"].add(canon_cita)
            if fila.get("resuelta"):
                evidencia["citas_detalle"].append(fila)
    elif nombre == "explorar_articulo":
        numero = str(datos.get("numero") or argumentos.get("numero") or "").upper()
        if numero:
            evidencia["articulos"].add(numero)
            evidencia["articulos_detalle"].setdefault(
                numero,
                {"numero": numero, "score": None, "bloque": datos.get("bloque")},
            )
    elif nombre == "validar_sentencia":
        id_detectado = datos.get("id_detectado")
        if id_detectado:
            evidencia["ids"].add(str(id_detectado))


def _articulos_citados(texto: str) -> list[str]:
    """Extrae los numeros de articulo mencionados en un texto.

    Args:
        texto: Texto donde buscar "articulo 13", "articulos 13 y 23", etc.

    Returns:
        Lista de numeros normalizados en mayuscula, sin duplicados.
    """
    numeros: list[str] = []
    for coincidencia in RE_LISTA_ARTICULOS.finditer(texto):
        for numero in re.findall(r"\d{1,3}[A-Za-z]?", coincidencia.group(1)):
            limpio = numero.upper()
            if limpio not in numeros:
                numeros.append(limpio)
    return numeros


def verificar_grounding(texto: str, evidencia: dict) -> dict:
    """Verifica que las citas de la respuesta existan en la evidencia.

    Args:
        texto: Respuesta redactada por el modelo.
        evidencia: Contenedor de evidencia acumulada de las herramientas.

    Returns:
        Dict con ok, citas_verificadas y citas_no_verificadas. Los canones se
        comparan normalizados (T-293/97 == T-293-1997) y los articulos en
        mayuscula (22a == 22A).
    """
    verificadas: list[str] = []
    no_verificadas: list[str] = []
    canones = evidencia.get("canones", set())
    articulos = evidencia.get("articulos", set())
    for coincidencia in core.RE_CANON.finditer(texto or ""):
        canon = core.normalizar_canon(coincidencia.group(0))
        if canon is None:
            continue
        destino = verificadas if canon in canones else no_verificadas
        if canon not in destino:
            destino.append(canon)
    for numero in _articulos_citados(texto or ""):
        destino = verificadas if numero in articulos else no_verificadas
        if numero not in destino:
            destino.append(numero)
    verificadas.sort()
    no_verificadas.sort()
    return {
        "ok": not no_verificadas,
        "citas_verificadas": verificadas,
        "citas_no_verificadas": no_verificadas,
    }


def _detalle_articulo(fila: dict) -> str:
    """Formatea un articulo de la evidencia con su bloque y su score."""
    detalle = str(fila.get("numero"))
    if fila.get("bloque"):
        detalle += f" ({fila['bloque']})"
    if fila.get("score") is not None:
        detalle += f" score {fila['score']}"
    return detalle


def _lineas_evidencia(evidencia: dict) -> list[str]:
    """Construye las lineas legibles de la evidencia acumulada.

    Args:
        evidencia: Contenedor de evidencia acumulada.

    Returns:
        Lista de frases (documento, articulos y sentencias citadas).
    """
    lineas: list[str] = []
    if evidencia["documentos"]:
        documento = evidencia["documentos"][0]
        etiqueta = documento.get("canon") or "sin canon detectado"
        if documento.get("id"):
            etiqueta += f" (id {documento['id']})"
        lineas.append(f"Sentencia analizada: {etiqueta}.")
    filas = list(evidencia["articulos_detalle"].values())
    recuperados = sorted(
        (fila for fila in filas if fila.get("score") is not None),
        key=lambda fila: -float(fila["score"]),
    )[:MAX_ARTICULOS_PROMPT]
    extraidos = [
        fila for fila in filas if fila.get("score") is None
    ][:MAX_ARTICULOS_PROMPT]
    if recuperados:
        lineas.append(
            "Artículos recuperados por la búsqueda: "
            + "; ".join(_detalle_articulo(fila) for fila in recuperados)
            + "."
        )
    if extraidos:
        lineas.append(
            "Artículos citados en el texto: "
            + "; ".join(_detalle_articulo(fila) for fila in extraidos)
            + "."
        )
    if evidencia["citas_detalle"]:
        items = []
        for fila in evidencia["citas_detalle"][:MAX_CITAS_PROMPT]:
            canon = core.normalizar_canon(fila.get("canon") or "") or fila.get(
                "canon"
            )
            detalle = str(canon)
            if fila.get("area"):
                detalle += f" (área: {fila['area']})"
            items.append(detalle)
        lineas.append("Sentencias citadas resueltas: " + "; ".join(items) + ".")
    return lineas


def _respuesta_desde_evidencia(evidencia: dict) -> str:
    """Redacta una respuesta determinista a partir de la evidencia cruda.

    Se usa cuando el modelo cita referencias no verificadas o cuando no hay
    LLM disponible: toda frase sale de la salida de las herramientas.

    Args:
        evidencia: Contenedor de evidencia acumulada.

    Returns:
        Respuesta breve o el mensaje de evidencia insuficiente.
    """
    lineas = _lineas_evidencia(evidencia)
    if not lineas:
        return MSG_SIN_EVIDENCIA
    return "Evidencia recuperada por las herramientas. " + " ".join(lineas)


def _evidencia_para_prompt(evidencia: dict) -> str:
    """Formatea la evidencia como texto legible para el LLM redactor.

    Args:
        evidencia: Contenedor de evidencia acumulada.

    Returns:
        Bloque de texto (una linea por hallazgo) o un aviso de evidencia vacia.
    """
    lineas = _lineas_evidencia(evidencia)
    return "\n".join(lineas) if lineas else "Sin evidencia recuperada."


# ===== SECCION 5: ejecucion de herramientas =====
def _texto_de_ai(mensaje: AIMessage) -> str:
    """Extrae y limpia el texto de una respuesta del modelo."""
    contenido = mensaje.content
    if isinstance(contenido, list):
        partes = []
        for parte in contenido:
            if isinstance(parte, dict):
                partes.append(str(parte.get("text") or ""))
            else:
                partes.append(str(parte))
        contenido = " ".join(partes)
    texto = RE_BLOQUE_PENSAMIENTO.sub("", str(contenido or ""))
    return RE_SALTOS.sub("\n\n", texto).strip()


def _ejecutar_herramienta(nombre: str, argumentos: dict) -> tuple[str, bool, int]:
    """Ejecuta una herramienta y devuelve salida, exito y duracion.

    Args:
        nombre: Nombre de la herramienta.
        argumentos: Argumentos para la herramienta.

    Returns:
        Tupla (salida_texto, ok, ms).
    """
    inicio = time.perf_counter()
    herramienta = MAPA_TOOLS.get(nombre)
    if herramienta is None:
        salida = json.dumps(
            {"error": f"herramienta desconocida: {nombre}"}, ensure_ascii=False
        )
        return salida, False, int((time.perf_counter() - inicio) * 1000)
    try:
        salida = herramienta.invoke(argumentos or {})
        ok = True
    except Exception as exc:  # noqa: BLE001 - la traza registra el fallo
        salida = json.dumps({"error": str(exc)}, ensure_ascii=False)
        ok = False
    return salida, ok, int((time.perf_counter() - inicio) * 1000)


def _deserializar(salida: str) -> Any:
    """Convierte la salida JSON de una herramienta en dict (o None)."""
    try:
        return json.loads(salida)
    except (TypeError, ValueError):
        return None


def _resumen_traza(nombre: str, datos: Any) -> str:
    """Construye un resumen de una linea para la traza de tool calls."""
    if not isinstance(datos, dict):
        return "salida no JSON"
    if "error" in datos:
        return f"error: {str(datos['error'])[:160]}"
    if nombre == "vincular_cp":
        numeros = [
            str(fila.get("numero"))
            for fila in (datos.get("articulos_recuperados") or [])[:6]
        ]
        return (
            f"id={datos.get('id')} canon={datos.get('canon_detectado')} "
            f"articulos={','.join(numeros)}"
        )
    if nombre == "contexto_citas":
        return (
            f"id={datos.get('id')} citas={datos.get('n_citas')} "
            f"resueltas={datos.get('n_resueltas')}"
        )
    if nombre == "explorar_articulo":
        return f"articulo={datos.get('numero')} bloque={datos.get('bloque')}"
    if nombre == "validar_sentencia":
        return (
            f"es_sentencia={datos.get('es_sentencia')} "
            f"canon={datos.get('canon_detectado')}"
        )
    return "ok"


def _registrar_llamada(
    traza: list[dict],
    evidencia: dict,
    nombre: str,
    argumentos: dict,
    salida: str,
    ok: bool,
    ms: int,
) -> None:
    """Registra una llamada en la traza y acumula su evidencia."""
    datos = _deserializar(salida)
    ok_final = bool(ok and not (isinstance(datos, dict) and "error" in datos))
    _acumular_evidencia(evidencia, nombre, datos, argumentos)
    traza.append(
        {
            "herramienta": nombre,
            "argumentos": argumentos,
            "ok": ok_final,
            "ms": ms,
            "resumen": _resumen_traza(nombre, datos),
        }
    )


# ===== SECCION 6: agente con tool calling =====
def _ejecutar_agente(
    pregunta: str, modelo: str, rerank: bool, k: int, max_pasos: int
) -> dict:
    """Ejecuta el bucle de tool calling del agente.

    Args:
        pregunta: Consulta del usuario.
        modelo: Modelo de ollama.
        rerank: Valor por defecto del flag rerank en las herramientas.
        k: Profundidad de recuperacion por defecto.
        max_pasos: Maximo de turnos del modelo.

    Returns:
        Dict con respuesta, modo, tool_calls, grounding y advertencia.

    Raises:
        Exception: Si el modelo no soporta tools o la llamada falla; el
            llamador activa el fallback documentado.
    """
    llm = _crear_llm(modelo).bind_tools(TOOLS)
    mensajes = [
        SystemMessage(content=PROMPT_SISTEMA),
        HumanMessage(content=pregunta),
    ]
    traza: list[dict] = []
    evidencia = _evidencia_vacia()
    respuesta = ""
    for _ in range(max_pasos):
        mensaje = llm.invoke(mensajes)
        mensajes.append(mensaje)
        llamadas = getattr(mensaje, "tool_calls", None) or []
        if not llamadas:
            respuesta = _texto_de_ai(mensaje)
            break
        for llamada in llamadas:
            nombre = str(llamada.get("name") or "")
            argumentos = dict(llamada.get("args") or {})
            if nombre == "vincular_cp":
                # El backend manda sobre el flag de recuperacion y la
                # profundidad; el modelo solo aporta la referencia.
                argumentos["rerank"] = rerank
                argumentos.setdefault("k", k)
            salida, ok, ms = _ejecutar_herramienta(nombre, argumentos)
            _registrar_llamada(
                traza, evidencia, nombre, argumentos, salida, ok, ms
            )
            mensajes.append(
                ToolMessage(
                    content=salida[:6000],
                    tool_call_id=str(llamada.get("id") or ""),
                    name=nombre,
                )
            )
    sin_evidencia_usable = (
        not evidencia["articulos_detalle"] and not evidencia["citas_detalle"]
    )
    sin_respuesta = not respuesta.strip() or respuesta.strip() == MSG_SIN_EVIDENCIA
    if not traza or (sin_respuesta and sin_evidencia_usable):
        # El modelo no llamo herramientas, o las llamo sin recuperar nada
        # usable: se ejecuta el plan determinista para no descartar la
        # evidencia recuperable. La redaccion ya existe, no se repite la
        # llamada al LLM.
        return _ejecutar_fallback(
            pregunta,
            modelo,
            rerank,
            k,
            motivo=(
                "El modelo respondio sin evidencia; se aplico el plan "
                "determinista sobre la evidencia."
            ),
            usar_llm=False,
        )
    grounding = verificar_grounding(respuesta, evidencia)
    advertencia = None
    if respuesta and not grounding["ok"]:
        respuesta = _respuesta_desde_evidencia(evidencia)
        advertencia = (
            "La redacción del modelo citó referencias no verificadas en las "
            "herramientas ("
            + ", ".join(grounding["citas_no_verificadas"][:5])
            + "); se devuelve la evidencia sin redacción libre."
        )
    if not respuesta:
        respuesta = _respuesta_desde_evidencia(evidencia)
    modo = "agente"
    if respuesta == MSG_SIN_EVIDENCIA:
        modo = "sin_evidencia"
    return {
        "respuesta": respuesta,
        "modo": modo,
        "tool_calls": traza,
        "grounding": grounding,
        "advertencia": advertencia,
    }


# ===== SECCION 7: fallback determinista =====
def _plan_determinista(pregunta: str, rerank: bool, k: int) -> list[tuple]:
    """Planea las herramientas que un router determinista ejecutaria.

    Args:
        pregunta: Consulta del usuario.
        rerank: Flag de rerank para vincular_cp.
        k: Profundidad de recuperacion.

    Returns:
        Lista de tuplas (herramienta, argumentos).
    """
    llamadas: list[tuple] = []
    canon = core.normalizar_canon(pregunta)
    if canon:
        llamadas.append(
            ("vincular_cp", {"referencia": canon, "k": k, "rerank": rerank})
        )
        llamadas.append(("contexto_citas", {"referencia": canon}))
    else:
        coincidencia_id = RE_ID_EN_PREGUNTA.search(pregunta)
        referencia = coincidencia_id.group(1) if coincidencia_id else pregunta
        llamadas.append(
            ("vincular_cp", {"referencia": referencia, "k": k, "rerank": rerank})
        )
        if coincidencia_id:
            llamadas.append(("contexto_citas", {"referencia": referencia}))
    for numero in _articulos_citados(pregunta)[:2]:
        llamadas.append(("explorar_articulo", {"numero": numero}))
    return llamadas


def _ejecutar_fallback(
    pregunta: str,
    modelo: str,
    rerank: bool,
    k: int,
    motivo: str,
    usar_llm: bool = True,
) -> dict:
    """Fallback documentado: plan determinista + LLM solo para redactar.

    Args:
        pregunta: Consulta del usuario.
        modelo: Modelo de ollama para redactar.
        rerank: Flag de rerank.
        k: Profundidad de recuperacion.
        motivo: Causa del fallback (se expone en 'advertencia').
        usar_llm: False devuelve la evidencia cruda sin redactor (timeouts).

    Returns:
        Dict con respuesta, modo, tool_calls, grounding y advertencia.
    """
    traza: list[dict] = []
    evidencia = _evidencia_vacia()
    for nombre, argumentos in _plan_determinista(pregunta, rerank, k):
        salida, ok, ms = _ejecutar_herramienta(nombre, argumentos)
        _registrar_llamada(
            traza, evidencia, nombre, argumentos, salida, ok, ms
        )
    if not evidencia["articulos"] and not evidencia["canones"]:
        return {
            "respuesta": MSG_SIN_EVIDENCIA,
            "modo": "sin_evidencia",
            "tool_calls": traza,
            "grounding": {"ok": True, "citas_verificadas": [], "citas_no_verificadas": []},
            "advertencia": motivo,
        }
    respuesta = ""
    advertencia = motivo
    if usar_llm:
        try:
            llm = _crear_llm(modelo)
            mensaje = llm.invoke(
                [
                    SystemMessage(content=PROMPT_REDACTOR),
                    HumanMessage(
                        content=(
                            f"Pregunta: {pregunta}\n\n"
                            "Evidencia (salida de las herramientas):\n"
                            f"{_evidencia_para_prompt(evidencia)}"
                        )
                    ),
                ]
            )
            respuesta = _texto_de_ai(mensaje)
        except Exception as exc:  # noqa: BLE001 - degradacion a evidencia cruda
            advertencia = f"{motivo} | redactor fallo: {type(exc).__name__}: {exc}"
    grounding = verificar_grounding(respuesta, evidencia)
    hay_evidencia = bool(
        evidencia["articulos_detalle"] or evidencia["citas_detalle"]
    )
    if not respuesta or (
        respuesta.strip() == MSG_SIN_EVIDENCIA and hay_evidencia
    ):
        # El redactor se nego pese a tener evidencia: se devuelve la
        # evidencia cruda, que es verificable por construccion.
        respuesta = _respuesta_desde_evidencia(evidencia)
    elif respuesta and not grounding["ok"]:
        advertencia = (
            f"{advertencia} | citas no verificadas: "
            + ", ".join(grounding["citas_no_verificadas"][:5])
        )
        respuesta = _respuesta_desde_evidencia(evidencia)
    return {
        "respuesta": respuesta,
        "modo": "fallback_sin_tools",
        "tool_calls": traza,
        "grounding": grounding,
        "advertencia": advertencia,
    }


def _es_timeout(exc: Exception) -> bool:
    """Indica si la excepcion corresponde a un timeout de red."""
    return isinstance(exc, httpx.TimeoutException) or "timeout" in str(exc).lower()


# ===== SECCION 8: punto de entrada =====
def responder(
    pregunta: str,
    modelo: str | None = None,
    rerank: bool = False,
    k: int = 6,
    max_pasos: int = MAX_PASOS,
) -> dict:
    """Responde una consulta aplicando las tres capas de guardrails.

    Args:
        pregunta: Consulta del usuario.
        modelo: Modelo de ollama; None usa MODELO_DEFECTO.
        rerank: Activa la ruta de recuperacion B2+B1 en las herramientas.
        k: Articulos a recuperar en la herramienta de vinculo.
        max_pasos: Maximo de turnos del agente.

    Returns:
        Dict con respuesta, modo, intencion, tool_calls, grounding, modelo,
        latencia_ms y advertencia.
    """
    inicio = time.perf_counter()
    nombre_modelo = (modelo or MODELO_DEFECTO).strip() or MODELO_DEFECTO
    intencion = clasificar_intencion(pregunta)
    if intencion in DIRECTO:
        return {
            "respuesta": DIRECTO[intencion],
            "modo": {
                "saludo": "small_talk",
                "identidad": "small_talk",
                "agradecimiento": "small_talk",
                "despedida": "small_talk",
                "fuera_alcance": "fuera_alcance",
                "ambiguo": "aclaracion",
            }[intencion],
            "intencion": intencion,
            "tool_calls": [],
            "grounding": {
                "ok": True,
                "citas_verificadas": [],
                "citas_no_verificadas": [],
            },
            "modelo": nombre_modelo,
            "latencia_ms": int((time.perf_counter() - inicio) * 1000),
            "advertencia": None,
        }
    try:
        resultado = _ejecutar_agente(pregunta, nombre_modelo, rerank, k, max_pasos)
    except Exception as exc:  # noqa: BLE001 - fallback documentado
        motivo = (
            "fallback_sin_tools: el modelo no emitio tool calls o la llamada "
            f"fallo ({type(exc).__name__}: {str(exc)[:240]})"
        )
        resultado = _ejecutar_fallback(
            pregunta,
            nombre_modelo,
            rerank,
            k,
            motivo,
            usar_llm=not _es_timeout(exc),
        )
    resultado["intencion"] = intencion
    resultado["modelo"] = nombre_modelo
    resultado["latencia_ms"] = int((time.perf_counter() - inicio) * 1000)
    return resultado
