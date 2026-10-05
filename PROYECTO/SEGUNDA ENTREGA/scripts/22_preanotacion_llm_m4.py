"""22_preanotacion_llm_m4.py
Fecha: 2026-10-03
Descripcion: Pre-anotacion AUTOMATICA del pack de validacion humana M4
(resultados/validacion_m4/muestra_200.jsonl) con un LLM local servido por
ollama. Escribe respuestas con el MISMO esquema del modo --anotar del script
21 (anotador configurable con --anotador; default "llm_preanotacion") para que
--calcular-acuerdo las lea como un anotador mas. La anotacion es automatica
(campos automatico=true y modelo) y NUNCA debe presentarse ni reportarse como
anotacion humana: es una senal previa para acelerar la revision.

Diseno:
    - Entrada: resultados/validacion_m4/muestra_200.jsonl (id, tipo, ano,
      area_silver, top3_json, n_citas_cp, via, particion, en_holdout,
      extracto, n_chars_texto).
    - Blinding estricto: el prompt solo incluye tipo, ano y texto del
      documento; area_silver, top3_json, via y n_citas_cp NUNCA entran al
      prompt (el id tampoco: evita que el modelo use conocimiento memorizado
      de sentencias famosas).
    - Texto: se lee del corpus data/corpus_sentencias.parquet SOLO por id con
      filtros pyarrow (dataset particionado tipo/ano) y se usa el inicio hasta
      --max-chars caracteres; si el corpus no tiene el id se usa el extracto
      del pack como respaldo.
    - Taxonomia: las 16 clases se describen en una linea por bloque derivada
      de data/constitucion_articulos.parquet (rango de articulos + ejemplos).
    - Salida: resultados/validacion_m4/respuestas_{anotador}.jsonl (con el
      default, respuestas_llm_preanotacion.jsonl), un registro por documento
      con los campos del script 21 (anotador, id, indice, respuesta,
      confianza, nota, fecha, tiempo_seg) mas automatico y modelo. Append
      incremental tras cada documento (reanudable).
    - Metadatos: resultados/validacion_m4/preanotacion_llm_meta.json (modelo,
      host, fecha, conteos, tiempos, fallos, md5 de la muestra).
    - Modelo qwen3.8 (modo pensamiento): el payload envia think=false para que
      el JSON no se pierda dentro de la cadena de razonamiento con
      num_predict=160; sin think=false el smoke devolvio 3/3 "json ilegible"
      (evidencia cruda conservada en /tmp/opencode/o1_llm_smoke/evidencia_run1_thinking/).
    - Fallos de transporte/HTTP tras los reintentos: NO se escriben (quedan
      pendientes para una corrida con --reanudar) y se registran en el meta;
      respuestas ilegibles del modelo SI se escriben con respuesta="invalido"
      para que el acuerdo las cuente como no validas (n_otras).

Estado del arte aplicado:
    - Anotacion asistida por LLM como senal previa / weak supervision local
      (sin API externa, temperatura 0 y semilla fija para reproducibilidad).
    - Mismo protocolo de registro del script 21 para reutilizar el acuerdo
      humano vs silver (Cohen 1960; Fleiss 1971) con un "anotador" automatico
      claramente etiquetado.

Uso:
    /home/carlos/venv/bin/python scripts/22_preanotacion_llm_m4.py --smoke
    /home/carlos/venv/bin/python scripts/22_preanotacion_llm_m4.py --smoke --resumen
    /home/carlos/venv/bin/python scripts/22_preanotacion_llm_m4.py --resumen
    /home/carlos/venv/bin/python scripts/22_preanotacion_llm_m4.py --limite 20
    /home/carlos/venv/bin/python scripts/22_preanotacion_llm_m4.py --anotador llm2 --modelo gemma4:26b-factual-128k --max-chars 12000
    /home/carlos/venv/bin/python scripts/22_preanotacion_llm_m4.py
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

CONFIG = {
    "semilla": 42,
    "anotador": "llm_preanotacion",
    "modelo_default": "qwen3.8:docs-32k-q3_k_s",
    "host_default": "http://127.0.0.1:11434",
    "max_chars_default": 6000,
    "n_smoke": 3,
    "timeout_lectura_seg": 300,
    "timeout_conexion_seg": 10,
    "timeout_tags_seg": 15,
    "reintentos": 2,
    "backoff_base_seg": 2,
    "num_ctx": 8192,
    "num_predict": 160,
    "think": False,
    "temperature": 0.0,
    "seed_llm": 42,
    "chars_nota_max": 200,
    "lote_lectura_corpus": 64,
    "jobs_max_recomendado": 2,
    "dir_smoke": "/tmp/opencode/o1_llm_smoke",
    "rutas_datos": {
        "muestra": "resultados/validacion_m4/muestra_200.jsonl",
        "corpus": "data/corpus_sentencias.parquet",
        "articulos": "data/constitucion_articulos.parquet",
    },
    "rutas_salida": {
        "respuestas": "resultados/validacion_m4/respuestas_{anotador}.jsonl",
        "meta": "resultados/validacion_m4/preanotacion_llm_meta.json",
    },
    "rutas_salida_smoke": {
        "respuestas": "/tmp/opencode/o1_llm_smoke/respuestas_{anotador}.jsonl",
        "meta": "/tmp/opencode/o1_llm_smoke/preanotacion_llm_meta.json",
    },
}

LOGGER_NOMBRE = "preanotacion_llm_m4_22"

# Cache del modulo 21 (su nombre empieza por digito: importlib por ruta).
_MODULO_PACK = None


# ===== SECCION 2: reutilizacion del modulo 21 y utilidades =====


def modulo_pack():
    """Importa scripts/21_pack_validacion_m4.py por ruta (cacheado).

    El nombre del modulo empieza por digito, por lo que no puede importarse
    con la sintaxis normal; se usa importlib.util.spec_from_file_location para
    reutilizar constantes (CLASES_M4), la lectura del corpus por id y los
    helpers de jsonl sin duplicarlos. Importarlo NO ejecuta ningun modo del
    script 21 (solo define funciones y constantes).

    Returns:
        Modulo 21_pack_validacion_m4 ya ejecutado.

    Raises:
        ImportError: Si el archivo no existe o no se puede cargar.
    """
    global _MODULO_PACK
    if _MODULO_PACK is None:
        ruta = RAIZ / "scripts" / "21_pack_validacion_m4.py"
        spec = importlib.util.spec_from_file_location("pack_validacion_m4_21", ruta)
        if spec is None or spec.loader is None:
            raise ImportError(f"no se pudo cargar {ruta}")
        modulo = importlib.util.module_from_spec(spec)
        sys.modules["pack_validacion_m4_21"] = modulo
        spec.loader.exec_module(modulo)
        _MODULO_PACK = modulo
    return _MODULO_PACK


def md5_archivo(ruta: str | Path) -> str | None:
    """Calcula el md5 de un archivo por bloques (delegado al modulo 21).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    return modulo_pack().md5_archivo(ruta)


def cargar_muestra(rutas: dict) -> list[dict]:
    """Carga el pack de la muestra activa (delegado al modulo 21).

    Args:
        rutas: Dict de rutas activas (clave "muestra").

    Returns:
        Lista de registros del pack en el orden de anotacion.

    Raises:
        FileNotFoundError: Si no existe la muestra.
    """
    return modulo_pack().cargar_muestra(rutas)


def abrir_corpus(ruta_corpus: str | Path):
    """Abre el corpus particionado tipo/ano sin cargarlo (modulo 21).

    Args:
        ruta_corpus: Ruta del dataset parquet particionado.

    Returns:
        Dataset pyarrow listo para leer por id con filtros.
    """
    return modulo_pack().abrir_corpus(ruta_corpus)


def leer_textos_por_ids(dataset, ids: list[str], lote: int, logger: logging.Logger) -> dict[str, str]:
    """Lee SOLO el texto de los ids pedidos con filtros pyarrow (modulo 21).

    Args:
        dataset: Dataset del corpus ya abierto.
        ids: Ids a leer (strings).
        lote: Tamano de lote del filtro isin.
        logger: Logger activo.

    Returns:
        Dict id -> texto (string vacio si el texto viniera nulo).
    """
    return modulo_pack().leer_textos_por_ids(dataset, ids, lote, logger)


def extracto_texto(texto: str, n_chars: int) -> str:
    """Extrae el inicio del documento normalizando finales de linea (modulo 21).

    Args:
        texto: Texto completo del documento.
        n_chars: Maximo de caracteres a conservar.

    Returns:
        Extracto del inicio (sin CR, con strip).
    """
    return modulo_pack().extracto_texto(texto, n_chars)


def normalizar_entrada(texto: str) -> str:
    """Normaliza texto: minusculas, sin tildes, espacios simples (modulo 21).

    Args:
        texto: Texto crudo.

    Returns:
        Texto normalizado (NFKD sin marcas combinantes, minusculas, strip).
    """
    return modulo_pack().normalizar_entrada(texto)


def normalizar_anotador(nombre: str) -> str:
    """Normaliza el nombre del anotador para usarlo como archivo (modulo 21).

    Args:
        nombre: Nombre crudo.

    Returns:
        Nombre con solo letras, digitos, guion y guion bajo.

    Raises:
        ValueError: Si el nombre queda vacio.
    """
    return modulo_pack().normalizar_anotador(nombre)


def escribir_jsonl_append(ruta: str | Path, registro: dict) -> None:
    """Agrega un registro al final de un JSON Lines con flush (modulo 21).

    Args:
        ruta: Ruta del archivo (se crea si no existe).
        registro: Dict serializable.
    """
    modulo_pack().escribir_jsonl_append(ruta, registro)


def configurar_logger() -> logging.Logger:
    """Configura el logger de consola (mismo formato del script 21).

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
        smoke: True para usar salidas exclusivamente en /tmp/opencode/o1_llm_smoke.
        anotador: Nombre del anotador para el archivo de respuestas (default:
            CONFIG["anotador"], que reproduce la ruta historica).

    Returns:
        Dict combinado de rutas de datos y de salidas.
    """
    rutas = {}
    rutas.update(CONFIG["rutas_datos"])
    rutas.update(CONFIG["rutas_salida_smoke"] if smoke else CONFIG["rutas_salida"])
    nombre = anotador or CONFIG["anotador"]
    rutas["respuestas"] = str(rutas["respuestas"]).format(anotador=nombre)
    return rutas


def cargar_respuestas_tolerante(ruta: str | Path) -> tuple[dict[str, dict], int]:
    """Carga respuestas id -> ultimo registro ignorando lineas corruptas.

    A diferencia del cargador estricto del script 21, tolera lineas JSON
    invalidas (por ejemplo, un corte de energia a mitad de linea) y las
    reporta como corruptas para no bloquear la reanudacion.

    Args:
        ruta: Ruta del jsonl de respuestas (si no existe, dict vacio).

    Returns:
        Tupla (dict id -> registro, numero de lineas corruptas ignoradas).
    """
    archivo = Path(ruta)
    respuestas: dict[str, dict] = {}
    corruptas = 0
    if not archivo.exists():
        return respuestas, 0
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
            if isinstance(registro, dict) and "id" in registro:
                respuestas[str(registro["id"])] = registro
    return respuestas, corruptas


# ===== SECCION 3: taxonomia desde el mapeo articulo -> bloque =====


def resumir_fragmento(texto: str, n_chars: int) -> str:
    """Colapsa espacios y recorta un fragmento de texto para la taxonomia.

    Args:
        texto: Texto crudo del articulo.
        n_chars: Maximo de caracteres a conservar.

    Returns:
        Fragmento en una linea.
    """
    limpio = " ".join(str(texto or "").split())
    return limpio[:n_chars]


def construir_taxonomia(ruta_articulos: str | Path) -> dict[str, str]:
    """Construye una linea descriptiva por bloque desde el parquet de articulos.

    Para cada una de las 16 clases calcula el rango de articulos de la CP
    (minimo-maximo), el numero de articulos mapeados y dos ejemplos con el
    inicio del texto del articulo. Todo se deriva del parquet
    data/constitucion_articulos.parquet; no hay descripciones inventadas.

    Args:
        ruta_articulos: Ruta del parquet con numero, texto y bloque_tematico.

    Returns:
        Dict clase -> linea descriptiva (una linea por bloque).

    Raises:
        FileNotFoundError: Si el parquet no existe.
    """
    tabla = pq.read_table(ruta_articulos, columns=["numero", "texto", "bloque_tematico"]).to_pandas()
    tabla["numero_txt"] = tabla["numero"].astype(str)
    tabla = tabla[tabla["numero_txt"].str.isdigit()].copy()
    tabla["n"] = tabla["numero_txt"].astype(int)
    lineas: dict[str, str] = {}
    for clase in modulo_pack().CLASES_M4:
        sub = tabla[tabla["bloque_tematico"] == clase].sort_values("n")
        if sub.empty:
            lineas[clase] = f"{clase}: sin articulos mapeados en el parquet"
            continue
        numeros = sub["n"].tolist()
        ejemplos = [
            f"art. {fila.n} ({resumir_fragmento(fila.texto, 70)})" for fila in sub.head(2).itertuples()
        ]
        lineas[clase] = (
            f"{clase}: arts. {numeros[0]}-{numeros[-1]} de la CP "
            f"({len(numeros)} articulos; ej. {'; '.join(ejemplos)})"
        )
    return lineas


# ===== SECCION 4: prompts (blinding estricto) =====


def construir_prompt_sistema(taxonomia: dict[str, str]) -> str:
    """Construye el mensaje de sistema con rol, reglas y taxonomia.

    Args:
        taxonomia: Dict clase -> linea descriptiva (salida de construir_taxonomia).

    Returns:
        Prompt de sistema listo para ollama.
    """
    menu = "\n".join(f"- {linea}" for linea in taxonomia.values())
    return f"""Eres un experto en derecho constitucional colombiano y en clasificacion tematica de sentencias de la Corte Constitucional (T, C y SU).

Tarea: dado el inicio de una sentencia, decide el BLOQUE TEMATICO PREDOMINANTE de la Constitucion Politica (CP) de Colombia, entendido como el bloque al que pertenecen la mayoria de los articulos DISTINTOS de la CP citados o discutidos en el documento.

Taxonomia de 16 bloques (derivada del mapeo articulo->bloque de la CP):
{menu}

Reglas:
1. Decide UNICAMENTE con el texto entregado; no uses conocimiento externo ni supongas hechos que el texto no menciona.
2. Si el texto no alcanza para decidir un bloque predominante claro, responde bloque "ambiguo".
3. Si ningun bloque constitucional predomina (por ejemplo, un documento procesal o administrativo sin anclaje tematico en la CP), responde bloque "ninguno".
4. Responde SOLO un objeto JSON valido, sin texto adicional, con las claves exactas: {{"bloque": "<bloque>", "confianza": "alta|media|baja", "nota": "<=200 caracteres"}}.
5. En "bloque" usa exactamente uno de los 16 nombres de la taxonomia, "ambiguo" o "ninguno"; en "nota" justifica en una frase (articulos o temas)."""


def construir_prompt_usuario(doc: dict, texto: str) -> str:
    """Construye el mensaje de usuario con tipo, ano y texto (sin silver).

    Blinding: NO se incluyen area_silver, top3_json, via, n_citas_cp, id,
    particion ni en_holdout; solo la metadata permitida (tipo y ano) y el
    texto del documento.

    Args:
        doc: Registro del pack de validacion.
        texto: Extracto del texto del documento.

    Returns:
        Prompt de usuario listo para ollama.
    """
    return (
        f"Documento de la Corte Constitucional de Colombia.\n"
        f"Tipo: {doc['tipo']} | Ano: {doc['ano']}\n"
        f"Texto (inicio del documento, {len(texto)} caracteres):\n"
        f"{texto}\n\n"
        "Clasifica el bloque constitucional predominante y responde SOLO el JSON pedido."
    )


def texto_para_prompt(doc: dict, textos: dict[str, str], max_chars: int) -> str:
    """Obtiene el texto del documento con respaldo al extracto del pack.

    Args:
        doc: Registro del pack de validacion.
        textos: Dict id -> texto completo leido del corpus.
        max_chars: Maximo de caracteres del inicio a usar.

    Returns:
        Extracto del corpus si existe; si no, extracto del pack (tambien
        recortado a max_chars).
    """
    texto = textos.get(doc["id"], "")
    if texto:
        return extracto_texto(texto, max_chars)
    return extracto_texto(doc.get("extracto", ""), max_chars)


# ===== SECCION 5: cliente ollama =====


def construir_payload(modelo: str, prompt_sistema: str, doc: dict, texto: str) -> dict:
    """Construye el cuerpo del POST /api/chat con opciones reproducibles.

    Se envia think=false porque los modelos qwen3.8 usan modo pensamiento: con
    num_predict=160 la cadena de razonamiento consumiria todos los tokens y el
    JSON de respuesta nunca se emitiria (verificado en el smoke; la respuesta
    cruda del modo pensamiento se conservo como evidencia).

    Args:
        modelo: Nombre del modelo en ollama.
        prompt_sistema: Mensaje de sistema (rol + taxonomia + reglas).
        doc: Registro del pack (para tipo/ano del mensaje de usuario).
        texto: Extracto del documento.

    Returns:
        Dict serializable con model, messages, format, think y options.
    """
    return {
        "model": modelo,
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": construir_prompt_usuario(doc, texto)},
        ],
        "format": "json",
        "stream": False,
        "think": CONFIG["think"],
        "options": {
            "temperature": CONFIG["temperature"],
            "seed": CONFIG["seed_llm"],
            "num_ctx": CONFIG["num_ctx"],
            "num_predict": CONFIG["num_predict"],
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


def consultar_ollama(host: str, payload: dict, logger: logging.Logger) -> str:
    """Hace POST /api/chat con reintentos y backoff exponencial.

    Prefiere message.content; si viniera vacio (modelos con modo pensamiento)
    intenta message.thinking como respaldo.

    Args:
        host: URL base de ollama.
        payload: Cuerpo del POST (ver construir_payload).
        logger: Logger activo.

    Returns:
        Contenido textual devuelto por el modelo.

    Raises:
        RuntimeError: Si todos los intentos fallan (transporte, HTTP o vacio).
    """
    url = f"{host}/api/chat"
    ultimo_error = "sin intentos"
    for intento in range(CONFIG["reintentos"] + 1):
        if intento > 0:
            espera = CONFIG["backoff_base_seg"] ** intento
            logger.warning(
                "reintento %d/%d en %.0f s (%s)", intento, CONFIG["reintentos"], espera, ultimo_error
            )
            time.sleep(espera)
        try:
            respuesta = requests.post(
                url,
                json=payload,
                timeout=(CONFIG["timeout_conexion_seg"], CONFIG["timeout_lectura_seg"]),
            )
            if respuesta.status_code != 200:
                ultimo_error = f"HTTP {respuesta.status_code}: {respuesta.text[:200]}"
                continue
            datos = respuesta.json()
            if isinstance(datos, dict) and datos.get("error"):
                ultimo_error = f"ollama error: {datos['error']}"
                continue
            mensaje = datos.get("message") or {}
            contenido = str(mensaje.get("content") or "").strip()
            if not contenido and mensaje.get("thinking"):
                contenido = str(mensaje["thinking"]).strip()
            if not contenido:
                ultimo_error = "respuesta sin message.content"
                continue
            return contenido
        except (requests.RequestException, ValueError) as exc:
            ultimo_error = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"ollama fallo tras {CONFIG['reintentos'] + 1} intentos: {ultimo_error}")


def extraer_json(contenido: str) -> dict | None:
    """Extrae el primer objeto JSON de un texto del modelo.

    Intenta json.loads directo y, si falla, busca el primer objeto con
    llaves balanceadas simples (respuesta tipica de format=json).

    Args:
        contenido: Texto crudo del modelo.

    Returns:
        Dict si se pudo parsear; None si el texto no contiene JSON valido.
    """
    try:
        datos = json.loads(contenido)
        return datos if isinstance(datos, dict) else None
    except (TypeError, ValueError):
        pass
    for patron in (r"\{.*?\}", r"\{.*\}"):
        coincidencia = re.search(patron, contenido, re.DOTALL)
        if not coincidencia:
            continue
        try:
            datos = json.loads(coincidencia.group(0))
        except ValueError:
            continue
        if isinstance(datos, dict):
            return datos
    return None


def normalizar_bloque(valor) -> str | None:
    """Normaliza el bloque devuelto por el modelo al espacio canonico.

    Args:
        valor: Valor crudo de la clave "bloque".

    Returns:
        Una de las 16 clases, "ambiguo" o "ninguno"; None si no es valido.
    """
    if valor is None:
        return None
    texto = normalizar_entrada(str(valor))
    if texto in ("ambiguo", "ambigua"):
        return "ambiguo"
    if texto in ("ninguno", "ninguna", "ningun"):
        return "ninguno"
    if texto in modulo_pack().CLASES_M4:
        return texto
    con_guion_bajo = texto.replace(" ", "_")
    if con_guion_bajo in modulo_pack().CLASES_M4:
        return con_guion_bajo
    return None


def normalizar_confianza(valor) -> str:
    """Normaliza la confianza devuelta por el modelo.

    Args:
        valor: Valor crudo de la clave "confianza".

    Returns:
        alta | media | baja; "no_reportada" si no es valida.
    """
    texto = normalizar_entrada(str(valor or ""))
    if texto in ("alta", "media", "baja"):
        return texto
    return "no_reportada"


def truncar_nota(valor) -> str:
    """Normaliza la nota a texto plano de a lo sumo chars_nota_max.

    Args:
        valor: Valor crudo de la clave "nota" (o texto de respaldo).

    Returns:
        Nota en una linea, recortada a 200 caracteres.
    """
    texto = " ".join(str(valor or "").split())
    return texto[: CONFIG["chars_nota_max"]]


def parsear_respuesta_llm(contenido: str) -> dict:
    """Valida y normaliza la respuesta JSON del modelo.

    Args:
        contenido: Texto crudo devuelto por el modelo.

    Returns:
        Dict con valido, respuesta (clase | ambiguo | ninguno | invalido),
        confianza, nota y motivo (None si es valida).
    """
    datos = extraer_json(contenido)
    if datos is None:
        return {
            "valido": False,
            "respuesta": "invalido",
            "confianza": "no_reportada",
            "nota": truncar_nota(contenido),
            "motivo": "json ilegible",
        }
    respuesta = normalizar_bloque(datos.get("bloque"))
    confianza = normalizar_confianza(datos.get("confianza"))
    nota = truncar_nota(datos.get("nota"))
    if respuesta is None:
        return {
            "valido": False,
            "respuesta": "invalido",
            "confianza": confianza,
            "nota": nota or truncar_nota(contenido),
            "motivo": f"bloque invalido: {datos.get('bloque')!r}",
        }
    return {
        "valido": True,
        "respuesta": respuesta,
        "confianza": confianza,
        "nota": nota,
        "motivo": None,
    }


# ===== SECCION 6: procesamiento de documentos =====


def construir_registro(doc: dict, parsed: dict, modelo: str, tiempo_seg: float, anotador: str) -> dict:
    """Construye el registro compatible con el script 21 mas trazabilidad LLM.

    Campos del script 21 (--anotar): anotador, id, indice, respuesta,
    confianza, nota, fecha, tiempo_seg. Se agregan automatico=true y modelo
    para que la preanotacion NUNCA se confunda con anotacion humana.

    Args:
        doc: Registro del pack de validacion.
        parsed: Salida de parsear_respuesta_llm.
        modelo: Nombre del modelo usado.
        tiempo_seg: Tiempo del documento (consulta + parseo).
        anotador: Nombre del anotador de la corrida (--anotador).

    Returns:
        Dict serializable listo para escribir_jsonl_append.
    """
    return {
        "anotador": anotador,
        "id": doc["id"],
        "indice": int(doc.get("indice", 0)),
        "respuesta": parsed["respuesta"],
        "confianza": parsed["confianza"],
        "nota": parsed["nota"],
        "fecha": pd.Timestamp.now().isoformat(timespec="seconds"),
        "tiempo_seg": round(tiempo_seg, 1),
        "automatico": True,
        "modelo": modelo,
    }


def procesar_documento(
    doc: dict,
    texto: str,
    prompt_sistema: str,
    modelo: str,
    host: str,
    logger: logging.Logger,
    anotador: str,
) -> tuple[dict | None, str, str | None]:
    """Consulta el LLM para un documento y construye su registro.

    Args:
        doc: Registro del pack de validacion.
        texto: Extracto del documento (max_chars).
        prompt_sistema: Prompt de sistema (rol + taxonomia + reglas).
        modelo: Nombre del modelo en ollama.
        host: URL base de ollama.
        logger: Logger activo.
        anotador: Nombre del anotador de la corrida (--anotador).

    Returns:
        Tupla (registro, estado, error): estado ok | invalido | fallo. En
        "fallo" el registro es None (el id queda pendiente para reanudar).
    """
    t_doc = time.perf_counter()
    payload = construir_payload(modelo, prompt_sistema, doc, texto)
    try:
        contenido = consultar_ollama(host, payload, logger)
    except Exception as exc:  # noqa: BLE001 - un documento no debe abortar la corrida
        return None, "fallo", f"{type(exc).__name__}: {exc}"
    parsed = parsear_respuesta_llm(contenido)
    registro = construir_registro(doc, parsed, modelo, time.perf_counter() - t_doc, anotador)
    if parsed["valido"]:
        return registro, "ok", None
    return registro, "invalido", parsed["motivo"]


def manejar_resultado(
    doc: dict,
    resultado: tuple[dict | None, str, str | None],
    resumen: dict,
    ruta_respuestas: Path,
    logger: logging.Logger,
    posicion: int | None,
    total: int,
) -> None:
    """Escribe (si aplica) el resultado de un documento y actualiza el resumen.

    Args:
        doc: Registro del pack de validacion.
        resultado: Tupla (registro, estado, error) de procesar_documento.
        resumen: Dict acumulador de la corrida (se muta).
        ruta_respuestas: Ruta del jsonl de respuestas.
        logger: Logger activo.
        posicion: Posicion 1..N en la corrida (None si jobs > 1).
        total: Total de documentos objetivo de la corrida.
    """
    registro, estado, error = resultado
    etiqueta = f"{posicion}/{total}" if posicion is not None else "job"
    if estado == "fallo":
        resumen["fallos"].append({"id": doc["id"], "error": error})
        logger.warning("[%s] id=%s sin respuesta tras reintentos: %s", etiqueta, doc["id"], error)
        return
    escribir_jsonl_append(ruta_respuestas, registro)
    resumen["escritos"] += 1
    resumen["tiempos_seg"].append(registro["tiempo_seg"])
    if estado == "ok":
        resumen["ok"] += 1
        logger.info(
            "[%s] id=%s -> %s (%s) en %.1f s",
            etiqueta,
            doc["id"],
            registro["respuesta"],
            registro["confianza"],
            registro["tiempo_seg"],
        )
    else:
        resumen["invalidos"] += 1
        logger.warning(
            "[%s] id=%s -> invalido (%s) en %.1f s",
            etiqueta,
            doc["id"],
            error,
            registro["tiempo_seg"],
        )


def procesar_pendientes(
    pendientes: list[dict],
    textos: dict[str, str],
    prompt_sistema: str,
    args: argparse.Namespace,
    logger: logging.Logger,
    ruta_respuestas: Path,
) -> dict:
    """Procesa los documentos pendientes y agrega cada respuesta al jsonl.

    Con jobs=1 recorre en orden del pack; con jobs>1 usa ThreadPoolExecutor y
    escribe desde el hilo principal a medida que terminan los futuros (el
    servidor ollama puede serializar las peticiones si no atiende en
    paralelo). Los fallos de transporte/HTTP no se escriben: quedan
    pendientes para una corrida con --reanudar.

    Args:
        pendientes: Documentos a procesar en esta corrida.
        textos: Dict id -> texto completo leido del corpus.
        prompt_sistema: Prompt de sistema.
        args: Argumentos CLI (jobs, max_chars, modelo, host, anotador).
        logger: Logger activo.
        ruta_respuestas: Ruta del jsonl de respuestas (append incremental).

    Returns:
        Dict con escritos, ok, invalidos, fallos (lista) y tiempos_seg.
    """
    resumen = {"escritos": 0, "ok": 0, "invalidos": 0, "fallos": [], "tiempos_seg": []}
    total = len(pendientes)
    if args.jobs <= 1:
        try:
            for posicion, doc in enumerate(pendientes, 1):
                resultado = procesar_documento(
                    doc,
                    texto_para_prompt(doc, textos, args.max_chars),
                    prompt_sistema,
                    args.modelo,
                    args.host,
                    logger,
                    args.anotador,
                )
                manejar_resultado(doc, resultado, resumen, ruta_respuestas, logger, posicion, total)
        except KeyboardInterrupt:
            logger.warning("interrumpido por el usuario: se conserva lo escrito (use --reanudar)")
        return resumen
    executor = ThreadPoolExecutor(max_workers=args.jobs)
    try:
        futuros = {
            executor.submit(
                procesar_documento,
                doc,
                texto_para_prompt(doc, textos, args.max_chars),
                prompt_sistema,
                args.modelo,
                args.host,
                logger,
                args.anotador,
            ): doc
            for doc in pendientes
        }
        for futuro in as_completed(futuros):
            doc = futuros[futuro]
            try:
                resultado = futuro.result()
            except Exception as exc:  # noqa: BLE001 - un fallo no debe abortar la corrida
                resultado = (None, "fallo", f"{type(exc).__name__}: {exc}")
            manejar_resultado(doc, resultado, resumen, ruta_respuestas, logger, None, total)
    except KeyboardInterrupt:
        logger.warning("interrumpido por el usuario: se conserva lo escrito (use --reanudar)")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return resumen


def construir_meta(
    args: argparse.Namespace,
    rutas: dict,
    muestra: list[dict],
    previos: dict[str, dict],
    pendientes: list[dict],
    resumen_corrida: dict,
    tiempo_total: float,
    corruptas: int,
) -> dict:
    """Construye el registro de metadatos de la corrida de preanotacion.

    Args:
        args: Argumentos CLI.
        rutas: Dict de rutas activas.
        muestra: Muestra completa del pack.
        previos: Respuestas previas (id -> registro) antes de la corrida.
        pendientes: Documentos objetivo de esta corrida.
        resumen_corrida: Salida de procesar_pendientes.
        tiempo_total: Tiempo total de la corrida en segundos.
        corruptas: Lineas corruptas ignoradas al cargar respuestas previas.

    Returns:
        Dict serializable de metadatos.
    """
    escritos = resumen_corrida["escritos"]
    tiempos = resumen_corrida["tiempos_seg"]
    media = (sum(tiempos) / len(tiempos)) if tiempos else None
    ruta_respuestas = Path(rutas["respuestas"])
    return {
        "script": "scripts/22_preanotacion_llm_m4.py",
        "tipo_anotacion": "automatica_llm",
        "advertencia": "Preanotacion automatica con LLM local; NO es anotacion humana",
        "fecha": pd.Timestamp.now().isoformat(timespec="seconds"),
        "modelo": args.modelo,
        "host": args.host,
        "smoke": bool(args.smoke),
        "anotador": args.anotador,
        "automatico": True,
        "max_chars": args.max_chars,
        "limite": CONFIG["n_smoke"] if args.smoke else args.limite,
        "jobs": args.jobs,
        "reanudar": bool(args.reanudar),
        "n_muestra": len(muestra),
        "n_respuestas_previas": len(previos),
        "n_objetivo_corrida": len(pendientes),
        "n_escritos": escritos,
        "n_ok": resumen_corrida["ok"],
        "n_invalidos": resumen_corrida["invalidos"],
        "n_fallos": len(resumen_corrida["fallos"]),
        "fallos": resumen_corrida["fallos"][:50],
        "lineas_corruptas_ignoradas": corruptas,
        "tiempo_total_seg": round(tiempo_total, 1),
        "tiempo_medio_doc_seg": (round(media, 2) if media is not None else None),
        "docs_por_minuto": (
            round(escritos / (tiempo_total / 60.0), 2) if (escritos and tiempo_total > 0) else None
        ),
        "md5_muestra": md5_archivo(rutas["muestra"]),
        "md5_respuestas": md5_archivo(ruta_respuestas) if ruta_respuestas.exists() else None,
        "ruta_respuestas": str(ruta_respuestas),
        "config_llm": {
            "format": "json",
            "think": CONFIG["think"],
            "temperature": CONFIG["temperature"],
            "seed": CONFIG["seed_llm"],
            "num_ctx": CONFIG["num_ctx"],
            "num_predict": CONFIG["num_predict"],
            "timeout_lectura_seg": CONFIG["timeout_lectura_seg"],
            "reintentos": CONFIG["reintentos"],
        },
        "blinding": (
            "el prompt usa solo tipo, ano y texto; excluye area_silver, top3_json, "
            "via y n_citas_cp (tampoco incluye el id)"
        ),
    }


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
    limite = CONFIG["n_smoke"] if args.smoke else args.limite
    if args.smoke and args.limite is not None and args.limite != CONFIG["n_smoke"]:
        logger.warning("--smoke fuerza limite=%d (--limite=%s ignorado)", CONFIG["n_smoke"], args.limite)
    if args.jobs > CONFIG["jobs_max_recomendado"]:
        logger.warning(
            "jobs=%d: con una sola GPU conviene jobs<=%d (ollama puede serializar o degradar)",
            args.jobs,
            CONFIG["jobs_max_recomendado"],
        )
    elif args.jobs > 1:
        logger.info("jobs=%d: ThreadPoolExecutor (acelera solo si ollama atiende en paralelo)", args.jobs)
    verificar_modelo_disponible(host, args.modelo, logger)
    muestra = cargar_muestra(rutas)
    previos, corruptas = cargar_respuestas_tolerante(ruta_respuestas)
    if corruptas:
        logger.warning("lineas corruptas ignoradas en %s: %d", ruta_respuestas, corruptas)
    ids_muestra = {doc["id"] for doc in muestra}
    huerfanos = sorted(set(previos) - ids_muestra)
    if huerfanos:
        logger.warning("respuestas previas de ids fuera de la muestra: %d", len(huerfanos))
    pendientes = [doc for doc in muestra if doc["id"] not in previos]
    if limite is not None:
        pendientes = pendientes[:limite]
    logger.info(
        "muestra=%d | ya respondidos=%d | pendientes objetivo=%d%s",
        len(muestra),
        len(previos),
        len(pendientes),
        " (smoke)" if args.smoke else "",
    )
    taxonomia = construir_taxonomia(rutas["articulos"])
    prompt_sistema = construir_prompt_sistema(taxonomia)
    logger.info("taxonomia construida: %d bloques", len(taxonomia))
    if pendientes:
        dataset = abrir_corpus(rutas["corpus"])
        textos = leer_textos_por_ids(
            dataset, [doc["id"] for doc in pendientes], CONFIG["lote_lectura_corpus"], logger
        )
        resumen_corrida = procesar_pendientes(
            pendientes, textos, prompt_sistema, args, logger, ruta_respuestas
        )
    else:
        resumen_corrida = {"escritos": 0, "ok": 0, "invalidos": 0, "fallos": [], "tiempos_seg": []}
        logger.info("no hay documentos pendientes: nada que preanotar")
    tiempo_total = time.perf_counter() - t0
    meta = construir_meta(
        args, rutas, muestra, previos, pendientes, resumen_corrida, tiempo_total, corruptas
    )
    escribir_meta(rutas["meta"], meta)
    logger.info("meta guardada: %s", rutas["meta"])
    logger.info(
        "corrida: escritos=%d ok=%d invalidos=%d fallos=%d | total=%d/%d | %.1f s (%.2f docs/min)",
        resumen_corrida["escritos"],
        resumen_corrida["ok"],
        resumen_corrida["invalidos"],
        len(resumen_corrida["fallos"]),
        len(previos) + resumen_corrida["escritos"],
        len(muestra),
        tiempo_total,
        meta["docs_por_minuto"] or 0.0,
    )


# ===== SECCION 7: modo resumen =====


def modo_resumen(logger: logging.Logger, rutas: dict, anotador: str) -> None:
    """Imprime el avance de la preanotacion (respondidos, faltantes, distribucion).

    Args:
        logger: Logger activo.
        rutas: Dict de rutas activas (smoke o real).
        anotador: Nombre del anotador de la corrida (--anotador).
    """
    t0 = time.perf_counter()
    muestra = cargar_muestra(rutas)
    ruta_respuestas = Path(rutas["respuestas"])
    registros, corruptas = cargar_respuestas_tolerante(ruta_respuestas)
    por_id = {doc["id"]: doc for doc in muestra}
    validos = {id_doc: registro for id_doc, registro in registros.items() if id_doc in por_id}
    clases = set(modulo_pack().CLASES_M4)
    especiales = set(modulo_pack().RESPUESTAS_ESPECIALES)
    distribucion = Counter(str(registro.get("respuesta", "?")) for registro in validos.values())
    no_validas = sum(n for respuesta, n in distribucion.items() if respuesta not in clases | especiales)
    print(f"Anotador: {anotador} (automatico, LLM local; no es anotacion humana)")
    print(f"Muestra: {len(muestra)} docs | {rutas['muestra']}")
    print(f"Respuestas: {ruta_respuestas} {'(existe)' if ruta_respuestas.exists() else '(no existe)'}")
    porcentaje = 100.0 * len(validos) / len(muestra) if muestra else 0.0
    print(
        f"Respondidos: {len(validos)}/{len(muestra)} ({porcentaje:.1f} %) | "
        f"pendientes: {len(muestra) - len(validos)}"
    )
    if registros and len(registros) != len(validos):
        print(f"aviso: {len(registros) - len(validos)} respuestas de ids fuera de la muestra")
    if corruptas:
        print(f"aviso: {corruptas} lineas corruptas ignoradas")
    print("Distribucion de respuesta:")
    for respuesta, conteo in distribucion.most_common():
        print(f"  {respuesta}: {conteo}")
    confianzas = Counter(str(registro.get("confianza", "?")) for registro in validos.values())
    print(f"Confianza: {dict(confianzas)}")
    print(f"No validas (fuera de 16 clases + ambiguo/ninguno): {no_validas}")
    tiempos = [
        float(registro["tiempo_seg"])
        for registro in validos.values()
        if isinstance(registro.get("tiempo_seg"), (int, float))
    ]
    if tiempos:
        print(
            f"Tiempo medio por doc: {sum(tiempos) / len(tiempos):.1f} s (n={len(tiempos)}) | "
            f"total registrado: {sum(tiempos):.1f} s"
        )
    modelos = Counter(str(registro.get("modelo", "?")) for registro in validos.values())
    print(f"Modelos usados: {dict(modelos)}")
    logger.info("--resumen completado en %.2f s", time.perf_counter() - t0)


# ===== SECCION 8: CLI y main =====


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Args:
        argv: Lista de argumentos (None usa sys.argv).

    Returns:
        Namespace con modelo, host, anotador, max_chars, limite, jobs, reanudar,
        smoke y resumen.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pre-anotacion AUTOMATICA (LLM local via ollama) del pack de validacion "
            "humana M4; compatible con --calcular-acuerdo del script 21. No es "
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
            '"anotador" del registro (default llm_preanotacion)'
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=CONFIG["max_chars_default"],
        dest="max_chars",
        help="caracteres del inicio del documento a enviar (default 6000)",
    )
    parser.add_argument("--limite", type=int, default=None, help="maximo de documentos a procesar")
    parser.add_argument("--jobs", type=int, default=1, help="peticiones concurrentes (default 1)")
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
    parser.add_argument("--smoke", action="store_true", help="3 docs y salida solo en /tmp/opencode/o1_llm_smoke")
    parser.add_argument("--resumen", action="store_true", help="muestra el avance y sale (no llama al LLM)")
    args = parser.parse_args(argv)
    try:
        args.anotador = normalizar_anotador(args.anotador)
    except ValueError as exc:
        parser.error(f"--anotador invalido: {exc}")
    if args.max_chars < 100:
        parser.error("--max-chars debe ser >= 100")
    if args.limite is not None and args.limite < 1:
        parser.error("--limite debe ser >= 1")
    if args.jobs < 1:
        parser.error("--jobs debe ser >= 1")
    return args


def main(argv: list[str] | None = None) -> None:
    """Orquesta el modo pedido (corrida de preanotacion o resumen).

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
        "=== INICIO preanotacion LLM M4 | smoke=%s resumen=%s modelo=%s host=%s anotador=%s ===",
        args.smoke,
        args.resumen,
        args.modelo,
        args.host,
        args.anotador,
    )
    if args.smoke:
        Path(CONFIG["dir_smoke"]).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.resumen:
        modo_resumen(logger, rutas, args.anotador)
    else:
        ejecutar_preanotacion(args, logger, rutas)
    logger.info("=== FIN preanotacion LLM M4 en %.1f s ===", time.perf_counter() - t0)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
