"""31_c2_llm_rerank.py
Fecha: 2026-10-04
Descripcion: C2(b) - rerank listwise del top-20 de candidatos con el LLM
    legal local alia-legal (ALIA-es-legal-administrative-7B-Instruct,
    GGUF Q4_K_M servido por ollama en http://127.0.0.1:11434).
    Hipotesis C2(b): razonar sobre el top-20 de candidatos con semantica
    juridica transfiere mejor que generar consultas cortas (C2(a) resulto
    NEGATIVO: la consulta del LLM perdio -0.198 nDCG@10 pareado frente al
    texto completo). Por cada caso se construye el top-20 de la 1a etapa
    (hibrido pre-registrado 0.5*BM25 + 0.5*B2 sobre el texto completo,
    SIN el reranker B1) y se pide al LLM un ranking listwise en JSON
    (orden + grados de relevancia 0/1/2) con el contexto del caso
    (problema juridico y hechos de la extraccion C2 cuando existen; si
    no, el encabezado de la sentencia). El parser es tolerante (cercas de
    codigo, bloque balanceado, regex del campo orden y, si todo falla,
    conserva el orden previo del hibrido marcandolo). Modos: --rerank
    (llama a ollama por caso y escribe un JSONL resumible), --evaluar
    (ranking del LLM k=10 frente a la ruta v13 recalculada con el
    reranker B1 en la misma corrida, comparacion PAREADA por caso) y
    --todo. Con --smoke/--limite todas las salidas van a
    /tmp/opencode/c2r_smoke/ y nunca se sobrescriben los artefactos
    reales. Reglas de salida: solo se escribe en resultados/c2_llm/,
    figuras/c2_rerank_llm.png y experimentos/31_c2_llm_rerank.log.
Estado del arte aplicado: patron de reranking listwise con LLM en
    pipelines de statute retrieval (LegalMALR arXiv:2601.17692; COLIEE
    arXiv:2607.11400) sobre la ruta de producto del proyecto
    (hibrido_b2_rerank_chunks_top50_b1, hubs alpha=1.5, k=10); bootstrap
    no parametrico pareado (1000 remuestreos, seed 42, ADR-16) y gate de
    latencia <= 30 s/eval del plan v3. Se reutiliza por importlib la
    maquinaria de scripts/30_c2_llm_query.py (evaluacion, IC bootstrap,
    consistencia congelada, capa ollama tolerante a think y parser de
    cercas) y de scripts/12_evaluar_vinculo_cp_v2.py (BM25 por chunks,
    denso B2, hibrido, hubs, metricas y bootstrap), sin duplicar logica
    y sin tocar los artefactos previos.
"""

import argparse
import contextlib
import importlib.util
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CONFIG = {
    "version": "c2-rerank-llm-v1",
    "semilla": 42,
    "modelo_llm": "alia-legal",
    "host_ollama": "http://127.0.0.1:11434",
    "temperature": 0.0,
    "num_ctx": 16384,
    "num_predict": 900,
    "think": False,
    "k_candidatos_llm": 20,
    "snippet_chars": 200,
    "max_chars_contexto": 1600,
    "max_chars_encabezado": 1200,
    "timeout_conexion_seg": 10,
    "timeout_lectura_seg": 600,
    "reintentos": 2,
    "backoff_base_seg": 2,
    "smoke_limite": 5,
    "ruta_script_12": "scripts/12_evaluar_vinculo_cp_v2.py",
    "ruta_script_30": "scripts/30_c2_llm_query.py",
    "ruta_benchmark_v2": "resultados/vinculo_cp_benchmark_v2.json",
    "ruta_benchmark_v13": "resultados/vinculo_cp_benchmark_v13.json",
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "estrategia_v13": "hibrido_b2_rerank_chunks_top50_b1",
    "alpha_hubs": 1.5,
    "k_candidatos_v13": 50,
    "k_medidas": 5,
    "k_ndcg": 10,
    "bootstrap": 1000,
    "contexto_v13_congelado": "con_hubs",
    "gate_latencia_seg": 30.0,
    "dir_salida": "resultados/c2_llm",
    "ruta_json": "resultados/c2_llm/eval_rerank_llm.json",
    "ruta_figura": "figuras/c2_rerank_llm.png",
    "ruta_log": "experimentos/31_c2_llm_rerank.log",
    "dir_smoke": "/tmp/opencode/c2r_smoke",
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

# System prompt del rerank listwise: contexto del caso + 20 candidatos.
PROMPT_SISTEMA = (
    "Eres un asistente juridico experto en derecho constitucional "
    "colombiano. Recibes el contexto de un caso de la Corte "
    "Constitucional de Colombia y una lista de articulos candidatos de "
    "la Constitucion Politica (numero y fragmento). Ordena los "
    "candidatos por relevancia para resolver el caso: primero los "
    "articulos que fundamentan la decision, despues los que se "
    "relacionan de forma secundaria y al final los poco relacionados.\n"
    "Responde UNICAMENTE con un objeto JSON valido, sin texto adicional "
    "ni bloques de codigo, con esta forma:\n"
    '{"orden": [numeros de articulo de mas a menos relevante], '
    '"relevancia": [{"articulo": numero, "grado": 2}]}\n'
    "Reglas:\n"
    "- orden: los numeros de articulo de los candidatos (como "
    "aparecen en la lista, por ejemplo 41 o 22A), NO las posiciones "
    "1-20 de la lista; de mas a menos relevante, sin repetir ni omitir "
    "ninguno.\n"
    "- relevancia: una entrada por articulo con grado 2 (fundamenta la "
    "decision), 1 (relevante o mencionado) o 0 (poco o nada "
    "relacionado).\n"
    "- Devuelve el JSON en una sola linea compacta, sin indentacion ni "
    "saltos de linea.\n"
    "- Ordena por criterio juridico sustantivo, no por el orden en que "
    "aparecen los candidatos.\n"
    "- No expliques tu respuesta; devuelve solo el JSON."
)


# =====================================================================
# SECCION 1: CONFIGURACION, IMPORTLIB Y UTILIDADES
# =====================================================================
def importar_modulo(ruta: Path, nombre: str) -> object:
    """Importa un script por ruta con importlib (nombre con digito inicial).

    Args:
        ruta: Ruta absoluta del script a importar.
        nombre: Nombre interno del modulo en sys.modules.

    Returns:
        Modulo cargado (sin ejecutar su main por el guard __main__).

    Raises:
        FileNotFoundError: Si la ruta no existe.
        RuntimeError: Si importlib no puede construir el spec o loader.
    """
    if not ruta.is_file():
        raise FileNotFoundError(f"no existe el modulo {nombre}: {ruta}")
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {nombre}: {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def resolver_rutas_salida(limite: int | None) -> dict:
    """Resuelve las rutas de salida segun el modo (real o smoke/limite).

    En modo smoke todas las salidas van al directorio temporal; el JSONL
    de contexto C2 (queries.jsonl) se lee del directorio real porque es
    un insumo de solo lectura, no un artefacto de esta corrida.

    Args:
        limite: Valor efectivo de --limite (None en el run real).

    Returns:
        Dict con rutas rerank_jsonl/rerank_meta/queries_c2/json/figura/
        log y flag sobrescribir.
    """
    if limite is None:
        return {
            "rerank_jsonl": ROOT / CONFIG["dir_salida"] / "rerank_llm.jsonl",
            "rerank_meta": ROOT / CONFIG["dir_salida"] / "meta_rerank_llm.json",
            "queries_c2": ROOT / CONFIG["dir_salida"] / "queries.jsonl",
            "json": ROOT / CONFIG["ruta_json"],
            "figura": ROOT / CONFIG["ruta_figura"],
            "log": ROOT / CONFIG["ruta_log"],
            "sobrescribir": False,
        }
    base = Path(CONFIG["dir_smoke"])
    return {
        "rerank_jsonl": base / "rerank_llm.jsonl",
        "rerank_meta": base / "meta_rerank_llm.json",
        "queries_c2": ROOT / CONFIG["dir_salida"] / "queries.jsonl",
        "json": base / "eval_rerank_llm.json",
        "figura": base / "c2_rerank_llm.png",
        "log": base / "31_c2_llm_rerank.log",
        "sobrescribir": True,
    }


def contar_por(filas: list[dict], campo: str) -> dict:
    """Cuenta filas por el valor de un campo (resumen de trazabilidad).

    Args:
        filas: Filas serializables (JSONL).
        campo: Nombre del campo a contar.

    Returns:
        Dict {valor: conteo} ordenado por clave.
    """
    conteo: dict = {}
    for fila in filas:
        valor = str(fila.get(campo, "desconocido"))
        conteo[valor] = conteo.get(valor, 0) + 1
    return dict(sorted(conteo.items()))


def resumen_latencia(filas: list[dict]) -> dict:
    """Resume la latencia por caso del rerank LLM y el gate del plan.

    Args:
        filas: Filas del JSONL de rerank con el campo tiempo_seg.

    Returns:
        Dict con n, media/mediana/p95/min/max, gate de 30 s, numero de
        casos que lo exceden y veredictos del gate por media y por p95.
    """
    tiempos = np.asarray(
        [float(fila.get("tiempo_seg", 0.0)) for fila in filas],
        dtype=np.float64,
    )
    if tiempos.size == 0:
        return {
            "n": 0,
            "media_seg": None,
            "mediana_seg": None,
            "p95_seg": None,
            "min_seg": None,
            "max_seg": None,
            "gate_seg": CONFIG["gate_latencia_seg"],
            "n_excede_gate": 0,
            "cumple_gate_media": None,
            "cumple_gate_p95": None,
        }
    p95 = float(np.percentile(tiempos, 95))
    return {
        "n": int(tiempos.size),
        "media_seg": round(float(tiempos.mean()), 1),
        "mediana_seg": round(float(np.median(tiempos)), 1),
        "p95_seg": round(p95, 1),
        "min_seg": round(float(tiempos.min()), 1),
        "max_seg": round(float(tiempos.max()), 1),
        "gate_seg": CONFIG["gate_latencia_seg"],
        "n_excede_gate": int(
            (tiempos > CONFIG["gate_latencia_seg"]).sum()
        ),
        "cumple_gate_media": bool(
            float(tiempos.mean()) <= CONFIG["gate_latencia_seg"]
        ),
        "cumple_gate_p95": bool(p95 <= CONFIG["gate_latencia_seg"]),
    }


# =====================================================================
# SECCION 2: CANDIDATOS TOP-20, SNIPPETS Y PROMPT LISTWISE
# =====================================================================
def preparar_contexto_caso(
    texto_crudo: str, consulta_c2: dict | None
) -> tuple[str, str]:
    """Construye el contexto del caso para el prompt de rerank.

    Usa problema_juridico y hechos de la extraccion C2(a) cuando existen
    (consulta_c2); si no hay campos utiles, cae al encabezado de la
    sentencia cruda (primeros max_chars_encabezado caracteres, que
    concentran el numero de sentencia y los titulos de estudio).

    Args:
        texto_crudo: Texto crudo de la sentencia en el corpus.
        consulta_c2: Fila del JSONL C2(a) o None si el caso no la tiene.

    Returns:
        Tupla (contexto, fuente) con fuente "c2_llm" o "encabezado".
    """
    partes: list[str] = []
    fuente = "encabezado"
    if consulta_c2:
        problema = " ".join(
            str(consulta_c2.get("problema_juridico") or "").split()
        )
        hechos = " ".join(
            str(consulta_c2.get("hechos") or "").split()
        )
        if problema:
            partes.append(f"Problema juridico: {problema}")
        if hechos:
            partes.append(f"Hechos: {hechos}")
        if partes:
            fuente = "c2_llm"
    if not partes:
        encabezado = " ".join(
            texto_crudo[: CONFIG["max_chars_encabezado"]].split()
        )
        partes = [f"Encabezado de la sentencia: {encabezado}"]
    contexto = " ".join(partes)
    if len(contexto) > CONFIG["max_chars_contexto"]:
        contexto = contexto[: CONFIG["max_chars_contexto"]].rstrip()
    return contexto, fuente


def obtener_candidatos(
    caso: dict, clave_hibrido: str, canones: list, k: int
) -> list:
    """Extrae el top-k de la 1a etapa (hibrido 0.5*BM25 + 0.5*B2).

    No aplica el reranker B1: el LLM sustituye esa etapa de rerank en la
    hipotesis C2(b).

    Args:
        caso: Caso con el vector de scores hibrido materializado.
        clave_hibrido: Clave del vector hibrido en el caso.
        canones: Canones alineados con el vector de scores.
        k: Numero de candidatos (20 en la spec).

    Returns:
        Lista de canones ordenados por score descendente (<= k).
    """
    scores = caso[clave_hibrido]
    orden = np.argsort(-scores, kind="stable")[:k]
    return [canones[int(indice)] for indice in orden]


def construir_snippets(
    candidatos: list,
    pasajes: object,
    posicion_canon: dict,
    grupos_indices: list,
) -> dict:
    """Construye el snippet (~200 chars) del primer pasaje de cada candidato.

    Args:
        candidatos: Canones del top-k.
        pasajes: DataFrame de pasajes del indice activo (columna texto).
        posicion_canon: Dict {canon: posicion} en canones/grupos_indices.
        grupos_indices: Arrays de posiciones de pasajes por articulo.

    Returns:
        Dict {canon: snippet} con el primer pasaje normalizado y truncado.
    """
    snippets: dict = {}
    for canon in candidatos:
        indices = grupos_indices[posicion_canon[canon]]
        texto = " ".join(str(pasajes["texto"].iloc[int(indices[0])]).split())
        if len(texto) > CONFIG["snippet_chars"]:
            texto = texto[: CONFIG["snippet_chars"]].rstrip() + "..."
        snippets[canon] = texto
    return snippets


def construir_prompt_usuario(
    contexto: str, candidatos: list, snippets: dict
) -> str:
    """Construye el mensaje de usuario con contexto y candidatos.

    Args:
        contexto: Contexto del caso (problema/hechos o encabezado).
        candidatos: Canones del top-20 en orden del hibrido.
        snippets: Dict {canon: snippet} de cada candidato.

    Returns:
        Prompt de usuario en espanol.
    """
    lineas = [
        f"- Articulo {canon}: {snippets[canon]}" for canon in candidatos
    ]
    return (
        "CONTEXTO DEL CASO:\n"
        f"{contexto}\n\n"
        f"ARTICULOS CANDIDATOS ({len(candidatos)}):\n"
        + "\n".join(lineas)
        + "\n\nDevuelve el objeto JSON con el orden de relevancia."
    )


def construir_payload(
    modelo: str, contexto: str, candidatos: list, snippets: dict
) -> dict:
    """Construye el cuerpo del POST /api/chat del rerank listwise.

    Se envia think=false y format="json" (gramatica JSON de ollama); la
    cadena tolerante de la capa reutilizada reintenta sin esas claves si
    el modelo no las soporta.

    Args:
        modelo: Nombre del modelo en ollama.
        contexto: Contexto del caso.
        candidatos: Canones del top-20.
        snippets: Dict {canon: snippet}.

    Returns:
        Dict serializable con model, messages, format, stream, think y
        options (temperature 0, seed 42, num_ctx, num_predict).
    """
    return {
        "model": modelo,
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {
                "role": "user",
                "content": construir_prompt_usuario(
                    contexto, candidatos, snippets
                ),
            },
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


# =====================================================================
# SECCION 3: PARSER TOLERANTE Y RERANK RESUMIBLE (OLLAMA)
# =====================================================================
def extraer_token(valor: object) -> str | None:
    """Extrae el token numerico (digitos + letra opcional) de un item crudo.

    Acepta enteros, flotantes enteros y cadenas con ruido ("Articulo 86",
    "22a", "86."). No valida contra los candidatos: la interpretacion
    (numero de articulo o posicion de lista) se decide por separado.

    Args:
        valor: Item crudo del campo orden/relevancia.

    Returns:
        Token en mayusculas (p. ej. "86", "22A") o None si no hay digitos.
    """
    if valor is None or isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        if isinstance(valor, float) and not float(valor).is_integer():
            return None
        return str(int(valor))
    texto = str(valor).strip().upper()
    coincidencia = re.search(r"(\d+)\s*([A-Z])?", texto)
    if coincidencia is None:
        return None
    return coincidencia.group(1) + (coincidencia.group(2) or "")


def decidir_interpretacion(
    tokens: list, candidatos: list, n_candidatos: int
) -> str:
    """Decide si los tokens son numeros de articulo o posiciones 1..n.

    El modelo puede devolver numeros de articulo (formato instruido) o,
    como se observo en el smoke, posiciones de la lista (1-20). Se
    comparan ambas lecturas por numero de coincidencias; los empates
    favorecen "articulo" (el formato pedido).

    Args:
        tokens: Tokens extraidos del orden o de relevancia.
        candidatos: Canones candidatos validos.
        n_candidatos: Tamano de la lista de candidatos.

    Returns:
        "posicion" si hay mas tokens que mapean a posiciones validas que
        a numeros de articulo; "articulo" en otro caso.
    """
    candidatos_set = set(candidatos)
    n_articulo = sum(1 for token in tokens if token in candidatos_set)
    n_posicion = sum(
        1
        for token in tokens
        if token.isdigit() and 1 <= int(token) <= n_candidatos
    )
    return "posicion" if n_posicion > n_articulo else "articulo"


def mapear_token(
    token: str | None,
    interpretacion: str,
    candidatos: list,
    candidatos_set: set,
) -> str | None:
    """Mapea un token a un canon candidato segun la interpretacion.

    Args:
        token: Token extraido ("86", "22A", "8").
        interpretacion: "articulo" (el token es un canon) o "posicion"
            (el token es la posicion 1..n de la lista).
        candidatos: Canones candidatos validos en orden del hibrido.
        candidatos_set: Conjunto de candidatos para la busqueda directa.

    Returns:
        Canon candidato o None si el token no mapea.
    """
    if token is None:
        return None
    if interpretacion == "posicion":
        if token.isdigit() and 1 <= int(token) <= len(candidatos):
            return candidatos[int(token) - 1]
        return None
    if token in candidatos_set:
        return token
    return None


def normalizar_grado(valor: object) -> int | None:
    """Normaliza un grado de relevancia a 0, 1 o 2.

    Args:
        valor: Grado crudo (entero, flotante o cadena con ruido).

    Returns:
        Grado entero en {0, 1, 2} o None si no es interpretable.
    """
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        numero = int(valor)
    elif isinstance(valor, str):
        coincidencia = re.search(r"[012]", valor)
        if coincidencia is None:
            return None
        numero = int(coincidencia.group(0))
    else:
        return None
    return numero if 0 <= numero <= 2 else None


def extraer_lista_orden(datos: dict) -> list | None:
    """Extrae el campo orden de un dict parseado (lista o cadena).

    Args:
        datos: Dict decodificado de la respuesta del modelo.

    Returns:
        Lista cruda de items del orden o None si no hay campo util.
    """
    orden = datos.get("orden")
    if isinstance(orden, list):
        return orden
    if isinstance(orden, str):
        partes = [parte for parte in re.split(r"[,;\s]+", orden) if parte]
        return partes or None
    return None


def extraer_tokens_relevancia(datos: dict) -> list:
    """Extrae los tokens de articulo del campo relevancia (sin mapear).

    Acepta lista de dicts {"articulo", "grado"}, lista de pares
    [articulo, grado] y dict {articulo: grado}.

    Args:
        datos: Dict decodificado de la respuesta del modelo.

    Returns:
        Lista de tokens crudos (para decidir la interpretacion).
    """
    relevancia = datos.get("relevancia")
    tokens: list = []
    if isinstance(relevancia, dict):
        for clave in relevancia:
            token = extraer_token(clave)
            if token is not None:
                tokens.append(token)
        return tokens
    if not isinstance(relevancia, list):
        return tokens
    for entrada in relevancia:
        valor = None
        if isinstance(entrada, dict):
            valor = entrada.get("articulo")
        elif isinstance(entrada, (list, tuple)) and entrada:
            valor = entrada[0]
        token = extraer_token(valor)
        if token is not None:
            tokens.append(token)
    return tokens


def extraer_grados(
    datos: dict, interpretacion: str, candidatos: list, candidatos_set: set
) -> dict:
    """Extrae los grados de relevancia del campo relevancia.

    Args:
        datos: Dict decodificado de la respuesta del modelo.
        interpretacion: "articulo" o "posicion" (ver mapear_token).
        candidatos: Canones candidatos validos en orden del hibrido.
        candidatos_set: Conjunto de candidatos para la busqueda directa.

    Returns:
        Dict {canon: grado 0/1/2} de las entradas interpretables.
    """
    relevancia = datos.get("relevancia")
    grados: dict = {}
    if isinstance(relevancia, dict):
        for clave, valor in relevancia.items():
            canon = mapear_token(
                extraer_token(clave),
                interpretacion,
                candidatos,
                candidatos_set,
            )
            grado = normalizar_grado(valor)
            if canon is not None and grado is not None:
                grados[canon] = grado
        return grados
    if not isinstance(relevancia, list):
        return grados
    for entrada in relevancia:
        if isinstance(entrada, dict):
            token = extraer_token(entrada.get("articulo"))
            grado = normalizar_grado(entrada.get("grado"))
        elif isinstance(entrada, (list, tuple)) and len(entrada) >= 2:
            token = extraer_token(entrada[0])
            grado = normalizar_grado(entrada[1])
        else:
            continue
        canon = mapear_token(
            token, interpretacion, candidatos, candidatos_set
        )
        if canon is not None and grado is not None:
            grados[canon] = grado
    return grados


def parsear_respuesta_orden(
    mod30: object, crudo: str, candidatos: list, orden_previo: list
) -> tuple[list, dict, str, int, str]:
    """Parsea la respuesta del LLM al esquema orden/relevancia.

    Cadena tolerante: json.loads directo, bloque {...} balanceado, regex
    del campo orden, orden derivado de los grados y, como ultimo recurso,
    el orden previo del hibrido marcado como fallback. Los tokens se
    mapean como numeros de articulo o como posiciones 1..n de la lista
    segun la interpretacion dominante (el modelo devuelve posiciones en
    parte de los casos). El orden final siempre es una permutacion de los
    candidatos: se descartan tokens invalidos o repetidos y los
    candidatos faltantes se completan al final en el orden previo.

    Args:
        mod30: Modulo 30 reutilizado (limpiar_cerca, _intentar_json,
            _extraer_bloque).
        crudo: Respuesta textual del modelo.
        candidatos: Canones del top-20 en orden del hibrido.
        orden_previo: Orden de respaldo (los propios candidatos).

    Returns:
        Tupla (orden, grados, fuente_parseo, n_completados,
        interpretacion_orden, interpretacion_relevancia) con orden de
        exactamente len(candidatos) canones y fuente en {json_directo,
        bloque_balanceado, regex_orden, grados_sin_orden,
        fallback_orden_previo}.
    """
    candidatos_set = set(candidatos)
    n_candidatos = len(candidatos)
    texto = mod30.limpiar_cerca(crudo)
    datos = mod30._intentar_json(texto)
    fuente = "json_directo"
    if datos is None:
        bloque = mod30._extraer_bloque(texto)
        if bloque is not None:
            datos = mod30._intentar_json(bloque)
            if datos is not None:
                fuente = "bloque_balanceado"
    orden_crudo = None
    if datos is not None:
        orden_crudo = extraer_lista_orden(datos)
    if not orden_crudo:
        coincidencia = re.search(
            r'"orden"\s*:\s*\[([^\]]*)\]', texto, re.DOTALL
        )
        if coincidencia is None:
            # Variante truncada por num_predict: sin corchete de cierre.
            coincidencia = re.search(
                r'"orden"\s*:\s*\[([^\]]*)', texto, re.DOTALL
            )
        if coincidencia is not None:
            orden_crudo = [
                parte
                for parte in re.split(r"[,;\s]+", coincidencia.group(1))
                if parte
            ]
            fuente = "regex_orden"
    tokens_orden = []
    if orden_crudo:
        for item in orden_crudo:
            token = extraer_token(item)
            if token is not None:
                tokens_orden.append(token)
    tokens_relevancia = (
        extraer_tokens_relevancia(datos) if datos is not None else []
    )
    # La interpretacion se decide por campo: el smoke mostro respuestas
    # mixtas (orden con posiciones 1-20 y relevancia con articulos).
    interpretacion_orden = (
        decidir_interpretacion(tokens_orden, candidatos, n_candidatos)
        if tokens_orden
        else None
    )
    interpretacion_relevancia = (
        decidir_interpretacion(tokens_relevancia, candidatos, n_candidatos)
        if tokens_relevancia
        else None
    )
    if interpretacion_orden is None:
        interpretacion_orden = interpretacion_relevancia or "articulo"
    if interpretacion_relevancia is None:
        interpretacion_relevancia = interpretacion_orden
    grados = (
        extraer_grados(
            datos, interpretacion_relevancia, candidatos, candidatos_set
        )
        if datos is not None
        else {}
    )
    if not tokens_orden:
        if grados:
            posicion_previa = {
                canon: posicion
                for posicion, canon in enumerate(orden_previo)
            }
            orden = sorted(
                candidatos,
                key=lambda canon: (
                    -grados.get(canon, 0),
                    posicion_previa.get(canon, n_candidatos),
                ),
            )
            return (
                orden,
                grados,
                "grados_sin_orden",
                0,
                interpretacion_orden,
                interpretacion_relevancia,
            )
        return (
            list(orden_previo),
            {},
            "fallback_orden_previo",
            0,
            "articulo",
            "articulo",
        )
    orden: list = []
    vistos: set = set()
    for token in tokens_orden:
        canon = mapear_token(
            token, interpretacion_orden, candidatos, candidatos_set
        )
        if canon is None or canon in vistos:
            continue
        orden.append(canon)
        vistos.add(canon)
    if not orden:
        return (
            list(orden_previo),
            {},
            "fallback_orden_previo",
            0,
            "articulo",
            "articulo",
        )
    n_completados = sum(1 for canon in orden_previo if canon not in vistos)
    for canon in orden_previo:
        if canon not in vistos:
            orden.append(canon)
            vistos.add(canon)
    return (
        orden,
        grados,
        fuente,
        n_completados,
        interpretacion_orden,
        interpretacion_relevancia,
    )


def rerankear_caso(
    mod30: object,
    caso_id: str,
    texto_crudo: str,
    consulta_c2: dict | None,
    candidatos: list,
    snippets: dict,
    host: str,
    modelo: str,
) -> dict:
    """Rerankea un caso con el LLM y construye su fila JSONL.

    Args:
        mod30: Modulo 30 reutilizado (llamar_ollama_tolerante).
        caso_id: Id del caso del benchmark.
        texto_crudo: Texto crudo de la sentencia (fallback de contexto).
        consulta_c2: Fila C2(a) del caso o None.
        candidatos: Canones del top-20 del hibrido.
        snippets: Dict {canon: snippet}.
        host: URL base de ollama.
        modelo: Nombre del modelo en ollama.

    Returns:
        Fila serializable con id, orden, grados, fuente_parseo, modelo,
        fecha, tiempo_seg y trazabilidad del prompt.

    Raises:
        RuntimeError: Si ollama falla tras la cadena de variantes.
    """
    inicio = time.time()
    contexto, fuente_contexto = preparar_contexto_caso(
        texto_crudo, consulta_c2
    )
    payload = construir_payload(modelo, contexto, candidatos, snippets)
    crudo, _ = mod30.llamar_ollama_tolerante(host, payload)
    (
        orden,
        grados,
        fuente_parseo,
        n_completados,
        interpretacion_orden,
        interpretacion_relevancia,
    ) = parsear_respuesta_orden(mod30, crudo, candidatos, candidatos)
    return {
        "id": caso_id,
        "orden": orden,
        "grados": grados,
        "fuente_parseo": fuente_parseo,
        "interpretacion_orden": interpretacion_orden,
        "interpretacion_relevancia": interpretacion_relevancia,
        "n_completados": n_completados,
        "n_candidatos": len(candidatos),
        "candidatos": list(candidatos),
        "contexto_fuente": fuente_contexto,
        "modelo": modelo,
        "fecha": datetime.now(timezone.utc).isoformat(),
        "tiempo_seg": round(time.time() - inicio, 1),
        "n_chars_contexto": len(contexto),
        "n_chars_prompt": len(payload["messages"][1]["content"]),
    }


def ejecutar_rerank(
    mod12: object, mod30: object, limite: int | None, rutas: dict
) -> dict:
    """Rerankea con el LLM el top-20 del hibrido (JSONL resumible).

    Estrategia: lee el JSONL existente, salta los ids ya rerankeados y
    procesa los pendientes anadiendo una linea por caso. Los fallos de
    transporte se registran y se dejan pendientes para una re-ejecucion;
    los fallos de parseo se escriben con el orden previo marcados como
    fallback_orden_previo. El meta se reescribe con el estado acumulado.

    Args:
        mod12: Modulo del evaluador v12 (BM25, B2, hibrido, pasajes).
        mod30: Modulo 30 reutilizado (ollama, JSONL, benchmark).
        limite: Limite efectivo de casos (None en el run real).
        rutas: Dict de rutas resuelto por resolver_rutas_salida.

    Returns:
        Resumen de la corrida (conteos, parseos, latencias y pendientes).
    """
    inicio = time.time()
    host = CONFIG["host_ollama"]
    modelo = CONFIG["modelo_llm"]
    bench = mod30.cargar_benchmark(
        ROOT / CONFIG["ruta_benchmark_v2"], ("muestra",)
    )
    muestra = mod30.recortar_muestra(mod30.normalizar_muestra(bench), limite)
    mod30.verificar_modelo_disponible(host, modelo)
    print("=" * 78)
    print(
        "31_c2_llm_rerank.py | C2(b): rerank listwise top-20 con "
        "alia-legal"
    )
    print(
        f"fecha: {datetime.now(timezone.utc).isoformat()} | modo: "
        f"{'smoke/limite' if limite is not None else 'real'} | semilla: "
        f"{CONFIG['semilla']}"
    )
    print(f"salidas: {rutas['rerank_jsonl']} | {rutas['rerank_meta']}")
    print("=" * 78)
    print(
        f"Rerank LLM: {len(muestra)} casos | modelo {modelo} | host "
        f"{host} | top-{CONFIG['k_candidatos_llm']} de la 1a etapa"
    )
    if limite is not None:
        print(
            f"[smoke] --limite/--smoke: muestra recortada a {len(muestra)} "
            "casos (primeros por orden del benchmark; NO es el run final)"
        )
    previos = mod30.cargar_queries_jsonl(rutas["rerank_jsonl"])
    consultas_c2 = mod30.cargar_queries_jsonl(rutas["queries_c2"])
    pendientes = [caso for caso in muestra if caso["id"] not in previos]
    print(
        f"JSONL: {len(previos)} filas previas | contexto C2(a) "
        f"disponible en {len(consultas_c2)} casos | pendientes en esta "
        f"corrida: {len(pendientes)}"
    )
    fallos: list[dict] = []
    tiempos_nuevos: dict = {}
    if pendientes:
        textos = mod30.leer_textos_por_id(
            mod12,
            ROOT / CONFIG["ruta_corpus"],
            [caso["id"] for caso in pendientes],
        )
        pasajes, canones, grupos_indices, _ = (
            mod12.preparar_pasajes_constitucion(False)
        )
        posicion_canon = {
            canon: posicion for posicion, canon in enumerate(canones)
        }
        from lib.reranker_bge import elegir_dispositivo

        dispositivo = elegir_dispositivo()
        modelo_b2 = mod12.resolver_modelo_b2(None)
        modelo_b1 = mod12.resolver_modelo_b1(None)
        mod12.aplicar_modo_b2(None, modelo_b2, modelo_b1)
        print(f"Retriever B2: {modelo_b2} | reranker B1 (solo v13): {modelo_b1}")
        casos = mod30.construir_casos_ruta(
            mod12, pendientes, textos, {}, usar_consulta_llm=False
        )
        _, seg_bm25 = mod30.calcular_bm25_ruta(
            mod12, casos, pasajes, canones, grupos_indices
        )
        stats_b2, clave_hibrido = mod30.calcular_denso_ruta(
            mod12, casos, pasajes, canones, grupos_indices, dispositivo
        )
        print(
            f"1a etapa: BM25 {seg_bm25} s | B2 "
            f"{stats_b2['segundos_total']} s "
            f"({stats_b2['n_chunks_query']} chunks de query, "
            f"{dispositivo}) | {len(canones)} canones indexados"
        )
        for posicion, caso in enumerate(casos, start=1):
            sid = caso["id"]
            candidatos = obtener_candidatos(
                caso,
                clave_hibrido,
                canones,
                CONFIG["k_candidatos_llm"],
            )
            snippets = construir_snippets(
                candidatos, pasajes, posicion_canon, grupos_indices
            )
            try:
                fila = rerankear_caso(
                    mod30,
                    caso["id"],
                    caso["texto"],
                    consultas_c2.get(sid),
                    candidatos,
                    snippets,
                    host,
                    modelo,
                )
            except Exception as exc:  # noqa: BLE001 - guard por caso
                fallos.append(
                    {"id": sid, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"[fallo] caso {sid}: {type(exc).__name__}: {exc}")
                continue
            mod30.agregar_query_jsonl(rutas["rerank_jsonl"], fila)
            previos[sid] = fila
            tiempos_nuevos[sid] = fila["tiempo_seg"]
            marcas = []
            if fila["fuente_parseo"] == "fallback_orden_previo":
                marcas.append("fallback_orden")
            if fila["n_completados"]:
                marcas.append(f"{fila['n_completados']} completados")
            sufijo = f" ({', '.join(marcas)})" if marcas else ""
            print(
                f"[{posicion}/{len(pendientes)}] caso {sid}: "
                f"{fila['tiempo_seg']} s | {fila['fuente_parseo']} | "
                f"contexto {fila['contexto_fuente']} | top-3: "
                f"{fila['orden'][:3]}{sufijo}"
            )
        del casos
        if dispositivo == "cuda":
            torch.cuda.empty_cache()
    ids_benchmark = {caso["id"] for caso in mod30.normalizar_muestra(bench)}
    ids_jsonl = set(previos)
    filas_totales = list(previos.values())
    resumen = {
        "n_casos_benchmark": len(ids_benchmark),
        "n_muestra": len(muestra),
        "n_previos": len(muestra) - len(pendientes),
        "n_rerankeados_nuevos": len(tiempos_nuevos),
        "n_fallos_corrida": len(fallos),
        "fallos_corrida": fallos,
        "n_filas_jsonl": len(ids_jsonl),
        "pendientes_benchmark": sorted(ids_benchmark - ids_jsonl),
        "fuentes_parseo": contar_por(filas_totales, "fuente_parseo"),
        "interpretaciones_orden": contar_por(
            filas_totales, "interpretacion_orden"
        ),
        "interpretaciones_relevancia": contar_por(
            filas_totales, "interpretacion_relevancia"
        ),
        "contextos": contar_por(filas_totales, "contexto_fuente"),
        "n_fallbacks_orden": sum(
            1
            for fila in filas_totales
            if fila.get("fuente_parseo") == "fallback_orden_previo"
        ),
        "n_con_completados": sum(
            1 for fila in filas_totales if int(fila.get("n_completados", 0))
        ),
        "latencias": resumen_latencia(filas_totales),
        "tiempos_nuevos_seg": tiempos_nuevos,
        "segundos_corrida": round(time.time() - inicio, 1),
    }
    meta = {
        "fecha": datetime.now(timezone.utc).isoformat(),
        "version": CONFIG["version"],
        "script": "scripts/31_c2_llm_rerank.py",
        "modelo": modelo,
        "host": host,
        "config": {
            "temperature": CONFIG["temperature"],
            "seed": CONFIG["semilla"],
            "num_ctx": CONFIG["num_ctx"],
            "num_predict": CONFIG["num_predict"],
            "think": CONFIG["think"],
            "k_candidatos_llm": CONFIG["k_candidatos_llm"],
            "snippet_chars": CONFIG["snippet_chars"],
            "max_chars_contexto": CONFIG["max_chars_contexto"],
            "max_chars_encabezado": CONFIG["max_chars_encabezado"],
        },
        "jsonl": str(rutas["rerank_jsonl"]),
        "md5_jsonl": mod30._md5(rutas["rerank_jsonl"]),
        **resumen,
        "nota": (
            "Rerank resumible: cada re-ejecucion salta los ids ya "
            "presentes en el JSONL y reintenta los pendientes; los fallos "
            "de parseo se guardan con el orden previo del hibrido "
            "(fuente_parseo=fallback_orden_previo); el meta se reescribe "
            "con el estado acumulado."
        ),
    }
    mod30.guardar_json(rutas["rerank_meta"], meta, sobrescribir=True)
    print(
        f"Rerank: {resumen['n_rerankeados_nuevos']} nuevos | "
        f"{resumen['n_fallos_corrida']} fallos | "
        f"{len(resumen['pendientes_benchmark'])} pendientes del benchmark"
    )
    latencias = resumen["latencias"]
    if latencias["n"]:
        print(
            f"Latencia rerank: media {latencias['media_seg']} s | mediana "
            f"{latencias['mediana_seg']} s | p95 {latencias['p95_seg']} s "
            f"| max {latencias['max_seg']} s | >30 s: "
            f"{latencias['n_excede_gate']}/{latencias['n']}"
        )
    print(f"JSONL: {rutas['rerank_jsonl']}")
    print(f"Meta: {rutas['rerank_meta']}")
    return resumen


# =====================================================================
# SECCION 4: EVALUACION - METRICAS, LATENCIAS Y PAREADO VS V13
# =====================================================================
def construir_filas_llm(
    mod12: object,
    mod30: object,
    muestra_eval: list[dict],
    textos: dict,
    rerank_filas: dict,
) -> list[dict]:
    """Construye las filas por caso de la ruta LLM (orden listwise, k=10).

    El ranking evaluado es el orden del LLM tal cual (sin prior de hubs);
    el truncamiento a k=10 usa la maquinaria de medidas del modulo 30.

    Args:
        mod12: Modulo del evaluador v12 (metricas).
        mod30: Modulo 30 reutilizado (medidas_caso).
        muestra_eval: Casos con fila de rerank disponible.
        textos: Dict {id: {"texto", "texto_limpio"}} del corpus.
        rerank_filas: Dict {id: fila JSONL} del rerank LLM.

    Returns:
        Lista de filas con id, ano, tipo, n_relevantes, n_palabras_texto
        y las seis medidas C2.
    """
    filas = []
    for caso in muestra_eval:
        sid = caso["id"]
        ranking = list(rerank_filas[sid]["orden"])[: CONFIG["k_ndcg"]]
        fila = {
            "id": sid,
            "ano": int(caso["ano"]),
            "tipo": caso["tipo"],
            "n_relevantes": len(caso["arts"]),
            "n_palabras_texto": len(textos[sid]["texto_limpio"].split()),
        }
        fila.update(mod30.medidas_caso(mod12, caso["arts"], ranking))
        filas.append(fila)
    return filas


def asignar_periodos(
    mod12: object, filas_llm: list[dict], filas_v13: list[dict]
) -> None:
    """Asigna el periodo cronologico a ambas rutas (in-place).

    Args:
        mod12: Modulo del evaluador v12 (asignar_periodo).
        filas_llm: Filas por caso de la ruta LLM.
        filas_v13: Filas por caso de la ruta v13.
    """
    for fila_llm, fila_v13 in zip(filas_llm, filas_v13, strict=True):
        periodo = mod12.asignar_periodo(int(fila_llm["ano"]))
        fila_llm["periodo"] = periodo
        fila_v13["periodo"] = periodo


def ejecutar_evaluacion(
    mod12: object, mod30: object, limite: int | None, rutas: dict
) -> dict:
    """Evalua el ranking LLM frente a la ruta v13 recalculada (pareado).

    Recalcula la ruta v13 en la misma corrida con la maquinaria del
    modulo 30 (BM25 + B2 + rerank B1 por chunks + hubs alpha=1.5, k=10),
    evalua el orden listwise del LLM a k=10, calcula IC 95% bootstrap e
    IC pareado LLM - v13 (global y por periodo), valida la ruta v13
    recalculada contra el per-case congelado del benchmark v13, resume
    la latencia por caso frente al gate de 30 s y guarda JSON + figura.

    Args:
        mod12: Modulo del evaluador v12.
        mod30: Modulo 30 reutilizado.
        limite: Limite efectivo de casos (None en el run real).
        rutas: Dict de rutas resuelto por resolver_rutas_salida.

    Returns:
        Dict de salida serializado en el JSON de evaluacion.

    Raises:
        FileNotFoundError: Si no existe el JSONL del rerank LLM.
        RuntimeError: Si ningun caso de la muestra tiene rerank.
    """
    inicio_total = time.time()
    if not rutas["rerank_jsonl"].is_file():
        raise FileNotFoundError(
            f"no existe {rutas['rerank_jsonl']}: ejecuta primero --rerank"
        )
    mod30.comprobar_salidas(rutas)
    print("=" * 78)
    print(
        "31_c2_llm_rerank.py | evaluacion C2(b): rerank LLM vs v13 pareada"
    )
    print(
        f"fecha: {datetime.now(timezone.utc).isoformat()} | modo: "
        f"{'smoke/limite' if limite is not None else 'real'} | semilla: "
        f"{CONFIG['semilla']}"
    )
    print(f"salidas: {rutas['json']} | {rutas['figura']}")
    print("=" * 78)

    estrategia = CONFIG["estrategia_v13"]
    modelo_b2 = mod12.resolver_modelo_b2(None)
    modelo_b1 = mod12.resolver_modelo_b1(None)
    mod12.aplicar_modo_b2(None, modelo_b2, modelo_b1)
    print(f"Ruta de producto v13: {estrategia} | hubs alpha={CONFIG['alpha_hubs']}")
    print(f"Retriever B2: {modelo_b2}")
    print(f"Reranker B1: {modelo_b1}")

    print("[1/6] Insumos, muestra y filas de rerank")
    bench_v2 = mod30.cargar_benchmark(
        ROOT / CONFIG["ruta_benchmark_v2"], ("muestra",)
    )
    bench_v13 = mod30.cargar_benchmark(
        ROOT / CONFIG["ruta_benchmark_v13"],
        ("muestra", "comparacion_b2"),
    )
    muestra = mod30.recortar_muestra(mod30.normalizar_muestra(bench_v2), limite)
    rerank_filas = mod30.cargar_queries_jsonl(rutas["rerank_jsonl"])
    muestra_eval = [caso for caso in muestra if caso["id"] in rerank_filas]
    sin_rerank = [
        caso["id"] for caso in muestra if caso["id"] not in rerank_filas
    ]
    if not muestra_eval:
        raise RuntimeError(
            "ningun caso de la muestra tiene rerank del LLM; ejecuta "
            "--rerank primero"
        )
    if sin_rerank:
        print(
            f"[aviso] {len(sin_rerank)} casos sin rerank quedan fuera de "
            f"la evaluacion: {sin_rerank[:5]}"
        )
    print(
        f"Muestra: {len(muestra)} casos | evaluados: {len(muestra_eval)} | "
        f"benchmark v13: {len(bench_v13['muestra'])} casos congelados"
    )
    n_orden_completo = sum(
        1
        for caso in muestra_eval
        if len(rerank_filas[caso["id"]]["orden"]) == CONFIG["k_candidatos_llm"]
    )
    print(
        f"Rerank LLM: orden completo (20 canones) en {n_orden_completo}/"
        f"{len(muestra_eval)} casos | fuentes: "
        f"{contar_por([rerank_filas[c['id']] for c in muestra_eval], 'fuente_parseo')}"
    )

    textos = mod30.leer_textos_por_id(
        mod12, ROOT / CONFIG["ruta_corpus"], [c["id"] for c in muestra_eval]
    )
    pasajes, canones, grupos_indices, canon_valido = (
        mod12.preparar_pasajes_constitucion(False)
    )
    prior = mod12.cargar_o_construir_prior_hubs()
    print(
        f"CP: {len(canones)} canones / {len(pasajes)} pasajes | prior de "
        f"hubs: {len(prior)} canones"
    )

    print("[2/6] Ruta v13: BM25 y denso B2 (texto completo)")
    casos_v13 = mod30.construir_casos_ruta(
        mod12, muestra_eval, textos, {}, usar_consulta_llm=False
    )
    _, seg_bm25 = mod30.calcular_bm25_ruta(
        mod12, casos_v13, pasajes, canones, grupos_indices
    )
    from lib.reranker_bge import RerankerBGE, elegir_dispositivo

    dispositivo = elegir_dispositivo()
    print(
        f"Dispositivo: {dispositivo} | torch.cuda disponible: "
        f"{torch.cuda.is_available()}"
    )
    stats_b2, clave_hibrido = mod30.calcular_denso_ruta(
        mod12, casos_v13, pasajes, canones, grupos_indices, dispositivo
    )
    print(
        f"BM25: {seg_bm25} s | B2: {stats_b2['n_chunks_query']} chunks de "
        f"query en {stats_b2['segundos_total']} s"
    )

    print("[3/6] Rerank B1 de la ruta v13 (GPU con guard de VRAM)")
    reranker = RerankerBGE(modelo_id=modelo_b1, dispositivo=dispositivo)
    try:
        stats_rr = mod30.rerankear_ruta(
            mod12,
            casos_v13,
            pasajes,
            canones,
            grupos_indices,
            reranker,
            estrategia,
            CONFIG["k_candidatos_v13"],
            clave_hibrido,
        )
    finally:
        del reranker
        if dispositivo == "cuda":
            torch.cuda.empty_cache()
    print(
        f"Rerank B1 v13: {stats_rr['segundos']} s | pares/caso: "
        f"{stats_rr['pares_por_caso']} | max tokens par: "
        f"{stats_rr['max_tokens_par']}"
    )

    print("[4/6] Metricas LLM vs v13 (hubs alpha=1.5, k=10)")
    filas_v13 = mod30.evaluar_ruta(
        mod12,
        casos_v13,
        estrategia,
        prior,
        canones,
        CONFIG["alpha_hubs"],
    )
    filas_llm = construir_filas_llm(
        mod12, mod30, muestra_eval, textos, rerank_filas
    )
    asignar_periodos(mod12, filas_llm, filas_v13)
    agregado_llm = mod30.agregar_medidas(
        mod12, filas_llm, CONFIG["bootstrap"], CONFIG["semilla"]
    )
    agregado_v13 = mod30.agregar_medidas(
        mod12, filas_v13, CONFIG["bootstrap"], CONFIG["semilla"]
    )
    pareado = mod30.pareado_global(mod12, filas_llm, filas_v13)
    pareado_periodo = mod30.pareado_por_grupo(
        mod12, filas_llm, filas_v13, "periodo"
    )
    consistencia = mod30.consistencia_congelada(
        bench_v13,
        filas_v13,
        estrategia,
        CONFIG["contexto_v13_congelado"],
    )
    latencias = resumen_latencia(
        [rerank_filas[caso["id"]] for caso in muestra_eval]
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
    print(
        f"Latencia rerank LLM: media {latencias['media_seg']} s | p95 "
        f"{latencias['p95_seg']} s | >{CONFIG['gate_latencia_seg']:.0f} s: "
        f"{latencias['n_excede_gate']}/{latencias['n']} | cumple gate "
        f"(media)={latencias['cumple_gate_media']} (p95)="
        f"{latencias['cumple_gate_p95']}"
    )

    print("[5/6] Figura")
    graficar_evaluacion(
        agregado_llm,
        agregado_v13,
        pareado,
        pareado_periodo,
        filas_llm,
        filas_v13,
        [float(rerank_filas[c["id"]]["tiempo_seg"]) for c in muestra_eval],
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
            "n_relevantes": int(fila_llm["n_relevantes"]),
            "n_palabras_texto": int(fila_llm["n_palabras_texto"]),
            "fuente_parseo": rerank_filas[fila_llm["id"]]["fuente_parseo"],
            "contexto_fuente": rerank_filas[fila_llm["id"]].get(
                "contexto_fuente"
            ),
            "tiempo_rerank_seg": float(
                rerank_filas[fila_llm["id"]].get("tiempo_seg", 0.0)
            ),
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
        "script": "scripts/31_c2_llm_rerank.py",
        "descripcion": (
            "C2(b) rerank listwise: el LLM legal local alia-legal reordena "
            "el top-20 de la 1a etapa (hibrido 0.5*BM25 + 0.5*B2, sin B1) "
            "con contexto del caso (problema juridico/hechos C2(a) o "
            "encabezado) y se evalua a k=10 frente a la ruta v13 "
            "(hibrido_b2_rerank_chunks_top50_b1 + hubs alpha=1.5) "
            "recalculada en la misma corrida."
        ),
        "terminologia": (
            "La referencia de relevancia son citas CP NER silver del "
            "benchmark M7; 'gold' queda reservado para anotacion humana "
            "(regla del proyecto)."
        ),
        "modo": {"smoke_o_limite": limite is not None, "limite": limite},
        "config": {
            "semilla": CONFIG["semilla"],
            "estrategia_v13": estrategia,
            "alpha_hubs": CONFIG["alpha_hubs"],
            "k_candidatos_v13": CONFIG["k_candidatos_v13"],
            "k_candidatos_llm": CONFIG["k_candidatos_llm"],
            "k_medidas": CONFIG["k_medidas"],
            "k_ndcg": CONFIG["k_ndcg"],
            "bootstrap": CONFIG["bootstrap"],
            "contexto_v13_congelado": CONFIG["contexto_v13_congelado"],
            "gate_latencia_seg": CONFIG["gate_latencia_seg"],
            "modelo_llm": CONFIG["modelo_llm"],
            "host_ollama": CONFIG["host_ollama"],
            "llm": {
                "temperature": CONFIG["temperature"],
                "seed": CONFIG["semilla"],
                "num_ctx": CONFIG["num_ctx"],
                "num_predict": CONFIG["num_predict"],
                "think": CONFIG["think"],
                "snippet_chars": CONFIG["snippet_chars"],
                "max_chars_contexto": CONFIG["max_chars_contexto"],
                "max_chars_encabezado": CONFIG["max_chars_encabezado"],
            },
            "chunk_bm25_chars": mod12.CONFIG["chunk_bm25_chars"],
            "solape_bm25_chars": mod12.CONFIG["solape_bm25_chars"],
            "ventana_query_chars_v13": mod12.CONFIG["ventana_query_chars"],
            "k_chunks_rerank": mod12.CONFIG["k_chunks_rerank"],
            "pesos_hibrido": dict(mod12.CONFIG["pesos_hibrido"]),
        },
        "insumos": {
            "benchmark_v2": str(ROOT / CONFIG["ruta_benchmark_v2"]),
            "benchmark_v13": str(ROOT / CONFIG["ruta_benchmark_v13"]),
            "corpus": str(ROOT / CONFIG["ruta_corpus"]),
            "rerank_jsonl": str(rutas["rerank_jsonl"]),
            "md5_rerank_jsonl": mod30._md5(rutas["rerank_jsonl"]),
            "queries_c2_jsonl": str(rutas["queries_c2"]),
            "canones_indexados": len(canones),
            "canones_validos_gold": len(canon_valido),
            "pasajes": len(pasajes),
            "prior_hubs_canones": len(prior),
        },
        "cobertura_rerank": {
            "n_muestra": len(muestra),
            "n_con_rerank": len(muestra_eval),
            "n_sin_rerank": len(sin_rerank),
            "ids_sin_rerank": sin_rerank,
            "n_orden_completo_20": n_orden_completo,
            "fuentes_parseo": contar_por(
                [rerank_filas[c["id"]] for c in muestra_eval],
                "fuente_parseo",
            ),
            "interpretaciones_orden": contar_por(
                [rerank_filas[c["id"]] for c in muestra_eval],
                "interpretacion_orden",
            ),
            "interpretaciones_relevancia": contar_por(
                [rerank_filas[c["id"]] for c in muestra_eval],
                "interpretacion_relevancia",
            ),
            "contextos": contar_por(
                [rerank_filas[c["id"]] for c in muestra_eval],
                "contexto_fuente",
            ),
        },
        "latencias_rerank": latencias,
        "evaluacion": {
            "llm": {
                "global": agregado_llm,
                "por_periodo": mod30.desglosar_por(
                    mod12,
                    filas_llm,
                    "periodo",
                    CONFIG["bootstrap"],
                    CONFIG["semilla"],
                ),
            },
            "v13": {
                "global": agregado_v13,
                "por_periodo": mod30.desglosar_por(
                    mod12,
                    filas_v13,
                    "periodo",
                    CONFIG["bootstrap"],
                    CONFIG["semilla"],
                ),
            },
            "pareado_llm_vs_v13": {
                "global": pareado,
                "por_periodo": pareado_periodo,
            },
            "consistencia_v13_congelado": consistencia,
            "por_caso": por_caso,
        },
        "rerank_stats_v13": stats_rr,
        "b2_stats_v13": stats_b2,
        "tiempos_segundos": {
            "bm25_v13": seg_bm25,
            "b2_v13": stats_b2["segundos_total"],
            "rerank_b1_v13": stats_rr["segundos"],
            "total": round(time.time() - inicio_total, 1),
        },
        "notas": [
            (
                "El ranking LLM se evalua TAL CUAL (orden listwise puro, "
                "sin prior de hubs) mientras que v13 es la ruta de "
                "producto recalculada en esta corrida (con hubs "
                "alpha=1.5); la diferencia es deliberada: mide si el "
                "criterio del LLM supera a la ruta vigente completa."
            ),
            (
                "Los candidatos del LLM son el top-20 del hibrido "
                "0.5*BM25 + 0.5*B2 sobre el texto completo (sin reranker "
                "B1); su recall@10 esta acotado por ese top-20."
            ),
            (
                "La ruta v13 se RECALCULA en la misma corrida (misma "
                "maquinaria y mismos casos) y se valida contra el "
                "per-case congelado de comparacion_b2 (contexto con_hubs) "
                "del benchmark v13 para nDCG@10 y F1@10."
            ),
            (
                "Parser tolerante: json directo, bloque balanceado, regex "
                "del campo orden y fallback al orden previo del hibrido "
                "marcado como fuente_parseo=fallback_orden_previo; el "
                "orden final siempre es una permutacion de los 20 "
                "candidatos."
            ),
            (
                "Gate de latencia del plan v3: <= 30 s/eval; se reportan "
                "media, mediana, p95 y el numero de casos que exceden el "
                "gate."
            ),
        ],
    }
    mod30.guardar_json(rutas["json"], salida, rutas["sobrescribir"])
    print(f"JSON: {rutas['json']}")
    print("-" * 78)
    print(
        f"Resumen C2(b) (hubs alpha={CONFIG['alpha_hubs']}, "
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
        f"  latencia rerank: media {latencias['media_seg']} s | p95 "
        f"{latencias['p95_seg']} s | >30 s: "
        f"{latencias['n_excede_gate']}/{latencias['n']}"
    )
    print(
        f"  tiempo total evaluacion: "
        f"{salida['tiempos_segundos']['total']} s"
    )
    print("=" * 78)
    return salida


# =====================================================================
# SECCION 5: FIGURA
# =====================================================================
def graficar_evaluacion(
    agregado_llm: dict,
    agregado_v13: dict,
    pareado: dict,
    pareado_periodo: dict,
    filas_llm: list[dict],
    filas_v13: list[dict],
    tiempos: list[float],
    ruta: Path,
    n_casos: int,
) -> None:
    """Genera la figura C2(b): metricas, dispersion, periodo y latencia.

    Paneles: (1) barras de las seis medidas con IC 95% bootstrap para las
    dos rutas; (2) dispersion nDCG@10 por caso con diagonal y conteo
    victorias/empates/derrotas; (3) delta pareado de nDCG@10 por periodo;
    (4) latencia del rerank LLM por caso con la linea del gate de 30 s.

    Args:
        agregado_llm: Agregado global de la ruta LLM.
        agregado_v13: Agregado global de la ruta v13.
        pareado: IC pareado global por medida.
        pareado_periodo: IC pareado por periodo y medida.
        filas_llm: Filas por caso de la ruta LLM.
        filas_v13: Filas por caso de la ruta v13.
        tiempos: Latencia por caso del rerank LLM (segundos).
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
        label="LLM rerank",
    )
    ejes[0, 0].bar(
        posiciones + ancho / 2,
        medias_v13,
        ancho,
        yerr=error_v13,
        capsize=3,
        color="#4c72b0",
        label="v13 (B1 + hubs)",
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
    ejes[0, 1].set_xlabel("nDCG@10 v13 (B1 + hubs)")
    ejes[0, 1].set_ylabel("nDCG@10 LLM rerank")
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

    # Panel 4: latencia por caso del rerank LLM con el gate del plan.
    arreglo = np.asarray(tiempos, dtype=np.float64)
    ordenados = np.sort(arreglo)[::-1]
    ejes[1, 1].bar(
        np.arange(ordenados.size),
        ordenados,
        color="#ccb974",
        width=0.8,
    )
    ejes[1, 1].axhline(
        CONFIG["gate_latencia_seg"],
        color="#c44e52",
        linestyle="--",
        linewidth=1.0,
        label=f"gate {CONFIG['gate_latencia_seg']:.0f} s",
    )
    if arreglo.size:
        ejes[1, 1].text(
            0.03,
            0.97,
            f"media: {arreglo.mean():.1f} s\n"
            f"mediana: {np.median(arreglo):.1f} s\n"
            f"p95: {np.percentile(arreglo, 95):.1f} s\n"
            f">30 s: {int((arreglo > CONFIG['gate_latencia_seg']).sum())}"
            f"/{arreglo.size}",
            transform=ejes[1, 1].transAxes,
            va="top",
            fontsize=9,
        )
    ejes[1, 1].set_xlabel("casos (ordenados por latencia)")
    ejes[1, 1].set_ylabel("segundos por caso")
    ejes[1, 1].set_title("Latencia del rerank LLM por caso")
    ejes[1, 1].legend(fontsize=9)
    ejes[1, 1].grid(axis="y", alpha=0.3)

    figura.suptitle(
        "C2(b) rerank listwise con LLM legal local (alia-legal)\n"
        "top-20 hibrido 0.5*BM25 + 0.5*B2, k=10 | "
        f"n={n_casos} casos del benchmark M7"
    )
    figura.tight_layout()
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


# =====================================================================
# SECCION 6: CLI Y ENTRADA PRINCIPAL
# =====================================================================
def parsear_args() -> argparse.Namespace:
    """Define y parsea la interfaz de linea de comandos.

    Returns:
        Namespace con rerank, evaluar, todo, limite y smoke.
    """
    parser = argparse.ArgumentParser(
        description=(
            "C2(b) rerank listwise con el LLM legal local alia-legal: "
            "reordena el top-20 de la 1a etapa por caso y lo evalua "
            "frente a la ruta v13 (B1 + hubs) de forma pareada."
        )
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Rerankea con el LLM el top-20 por caso (JSONL resumible en "
            "resultados/c2_llm/)."
        ),
    )
    parser.add_argument(
        "--evaluar",
        action="store_true",
        help=(
            "Evalua el rerank del LLM frente a v13 (requiere --rerank "
            "previo)."
        ),
    )
    parser.add_argument(
        "--todo",
        action="store_true",
        help="Ejecuta rerank y evaluacion en una sola corrida.",
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


def _ejecutar(
    args: argparse.Namespace, rutas: dict, limite: int | None, mod30: object
) -> None:
    """Orquesta los modos solicitados (secciones 1 a 5).

    Args:
        args: Namespace del CLI.
        rutas: Dict de rutas de salida.
        limite: Limite efectivo de casos (None en el run real).
        mod30: Modulo 30 reutilizado por importlib.
    """
    inicio_total = time.time()
    mod12 = importar_modulo(ROOT / CONFIG["ruta_script_12"], "evaluador_v12")
    mod12.resolver_rutas()
    mod12.configurar_semillas(CONFIG["semilla"])
    if args.rerank or args.todo:
        ejecutar_rerank(mod12, mod30, limite, rutas)
    if args.evaluar or args.todo:
        ejecutar_evaluacion(mod12, mod30, limite, rutas)
    print(f"Tiempo total del proceso: {round(time.time() - inicio_total, 1)} s")


def main() -> None:
    """Punto de entrada: CLI, log duplicado y manejo de errores."""
    args = parsear_args()
    if not (args.rerank or args.evaluar or args.todo):
        print(
            "Nada que ejecutar: usa --rerank, --evaluar o --todo "
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
    mod30 = importar_modulo(ROOT / CONFIG["ruta_script_30"], "c2_query_llm")
    with (
        rutas["log"].open("w", encoding="utf-8") as archivo,
        contextlib.redirect_stdout(mod30._DuplicadorSalida(archivo)),
        contextlib.redirect_stderr(mod30._DuplicadorSalida(archivo)),
    ):
        try:
            _ejecutar(args, rutas, limite, mod30)
        except Exception as exc:
            # El traceback va al log via stderr redirigido; la marca evita
            # que el manejador externo lo duplique en consola.
            traceback.print_exc()
            exc._reportado_c2b = True
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - reporta el fallo y sale con 1
        if not getattr(exc, "_reportado_c2b", False):
            traceback.print_exc()
        sys.exit(1)
