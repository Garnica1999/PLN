"""evaluar_calidad_respuesta.py
Fecha: 2026-11-01
Descripcion: Evaluacion de la calidad de respuesta de la capa RAG del agente
    (/v1/ask). Para 30 consultas en alcance del gold set mide tres cosas con
    un modelo juez distinto del generador:

    - fidelidad: que las afirmaciones de la respuesta esten respaldadas por
      el contexto recuperado (articulos y fragmentos de la CP);
    - relevancia: que la respuesta ataque la pregunta formulada;
    - precision del contexto: de los articulos que el sistema presenta como
      recuperados, cuantos son pertinentes para la pregunta.

    El juez es `alia-legal:latest` (familia llama, distinta del generador
    qwen) y su veredicto se declara como gold-LLM, no como anotacion humana.
    Por defecto genera respuestas en vivo con el backend; con --desde-gold
    reutiliza las respuestas ya registradas en agente_eval_v3.json, util
    cuando la GPU esta ocupada por otro proyecto.

Estado del arte aplicado: evaluacion tipo RAGAS (Es et al., 2024) con juez
    LLM y rubrica fija, IC 95 % por bootstrap de casos y separacion explicita
    entre contexto recuperado y respuesta.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

BASE = Path(__file__).resolve().parent
GOLD = BASE / "gold_agente_v1.jsonl"
RESPUESTAS_V3 = BASE / "agente_eval_v3.json"
SALIDA_JSON = BASE / "calidad_respuesta_v1.json"
SALIDA_MD = BASE / "calidad_respuesta_v1.md"

API = "http://127.0.0.1:8123"
OLLAMA = "http://127.0.0.1:11434"
MODELO_GENERADOR = "qwen3.8:diario-8k-q3_k_s"
MODELO_JUEZ = os.getenv("JURIS_JUEZ", "alia-legal:latest")
N_CASOS = 30
SEMILLA = 42
REMUESTREOS = 1000
TIMEOUT_ASK = 300.0
TIMEOUT_JUEZ = 180.0

RUBRICA = """Eres un evaluador de respuestas juridicas. Recibes una pregunta,
la respuesta de un asistente y el contexto recuperado (articulos de la
Constitucion con su texto). Califica con numeros entre 0 y 1:

- fidelidad: fraccion de las afirmaciones de la respuesta que el contexto
  respalda. Si la respuesta inventa articulos, sentencias o contenidos que no
  estan en el contexto, baja la nota en proporcion.
- relevancia: cuanto responde la pregunta formulada. 1 si la responde
  directamente, 0 si la evade o responde otra cosa.
- precision_contexto: de los articulos del contexto, fraccion que son
  pertinentes para la pregunta.

Devuelve SOLO un objeto JSON con las claves "fidelidad", "relevancia",
"precision_contexto" y "justificacion" (maximo 30 palabras)."""


def cargar_prompts() -> list[dict]:
    """Toma 30 consultas en alcance del gold set, con muestra determinista."""
    casos = [
        json.loads(linea)
        for linea in GOLD.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]
    en_alcance = sorted(
        (caso for caso in casos if caso["ruta_esperada"] == "en_alcance"),
        key=lambda caso: caso["id"],
    )
    generador = np.random.default_rng(SEMILLA)
    elegidos = generador.choice(len(en_alcance), size=N_CASOS, replace=False)
    return [en_alcance[int(indice)] for indice in sorted(elegidos)]


def respuestas_grabadas() -> list[dict]:
    """Carga las respuestas en alcance registradas en la corrida v3.

    Se reutilizan para juzgar la calidad de respuesta sin volver a cargar el
    generador: el juez solo necesita la pregunta, la respuesta y el contexto.

    Returns:
        Lista de respuestas con la misma forma que la ruta en vivo.
    """
    datos = json.loads(RESPUESTAS_V3.read_text(encoding="utf-8"))
    en_alcance = sorted(
        (
            fila
            for fila in datos["resultados"]
            if fila["ruta_esperada"] == "en_alcance"
        ),
        key=lambda fila: fila["id"],
    )
    generador = np.random.default_rng(SEMILLA)
    elegidos = generador.choice(len(en_alcance), size=N_CASOS, replace=False)
    return [
        {
            "id": en_alcance[int(indice)]["id"],
            "prompt": en_alcance[int(indice)]["prompt"],
            "respuesta": en_alcance[int(indice)].get("respuesta"),
            "tool_calls_resumen": [
                llamada.get("resumen")
                for llamada in en_alcance[int(indice)].get("tool_calls") or []
            ],
            "modo": en_alcance[int(indice)].get("modo"),
            "grounding_ok": (
                en_alcance[int(indice)].get("grounding") or {}
            ).get("ok"),
            "ms": en_alcance[int(indice)].get("ms_api"),
        }
        for indice in sorted(elegidos)
    ]


def preguntar(cliente: httpx.Client, pregunta: str) -> dict:
    """Ejecuta /v1/ask y devuelve la respuesta con su traza."""
    inicio = time.perf_counter()
    respuesta = cliente.post(
        "/v1/ask", json={"pregunta": pregunta}, timeout=TIMEOUT_ASK
    )
    respuesta.raise_for_status()
    datos = respuesta.json()
    datos["_ms"] = int((time.perf_counter() - inicio) * 1000)
    return datos


def contexto(cliente: httpx.Client, pregunta: str, respuesta: dict) -> list[dict]:
    """Arma el contexto recuperado a partir de la traza y del indice CP.

    Usa la sentencia que resolvio la herramienta y los articulos que el
    sistema presenta, con su texto constitucional.
    """
    articulos: dict[str, dict] = {}
    sentencia = None
    resumenes = respuesta.get("tool_calls_resumen") or [
        llamada.get("resumen") for llamada in respuesta.get("tool_calls") or []
    ]
    for resumen_bruto in resumenes:
        resumen = str(resumen_bruto or "")
        for parte in resumen.replace("articulos=", ";").split(";"):
            valor = parte.strip()
            if valor.isdigit() and valor not in articulos:
                articulos[valor] = {}
        if "id=" in resumen and sentencia is None:
            sentencia = resumen.split("id=")[1].split()[0].strip(",")
    if not articulos:
        try:
            enlace = cliente.post(
                "/v1/link", json={"texto": pregunta, "k": 5}, timeout=120.0
            )
            if enlace.status_code == 200:
                for fila in enlace.json().get("articulos_recuperados") or []:
                    articulos[str(fila["numero"])] = {}
        except httpx.HTTPError:
            pass
    detalle = []
    for numero in list(articulos)[:5]:
        try:
            info = cliente.get(f"/v1/article/{numero}", timeout=30.0)
            if info.status_code == 200:
                datos = info.json()
                detalle.append(
                    {
                        "numero": numero,
                        "bloque": datos.get("bloque"),
                        "texto": datos.get("texto_150chars"),
                    }
                )
        except httpx.HTTPError:
            continue
    return [{"sentencia": sentencia, "articulos": detalle}] if detalle else []


def juzgar(pregunta: str, respuesta: str, contexto_datos: list[dict]) -> dict:
    """Pide al modelo juez las tres notas y devuelve el JSON parseado."""
    cuerpo = {
        "model": MODELO_JUEZ,
        "messages": [
            {"role": "system", "content": RUBRICA},
            {
                "role": "user",
                "content": (
                    f"Pregunta: {pregunta}\n\n"
                    f"Respuesta del asistente:\n{respuesta}\n\n"
                    "Contexto recuperado:\n"
                    + json.dumps(contexto_datos, ensure_ascii=False)
                ),
            },
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "seed": 42, "num_predict": 300},
    }
    with httpx.Client(timeout=TIMEOUT_JUEZ) as cliente:
        datos = cliente.post(f"{OLLAMA}/api/chat", json=cuerpo).json()
    texto = (datos.get("message") or {}).get("content") or "{}"
    notas = json.loads(texto)
    valores = {
        clave: float(notas.get(clave, 0.0))
        for clave in ("fidelidad", "relevancia", "precision_contexto")
    }
    fuera = [clave for clave, valor in valores.items() if not 0.0 <= valor <= 1.0]
    if fuera:
        raise ValueError(f"juez fuera de escala en {', '.join(fuera)}: {valores}")
    return {
        **valores,
        "justificacion": str(notas.get("justificacion", ""))[:200],
    }


def ic95(valores: list[float]) -> list[float]:
    """IC 95 % por bootstrap de casos con semilla 42."""
    arreglo = np.asarray(valores, dtype=float)
    generador = np.random.default_rng(SEMILLA)
    medias = [
        arreglo[generador.integers(0, len(arreglo), len(arreglo))].mean()
        for _ in range(REMUESTREOS)
    ]
    return [
        round(float(np.percentile(medias, 2.5)), 4),
        round(float(np.percentile(medias, 97.5)), 4),
    ]


def detener(modelo: str) -> None:
    """Descarga un modelo de ollama para no dejar la GPU ocupada."""
    subprocess.run(
        ["ollama", "stop", modelo], capture_output=True, text=True, timeout=60, check=False
    )


def main() -> int:
    """Ejecuta la evaluacion completa y escribe JSON y Markdown."""
    desde_gold = "--desde-gold" in sys.argv
    registros = []
    with httpx.Client(base_url=API, timeout=TIMEOUT_ASK) as cliente:
        if desde_gold:
            casos = [
                {"id": fila["id"], "prompt": fila["prompt"], "grabada": fila}
                for fila in respuestas_grabadas()
            ]
        else:
            casos = [
                {"id": fila["id"], "prompt": fila["prompt"]} for fila in cargar_prompts()
            ]
        for caso in casos:
            try:
                respuesta = (
                    caso["grabada"]
                    if desde_gold
                    else preguntar(cliente, caso["prompt"])
                )
                contexto_datos = contexto(cliente, caso["prompt"], respuesta)
                if not contexto_datos:
                    registros.append(
                        {
                            "id": caso["id"],
                            "prompt": caso["prompt"],
                            "error": "sin contexto recuperado",
                        }
                    )
                    continue
                notas = juzgar(
                    caso["prompt"], respuesta.get("respuesta") or "", contexto_datos
                )
                registros.append(
                    {
                        "id": caso["id"],
                        "prompt": caso["prompt"],
                        "modo": respuesta.get("modo"),
                        "respuesta": respuesta.get("respuesta"),
                        "tool_calls": (
                            [
                                llamada["herramienta"]
                                for llamada in respuesta.get("tool_calls") or []
                            ]
                            if respuesta.get("tool_calls")
                            else respuesta.get("tool_calls_resumen") or []
                        ),
                        "grounding_ok": (respuesta.get("grounding") or {}).get("ok"),
                        "ms": respuesta.get("_ms"),
                        "contexto": contexto_datos,
                        **notas,
                    }
                )
            except (httpx.HTTPError, ValueError) as exc:
                registros.append(
                    {
                        "id": caso["id"],
                        "prompt": caso["prompt"],
                        "error": str(exc)[:200],
                        "descartado": True,
                    }
                )

    detener(MODELO_GENERADOR)
    detener(MODELO_JUEZ)

    validos = [r for r in registros if "error" not in r]
    metricas = {}
    for clave in ("fidelidad", "relevancia", "precision_contexto"):
        valores = [r[clave] for r in validos]
        metricas[clave] = {
            "media": round(float(np.mean(valores)), 4) if valores else None,
            "ic95": ic95(valores) if valores else None,
            "min": round(float(np.min(valores)), 4) if valores else None,
            "max": round(float(np.max(valores)), 4) if valores else None,
        }
    resultado = {
        "experimento": "calidad_respuesta_rag_v1",
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_casos": len(casos),
        "n_validos": len(validos),
        "generador": MODELO_GENERADOR,
        "juez": MODELO_JUEZ,
        "procedencia": (
            "Juez LLM de familia distinta al generador con rubrica fija; es gold-LLM, "
            "no anotacion humana. Las consultas salen del gold set del agente (ruta en_alcance)."
        ),
        "metricas": metricas,
        "resultados": registros,
    }
    SALIDA_JSON.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")

    filas = "\n".join(
        f"| {r['id']} | {r.get('fidelidad', 'error')} | {r.get('relevancia', '')} | "
        f"{r.get('precision_contexto', '')} | {(r.get('respuesta') or r.get('error') or '')[:90]} |"
        for r in registros
    )
    lineas = [
        "# Calidad de respuesta del RAG (v1)",
        "",
        f"Fecha: {resultado['fecha']}. Casos: {len(casos)} ({len(validos)} validos).",
        f"Generador: `{MODELO_GENERADOR}`. Juez: `{MODELO_JUEZ}` (gold-LLM, no humano).",
        "",
        "## Metricas",
        "",
        "| Metrica | Media | IC 95 % | Min | Max |",
        "|---|---:|---|---:|---:|",
    ]
    for clave, datos in metricas.items():
        lineas.append(
            f"| {clave} | {datos['media']} | {datos['ic95']} | {datos['min']} | {datos['max']} |"
        )
    lineas += [
        "",
        "## Detalle",
        "",
        "| Caso | Fidelidad | Relevancia | Precision contexto | Respuesta (extracto) |",
        "|---|---:|---:|---:|---|",
        filas,
        "",
    ]
    SALIDA_MD.write_text("\n".join(lineas), encoding="utf-8")

    print(f"casos validos: {len(validos)}/{len(casos)}")
    for clave, datos in metricas.items():
        print(f"  {clave}: {datos['media']} {datos['ic95']}")
    print(f"JSON: {SALIDA_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
