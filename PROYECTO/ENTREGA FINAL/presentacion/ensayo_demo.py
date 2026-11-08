"""ensayo_demo.py
Fecha: 2026-10-31
Descripcion: Ensayo de la demo de la Entrega 3 contra el backend real. Recorre
    los mismos flujos que la interfaz (validacion, enlace, evidencia,
    jurisprudencia, similares, metricas y chat) y escribe el guion de la demo
    con las latencias medidas. Antes de usar el LLM verifica la VRAM y al
    terminar descarga el modelo para no dejar la GPU ocupada.
Estado del arte aplicado: medicion de latencia en pared por paso, umbral de
    VRAM antes de cargar el modelo y guion reproducible generado desde la
    corrida, no escrito a mano.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

BASE = Path(__file__).resolve().parent
GUION = BASE / "guion_demo.md"
API = "http://127.0.0.1:8123"
MODELO = "qwen3.8:diario-8k-q3_k_s"
TIMEOUT = 300.0

CABECERA = """Sentencia T-293/97
República de Colombia
Corte Constitucional
Magistrado Ponente: Jorge Arango Mejía

La Sala Tercera de Revisión de la Corte Constitucional decide la acción de
tutela interpuesta contra la Gobernación de Cundinamarca.
"""


def vram_libre_gib() -> float | None:
    """Devuelve la VRAM libre en GiB, o None si rocm-smi no responde."""
    try:
        salida = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    total = usado = None
    for linea in salida.splitlines():
        if "Total Memory" in linea:
            total = int(linea.rsplit(":", 1)[1].strip())
        elif "Used Memory" in linea:
            usado = int(linea.rsplit(":", 1)[1].strip())
    if total is None or usado is None:
        return None
    return (total - usado) / 1024**3


def paso(nombre: str, metodo: str, ruta: str, cuerpo: dict | None = None) -> dict:
    """Ejecuta una peticion y devuelve el JSON con su latencia en ms."""
    inicio = time.perf_counter()
    with httpx.Client(base_url=API, timeout=TIMEOUT) as cliente:
        respuesta = cliente.request(metodo, ruta, json=cuerpo)
        respuesta.raise_for_status()
        datos = respuesta.json()
    datos["_ms"] = int((time.perf_counter() - inicio) * 1000)
    datos["_paso"] = nombre
    return datos


def main() -> int:
    """Recorre la demo completa y escribe el guion con lo medido."""
    sin_llm = "--sin-llm" in sys.argv
    libre = vram_libre_gib()
    print(f"[vram] libre antes del ensayo: {libre:.1f} GiB" if libre else "[vram] sin dato")
    if not sin_llm and libre is not None and libre < 14.0:
        print("[vram] insuficiente para el LLM; use --sin-llm o libere la GPU.")
        return 1
    pasos: list[dict] = []
    pasos.append(paso("Salud del servicio", "GET", "/health"))
    pasos.append(paso("Validación M0", "POST", "/v1/validate", {"texto": CABECERA}))
    pasos.append(paso("Enlace sentencia 985", "POST", "/v1/link", {"id": "985", "k": 10}))
    pasos.append(paso("Enlace sentencia 2195", "POST", "/v1/link", {"id": "2195", "k": 10}))
    pasos.append(
        paso(
            "Evidencia del artículo 86",
            "POST",
            "/v1/evidencia",
            {"id": "2195", "articulos": ["86", "13"]},
        )
    )
    pasos.append(
        paso("Jurisprudencia del artículo 86", "GET", "/v1/article/86/jurisprudencia")
    )
    pasos.append(paso("Sentencias similares a 2195", "GET", "/v1/similar/2195?k=5"))
    pasos.append(paso("Resumen de evaluación", "GET", "/v1/eval/summary"))
    pasos.append(
        paso("Chat: saludo", "POST", "/v1/ask", {"pregunta": "Muy buenas"})
    )
    pasos.append(
        paso(
            "Chat: fuera de alcance",
            "POST",
            "/v1/ask",
            {"pregunta": "¿Quién ganó el mundial 2022?"},
        )
    )
    if sin_llm:
        print("[llm] omitido: no hay VRAM suficiente en este momento")
    else:
        pasos.append(
            paso(
                "Chat: consulta en alcance",
                "POST",
                "/v1/ask",
                {
                    "pregunta": (
                        "¿Qué artículos de la CP fundamentan la sentencia T-293/97?"
                    )
                },
            )
        )
        pasos.append(
            paso(
                "Chat: segunda consulta en alcance",
                "POST",
                "/v1/ask",
                {"pregunta": "¿Qué sentencias cita la C-355/06?"},
            )
        )

    subprocess.run(
        ["ollama", "stop", MODELO], capture_output=True, text=True, timeout=60, check=False
    )
    print("[vram] modelo descargado al terminar")

    filas = []
    for datos in pasos:
        detalle = ""
        if datos["_paso"] == "Validación M0":
            detalle = (
                f"es_sentencia={datos.get('es_sentencia')} "
                f"canon={datos.get('canon_detectado')}"
            )
        elif datos["_paso"].startswith("Enlace"):
            detalle = (
                f"canon={datos.get('canon_detectado')} "
                f"extraídos={len(datos.get('articulos_extraidos') or [])} "
                f"recuperados={len(datos.get('articulos_recuperados') or [])} "
                f"citadas={len(datos.get('sentencias_citadas') or [])}"
            )
        elif datos["_paso"].startswith("Evidencia"):
            detalle = ", ".join(
                f"{a['numero']}: {len(a['fragmentos'])} fragmentos"
                for a in datos.get("articulos") or []
            )
        elif datos["_paso"].startswith("Jurisprudencia"):
            detalle = f"total={datos.get('total')} top={datos.get('top', [{}])[0].get('canon')}"
        elif datos["_paso"].startswith("Sentencias similares"):
            detalle = "vecinas=" + ", ".join(
                fila["canon"] for fila in (datos.get("similares") or [])[:3]
            )
        elif datos["_paso"] == "Resumen de evaluación":
            detalle = "comparativa y guardrails disponibles"
        elif datos["_paso"].startswith("Chat"):
            herramientas = ",".join(
                fila["herramienta"] for fila in datos.get("tool_calls") or []
            )
            grounding = (datos.get("grounding") or {}).get("ok")
            detalle = (
                f"modo={datos.get('modo')} intención={datos.get('intencion')}"
                + (f" tools={herramientas}" if herramientas else "")
                + f" grounding={'ok' if grounding else 'revisar'}"
            )
        filas.append((datos["_paso"], datos["_ms"], detalle))

    lineas = [
        "# Guion de la demo (Entrega 3)",
        "",
        (
            f"Ensayo ejecutado el "
            f"{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')} "
            f"contra `{API}` con el modelo `{MODELO}`."
        ),
        "",
        "## Antes de empezar",
        "",
        "1. Verificar la GPU: `rocm-smi --showmeminfo vram` (se necesitan 14 GB libres).",
        "2. Levantar el backend: `uvicorn main:app --host 127.0.0.1 --port 8123`.",
        "3. Levantar la interfaz: `streamlit run streamlit_app.py --server.port 8502`.",
        "4. Abrir `http://127.0.0.1:8502` y dejar la barra lateral visible.",
        "",
        "La primera consulta en alcance carga el modelo (unos 20 s); la segunda",
        "responde en unos 9 s. Conviene hacer una consulta de calentamiento antes",
        "de la demo.",
        "",
        "## Recorrido medido",
        "",
        "| Paso | Latencia | Resultado |",
        "|---|---:|---|",
    ]
    for nombre, ms, detalle in filas:
        lineas.append(f"| {nombre} | {ms} ms | {detalle} |")

    respuesta = next(
        (dato for dato in pasos if dato["_paso"] == "Chat: consulta en alcance"),
        None,
    )
    if respuesta:
        texto_respuesta = " ".join(str(respuesta.get("respuesta") or "").split())
        traza = "; ".join(
            f"{fila['herramienta']} ({'ok' if fila['ok'] else 'fallo'}, {fila['ms']} ms)"
            for fila in respuesta.get("tool_calls") or []
        )
        grounding = (respuesta.get("grounding") or {}).get("ok")
        lineas.extend(
            [
                "",
                "## Respuesta real del agente (consulta en alcance)",
                "",
                f"> {texto_respuesta[:600]}",
                "",
                (
                    f"Traza: {traza or 'sin herramientas'}. Grounding: "
                    f"{'verificado' if grounding else 'con citas sin verificar'}."
                ),
            ]
        )

    guardrails = next(
        (dato for dato in pasos if dato["_paso"] == "Resumen de evaluación"), {}
    ).get("agente_guardrails") or {}
    metricas = guardrails.get("metricas") or []
    if metricas:
        lineas.extend(["", "## Cifras de guardrails para decir en voz alta", ""])
        for metrica in metricas:
            lineas.append(
                f"- {metrica['etiqueta']}: {str(round(float(metrica['valor']) * 100, 1)).replace('.', ',')} %"
            )
        latencia = {fila["clave"]: fila["valor"] for fila in guardrails.get("latencia") or []}
        p50 = str(round(latencia.get("p50", 0) / 1000, 1)).replace(".", ",")
        p95 = str(round(latencia.get("p95", 0) / 1000, 1)).replace(".", ",")
        lineas.append(
            f"- Latencia del agente: p50 {p50} s, p95 {p95} s"
        )

    if sin_llm:
        lineas.extend(
            [
                "",
                "## Pendiente",
                "",
                "El chat en alcance no se midió en este ensayo: la GPU estaba ocupada",
                "por otro proyecto. Repetir `python ensayo_demo.py` con 14 GB libres",
                "antes de la demo.",
            ]
        )

    lineas.extend(
        [
            "",
            "## Orden de la presentación",
            "",
            "1. Portada y problema (2 min): 23.742 sentencias, 261 M palabras, la tarea manual.",
            "2. Datos y método (2 min): corpus, MTLD, benchmark de 90 casos y gold humano.",
            "3. Resultados (2 min): 0,6013 de la línea base a 0,7852 con gold humano.",
            "4. Producto (3 min): P1 con el id 985, P2 con el artículo 86, P3 y chat.",
            "5. Cierre (1 min): límites, trabajo futuro y la frase de cierre.",
            "",
            "## Contingencia",
            "",
            "- Si el LLM tarda o la GPU está ocupada, el agente responde con el plan",
            "  determinista y evidencia verificable; el modo aparece en la interfaz.",
            "- Si el backend no responde, revisar `/health` y reiniciar uvicorn.",
            "- Usar el id 985 para la demo con citas: el 2195 solo tiene autocitas.",
            "",
        ]
    )
    GUION.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Guion escrito en {GUION}")
    for nombre, ms, detalle in filas:
        print(f"  {nombre:32s} {ms:6d} ms  {detalle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
