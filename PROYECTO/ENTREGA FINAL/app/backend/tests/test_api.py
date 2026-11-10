"""test_api.py
Fecha: 2026-10-31
Descripcion: Pruebas del backend Juris-PLN. Las de API y router no usan LLM
    (rapidas, deterministas). La prueba del agente esta marcada como
    integracion y se salta si no hay ollama, si la VRAM libre es menor a la
    que necesita el modelo por defecto o si no se define JURIS_LLM_TESTS=1.
Estado del arte aplicado: TestClient de FastAPI, verificacion de grounding
    con evidencia fabricada y skip controlado por recursos reales.
"""

import os
import subprocess

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import agent
import core
import main

TEXTO_SENTENCIA = """Sentencia T-293/97
República de Colombia
Corte Constitucional
Magistrado Ponente: Eduardo Cifuentes Muñoz

La Corte estudia el derecho fundamental a la dignidad humana y cita la
Sentencia T-406/92 y los artículos 1 y 13 de la Constitución Política.
"""

TEXTO_NO_SENTENCIA = "Estimado usuario: adjunto el informe de gestión anual."


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    """Cliente HTTP de la app FastAPI."""
    return TestClient(main.app)


# ===== SECCION 1: salud y catalogo =====
def test_health(cliente: TestClient) -> None:
    """La app responde con el estado de artefactos y componentes."""
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["status"] == "ok"
    assert "ready" in datos and "cargados" in datos
    assert datos["modelo_defecto"] == agent.MODELO_DEFECTO


def test_models(cliente: TestClient) -> None:
    """El catalogo lista modelos y marca el defecto (o informa su ausencia)."""
    respuesta = cliente.get("/v1/models")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["defecto"] == agent.MODELO_DEFECTO
    assert isinstance(datos["modelos"], list)
    if datos["disponible"]:
        assert any(modelo["defecto"] for modelo in datos["modelos"])
        assert all("tool_calling" in modelo for modelo in datos["modelos"])


# ===== SECCION 2: validacion M0 =====
def test_validate_sentencia(cliente: TestClient) -> None:
    """Un encabezado de providencia cumple las tres reglas de M0."""
    respuesta = cliente.post("/v1/validate", json={"texto": TEXTO_SENTENCIA})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["es_sentencia"] is True
    assert datos["score_validacion"] == 1.0
    assert datos["canon_detectado"] == "T-293-1997"
    assert datos["id_detectado"] == "2195"


def test_validate_no_sentencia(cliente: TestClient) -> None:
    """Un texto administrativo no pasa la heuristica."""
    respuesta = cliente.post("/v1/validate", json={"texto": TEXTO_NO_SENTENCIA})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["es_sentencia"] is False
    assert datos["canon_detectado"] is None


def test_validate_requiere_una_entrada(cliente: TestClient) -> None:
    """Enviar texto e id a la vez es un error de validacion (422)."""
    respuesta = cliente.post(
        "/v1/validate", json={"texto": TEXTO_SENTENCIA, "id": "2195"}
    )
    assert respuesta.status_code == 422


# ===== SECCION 3: vinculo con la CP =====
def test_link_texto_corto(cliente: TestClient) -> None:
    """Un texto corto recupera articulos de la Constitucion."""
    respuesta = cliente.post(
        "/v1/link",
        json={"texto": "derecho fundamental a la vida y a la dignidad humana", "k": 3},
    )
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["origen"] == "texto"
    assert len(datos["articulos_recuperados"]) == 3
    assert all(articulo["numero"] for articulo in datos["articulos_recuperados"])


def test_link_por_id(cliente: TestClient) -> None:
    """Un id del corpus resuelve a la sentencia y recupera articulos."""
    respuesta = cliente.post("/v1/link", json={"id": "2195", "k": 3})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["id"] == "2195"
    assert datos["canon_detectado"] == "T-293-1997"
    assert len(datos["articulos_recuperados"]) == 3


def test_link_contrato_extendido(cliente: TestClient) -> None:
    """El vinculo incluye area exploratoria M4, citadas M5 y resumen."""
    respuesta = cliente.post("/v1/link", json={"id": "985", "k": 3})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["area"]["exploratoria"] is True
    assert datos["area"]["via"]
    assert len(datos["area"]["top3"]) == 3
    assert all(0.0 <= item["prob"] <= 1.0 for item in datos["area"]["top3"])
    assert datos["sentencias_citadas"], "T-225-1993 cita sentencias del corpus"
    assert all(
        cita["canon"] != datos["canon_detectado"]
        for cita in datos["sentencias_citadas"]
    )
    assert any(cita["resuelta"] for cita in datos["sentencias_citadas"])
    assert datos["resumen"].startswith("Providencia valida")
    assert "articulos CP" in datos["resumen"]


def test_link_id_inexistente(cliente: TestClient) -> None:
    """Un id fuera del corpus devuelve 404."""
    respuesta = cliente.post("/v1/link", json={"id": "999999"})
    assert respuesta.status_code == 404


def test_link_id_c3(cliente: TestClient) -> None:
    """Un id del corpus 2022-2026 se resuelve y se vincula con la CP."""
    respuesta = cliente.post("/v1/link", json={"id": "30000", "k": 10})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["canon_detectado"] == "T-1-2022"
    assert datos["articulos_extraidos"]
    assert datos["articulos_recuperados"]


def test_jurisprudencia_incluye_2022_2026(cliente: TestClient) -> None:
    """El indice de articulos incluye las sentencias del corpus extendido."""
    datos = cliente.get("/v1/article/86/jurisprudencia").json()
    assert datos["total"] > 16405
    assert max(int(ano) for ano in datos["por_ano"]) == 2026


def test_similar_id_c3(cliente: TestClient) -> None:
    """El indice de similitud incluye los documentos 2022-2026."""
    datos = cliente.get("/v1/similar/30000?k=5").json()
    assert datos["canon"] == "T-1-2022"
    assert datos["similares"]


def test_article_detalle(cliente: TestClient) -> None:
    """El detalle de un articulo trae texto y bloque tematico."""
    respuesta = cliente.get("/v1/article/13")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["numero"] == "13"
    assert datos["texto_150chars"]
    assert datos["bloque"]


def test_article_inexistente(cliente: TestClient) -> None:
    """Un articulo que no existe devuelve 404."""
    respuesta = cliente.get("/v1/article/9999")
    assert respuesta.status_code == 404


# ===== SECCION 3b: jurisprudencia por articulo (P2) =====
def test_jurisprudencia_articulo(cliente: TestClient) -> None:
    """El articulo 86 tiene sentencias del corpus y un top con PageRank."""
    respuesta = cliente.get("/v1/article/86/jurisprudencia", params={"top": 5})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["numero"] == "86"
    assert datos["total"] > 0
    assert datos["por_ano"]
    assert len(datos["top"]) == 5
    assert all(fila["canon"] and fila["tipo"] for fila in datos["top"])
    assert all("area_silver" in fila for fila in datos["top"])
    assert "silver" in datos["aviso"]
    pageranks = [fila["pagerank"] for fila in datos["top"]]
    assert pageranks == sorted(pageranks, reverse=True)


def test_jurisprudencia_articulo_inexistente(cliente: TestClient) -> None:
    """Un articulo fuera de la CP devuelve 404 con mensaje explicito."""
    respuesta = cliente.get("/v1/article/9999/jurisprudencia")
    assert respuesta.status_code == 404
    assert "no existe" in respuesta.json()["detail"]


# ===== SECCION 3c: evidencia por articulo (P1) =====
def test_evidencia_por_articulo(cliente: TestClient) -> None:
    """El articulo 86 devuelve fragmentos no vacios con score positivo."""
    respuesta = cliente.post(
        "/v1/evidencia",
        json={"id": "2195", "articulos": ["86"], "rerank": False},
    )
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["id"] == "2195"
    assert datos["canon_detectado"] == "T-293-1997"
    assert len(datos["articulos"]) == 1
    evidencias = datos["articulos"][0]
    assert evidencias["numero"] == "86"
    assert evidencias["fragmentos"]
    assert all(frag["texto"].strip() for frag in evidencias["fragmentos"])
    assert all(frag["score"] > 0 for frag in evidencias["fragmentos"])


def test_evidencia_articulo_inexistente(cliente: TestClient) -> None:
    """Un articulo fuera del indice de la CP devuelve 404."""
    respuesta = cliente.post(
        "/v1/evidencia", json={"id": "2195", "articulos": ["9999"]}
    )
    assert respuesta.status_code == 404


def test_evidencia_requiere_una_entrada(cliente: TestClient) -> None:
    """Enviar texto e id a la vez es un error de validacion (422)."""
    respuesta = cliente.post(
        "/v1/evidencia",
        json={"id": "2195", "texto": "texto de prueba", "articulos": ["86"]},
    )
    assert respuesta.status_code == 422


# ===== SECCION 4: router y grounding (sin LLM) =====
@pytest.mark.parametrize(
    ("pregunta", "intencion"),
    [
        ("Hola, cómo estás", "saludo"),
        ("Muy buenas", "saludo"),
        ("Buenas", "saludo"),
        ("Gracias por la ayuda", "agradecimiento"),
        ("¿Quién eres?", "identidad"),
        ("¿En qué puedes ayudarme?", "identidad"),
        ("¿Qué puedes hacer?", "identidad"),
        ("¿Quién ganó el mundial 2022?", "fuera_alcance"),
        ("¿Cuál es el precio del dólar?", "fuera_alcance"),
        ("¿Cómo se dice hola en japonés?", "fuera_alcance"),
        ("Háblame de la sentencia T-293/97", "en_alcance"),
        ("¿Qué artículos de la CP fundamentan la sentencia T-293/97?", "en_alcance"),
        ("¿Qué es el debido proceso?", "en_alcance"),
        ("¿Qué dice sobre eso?", "ambiguo"),
        ("¿Puedes revisarlo?", "ambiguo"),
        ("cuéntame algo", "ambiguo"),
    ],
)
def test_router_intencion(pregunta: str, intencion: str) -> None:
    """El router determinista clasifica saludo, alcance y ambiguedad."""
    assert agent.clasificar_intencion(pregunta) == intencion


class _FalsoLLM:
    """Modelo falso que responde texto plano, sin tool calls."""

    def bind_tools(self, _herramientas: object) -> "_FalsoLLM":
        """Devuelve el mismo objeto: el agente lo usa como chat model."""
        return self

    def invoke(self, _mensajes: object) -> AIMessage:
        """Simula una respuesta vacia del modelo, sin herramientas."""
        return AIMessage(content="")


def test_agente_sin_tool_calls_usa_plan_determinista(monkeypatch) -> None:
    """Si el modelo no recupera evidencia, responde el plan determinista.

    Cubre la rama que convierte las respuestas vacias o sin evidencia en una
    respuesta con articulos verificables, sin depender del LLM ni de la GPU.
    """
    monkeypatch.setattr(agent, "_crear_llm", lambda _modelo: _FalsoLLM())
    resultado = agent.responder(
        "¿Qué artículos de la CP fundamentan la sentencia T-293/97?"
    )
    assert resultado["modo"] == "fallback_sin_tools"
    assert resultado["tool_calls"], "el plan determinista debe ejecutar herramientas"
    assert resultado["grounding"]["ok"] is True
    assert "T-293" in resultado["respuesta"]


def test_crear_llm_sin_tope_y_con_pensamiento() -> None:
    """La salida no tiene tope artificial y el pensamiento es opcional."""
    normal = agent._crear_llm("modelo-demo", False)
    pensando = agent._crear_llm("modelo-demo", True)
    assert normal.reasoning is False
    assert pensando.reasoning is True
    assert normal.num_predict == -1
    assert pensando.num_predict == -1


def test_ask_acepta_pensar_sin_cargar_modelo(cliente: TestClient) -> None:
    """El router acepta el flag de pensamiento y no devuelve razonamiento."""
    respuesta = cliente.post(
        "/v1/ask", json={"pregunta": "Muy buenas", "pensar": True}
    )
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["modo"] == "small_talk"
    assert datos["pensamiento"] is None


def test_texto_constitucional_por_frases() -> None:
    """El agente recibe la frase completa del articulo, no un corte ciego."""
    articulo_49 = core.texto_constitucional("49")
    assert "servicios públicos a cargo del Estado" in articulo_49
    assert "derechos fundamentales" not in articulo_49
    articulo_86 = core.texto_constitucional("86")
    assert "acción de tutela" in articulo_86
    assert len(articulo_86) > 250


def test_evidencia_incluye_texto_de_articulo() -> None:
    """La evidencia del redactor incluye el texto del artículo cuando existe."""
    evidencia = agent._evidencia_vacia()
    evidencia["articulos_detalle"]["11"] = {
        "numero": "11",
        "score": None,
        "bloque": "derechos_fundamentales",
        "texto": "El derecho a la vida es inviolable.",
    }
    lineas = agent._lineas_evidencia(evidencia)
    assert "El derecho a la vida es inviolable." in " ".join(lineas)


def test_grounding_detecta_citas_no_verificadas() -> None:
    """Una cita o un articulo ausente de la evidencia se marca como no verificado."""
    evidencia = {
        "canones": {"T-293-1997"},
        "articulos": {"13"},
        "ids": {"2195"},
        "documentos": [],
        "articulos_detalle": {},
        "citas_detalle": [],
    }
    texto = (
        "La sentencia T-293/97 aplica el artículo 13, pero también cita la "
        "T-999/99 y el artículo 99."
    )
    resultado = agent.verificar_grounding(texto, evidencia)
    assert resultado["ok"] is False
    assert resultado["citas_verificadas"] == ["13", "T-293-1997"]
    assert resultado["citas_no_verificadas"] == ["99", "T-999-1999"]


def test_grounding_respuesta_sin_citas() -> None:
    """Una respuesta sin citas no se bloquea por grounding."""
    evidencia = {
        "canones": set(),
        "articulos": set(),
        "ids": set(),
        "documentos": [],
        "articulos_detalle": {},
        "citas_detalle": [],
    }
    resultado = agent.verificar_grounding("No hay evidencia disponible.", evidencia)
    assert resultado["ok"] is True


def test_plan_determinista_usa_canon() -> None:
    """El plan del fallback resuelve el canon de la pregunta."""
    plan = agent._plan_determinista(
        "¿Qué artículos de la CP fundamentan la sentencia T-293/97?", False, 5
    )
    assert plan[0][0] == "vincular_cp"
    assert plan[0][1]["referencia"] == "T-293-1997"
    assert any(nombre == "contexto_citas" for nombre, _ in plan)


def test_ask_saludo_no_usa_llm(cliente: TestClient) -> None:
    """El saludo se responde por router, sin herramientas ni LLM."""
    respuesta = cliente.post("/v1/ask", json={"pregunta": "Hola, cómo estás"})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["modo"] == "small_talk"
    assert datos["tool_calls"] == []


def test_ask_fuera_de_alcance(cliente: TestClient) -> None:
    """Un tema ajeno devuelve el mensaje de alcance, no un error de corpus."""
    respuesta = cliente.post(
        "/v1/ask", json={"pregunta": "¿Quién ganó el mundial 2022?"}
    )
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["modo"] == "fuera_alcance"
    assert "fuera de mi alcance" in datos["respuesta"]
    assert datos["tool_calls"] == []


# ===== SECCION 5: resumen de evaluacion =====
def test_eval_summary(cliente: TestClient) -> None:
    """El resumen lee las metricas congeladas sin inventar valores."""
    respuesta = cliente.get("/v1/eval/summary")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["fuentes"]
    assert datos["metricas"]
    assert datos["procedencia"]
    assert datos["metricas"]["m7_ndcg10_bm25_hubs"] is not None


def test_eval_summary_comparativa_rutas(cliente: TestClient) -> None:
    """La comparativa de rutas trae las cuatro rutas con su procedencia."""
    respuesta = cliente.get("/v1/eval/summary")
    assert respuesta.status_code == 200
    rutas = respuesta.json()["comparativa_rutas"]
    assert len(rutas) == 4
    assert all(ruta["metrica"] == "nDCG@10" for ruta in rutas)
    valores = {ruta["ruta"]: ruta["valor"] for ruta in rutas}
    assert valores["v2.2 con gold humano (B3)"] == 0.7852
    assert valores["v2.2, retriever B2 + reranker B1"] == 0.6907
    assert valores["C3, corpus 2022-2026"] == 0.7472
    assert valores["BM25 con prior de hubs (referencia)"] == 0.6013
    assert all("metricas.json:" in ruta["procedencia"] for ruta in rutas)


def test_eval_summary_guardrails_agente(cliente: TestClient) -> None:
    """Los guardrails del agente exponen tasas, latencia p50/p95 y aviso."""
    respuesta = cliente.get("/v1/eval/summary")
    assert respuesta.status_code == 200
    guardrails = respuesta.json()["agente_guardrails"]
    assert guardrails is not None
    claves = {item["clave"] for item in guardrails["metricas"]}
    assert {
        "ruteo_exacto",
        "saludos_no_rechazados",
        "rechazo_correcto_fuera_alcance",
        "adversarios_contenidos",
        "tool_call_en_alcance",
        "grounding_ok_global",
    } <= claves
    assert all(0.0 <= item["valor"] <= 1.0 for item in guardrails["metricas"])
    latencia = {item["clave"]: item["valor"] for item in guardrails["latencia"]}
    assert latencia["p50"] == 13444.0
    assert latencia["p95"] == 60799.0
    assert "agente_guardrails_eval" in guardrails["procedencia"]
    assert guardrails["aviso"]


def test_eval_summary_avisos_procedencia(cliente: TestClient) -> None:
    """Los avisos distinguen gold humano, proxy LLM y etiquetas automaticas."""
    respuesta = cliente.get("/v1/eval/summary")
    assert respuesta.status_code == 200
    avisos = respuesta.json()["avisos_procedencia"]
    temas = {aviso["tema"] for aviso in avisos}
    assert {"Gold humano", "Gold-LLM (proxy)", "Etiquetas del producto"} <= temas
    humano = next(aviso for aviso in avisos if aviso["tema"] == "Gold humano")
    assert "820" in humano["texto"]


# ===== SECCION 5b: sentencias similares (P5) =====
def test_similar_por_id(cliente: TestClient) -> None:
    """El id 2195 devuelve cinco vecinas del corpus sin incluirse a si mismo."""
    respuesta = cliente.get("/v1/similar/2195", params={"k": 5})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["id"] == "2195"
    assert datos["canon"] == "T-293-1997"
    assert len(datos["similares"]) == 5
    assert all(fila["id"] != "2195" for fila in datos["similares"])
    assert all(fila["canon"] and fila["tipo"] for fila in datos["similares"])
    assert all(fila["score"] > 0 for fila in datos["similares"])
    scores = [fila["score"] for fila in datos["similares"]]
    assert scores == sorted(scores, reverse=True)
    assert datos["aviso"]
    assert datos["ms"] >= 0


def test_similar_id_inexistente(cliente: TestClient) -> None:
    """Un id fuera del corpus devuelve 404 con mensaje explicito."""
    respuesta = cliente.get("/v1/similar/999999", params={"k": 5})
    assert respuesta.status_code == 404
    assert "no existe" in respuesta.json()["detail"].lower()


def test_similar_k_fuera_de_rango(cliente: TestClient) -> None:
    """k=0 es una entrada invalida para el contrato (422)."""
    respuesta = cliente.get("/v1/similar/2195", params={"k": 0})
    assert respuesta.status_code == 422


# ===== SECCION 6: integracion con el LLM (opt-in) =====
def _vram_libre_gb() -> float | None:
    """Devuelve la VRAM libre en GB segun rocm-smi, o None si no hay GPU."""
    try:
        salida = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    total = usado = None
    for linea in salida.splitlines():
        if "VRAM Total Memory (B)" in linea:
            total = int(linea.rsplit(":", 1)[1].strip())
        elif "VRAM Total Used Memory (B)" in linea:
            usado = int(linea.rsplit(":", 1)[1].strip())
    if total is None or usado is None:
        return None
    return (total - usado) / 1e9


def _motivo_skip_integracion() -> str | None:
    """Explica por que la prueba de integracion no puede correr."""
    if os.getenv("JURIS_LLM_TESTS") != "1":
        return "defina JURIS_LLM_TESTS=1 para ejecutar la prueba con LLM"
    if not agent.ollama_disponible():
        return "ollama no responde en http://127.0.0.1:11434"
    libre = _vram_libre_gb()
    if libre is None:
        return "rocm-smi no disponible para verificar la VRAM"
    if libre < 14.0:
        return f"VRAM libre insuficiente: {libre:.1f} GB (se requieren 14 GB)"
    return None


MOTIVO_SKIP = _motivo_skip_integracion()


@pytest.mark.integration
@pytest.mark.skipif(MOTIVO_SKIP is not None, reason=str(MOTIVO_SKIP))
def test_ask_en_alcance_llama_herramientas() -> None:
    """El agente llama herramientas y cita evidencia verificada."""
    resultado = agent.responder(
        "¿Qué artículos de la CP fundamentan la sentencia T-293/97?"
    )
    assert resultado["modo"] in {"agente", "fallback_sin_tools"}
    assert resultado["tool_calls"], "el agente debe registrar tool calls"
    assert resultado["grounding"]["ok"] is True
    assert "T-293" in resultado["respuesta"]
