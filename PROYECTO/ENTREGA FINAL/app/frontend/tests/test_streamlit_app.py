"""test_streamlit_app.py
Fecha: 2026-10-31
Descripción: Pruebas de interfaz con streamlit.testing.v1.AppTest. Renderizan
    la app real y ejecutan el flujo P1 con el id 2195 contra el backend en
    API_URL, además de las pantallas P2, P3 y el router del chat (saludo y
    fuera de alcance, que no invocan el LLM). Se saltan si el backend no
    responde.
Estado del arte aplicado: pruebas de extremo a extremo en proceso con
    AppTest, sin navegador y sin cargar el modelo.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8123")
TIMEOUT = 180.0


def backend_ready() -> bool:
    """Comprueba si el backend está listo para las pruebas de integración.

    Returns:
        True si /health responde 200 con ready=True.
    """
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        return response.status_code == 200 and bool(response.json().get("ready"))
    except (httpx.HTTPError, ValueError):
        return False


@pytest.fixture()
def app() -> AppTest:
    """Crea y renderiza la app; se salta si el backend no está disponible."""
    if not backend_ready():
        pytest.skip(f"Backend no disponible en {API_URL}; se omite AppTest.")
    app = AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT)
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    return app


def markdown_text(app: AppTest) -> str:
    """Concatena el markdown renderizado por la app.

    Args:
        app: Aplicación bajo prueba.

    Returns:
        Todo el markdown y las leyendas como una cadena.
    """
    partes = [str(item.value) for item in app.markdown]
    partes += [str(item.value) for item in app.caption]
    return "\n".join(partes)


def test_p1_texto_vacio_avisa_sin_peticion(app: AppTest) -> None:
    """Con la pestaña de texto vacía se avisa y no se envía una petición vacía."""
    app.radio(key="p1_mode").set_value("Texto de la providencia")
    app.run()
    app.button(key="p1_run").click()
    app.run(timeout=TIMEOUT)

    assert not app.exception, [item.value for item in app.exception]
    assert any("pegue el texto" in str(item.value).lower() for item in app.warning)
    assert not app.error


def test_p1_flow_2195(app: AppTest) -> None:
    """El flujo P1 con el id 2195 muestra el canon y la evidencia textual."""
    app.text_input(key="p1_id").set_value("2195")
    app.run()

    app.button(key="p1_run").click()
    app.run(timeout=TIMEOUT)

    assert not app.exception, [item.value for item in app.exception]
    assert not app.error
    content = markdown_text(app)
    assert "T-293-1997" in content
    assert any("Artículos recuperados" in item.value for item in app.subheader)
    assert any("Área temática" in item.value for item in app.subheader)
    assert any(
        "Evidencia textual (fragmentos que activan el artículo)" in item.value
        for item in app.subheader
    )
    assert "Artículo 86" in content
    assert "149.1889" in content
    assert any(
        button.label == "Descargar ficha PDF" for button in app.download_button
    )
    assert any("Sentencias similares" in item.value for item in app.subheader)
    assert "Similitud" in content


def test_p2_article_86_shows_counts_and_filter(app: AppTest) -> None:
    """El artículo 86 devuelve el total real y el top filtrable por tipo."""
    app.radio(key="screen").set_value("Artículo a jurisprudencia")
    app.run()

    app.button(key="p2_run").click()
    app.run(timeout=TIMEOUT)

    assert not app.exception, [item.value for item in app.exception]
    assert not app.error
    content = markdown_text(app)
    assert "16405" in content
    assert "C-543-1992" in content
    assert "referencia automática" in content
    assert "PageRank" in content

    app.radio(key="p2_tipo").set_value("SU")
    app.run(timeout=TIMEOUT)

    assert not app.exception, [item.value for item in app.exception]
    content = markdown_text(app)
    assert "SU-480-1997" in content
    assert "C-543-1992" not in content


def test_p2_unknown_article_reports_404(app: AppTest) -> None:
    """Un artículo inexistente muestra el aviso de la API, sin datos."""
    app.radio(key="screen").set_value("Artículo a jurisprudencia")
    app.run()

    app.text_input(key="p2_numero").set_value("999")
    app.run()
    app.button(key="p2_run").click()
    app.run(timeout=TIMEOUT)

    assert not app.exception
    assert any("no existe" in item.value for item in app.warning)


def test_p3_shows_real_metrics_with_provenance(app: AppTest) -> None:
    """La pantalla de evaluación muestra cifras reales y su procedencia."""
    app.radio(key="screen").set_value("Evaluación")
    app.run()

    assert not app.exception
    content = markdown_text(app)
    assert "0.7852" in content
    assert "SEGUNDA ENTREGA/resultados/metricas.json" in content
    assert "ADR-19" in content
    assert any(
        "Comparación de rutas de recuperación" in item.value
        for item in app.subheader
    )
    assert any("Guardrails del agente" in item.value for item in app.subheader)
    assert "0.6013" in content
    assert "0.6907" in content
    assert "13444 ms" in content
    assert "agente_guardrails_eval" in content
    assert "Gold humano" in content


def test_chat_small_talk_uses_router(app: AppTest) -> None:
    """El saludo se responde por router, sin cargar el modelo."""
    app.radio(key="screen").set_value("Consulta asistida")
    app.run()

    app.text_area(key="chat_question").set_value("Hola")
    app.run()
    app.button(key="chat_ask").click()
    app.run(timeout=TIMEOUT)

    assert not app.exception
    content = markdown_text(app)
    assert "Saludo" in content
    assert "asistente de Juris-PLN" in content


def test_chat_out_of_scope_uses_router(app: AppTest) -> None:
    """Una pregunta ajena al dominio se responde por router, sin LLM."""
    app.radio(key="screen").set_value("Consulta asistida")
    app.run()

    app.text_area(key="chat_question").set_value("¿Cuál es la capital de Francia?")
    app.run()
    app.button(key="chat_ask").click()
    app.run(timeout=TIMEOUT)

    assert not app.exception
    content = markdown_text(app)
    assert "Fuera de alcance" in content
