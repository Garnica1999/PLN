"""streamlit_app.py
Fecha: 2026-10-31
Descripción: Frontend Streamlit de Juris-PLN (Entrega Final). Habla con la
    API FastAPI solo por HTTP (api_client) y presenta cuatro pantallas:
    sentencia a Constitución con evidencia textual por artículo, artículo a
    jurisprudencia, evaluación con procedencia y consulta asistida. No
    importa el núcleo ni los modelos del backend.
Estado del arte aplicado: separación frontend/backend por contrato HTTP,
    estados vacío, carga y error explícitos, procedencia visible en cada
    cifra, paleta y tipografía de DESIGN.md, contraste AA y controles con
    comportamiento real.
"""

from __future__ import annotations

import html
import json
import re
import sys
import traceback
from typing import Any

import streamlit as st

from api_client import ApiClient, ApiError
from pdf_ficha import construir_ficha, nombre_ficha

CONFIG: dict[str, Any] = {
    "title": "Juris-PLN",
    "subtitle": "Investigación asistida sobre jurisprudencia constitucional",
    "default_model": "qwen3.8:diario-8k-q3_k_s",
    "screens": [
        "Sentencia a Constitución",
        "Artículo a jurisprudencia",
        "Evaluación",
        "Consulta asistida",
    ],
    "k_articles": 10,
    "evidence_articles": 3,
    "evidence_max": 5,
    "jurisprudencia_top": 10,
    "default_article": "86",
    "similares_k": 5,
}

METRIC_LABELS: dict[str, str] = {
    "m0_accuracy": "M0, exactitud de validación",
    "m4_macro_f1_test": "M4, macro-F1 en test (diagnóstico)",
    "m4_accuracy_test": "M4, exactitud en test (diagnóstico)",
    "m7_ndcg10_bm25_hubs": "M7, nDCG@10 con BM25 y hubs",
    "m7_ndcg10_rerank_v2_2": "M7, nDCG@10 con rerank v2.2",
    "m7_ndcg10_gold_humano": "M7, nDCG@10 con gold humano",
    "c3_ndcg10_temporal": "C3, nDCG@10 en deriva temporal",
}

RESPONSE_MODES: dict[str, str] = {
    "agente": "Agente con herramientas",
    "fallback_sin_tools": "Fallback sin tool calls",
    "sin_evidencia": "Sin evidencia suficiente",
    "small_talk": "Saludo",
    "fuera_alcance": "Fuera de alcance",
    "aclaracion": "Aclaración",
}

CSS = """
<style>
:root {
    --azul: #2F75B5;
    --azul-oscuro: #28649B;
    --tinta: #162536;
    --gris: #425466;
    --fondo: #FFFFFF;
    --superficie: #F5F7FA;
    --borde: #DCE3EB;
    --borde-suave: #E7ECF2;
}
html, body, [data-testid="stAppViewContainer"] {
    font-family: "Source Sans 3", "Source Sans Pro", "Segoe UI", system-ui, sans-serif;
    color: var(--tinta);
    font-size: 16px;
}
[data-testid="stAppViewContainer"] { background: var(--fondo); }
[data-testid="stHeader"] { background: var(--fondo); }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    max-width: 1080px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}
h1, h2, h3, .jp-titulo, .jp-subtitulo {
    font-family: "Source Serif 4", "Source Serif Pro", Georgia, serif;
    color: var(--tinta);
    font-weight: 600;
}
div[data-testid="stHeadingWithActionElements"] h1,
div[data-testid="stHeadingWithActionElements"] h2,
div[data-testid="stHeadingWithActionElements"] h3,
h1.jp-titulo, h2.jp-titulo, h3.jp-titulo {
    font-family: "Source Serif 4", "Source Serif Pro", Georgia, serif;
    color: var(--tinta);
    font-weight: 600;
}
div[data-testid="stHeadingWithActionElements"] h1.jp-titulo {
    font-size: 2rem;
    line-height: 1.2;
    margin: 0 0 .35rem 0;
}
div[data-testid="stHeadingWithActionElements"] h3 {
    font-size: 1.35rem;
    line-height: 1.3;
    margin-top: 1.2rem;
}
.jp-titulo { font-size: 2rem; line-height: 1.2; margin: 0 0 .35rem 0; }
.jp-lede {
    color: var(--gris);
    font-size: 1.02rem;
    line-height: 1.55;
    max-width: 68ch;
    margin: 0 0 1.8rem 0;
}
.jp-marca {
    font-family: "Source Serif 4", Georgia, serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--tinta);
    padding-top: .4rem;
}
.jp-regla { border-top: 1px solid var(--borde); margin: 1.6rem 0; }
.jp-card {
    background: var(--fondo);
    border: 1px solid var(--borde);
    border-radius: 3px;
    padding: 1rem 1.1rem;
    margin: .4rem 0 1rem 0;
}
.jp-aviso {
    background: var(--superficie);
    border: 1px solid var(--borde);
    border-radius: 3px;
    padding: .85rem 1rem;
    color: var(--tinta);
    font-size: .94rem;
    line-height: 1.5;
    margin: .4rem 0 1rem 0;
}
.jp-aviso p { margin: 0; }
.jp-etiqueta {
    color: var(--gris);
    font-size: .82rem;
    line-height: 1.4;
    margin-bottom: .25rem;
}
.jp-dato {
    font-family: "Source Code Pro", "JetBrains Mono", ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    color: var(--tinta);
}
.jp-meta-fila {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: .35rem 0;
    border-bottom: 1px solid var(--borde-suave);
    font-size: .94rem;
}
.jp-meta-fila:last-child { border-bottom: none; }
.jp-meta-clave { color: var(--gris); }
.jp-tabla-envoltura {
    overflow-x: auto;
    border: 1px solid var(--borde);
    border-radius: 3px;
    margin: .3rem 0 1.2rem 0;
}
.jp-tabla {
    width: 100%;
    border-collapse: collapse;
    font-size: .92rem;
    background: var(--fondo);
}
.jp-tabla th {
    text-align: left;
    font-weight: 600;
    color: var(--tinta);
    background: var(--superficie);
    padding: .5rem .65rem;
    border-bottom: 1px solid var(--borde);
    white-space: nowrap;
}
.jp-tabla td {
    padding: .5rem .65rem;
    border-bottom: 1px solid var(--borde-suave);
    color: var(--tinta);
    vertical-align: top;
    line-height: 1.45;
}
.jp-tabla tr:last-child td { border-bottom: none; }
.jp-tabla .jp-dato { white-space: nowrap; }
.jp-ruta {
    font-family: "Source Code Pro", ui-monospace, monospace;
    font-size: .78rem;
    color: var(--gris);
    word-break: break-word;
}
.jp-vacio {
    border: 1px solid var(--borde);
    border-radius: 3px;
    padding: 1.2rem 1.1rem;
    background: var(--superficie);
}
.jp-vacio-titulo {
    font-family: "Source Serif 4", Georgia, serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--tinta);
    margin: 0 0 .35rem 0;
}
.jp-vacio p { margin: 0; color: var(--gris); line-height: 1.5; }
.jp-frag-texto {
    margin: 0;
    color: var(--tinta);
    font-size: .93rem;
    line-height: 1.55;
}
.jp-figura {
    border: 1px solid var(--borde);
    border-radius: 3px;
    background: var(--fondo);
    padding: .85rem 1rem .5rem 1rem;
    margin: .3rem 0 1.2rem 0;
}
.jp-figura .jp-etiqueta { margin-bottom: .5rem; }
.jp-barras {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 130px;
    border-bottom: 1px solid var(--borde);
}
.jp-barra {
    flex: 1 1 0;
    background: var(--azul);
    min-height: 1px;
}
.jp-ejes {
    display: flex;
    gap: 2px;
    padding-top: .2rem;
}
.jp-eje {
    flex: 1 1 0;
    text-align: center;
    font-size: .68rem;
    color: var(--gris);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

[data-testid="stSidebar"] {
    background: var(--superficie);
    border-right: 1px solid var(--borde);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { font-size: 1.05rem; }

.stButton > button, [data-testid="stBaseButton-secondary"] {
    border-radius: 3px;
    border: 1px solid #C9D3DE;
    background: var(--fondo);
    color: var(--tinta);
    font-family: inherit;
    min-height: 2.5rem;
}
.stButton > button:hover, [data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--azul);
    color: var(--tinta);
}
[data-testid="stBaseButton-primary"] {
    background: var(--azul);
    border: 1px solid var(--azul);
    border-radius: 3px;
    color: #FFFFFF;
    min-height: 2.5rem;
}
[data-testid="stBaseButton-primary"]:hover {
    background: var(--azul-oscuro);
    border-color: var(--azul-oscuro);
}
[data-testid="stBaseButton-primary"] p { color: #FFFFFF; }
[data-testid="stDownloadButton"] button {
    border-radius: 3px;
    border: 1px solid #C9D3DE;
    background: var(--fondo);
    color: var(--tinta);
    min-height: 2.5rem;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: var(--azul);
    color: var(--tinta);
}
button:focus-visible, a:focus-visible, input:focus-visible,
textarea:focus-visible, select:focus-visible, [role="radio"]:focus-visible {
    outline: 2px solid var(--azul);
    outline-offset: 2px;
}
a { color: var(--azul); }
[data-testid="stAlert"] {
    background: var(--superficie);
    border: 1px solid #C9D3DE;
    border-radius: 3px;
    color: var(--tinta);
}
div[data-testid="stAlertContainer"] {
    background: var(--superficie);
    color: var(--tinta);
}
[data-testid="stAlert"] p,
div[data-testid="stAlertContainer"] p { color: var(--tinta); }
[data-testid="stAlert"] svg,
div[data-testid="stAlertContainer"] svg { color: var(--gris); }
[data-testid="stExpander"] details {
    border: 1px solid var(--borde);
    border-radius: 3px;
    background: var(--fondo);
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    border-radius: 3px;
    border: 1px solid #C9D3DE;
    color: var(--tinta);
    background: var(--fondo);
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder { color: #5D6D7E; }
[data-testid="stCaptionContainer"] p { color: var(--gris); }
[data-testid="stSpinner"] p { color: var(--gris); }
</style>
"""


# ===== SECCIÓN 1: acceso a la API =====
@st.cache_resource(show_spinner=False)
def get_client() -> ApiClient:
    """Crea un único cliente HTTP reutilizable durante la sesión.

    Returns:
        Cliente configurado con API_URL o el valor por defecto.
    """
    return ApiClient()


@st.cache_data(ttl=30, show_spinner=False)
def read_health(_client: ApiClient, base_url: str) -> dict[str, Any]:
    """Consulta /health con una caché corta.

    Args:
        _client: Cliente HTTP; el guion bajo lo excluye del hash de caché.
        base_url: URL base para separar la caché si cambia el destino.

    Returns:
        Respuesta de /health.
    """
    return _client.health()


@st.cache_data(ttl=60, show_spinner=False)
def read_models(_client: ApiClient, base_url: str) -> dict[str, Any]:
    """Consulta /v1/models con una caché de un minuto.

    Args:
        _client: Cliente HTTP; el guion bajo lo excluye del hash de caché.
        base_url: URL base para separar la caché si cambia el destino.

    Returns:
        Respuesta de /v1/models.
    """
    return _client.models()


# ===== SECCIÓN 2: componentes de presentación =====
def render_header(title: str, lede: str) -> None:
    """Dibuja el encabezado serif de una pantalla.

    Args:
        title: Título principal.
        lede: Texto de apoyo bajo el título.
    """
    st.markdown(
        f'<h1 class="jp-titulo">{html.escape(title)}</h1>'
        f'<p class="jp-lede">{html.escape(lede)}</p>',
        unsafe_allow_html=True,
    )


def render_divider() -> None:
    """Dibuja una línea divisoria de la paleta."""
    st.markdown('<div class="jp-regla"></div>', unsafe_allow_html=True)


def render_meta_card(pairs: list[tuple[str, str]]) -> None:
    """Dibuja una tarjeta de pares clave y valor.

    Args:
        pairs: Etiquetas y valores ya formateados.
    """
    rows = "".join(
        '<div class="jp-meta-fila">'
        f'<span class="jp-meta-clave">{html.escape(label)}</span>'
        f'<span class="jp-dato">{html.escape(value)}</span>'
        "</div>"
        for label, value in pairs
    )
    st.markdown(f'<div class="jp-card">{rows}</div>', unsafe_allow_html=True)


def render_notice(text: str) -> None:
    """Dibuja un aviso informativo con la paleta del informe.

    Args:
        text: Contenido del aviso.
    """
    st.markdown(
        f'<div class="jp-aviso"><p>{html.escape(text)}</p></div>',
        unsafe_allow_html=True,
    )


def render_table(
    columns: list[str],
    rows: list[list[str]],
    *,
    numeric: set[int] | None = None,
) -> None:
    """Dibuja una tabla HTML con la paleta y la tipografía del informe.

    Args:
        columns: Cabeceras de la tabla.
        rows: Celdas ya formateadas.
        numeric: Índices de columnas que usan monoespaciada tabular.
    """
    numeric = numeric or set()
    header = "".join(
        f'<th scope="col">{html.escape(column)}</th>' for column in columns
    )
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            css_class = ' class="jp-dato"' if index in numeric else ""
            cells.append(f"<td{css_class}>{html.escape(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        '<div class="jp-tabla-envoltura"><table class="jp-tabla">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def as_text(value: Any, default: str = "No disponible") -> str:
    """Convierte un valor de la API en texto seguro para mostrar.

    Args:
        value: Valor recibido.
        default: Texto cuando el valor es None o vacío.

    Returns:
        Cadena legible.
    """
    if value is None or value == "":
        return default
    return str(value)


def format_score(value: Any) -> str:
    """Formatea un puntaje con cuatro decimales.

    Args:
        value: Número recibido o None.

    Returns:
        Puntaje formateado o "No disponible".
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4f}"
    return "No disponible"


def format_milisegundos(value: Any) -> str:
    """Formatea una latencia en milisegundos sin decimales.

    Args:
        value: Número recibido o None.

    Returns:
        Latencia formateada o "No disponible".
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.0f} ms"
    return "No disponible"


def format_block(value: Any) -> str:
    """Humaniza el nombre de un bloque temático.

    Args:
        value: Identificador del bloque.

    Returns:
        Etiqueta con espacios y mayúscula inicial.
    """
    if not value:
        return "Sin bloque"
    return str(value).replace("_", " ").capitalize()


def area_items(area: Any) -> list[str]:
    """Normaliza la forma del área temática sin inventar valores.

    Args:
        area: Valor recibido del backend: lista, diccionario o escalar.

    Returns:
        Hasta tres etiquetas legibles.
    """
    if isinstance(area, dict):
        top3 = area.get("top3")
        if isinstance(top3, list):
            labels = []
            for item in top3[:3]:
                if isinstance(item, dict):
                    nombre = as_text(
                        item.get("area")
                        or item.get("bloque")
                        or item.get("nombre")
                        or item.get("etiqueta"),
                        "Sin etiqueta",
                    )
                    prob = item.get("prob")
                    if not isinstance(prob, (int, float)):
                        prob = item.get("score")
                    labels.append(
                        f"{nombre}: {format_score(prob)}"
                        if isinstance(prob, (int, float))
                        else nombre
                    )
                else:
                    labels.append(str(item))
            return labels
        labels = []
        for key, value in list(area.items())[:3]:
            score = format_score(value) if isinstance(value, (int, float)) else value
            labels.append(f"{key}: {score}")
        return labels
    if isinstance(area, list):
        labels = []
        for item in area[:3]:
            if isinstance(item, dict):
                name = (
                    item.get("bloque")
                    or item.get("nombre")
                    or item.get("etiqueta")
                    or item.get("area")
                )
                score = item.get("score") or item.get("probabilidad")
                label = as_text(name, "Sin etiqueta")
                if isinstance(score, (int, float)):
                    label = f"{label}: {format_score(score)}"
                labels.append(label)
            else:
                labels.append(str(item))
        return labels
    return [str(area)]


def render_sidebar(client: ApiClient) -> str:
    """Dibuja la navegación, el selector de modelo y el estado de la API.

    Args:
        client: Cliente HTTP de la sesión.

    Returns:
        Nombre del modelo seleccionado.
    """
    with st.sidebar:
        st.markdown('<div class="jp-marca">Juris-PLN</div>', unsafe_allow_html=True)
        st.caption("Corte Constitucional de Colombia, 1992-2021")
        st.radio("Pantallas", CONFIG["screens"], key="screen")
        st.divider()

        models: dict[str, Any] | None = None
        try:
            models = read_models(client, client.base_url)
        except ApiError as exc:
            st.warning(f"Aviso: {exc}")

        names = [
            str(item.get("nombre"))
            for item in (models or {}).get("modelos", [])
            if item.get("nombre")
        ]
        default_model = (models or {}).get("defecto") or CONFIG["default_model"]
        if default_model not in names:
            names = [default_model, *names]
        index = names.index(default_model) if default_model in names else 0

        model = st.selectbox(
            "Modelo del asistente",
            names,
            index=index,
            key="llm_model",
            help="Modelo de ollama que usará /v1/ask.",
        )
        if models is None:
            st.caption(f"Catálogo no disponible. Se usará {default_model}.")
        info = next(
            (
                item
                for item in (models or {}).get("modelos", [])
                if item.get("nombre") == model
            ),
            None,
        )
        if info is not None and not info.get("tool_calling", False):
            st.warning(
                "Aviso: este modelo no declara tool calling. El backend usará "
                "el fallback determinista y lo indicará en la respuesta."
            )

        st.divider()
        try:
            health = read_health(client, client.base_url)
            if health.get("ready"):
                st.caption(f"API en {client.base_url}. Componentes cargados.")
            else:
                st.warning(
                    "Aviso: la API responde, pero sus componentes no están "
                    "cargados. La primera consulta puede fallar."
                )
            if not health.get("ollama", False):
                st.caption(
                    "ollama no está disponible: las pantallas que usan el "
                    "modelo no podrán responder."
                )
        except ApiError as exc:
            st.warning(f"Aviso: {exc}")
    return model


# ===== SECCIÓN 3: P1, sentencia a Constitución =====
def render_p1_empty() -> None:
    """Muestra el estado vacío de la pantalla P1."""
    st.markdown(
        '<div class="jp-vacio">'
        '<p class="jp-vacio-titulo">Sin análisis todavía</p>'
        "<p>Escriba un id del corpus o un canon y pulse Analizar sentencia, "
        "o pegue el texto de la providencia. Ejemplo del corpus: 2195, "
        "canon T-293/97.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def run_p1(
    client: ApiClient,
    *,
    sentence_id: str | None,
    text: str | None,
) -> None:
    """Ejecuta la validación M0, el vínculo con la Constitución y la evidencia.

    Args:
        client: Cliente HTTP.
        sentence_id: Id o canon, si la entrada es por referencia.
        text: Texto de la providencia, si la entrada es literal.
    """
    with st.status("Validando la providencia y recuperando artículos...") as status:
        try:
            validation = client.validate(text=text, sentence_id=sentence_id)
            link = None
            if validation.get("es_sentencia"):
                link = client.link(
                    text=text,
                    sentence_id=sentence_id,
                    k=CONFIG["k_articles"],
                )
        except ApiError as exc:
            status.update(label="El análisis no se completó", state="error")
            st.error(f"Error: {exc}")
            st.caption(
                "Compruebe que el backend responda en /health y que el id "
                "exista en el corpus."
            )
            return
        targets = evidence_targets(link) if link else []
        evidence = None
        evidence_error = None
        if targets:
            status.update(label="Recuperando la evidencia textual...")
            evidence, evidence_error = fetch_evidence(
                client,
                sentence_id=sentence_id,
                text=text,
                numbers=targets,
            )
        similares_data, similares_error = (None, None)
        if link and link.get("id"):
            status.update(label="Buscando sentencias similares...")
            similares_data, similares_error = fetch_similares(
                client, str(link["id"])
            )
        status.update(label="Análisis completado", state="complete")
    st.session_state["p1_result"] = {
        "validation": validation,
        "link": link,
        "evidence": evidence,
        "evidence_error": evidence_error,
        "evidence_targets": targets,
        "similares": similares_data,
        "similares_error": similares_error,
    }


def render_area(link: dict[str, Any]) -> None:
    """Dibuja el panel de área temática con su aviso de ADR-19.

    Args:
        link: Respuesta de /v1/link.
    """
    st.subheader("Área temática (exploratoria)")
    render_notice(
        "Exploratoria por ADR-19: el clasificador se entrenó con etiquetas "
        "silver que la validación humana no confirmó como gold. Orienta la "
        "lectura de la providencia, no la clasifica."
    )
    area = link.get("area")
    if not area:
        st.info(
            "La API en uso no devuelve el área temática. El panel quedará "
            "vacío hasta que /v1/link exponga ese campo."
        )
        return
    labels = area_items(area)
    render_meta_card(
        [(f"Opción {index}", label) for index, label in enumerate(labels, start=1)]
    )


def render_citations(link: dict[str, Any]) -> None:
    """Dibuja las sentencias citadas si el contrato las incluye.

    Args:
        link: Respuesta de /v1/link.
    """
    st.subheader("Sentencias citadas")
    citations = link.get("sentencias_citadas")
    if not citations:
        st.info(
            "Pendiente de API. El contrato actual de /v1/link no devuelve las "
            "sentencias citadas con id, año y PageRank; no se muestran datos "
            "para no inventarlos."
        )
        return
    if not isinstance(citations, list):
        st.json(citations)
        return
    rows = []
    for citation in citations:
        if not isinstance(citation, dict):
            rows.append([str(citation), "", "", ""])
            continue
        rows.append(
            [
                as_text(citation.get("canon")),
                as_text(citation.get("id")),
                as_text(citation.get("ano")),
                format_score(citation.get("pagerank")),
            ]
        )
    render_table(
        ["Canon", "Id", "Año", "PageRank"],
        rows,
        numeric={1, 2, 3},
    )


def render_articles(link: dict[str, Any]) -> None:
    """Dibuja las tablas de artículos extraídos y recuperados.

    Args:
        link: Respuesta de /v1/link.
    """
    extracted = link.get("articulos_extraidos") or []
    retrieved = link.get("articulos_recuperados") or []

    st.subheader("Artículos extraídos del texto")
    st.caption("Menciones explícitas detectadas en la providencia.")
    if not extracted:
        st.info("No se extrajeron artículos de la entrada.")
    else:
        render_table(
            ["Artículo", "Bloque", "Texto (150 caracteres)"],
            [
                [
                    as_text(item.get("numero")),
                    format_block(item.get("bloque")),
                    as_text(item.get("texto_150chars"), "Sin texto"),
                ]
                for item in extracted
            ],
            numeric={0},
        )

    st.subheader("Artículos recuperados")
    st.caption(
        "Ranking BM25 con prior de hubs, ejecutado sobre el texto de la "
        "providencia (ruta por CPU, sin rerank)."
    )
    if not retrieved:
        st.info("No se recuperaron artículos para esta entrada.")
    else:
        render_table(
            ["Artículo", "Score", "Bloque", "Texto (150 caracteres)"],
            [
                [
                    as_text(item.get("numero")),
                    format_score(item.get("score")),
                    format_block(item.get("bloque")),
                    as_text(item.get("texto_150chars"), "Sin texto"),
                ]
                for item in retrieved
            ],
            numeric={0, 1},
        )


def evidence_targets(link: dict[str, Any]) -> list[str]:
    """Selecciona los artículos que se evidencian tras el vínculo.

    Toma los tres primeros artículos extraídos y los tres primeros
    recuperados, sin duplicados, hasta el máximo de cinco que acepta
    /v1/evidencia.

    Args:
        link: Respuesta de /v1/link.

    Returns:
        Números de artículo en orden de aparición.
    """
    numbers: list[str] = []
    for key in ("articulos_extraidos", "articulos_recuperados"):
        for item in (link.get(key) or [])[: CONFIG["evidence_articles"]]:
            if not isinstance(item, dict):
                continue
            number = str(item.get("numero") or "").strip()
            if number and number not in numbers:
                numbers.append(number)
            if len(numbers) >= CONFIG["evidence_max"]:
                return numbers
    return numbers


def fetch_evidence(
    client: ApiClient,
    *,
    sentence_id: str | None,
    text: str | None,
    numbers: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Recupera los fragmentos de la sentencia para los artículos dados.

    Args:
        client: Cliente HTTP.
        sentence_id: Id o canon de la sentencia analizada.
        text: Texto de la sentencia, alternativo al id.
        numbers: Artículos a evidenciar.

    Returns:
        Tupla (respuesta, error): la respuesta de /v1/evidencia y None, o
        None y el mensaje accionable si la petición falló.
    """
    try:
        evidence = client.evidencia(
            sentence_id=sentence_id,
            text=text,
            articles=numbers,
        )
    except ApiError as exc:
        return None, str(exc)
    return evidence, None


def fetch_similares(
    client: ApiClient, sentence_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Busca las sentencias parecidas de una sentencia del corpus.

    Args:
        client: Cliente HTTP.
        sentence_id: Id del corpus de la sentencia.

    Returns:
        Tupla (respuesta, error): la respuesta de /v1/similar/{id} y None, o
        None y el mensaje accionable si la petición falló.
    """
    try:
        similares = client.similares(sentence_id, k=CONFIG["similares_k"])
    except ApiError as exc:
        return None, str(exc)
    return similares, None


def render_fragments(fragments: list[Any]) -> None:
    """Dibuja los fragmentos de un artículo con su score.

    Args:
        fragments: Lista de fragmentos {texto, score} de la API.
    """
    blocks = []
    for index, fragment in enumerate(fragments[:3], start=1):
        if not isinstance(fragment, dict):
            continue
        score = format_score(fragment.get("score"))
        body = as_text(fragment.get("texto"), "Fragmento sin texto.")
        blocks.append(
            '<div class="jp-card">'
            '<p class="jp-etiqueta">'
            f"Fragmento {index} · score {html.escape(score)}</p>"
            f'<p class="jp-frag-texto">{html.escape(body)}</p>'
            "</div>"
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_evidence(result: dict[str, Any]) -> None:
    """Dibuja la evidencia textual de los artículos del vínculo.

    Args:
        result: Estado guardado por run_p1.
    """
    st.subheader("Evidencia textual (fragmentos que activan el artículo)")
    st.caption(
        "Fragmentos de la propia sentencia con mayor score BM25 hacia cada "
        "artículo; el score mide la recuperación, no una probabilidad."
    )
    targets = result.get("evidence_targets") or []
    if not targets:
        st.info("El vínculo no devolvió artículos que evidenciar.")
        return

    error = result.get("evidence_error")
    if error:
        st.error(f"Error: {error}")
        st.caption(
            "El vínculo con los artículos sí está disponible; reintente la "
            "evidencia cuando la API responda."
        )
    evidence = result.get("evidence") or {}
    by_number = {
        str(item.get("numero")): item.get("fragmentos") or []
        for item in evidence.get("articulos") or []
        if isinstance(item, dict)
    }
    for number in targets:
        st.markdown(f"**Artículo {number}**")
        if error:
            st.caption("La evidencia de este artículo no se pudo cargar.")
            continue
        fragments = by_number.get(str(number)) or []
        if not fragments:
            st.caption(
                "La API no devolvió fragmentos para este artículo; no se "
                "muestra evidencia inventada."
            )
            continue
        render_fragments(fragments)


def render_similares(result: dict[str, Any]) -> None:
    """Dibuja la tabla de sentencias similares del corpus (P5).

    Args:
        result: Estado guardado por run_p1.
    """
    st.subheader("Sentencias similares")
    st.caption(
        "Similitud TF-IDF (coseno) sobre el texto limpio del corpus: mide "
        "parecido lexical, no equivalencia doctrinal ni precedente."
    )
    error = result.get("similares_error")
    if error:
        st.warning(f"Aviso: {error}")
        st.caption(
            "El vínculo y la evidencia siguen disponibles. La API puede "
            "estar construyendo el índice la primera vez; reintente."
        )
        return
    data = result.get("similares") or {}
    filas = data.get("similares") or []
    if not filas:
        st.info(
            "La API no devolvió sentencias similares para este id; no se "
            "muestran resultados inventados."
        )
        return
    rows = [
        [
            as_text(item.get("id")),
            as_text(item.get("canon")),
            as_text(item.get("ano")),
            as_text(item.get("tipo")),
            format_score(item.get("pagerank")),
            format_score(item.get("score")),
        ]
        for item in filas
        if isinstance(item, dict)
    ]
    render_table(
        ["Id", "Canon", "Año", "Tipo", "PageRank", "Similitud"],
        rows,
        numeric={0, 2, 4, 5},
    )


def render_p1_result(result: dict[str, Any]) -> None:
    """Dibuja la validación M0 y, si existe, el vínculo con la CP.

    Args:
        result: Estado guardado por run_p1.
    """
    validation = result.get("validation") or {}
    link = result.get("link")
    canon = validation.get("canon_detectado") or (link or {}).get("canon_detectado")
    is_sentence = bool(validation.get("es_sentencia"))

    pairs = [
        (
            "Validación M0",
            "Providencia reconocida" if is_sentence else "No reconocida",
        ),
        ("Score M0", format_score(validation.get("score_validacion"))),
        ("Canon detectado", as_text(canon, "No detectado")),
    ]
    if link:
        pairs.extend(
            [
                ("Origen del vínculo", as_text(link.get("origen"))),
                ("Latencia del vínculo", f"{as_text(link.get('ms'))} ms"),
            ]
        )
    render_meta_card(pairs)

    rules = validation.get("reglas") or {}
    if rules:
        rule_names = {
            "encabezado": "Encabezado",
            "marcador_corte": "Marcador de Corte",
            "marcador_ponente": "Marcador de ponente",
        }
        detail = "; ".join(
            f"{rule_names.get(key, key)}: {'cumple' if value else 'no cumple'}"
            for key, value in rules.items()
        )
        st.caption(f"Reglas M0. {detail}.")

    if not is_sentence or not link:
        st.error(
            "Error: M0 no reconoce la entrada como providencia. No se ejecutó "
            "el vínculo con la Constitución."
        )
        return

    render_divider()
    render_area(link)
    render_citations(link)
    render_divider()
    render_articles(link)
    render_divider()
    render_evidence(result)
    render_divider()
    render_similares(result)

    safe_canon = re.sub(r"[^A-Za-z0-9_-]+", "_", str(canon or "sentencia"))
    st.download_button(
        "Descargar JSON del vínculo",
        data=json.dumps(link, ensure_ascii=False, indent=2),
        file_name=f"vinculo_{safe_canon}.json",
        mime="application/json",
        key="p1_download",
    )
    try:
        ficha = construir_ficha(link)
    except ValueError as exc:
        st.warning(f"Aviso: no se pudo generar la ficha PDF: {exc}")
    else:
        st.download_button(
            "Descargar ficha PDF",
            data=ficha,
            file_name=nombre_ficha(link),
            mime="application/pdf",
            key="p1_ficha_pdf",
        )


def page_sentence(client: ApiClient) -> None:
    """Pantalla P1: sentencia a Constitución.

    Args:
        client: Cliente HTTP.
    """
    render_header(
        "Sentencia a Constitución",
        "Valide una providencia y vea los artículos de la Constitución que el "
        "sistema le atribuye, con la evidencia de recuperación.",
    )

    mode = st.radio(
        "Tipo de entrada",
        ["Id o canon", "Texto de la providencia"],
        horizontal=True,
        key="p1_mode",
    )
    sentence_id: str | None = None
    text: str | None = None
    if mode == "Id o canon":
        sentence_id = st.text_input(
            "Id del corpus o canon",
            key="p1_id",
            placeholder="2195 o T-293/97",
        )
        st.caption("Ejemplo del corpus: 2195, canon T-293/97.")
    else:
        text = st.text_area(
            "Texto de la providencia",
            key="p1_text",
            height=180,
            placeholder="Pegue el encabezado y la parte resolutiva.",
        )

    if st.button("Analizar sentencia", type="primary", key="p1_run"):
        if mode == "Id o canon" and not (sentence_id or "").strip():
            st.warning("Aviso: escriba un id del corpus o un canon.")
        elif mode == "Texto de la providencia" and not (text or "").strip():
            st.warning("Aviso: pegue el texto de la providencia.")
        else:
            run_p1(
                client,
                sentence_id=(sentence_id or "").strip() or None,
                text=(text or "").strip() or None,
            )

    result = st.session_state.get("p1_result")
    if not result:
        render_p1_empty()
        return
    render_divider()
    render_p1_result(result)


# ===== SECCIÓN 4: P2, artículo a jurisprudencia =====
def run_p2(client: ApiClient, number: str) -> None:
    """Consulta la jurisprudencia de un artículo y guarda el resultado.

    Args:
        client: Cliente HTTP.
        number: Número del artículo tal como lo escribió el usuario.
    """
    try:
        with st.spinner(f"Leyendo las sentencias que citan el artículo {number}..."):
            data = client.jurisprudencia(
                number, top=CONFIG["jurisprudencia_top"]
            )
    except ApiError as exc:
        st.session_state.pop("p2_result", None)
        if exc.status_code == 404:
            st.warning(f"Aviso: {exc}")
            st.caption(
                "El número no existe en la Constitución indexada. Use un "
                "artículo de la CP, por ejemplo 86."
            )
        else:
            st.error(f"Error: {exc}")
            st.caption(
                "Compruebe que el backend responda en /health y que existan "
                "los artefactos de la segunda entrega."
            )
        return
    st.session_state["p2_result"] = data


def render_year_chart(por_ano: dict[str, Any]) -> None:
    """Dibuja la distribución anual de sentencias como barras.

    Args:
        por_ano: Mapa año -> número de sentencias.
    """
    pairs = sorted(
        (str(ano), int(count))
        for ano, count in (por_ano or {}).items()
        if str(ano).isdigit()
        and isinstance(count, int)
        and not isinstance(count, bool)
    )
    if not pairs:
        st.info("La API no devolvió la distribución por año.")
        return
    peak = max(count for _, count in pairs)
    peak_years = ", ".join(ano for ano, count in pairs if count == peak)
    total = sum(count for _, count in pairs)
    bars = "".join(
        f'<div class="jp-barra" style="height:{max(2, round(100 * count / peak))}%"'
        f' title="{ano}: {count} sentencias"></div>'
        for ano, count in pairs
    )
    labels = "".join(
        f'<div class="jp-eje">{ano if int(ano) % 5 == 0 else ""}</div>'
        for ano, _ in pairs
    )
    summary = (
        f"Sentencias por año, de {pairs[0][0]} a {pairs[-1][0]}: máximo "
        f"{peak} en {peak_years}, total {total}."
    )
    st.markdown(
        f'<div class="jp-figura" role="img" aria-label="{html.escape(summary)}">'
        f'<p class="jp-etiqueta">Máximo {peak} sentencias en {peak_years}</p>'
        f'<div class="jp-barras">{bars}</div>'
        f'<div class="jp-ejes">{labels}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"n = {total} sentencias; corpus 1992-2021; etiquetas débiles de M2."
    )


def render_p2_result(data: dict[str, Any]) -> None:
    """Dibuja el total, la distribución anual y el top de jurisprudencia.

    Args:
        data: Respuesta de /v1/article/{numero}/jurisprudencia.
    """
    render_meta_card(
        [
            ("Artículo", as_text(data.get("numero"))),
            ("Sentencias que lo citan", as_text(data.get("total"))),
            ("Años con citas", str(len(data.get("por_ano") or {}))),
            ("Criterio de orden", "PageRank en la red de citas"),
        ]
    )
    render_notice(
        as_text(
            data.get("aviso"),
            "Los conteos provienen de las etiquetas débiles de M2 (bloque "
            "arts_cp), no de gold ni de citas verificadas.",
        )
    )

    st.subheader("Distribución por año")
    render_year_chart(data.get("por_ano") or {})

    st.subheader("Sentencias con mayor PageRank")
    top = [item for item in (data.get("top") or []) if isinstance(item, dict)]
    if not top:
        st.info("La API no devolvió sentencias en el top para este artículo.")
        return
    kind = st.radio(
        "Tipo de sentencia",
        ["Todas", "T", "C", "SU"],
        horizontal=True,
        key="p2_tipo",
    )
    filtered = [
        item
        for item in top
        if kind == "Todas" or str(item.get("tipo")) == kind
    ]
    st.caption(
        f"Top {len(filtered)} de {len(top)} sentencias por PageRank; el filtro "
        "se aplica en el cliente sobre este top."
    )
    if not filtered:
        st.info(f"Ninguna sentencia del top es de tipo {kind}.")
        return
    render_table(
        [
            "Id",
            "Canon",
            "Año",
            "Tipo",
            "PageRank",
            "Área silver (referencia automática)",
        ],
        [
            [
                as_text(item.get("id")),
                as_text(item.get("canon")),
                as_text(item.get("ano")),
                as_text(item.get("tipo")),
                format_score(item.get("pagerank")),
                as_text(item.get("area_silver"), "Sin etiqueta"),
            ]
            for item in filtered
        ],
        numeric={0, 2, 4},
    )


def page_article(client: ApiClient) -> None:
    """Pantalla P2: artículo de la CP a jurisprudencia.

    Args:
        client: Cliente HTTP.
    """
    render_header(
        "Artículo a jurisprudencia",
        "Sentencias del corpus que citan un artículo de la Constitución, "
        "ordenadas por PageRank y con la etiqueta automática del área.",
    )
    number = st.text_input(
        "Número del artículo",
        value=CONFIG["default_article"],
        key="p2_numero",
        help="Número de la Constitución de 1991, por ejemplo 86.",
    )
    st.caption(
        "El total cuenta sentencias del corpus 1992-2021 cuyo bloque arts_cp "
        "(M2) incluye el artículo: etiquetas débiles, no gold ni citas "
        "verificadas."
    )
    if st.button("Consultar jurisprudencia", type="primary", key="p2_run"):
        clean = (number or "").strip()
        if not clean:
            st.warning("Aviso: escriba un número de artículo de la CP.")
        else:
            run_p2(client, clean)

    data = st.session_state.get("p2_result")
    if not data:
        st.markdown(
            '<div class="jp-vacio">'
            '<p class="jp-vacio-titulo">Sin consulta todavía</p>'
            "<p>Escriba un número de artículo de la Constitución y pulse "
            "Consultar jurisprudencia. Ejemplo: 86.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return
    render_divider()
    render_p2_result(data)


# ===== SECCIÓN 5: P3, evaluación =====
def split_provenance(value: str) -> tuple[str, str]:
    """Separa archivo y clave de una ruta de procedencia.

    Args:
        value: Cadena "archivo:clave.con.puntos".

    Returns:
        Archivo y clave; la clave queda vacía si no hay separador.
    """
    path, separator, key = value.partition(":")
    return path, key if separator else ""


def page_evaluation(client: ApiClient) -> None:
    """Pantalla P3: métricas congeladas con su procedencia.

    Args:
        client: Cliente HTTP.
    """
    render_header(
        "Evaluación",
        "Métricas congeladas de la segunda entrega, con el archivo y la clave "
        "exacta de cada valor.",
    )
    try:
        with st.spinner("Leyendo las métricas de la API..."):
            summary = client.eval_summary()
    except ApiError as exc:
        st.error(f"Error: {exc}")
        st.caption(
            "La pantalla no inventa cifras. Revise que el backend lea los JSON "
            "de resultados de la segunda entrega."
        )
        return

    metrics = summary.get("metricas") or {}
    if not metrics:
        st.warning(
            "Aviso: la API respondió sin métricas. Revise que existan los JSON "
            "de resultados de la segunda entrega."
        )
        return

    updated = as_text(summary.get("actualizado"), "no declarado")
    sources = summary.get("fuentes") or []
    st.caption(f"Actualizado: {updated}. Fuentes declaradas: {len(sources)}.")
    render_notice(
        "M4 se reporta como diagnóstico: ADR-19 concluyó que su referencia "
        "silver no es un gold fiable. Las demás cifras son las registradas en "
        "los JSON de la segunda entrega."
    )

    provenance = summary.get("procedencia") or {}
    rows = []
    for key, label in METRIC_LABELS.items():
        path = as_text(provenance.get(key), "")
        file, detail = split_provenance(path) if path else ("", "")
        rows.append(
            [
                label,
                format_score(metrics.get(key)),
                file or "Sin procedencia",
                detail or "Sin clave",
            ]
        )
    render_table(
        ["Métrica", "Valor", "Archivo", "Clave"],
        rows,
        numeric={1},
    )

    comparativa = summary.get("comparativa_rutas") or []
    if comparativa:
        st.subheader("Comparación de rutas de recuperación")
        st.caption(
            "nDCG@10 de cada ruta y el archivo del que sale la cifra. La "
            "referencia usa el benchmark M7; el gold humano es la anotación "
            "de B3."
        )
        render_table(
            ["Ruta", "Métrica", "Valor", "Procedencia"],
            [
                [
                    as_text(item.get("ruta")),
                    as_text(item.get("metrica")),
                    format_score(item.get("valor")),
                    as_text(item.get("procedencia"), "Sin procedencia"),
                ]
                for item in comparativa
                if isinstance(item, dict)
            ],
            numeric={2},
        )

    guardrails = summary.get("agente_guardrails") or {}
    if guardrails:
        st.subheader("Guardrails del agente")
        st.caption(
            f"Procedencia: {as_text(guardrails.get('procedencia'))}."
        )
        render_table(
            ["Métrica de guardrail", "Tasa"],
            [
                [as_text(item.get("etiqueta")), format_score(item.get("valor"))]
                for item in guardrails.get("metricas") or []
                if isinstance(item, dict)
            ],
            numeric={1},
        )
        render_table(
            ["Latencia de la ruta del agente", "Valor"],
            [
                [
                    as_text(item.get("etiqueta")),
                    format_milisegundos(item.get("valor")),
                ]
                for item in guardrails.get("latencia") or []
                if isinstance(item, dict)
            ],
            numeric={1},
        )
        aviso = guardrails.get("aviso")
        if aviso:
            render_notice(as_text(aviso))

    avisos = summary.get("avisos_procedencia") or []
    if avisos:
        st.subheader("Procedencia de etiquetas y gold sets")
        for aviso in avisos:
            if not isinstance(aviso, dict):
                continue
            render_notice(
                f"{as_text(aviso.get('tema'), 'Procedencia')}: "
                f"{as_text(aviso.get('texto'), 'Sin detalle')}"
            )

    with st.expander("Fuentes y latencias registradas"):
        st.markdown(
            "\n".join(f"- `{as_text(source)}`" for source in sources)
            or "Sin fuentes declaradas."
        )
        st.json(summary.get("latencias_s") or {})


# ===== SECCIÓN 6: consulta asistida =====
def set_question(text: str) -> None:
    """Carga un ejemplo en el campo de pregunta.

    Args:
        text: Pregunta de ejemplo.
    """
    st.session_state["chat_question"] = text


def render_tool_calls(answer: dict[str, Any]) -> None:
    """Dibuja la traza de herramientas del agente.

    Args:
        answer: Respuesta de /v1/ask.
    """
    st.subheader("Tool calls")
    calls = answer.get("tool_calls") or []
    if not calls:
        st.caption("Sin llamadas a herramientas en esta respuesta.")
        return
    rows = [
        [
            as_text(item.get("herramienta")),
            json.dumps(item.get("argumentos") or {}, ensure_ascii=False),
            "sí" if item.get("ok") else "no",
            as_text(item.get("ms")),
            as_text(item.get("resumen")),
        ]
        for item in calls
    ]
    render_table(
        ["Herramienta", "Argumentos", "Ok", "ms", "Resumen"],
        rows,
        numeric={3},
    )


def render_chat_answer(answer: dict[str, Any]) -> None:
    """Dibuja una respuesta del agente con su trazabilidad.

    Args:
        answer: Respuesta de /v1/ask.
    """
    mode = str(answer.get("modo", ""))
    render_meta_card(
        [
            ("Modo", RESPONSE_MODES.get(mode, mode or "No declarado")),
            ("Intención", as_text(answer.get("intencion"))),
            ("Modelo", as_text(answer.get("modelo"))),
            ("Latencia", f"{as_text(answer.get('latencia_ms'))} ms"),
        ]
    )
    with st.container(border=True):
        st.markdown(as_text(answer.get("respuesta"), "Sin respuesta."))

    warning = answer.get("advertencia")
    if warning:
        st.warning(f"Aviso del backend: {warning}")

    pensamiento = answer.get("pensamiento")
    if pensamiento:
        with st.expander("Razonamiento del modelo", expanded=False):
            st.text(str(pensamiento))

    grounding = answer.get("grounding") or {}
    verified = grounding.get("citas_verificadas") or []
    unverified = grounding.get("citas_no_verificadas") or []
    if grounding.get("ok"):
        detail = ", ".join(str(citation) for citation in verified)
        prefix = f"Grounding verificado. Citas comprobadas: {len(verified)}"
        st.caption(f"{prefix}. {detail}" if detail else f"{prefix}.")
    else:
        st.warning(
            "Aviso: grounding no verificado. Citas sin respaldo en la "
            f"evidencia: {', '.join(str(c) for c in unverified)}."
        )

    render_tool_calls(answer)


def page_chat(client: ApiClient, model: str) -> None:
    """Pantalla de consulta asistida con el agente.

    Args:
        client: Cliente HTTP.
        model: Modelo seleccionado en la barra lateral.
    """
    render_header(
        "Consulta asistida",
        "Preguntas sobre el corpus. El asistente responde con evidencia de sus "
        "herramientas y declara cuando no la encuentra.",
    )
    st.text_area(
        "Pregunta",
        key="chat_question",
        height=110,
        placeholder=(
            "Ejemplo: ¿qué artículos de la Constitución vincula la sentencia 2195?"
        ),
    )
    with st.expander("Ejemplos de consulta"):
        st.button(
            "¿Qué artículos de la Constitución vincula la sentencia 2195?",
            key="chat_example_1",
            on_click=set_question,
            args=("¿Qué artículos de la Constitución vincula la sentencia 2195?",),
        )
        st.button(
            "Valida la sentencia 2195.",
            key="chat_example_2",
            on_click=set_question,
            args=("Valida la sentencia 2195.",),
        )
        st.button(
            "¿Qué sentencias cita la T-293/97?",
            key="chat_example_3",
            on_click=set_question,
            args=("¿Qué sentencias cita la T-293/97?",),
        )

    st.caption(f"Modelo seleccionado: {model}.")
    pensar = st.checkbox(
        "Razonamiento del modelo (thinking)",
        key="chat_pensar",
        help=(
            "El modelo razona antes de responder y el pensamiento se muestra "
            "aparte. Cuesta más tiempo: con el modelo por defecto la respuesta "
            "pasa de unos 10 s a unos 40 s, y consume más contexto."
        ),
    )
    if pensar:
        st.caption(
            "Razonamiento activado: la respuesta tarda más y el pensamiento se "
            "muestra al final, en un panel aparte."
        )
    if st.button("Consultar", type="primary", key="chat_ask"):
        question = (st.session_state.get("chat_question") or "").strip()
        if len(question) < 2:
            st.warning("Aviso: escriba una pregunta de al menos dos caracteres.")
        else:
            with st.spinner(
                f"Consultando a {model}. La primera respuesta puede tardar."
            ):
                try:
                    answer = client.ask(question, model=model, pensar=pensar)
                except ApiError as exc:
                    st.error(f"Error: {exc}")
                    st.caption(
                        "Si el modelo no está cargado, revise ollama y la VRAM "
                        "disponible antes de reintentar."
                    )
                else:
                    st.session_state["chat_answer"] = answer

    answer = st.session_state.get("chat_answer")
    if answer:
        render_divider()
        render_chat_answer(answer)


# ===== SECCIÓN 7: arranque =====
def main() -> None:
    """Configura la página, dibuja la barra lateral y despacha la pantalla."""
    st.set_page_config(
        page_title="Juris-PLN",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    client = get_client()
    model = render_sidebar(client)
    screen = st.session_state.get("screen", CONFIG["screens"][0])

    if screen == CONFIG["screens"][0]:
        page_sentence(client)
    elif screen == CONFIG["screens"][1]:
        page_article(client)
    elif screen == CONFIG["screens"][2]:
        page_evaluation(client)
    else:
        page_chat(client, model)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        st.error("Error inesperado en la interfaz. Revise la consola del servidor.")
        sys.exit(1)
