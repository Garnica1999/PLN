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
/* Material Design 3, tema claro con semilla azul institucional.
   Roles tonales, tipografia Roboto, formas de 4 a 28 dp y elevacion tonal. */
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap');

:root {
    /* Roles de color M3 (claro) */
    --md-primary: #3B5BA9;
    --md-on-primary: #FFFFFF;
    --md-primary-container: #DBE1FF;
    --md-on-primary-container: #00174B;
    --md-secondary: #565E71;
    --md-on-secondary: #FFFFFF;
    --md-secondary-container: #DAE2F9;
    --md-on-secondary-container: #131C2B;
    --md-tertiary: #3A6475;
    --md-on-tertiary: #FFFFFF;
    --md-tertiary-container: #BEEAFE;
    --md-on-tertiary-container: #001F29;
    --md-error: #BA1A1A;
    --md-on-error: #FFFFFF;
    --md-error-container: #FFDAD6;
    --md-on-error-container: #410002;
    --md-surface: #F9F9FF;
    --md-on-surface: #1A1B21;
    --md-surface-variant: #E1E2EC;
    --md-on-surface-variant: #44474F;
    --md-outline: #74777F;
    --md-outline-variant: #C4C6D0;
    --md-surface-container-lowest: #FFFFFF;
    --md-surface-container-low: #F3F3FA;
    --md-surface-container: #EDEDF4;
    --md-surface-container-high: #E7E8EE;
    --md-surface-container-highest: #E2E2E9;
    --md-inverse-surface: #2F3036;
    --md-inverse-on-surface: #F1F0F7;
    /* Formas */
    --md-xs: 4px;
    --md-sm: 8px;
    --md-md: 12px;
    --md-lg: 16px;
    --md-xl: 28px;
    --md-full: 999px;
    /* Elevacion tonal */
    --md-elev-1: 0 1px 2px rgba(0, 0, 0, .07), 0 1px 3px 1px rgba(0, 0, 0, .04);
    --md-elev-2: 0 1px 2px rgba(0, 0, 0, .10), 0 2px 6px 2px rgba(0, 0, 0, .05);
    /* Tipografia */
    --md-font: 'Roboto', 'Noto Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
    --md-mono: 'Roboto Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace;
}

/* ---------- Base ---------- */
html, body, [data-testid="stAppViewContainer"] {
    font-family: var(--md-font);
    color: var(--md-on-surface);
    font-size: 16px;
    background: var(--md-surface);
    -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"] { background: var(--md-surface); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stHeaderActionElements"] { display: none; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stMainBlockContainer"] {
    max-width: 1180px;
    padding: 2.25rem 2.5rem 4rem;
}
[data-testid="stVerticalBlock"] { gap: 1rem; }
::selection {
    background: var(--md-primary-container);
    color: var(--md-on-primary-container);
}
* { scrollbar-width: thin; scrollbar-color: var(--md-outline-variant) transparent; }
a { color: var(--md-primary); text-underline-offset: 2px; }
:focus-visible { outline: 2px solid var(--md-primary); outline-offset: 2px; }

/* ---------- Tipografia ---------- */
h1, h2, h3, h4, h5, h6, p, li, label, span, div, button, input, textarea {
    font-family: var(--md-font);
}
h1 {
    font-size: 2.25rem;
    line-height: 2.75rem;
    font-weight: 400;
    letter-spacing: -.01em;
    color: var(--md-on-surface);
    margin: 0 0 .5rem;
}
h2 {
    font-size: 1.5rem;
    line-height: 2rem;
    font-weight: 500;
    color: var(--md-on-surface);
    margin: 2rem 0 .75rem;
}
h3 {
    font-size: 1.25rem;
    line-height: 1.75rem;
    font-weight: 500;
    color: var(--md-on-surface);
    margin: 1.5rem 0 .5rem;
}
[data-testid="stMarkdownContainer"] p {
    font-size: 1rem;
    line-height: 1.5rem;
    color: var(--md-on-surface);
}
[data-testid="stCaptionContainer"] {
    color: var(--md-on-surface-variant);
    font-size: .875rem;
    line-height: 1.25rem;
}
[data-testid="stWidgetLabel"] p {
    font-size: .875rem;
    line-height: 1.25rem;
    font-weight: 500;
    color: var(--md-on-surface-variant);
}
[data-testid="stSubheader"] h2, [data-testid="stSubheader"] h3 {
    font-size: 1.375rem;
    line-height: 1.75rem;
    font-weight: 500;
}

/* Encabezado de pantalla */
.jp-overline {
    font-size: .75rem;
    line-height: 1rem;
    font-weight: 500;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--md-primary);
    margin: 0 0 .5rem;
}
.jp-titulo {
    font-size: 2.25rem;
    line-height: 2.75rem;
    font-weight: 400;
    letter-spacing: -.01em;
    color: var(--md-on-surface);
    margin: 0 0 .5rem;
}
.jp-lede {
    font-size: 1rem;
    line-height: 1.5rem;
    color: var(--md-on-surface-variant);
    max-width: 68ch;
    margin: 0 0 1.75rem;
}

/* ---------- Barra lateral: cajon de navegacion ---------- */
[data-testid="stSidebar"] {
    background: var(--md-surface-container-low);
    border-right: 1px solid var(--md-outline-variant);
    min-width: 272px;
    max-width: 296px;
}
[data-testid="stSidebarContent"] { padding: 0 .75rem 1.5rem; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding: 0 .5rem 1.5rem;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
[data-testid="stSidebar"] [data-testid="stElementContainer"],
[data-testid="stSidebar"] [data-testid="stRadio"],
[data-testid="stSidebar"] [data-testid="stRadioGroup"],
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {
    width: 100%;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .5rem; }
[data-testid="stSidebar"] hr { display: none; }
.jp-marca {
    font-size: 1.375rem;
    line-height: 1.75rem;
    font-weight: 500;
    color: var(--md-on-surface);
    padding: 1.25rem .75rem .25rem;
    letter-spacing: -.01em;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: .6875rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--md-on-surface-variant);
    padding: 0 .75rem;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    padding: 0 .75rem;
    color: var(--md-on-surface-variant);
}

/* Items de navegacion como active indicator de M3 */
[data-testid="stSidebar"] [data-testid="stRadioGroup"] { gap: 2px; }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div { width: 100%; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
    display: flex;
    align-items: center;
    width: 100%;
    min-height: 48px;
    padding: 0 1rem;
    margin: 0;
    border-radius: var(--md-full);
    color: var(--md-on-surface-variant);
    font-size: .875rem;
    font-weight: 500;
    cursor: pointer;
    transition: background-color .15s ease, color .15s ease;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {
    background: var(--md-surface-container-high);
    color: var(--md-on-surface);
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {
    background: var(--md-secondary-container);
    color: var(--md-on-secondary-container);
}
/* Ocultar el circulo del radio solo en la navegacion */
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {
    display: none;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div {
    padding: 0;
}

/* Radios del contenido: circulo M3 */
[data-testid="stMain"] label[data-testid="stRadioOption"] > div {
    gap: .625rem;
}
[data-testid="stMain"] label[data-testid="stRadioOption"] > div > div:first-child {
    width: 20px;
    height: 20px;
    min-width: 20px;
    border-radius: 50%;
    border: 2px solid var(--md-on-surface-variant);
    display: flex;
    align-items: center;
    justify-content: center;
}
[data-testid="stMain"] label[data-testid="stRadioOption"][data-selected="true"]
    > div > div:first-child {
    border-color: var(--md-primary);
}
[data-testid="stMain"] label[data-testid="stRadioOption"]
    > div > div:first-child > div {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: transparent;
}
[data-testid="stMain"] label[data-testid="stRadioOption"][data-selected="true"]
    > div > div:first-child > div {
    background: var(--md-primary);
}
[data-testid="stMain"] label[data-testid="stRadioOption"] p {
    font-size: .9375rem;
    color: var(--md-on-surface);
}

/* ---------- Botones M3 ---------- */
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
    min-height: 40px;
    padding: 0 24px;
    border-radius: var(--md-full);
    border: none;
    font-size: .875rem;
    font-weight: 500;
    letter-spacing: .01em;
    box-shadow: none;
    transition: background-color .15s ease, box-shadow .15s ease, color .15s ease;
}
[data-testid="stBaseButton-primary"] {
    background: var(--md-primary) !important;
    color: var(--md-on-primary) !important;
}
[data-testid="stBaseButton-primary"]:hover {
    background: #33539B !important;
    box-shadow: var(--md-elev-1);
}
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"] {
    background: var(--md-secondary-container) !important;
    color: var(--md-on-secondary-container) !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: #CED8F2 !important;
}
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-secondary"] p {
    font-size: .875rem;
    font-weight: 500;
    margin: 0;
    color: inherit;
}

/* ---------- Campos de texto: variante filled ---------- */
[data-testid="stTextInputRootElement"],
[data-testid="stTextAreaRootElement"],
[data-testid="stNumberInputContainer"] {
    background: var(--md-surface-container-highest);
    border: none;
    border-bottom: 1px solid var(--md-outline);
    border-radius: var(--md-xs) var(--md-xs) 0 0;
    box-shadow: none;
    transition: border-color .15s ease, background-color .15s ease;
}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within {
    border-bottom: 2px solid var(--md-primary);
    background: var(--md-surface-container-high);
}
[data-testid="stTextInputField"],
[data-testid="stTextAreaField"] textarea,
[data-testid="stNumberInputField"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--md-on-surface);
    font-size: 1rem;
    padding: .75rem 1rem;
}
[data-testid="stTextInputField"]::placeholder,
[data-testid="stTextAreaField"] textarea::placeholder {
    color: var(--md-on-surface-variant);
    opacity: .8;
}
[data-testid="stTextAreaRootElement"] textarea {
    min-height: 120px;
    resize: none;
}
[data-testid="stTextAreaRootElement"]::after,
[data-testid="stTextAreaRootElement"]::before {
    display: none !important;
    content: none !important;
}

/* Selector: mismo lenguaje que el campo filled */
[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"] {
    background: var(--md-surface-container-highest);
    border: none;
    border-bottom: 1px solid var(--md-outline);
    border-radius: var(--md-xs) var(--md-xs) 0 0;
    min-height: 56px;
    padding: 0 .5rem 0 1rem;
    box-shadow: none;
    align-items: center;
}
[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"]:focus-within {
    border-bottom: 2px solid var(--md-primary);
}
[data-testid="stSelectbox"] .react-aria-ComboBox input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--md-on-surface);
    font-size: .8125rem;
}
[data-testid="stSelectbox"] button[aria-label] {
    color: var(--md-on-surface-variant);
    background: transparent;
}

/* Radios y casillas del contenido */
[data-testid="stRadio"] input[type="radio"],
[data-testid="stCheckbox"] input[type="checkbox"] {
    accent-color: var(--md-primary);
}
[data-testid="stCheckbox"] label {
    font-size: .875rem;
    color: var(--md-on-surface);
    gap: .625rem;
}
[data-testid="stCheckbox"] label > div:first-of-type {
    width: 18px;
    height: 18px;
    min-width: 18px;
    border: 2px solid var(--md-on-surface-variant);
    border-radius: 3px;
    background: transparent;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color .15s ease, border-color .15s ease;
}
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {
    background: var(--md-primary);
    border-color: var(--md-primary);
}
[data-testid="stCheckbox"] label > div:first-of-type svg {
    width: 11px;
    height: 9px;
    fill: none;
}
[data-testid="stCheckbox"] label[data-selected="true"] svg polyline {
    stroke: var(--md-on-primary);
    stroke-width: 1.8;
}

/* ---------- Tarjetas, tablas y avisos ---------- */
.jp-card {
    background: var(--md-surface-container-lowest);
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-md);
    padding: .25rem 0;
    margin: .25rem 0;
}
.jp-card:has(.jp-etiqueta) { padding: 1rem 1.25rem; }
.jp-meta-fila {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
    padding: .75rem 1.25rem;
    border-bottom: 1px solid var(--md-outline-variant);
}
.jp-meta-fila:last-child { border-bottom: none; }
.jp-meta-clave {
    color: var(--md-on-surface-variant);
    font-size: .875rem;
    line-height: 1.25rem;
}
.jp-dato {
    font-family: var(--md-mono);
    font-variant-numeric: tabular-nums;
    font-size: .875rem;
    color: var(--md-on-surface);
    text-align: right;
}
.jp-etiqueta {
    font-size: .6875rem;
    line-height: 1rem;
    font-weight: 500;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--md-primary);
    margin: 0 0 .25rem;
}
.jp-frag-texto {
    font-size: .9375rem;
    line-height: 1.5rem;
    color: var(--md-on-surface);
    margin: 0;
}
.jp-regla {
    height: 1px;
    background: var(--md-outline-variant);
    margin: 1.75rem 0;
    border: none;
}
.jp-aviso {
    background: var(--md-secondary-container);
    color: var(--md-on-secondary-container);
    border-radius: var(--md-md);
    padding: .875rem 1rem;
    margin: .25rem 0;
}
.jp-aviso p {
    margin: 0;
    color: inherit;
    font-size: .9375rem;
    line-height: 1.375rem;
}
.jp-vacio {
    background: var(--md-surface-container-low);
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-lg);
    padding: 1.75rem 2rem;
}
.jp-vacio-titulo {
    font-size: 1.25rem;
    line-height: 1.75rem;
    font-weight: 500;
    color: var(--md-on-surface);
    margin: 0 0 .5rem;
}
.jp-vacio p { color: var(--md-on-surface-variant); margin: 0; }

/* Tablas como card M3 */
.jp-tabla-envoltura {
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-md);
    overflow: hidden;
    background: var(--md-surface-container-lowest);
    margin: .5rem 0;
}
.jp-tabla {
    width: 100%;
    border-collapse: collapse;
    font-size: .875rem;
}
.jp-tabla th {
    background: var(--md-surface-container);
    color: var(--md-on-surface-variant);
    font-size: .6875rem;
    font-weight: 500;
    letter-spacing: .08em;
    text-transform: uppercase;
    text-align: left;
    padding: .75rem 1rem;
}
.jp-tabla th, .jp-tabla td { text-align: left; }
.jp-tabla td {
    padding: .75rem 1rem;
    border-top: 1px solid var(--md-outline-variant);
    color: var(--md-on-surface);
    vertical-align: top;
}
.jp-tabla td.jp-mono, .jp-tabla td.jp-mono.jp-dato {
    font-family: var(--md-mono);
    font-size: .8125rem;
    overflow-wrap: anywhere;
}
.jp-tabla tbody tr:hover { background: var(--md-surface-container-low); }
.jp-tabla td.jp-dato { text-align: right; }

/* ---------- Grafico de barras ---------- */
.jp-figura {
    background: var(--md-surface-container-lowest);
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-md);
    padding: 1rem 1.25rem 1.25rem;
    margin: .5rem 0;
}
.jp-barras {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 168px;
    margin: .75rem 0 .25rem;
}
.jp-barra {
    flex: 1;
    background: var(--md-primary);
    border-radius: var(--md-xs) var(--md-xs) 0 0;
    min-height: 2px;
    transition: background-color .15s ease;
}
.jp-barra:hover { background: var(--md-tertiary); }
.jp-ejes { display: flex; gap: 3px; }
.jp-eje {
    flex: 1;
    text-align: center;
    font-size: .625rem;
    color: var(--md-on-surface-variant);
    white-space: nowrap;
    overflow: visible;
}

/* ---------- Widgets nativos: alertas, estados, expanders ---------- */
[data-testid="stAlert"] {
    border-radius: var(--md-md);
    border: none;
    padding: 0;
    background: transparent;
}
[data-testid="stAlertContainer"] {
    padding: 0;
    background: transparent;
}
[data-testid="stAlertContentInfo"],
[data-testid="stAlertContentSuccess"],
[data-testid="stAlertContentWarning"],
[data-testid="stAlertContentError"] {
    border-radius: var(--md-md);
    padding: .875rem 1rem;
}
[data-testid="stAlertContentInfo"] {
    background: var(--md-secondary-container);
    color: var(--md-on-secondary-container);
}
[data-testid="stAlertContentSuccess"] {
    background: var(--md-primary-container);
    color: var(--md-on-primary-container);
}
[data-testid="stAlertContentWarning"] {
    background: var(--md-tertiary-container);
    color: var(--md-on-tertiary-container);
}
[data-testid="stAlertContentError"] {
    background: var(--md-error-container);
    color: var(--md-on-error-container);
}
[data-testid="stAlert"] p {
    font-size: .9375rem;
    margin: 0;
    color: inherit;
}
[data-testid="stExpander"] {
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-md);
    background: var(--md-surface-container-lowest);
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-size: .875rem;
    font-weight: 500;
    color: var(--md-on-surface);
    padding: .875rem 1rem;
}
[data-testid="stExpander"] summary:hover {
    background: var(--md-surface-container-low);
}
[data-testid="stStatusWidget"] {
    background: var(--md-surface-container);
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-md);
    padding: .25rem .75rem;
}
[data-testid="stSpinner"] svg { color: var(--md-primary); }
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: var(--md-md);
}

/* Contenedor con borde (tarjeta de respuesta del chat) */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--md-outline-variant) !important;
    border-radius: var(--md-md) !important;
    background: var(--md-surface-container-lowest);
}

/* ---------- Responsive ---------- */
@media (min-width: 900px) {
    [data-testid="stSidebar"] {
        width: 336px !important;
        min-width: 336px !important;
        max-width: 336px !important;
    }
}
@media (max-width: 1024px) {
    [data-testid="stMainBlockContainer"] { padding: 1.75rem 1.5rem 3rem; }
}
@media (max-width: 720px) {
    [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 2.5rem; }
    .jp-titulo { font-size: 1.75rem; line-height: 2.25rem; }
    .jp-lede { font-size: .9375rem; }
    .jp-barras { height: 128px; }
    .jp-meta-fila { flex-direction: column; gap: .25rem; }
    .jp-dato { text-align: left; }
}

/* Movimiento reducido */
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
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
        '<p class="jp-overline">Juris-PLN · Entrega 3</p>'
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
    mono: set[int] | None = None,
    anchos: list[int] | None = None,
) -> None:
    """Dibuja una tabla HTML con la paleta y la tipografía del informe.

    Args:
        columns: Cabeceras de la tabla.
        rows: Celdas ya formateadas.
        numeric: Índices de columnas que usan monoespaciada tabular.
        mono: Índices de columnas de texto técnico (rutas, claves) que
            también usan monoespaciada.
        anchos: Anchos relativos por columna, en porcentaje.
    """
    numeric = numeric or set()
    mono = mono or set()
    header = "".join(
        f'<th scope="col">{html.escape(column)}</th>' for column in columns
    )
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            clases = []
            if index in numeric:
                clases.append("jp-dato")
            if index in mono:
                clases.append("jp-mono")
            css_class = f' class="{" ".join(clases)}"' if clases else ""
            cells.append(f"<td{css_class}>{html.escape(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    colgroup = ""
    if anchos and len(anchos) == len(columns):
        columnas = "".join(
            f'<col style="width:{ancho}%">' for ancho in anchos
        )
        colgroup = f"<colgroup>{columnas}</colgroup>"
    st.markdown(
        '<div class="jp-tabla-envoltura"><table class="jp-tabla">'
        f"{colgroup}<thead><tr>{header}</tr></thead>"
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
        return f"{value:.4f}".replace(".", ",")
    return "No disponible"


def format_milisegundos(value: Any) -> str:
    """Formatea una latencia en milisegundos sin decimales.

    Args:
        value: Número recibido o None.

    Returns:
        Latencia formateada o "No disponible".
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:,.0f}".replace(",", ".") + " ms"
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
        st.caption("Corte Constitucional de Colombia, 1992-2026")
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
        f"n = {total} sentencias; corpus 1992-2026; etiquetas débiles de M2."
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
        "El total cuenta sentencias del corpus 1992-2026 cuyo bloque arts_cp "
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
        mono={2, 3},
        anchos=[26, 9, 19, 46],
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
            mono={3},
            anchos=[24, 9, 9, 58],
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
                [
                    as_text(item.get("etiqueta")),
                    (
                        f"{float(item['valor']) * 100:.1f} %".replace(".", ",")
                        if isinstance(item.get("valor"), (int, float))
                        and not isinstance(item.get("valor"), bool)
                        else "No disponible"
                    ),
                ]
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
