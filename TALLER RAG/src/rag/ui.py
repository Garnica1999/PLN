"""Interfaz de chat sobre la API."""
from __future__ import annotations

import os
import random
from pathlib import Path

import requests
import streamlit as st

from src.rag.ui_prefs import DEFAULTS, read_prefs, write_prefs

API = os.getenv("API_URL", "http://localhost:8010").rstrip("/")
DEFAULT_MODEL = "gemma4:e4b"
EXAMPLE_POOL = [
    "¿Cuántos municipios reportaron el Retamo Espinoso?",
    "¿Qué pasó con la moneda de 1.000 pesos?",
    "¿Cuándo se celebra el Global Big Day?",
    "¿Dónde se encuentra la Marianne North Gallery?",
    "¿Cuántas rutas de aves habrá en el Global Big Day 2026?",
    "¿Cuántas toneladas de desechos se recolectaron en la jornada de limpieza?",
    "¿Cuántas horas se destina a reflexionar sobre los océanos?",
    "¿Cuándo es la Feria Turística Caldas es Natural?",
    "Hola",
]

st.set_page_config(page_title="Asistente RAG", layout="wide")

PREFS_PATH = Path(os.getenv("UI_PREFS_PATH", "config/ui_prefs.json"))


def save_prefs() -> None:
    """Guarda las preferencias tras cambiar un interruptor."""
    write_prefs(
        PREFS_PATH,
        {key: st.session_state.get(key, default) for key, default in DEFAULTS.items()},
    )


for pref_key, pref_value in read_prefs(PREFS_PATH).items():
    if pref_key not in st.session_state:
        st.session_state[pref_key] = pref_value

DARK = bool(st.session_state.get("modo_noche", False))
st.markdown(
    """
    <style>
    /* Material 3, esquema claro generado desde la semilla verde del corpus. */
    :root {
        --md-primary: #1f6b57;
        --md-on-primary: #ffffff;
        --md-primary-container: #a6f2d9;
        --md-on-primary-container: #00201a;
        --md-secondary: #4d635a;
        --md-on-secondary: #ffffff;
        --md-secondary-container: #cfe8dd;
        --md-on-secondary-container: #0a1f18;
        --md-tertiary: #8a3b22;
        --md-on-tertiary: #ffffff;
        --md-tertiary-container: #ffdbd0;
        --md-on-tertiary-container: #3a0b00;
        --md-error: #ba1a1a;
        --md-on-error: #ffffff;
        --md-error-container: #ffdad6;
        --md-on-error-container: #410002;
        --md-surface: #fbf8f2;
        --md-surface-container-lowest: #ffffff;
        --md-surface-container-low: #f6f3ec;
        --md-surface-container: #f0ece5;
        --md-surface-container-high: #eae6df;
        --md-surface-container-highest: #e4e0d9;
        --md-on-surface: #1b1c19;
        --md-on-surface-variant: #434944;
        --md-outline: #737971;
        --md-outline-variant: #c3c8c1;
        --md-inverse-surface: #30312d;
        --md-inverse-on-surface: #f2f1ea;

        --md-shape-xs: 4px;
        --md-shape-sm: 8px;
        --md-shape-md: 12px;
        --md-shape-lg: 16px;
        --md-shape-xl: 28px;
        --md-shape-full: 9999px;

        --md-font: "Roboto", "Roboto Flex", "Noto Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
        --md-display-sm: 2.25rem;
        --md-headline-md: 1.75rem;
        --md-title-lg: 1.375rem;
        --md-title-md: 1rem;
        --md-title-sm: 0.875rem;
        --md-body-lg: 1rem;
        --md-body-md: 0.875rem;
        --md-body-sm: 0.75rem;
        --md-label-lg: 0.875rem;
        --md-label-md: 0.75rem;
        --md-label-sm: 0.6875rem;

        --md-elev-1: 0 1px 2px rgba(27, 28, 25, 0.10), 0 1px 3px 1px rgba(27, 28, 25, 0.06);
        --md-elev-2: 0 1px 2px rgba(27, 28, 25, 0.10), 0 2px 6px 2px rgba(27, 28, 25, 0.08);
        --md-elev-3: 0 1px 3px rgba(27, 28, 25, 0.12), 0 4px 8px 3px rgba(27, 28, 25, 0.08);
        --md-ease-standard: cubic-bezier(0.2, 0, 0, 1);
        --md-ease-emphasized: cubic-bezier(0.05, 0.7, 0.1, 1);
        --md-dur-short: 150ms;
        --md-dur-medium: 250ms;
        --md-hover: color-mix(in srgb, var(--md-primary) 8%, transparent);
        --md-focus: color-mix(in srgb, var(--md-primary) 10%, transparent);
        --md-pressed: color-mix(in srgb, var(--md-primary) 10%, transparent);
    }

    html { scrollbar-color: var(--md-outline-variant) transparent; scrollbar-width: thin; }
    .stApp,
    .stApp button,
    .stApp input,
    .stApp textarea {
        font-family: var(--md-font);
    }
    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--md-surface) !important;
        color: var(--md-on-surface);
        -webkit-font-smoothing: antialiased;
    }
    [data-testid="stDecoration"], [data-testid="stLogoSpacer"],
    [data-testid="stToolbarActions"], [data-testid="stMainMenu"],
    [data-testid="stAppDeployButton"], #MainMenu, footer {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        background: transparent !important;
        display: flex !important;
    }
    /* La cabecera queda transparente porque aloja el boton de reapertura del panel. */
    [data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none;
    }
    [data-testid="stExpandSidebarButton"] {
        pointer-events: auto;
        display: flex !important;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border: none !important;
        border-radius: var(--md-shape-full) !important;
        background: var(--md-surface-container-high) !important;
        color: var(--md-on-surface) !important;
        box-shadow: var(--md-elev-1);
        transition: background-color var(--md-dur-short) var(--md-ease-standard);
    }
    [data-testid="stExpandSidebarButton"] span { color: inherit !important; }
    [data-testid="stExpandSidebarButton"]:hover {
        background: color-mix(in srgb, var(--md-primary) 8%, var(--md-surface-container-high)) !important;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1040px;
        margin: 0 auto;
        padding: 2.5rem 2rem 1rem;
    }
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div { background: var(--md-surface) !important; }
    [data-testid="stBottomBlockContainer"] {
        max-width: 1040px;
        margin: 0 auto;
        padding: 0 2rem 1.5rem;
        background: var(--md-surface);
    }

    /* Tipografia: escala M3 con estilos enfatizados para titulos. */
    h1, h2, h3, h4,
    [data-testid="stHeadingWithActionElements"] h1,
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3 {
        font-family: var(--md-font) !important;
        color: var(--md-on-surface) !important;
        letter-spacing: 0 !important;
        font-weight: 500 !important;
    }
    h1 { font-size: var(--md-headline-md) !important; line-height: 2.25rem !important; }
    h2 { font-size: var(--md-title-lg) !important; line-height: 1.75rem !important; }
    h3 { font-size: var(--md-title-lg) !important; line-height: 1.75rem !important; }
    h4 { font-size: var(--md-title-md) !important; line-height: 1.5rem !important; }
    p, label, .stMarkdown { color: var(--md-on-surface); }
    [data-testid="stMarkdown"] p { line-height: 1.55; }
    .m3-headline {
        font-size: var(--md-headline-md) !important;
        line-height: 2.25rem !important;
        font-weight: 500 !important;
        color: var(--md-on-surface) !important;
        margin: 0 !important;
    }
    .m3-supporting {
        font-size: var(--md-body-lg);
        line-height: 1.5rem;
        color: var(--md-on-surface-variant);
        margin: 0.25rem 0 0;
    }
    .hint {
        font-size: var(--md-title-sm) !important;
        font-weight: 500 !important;
        color: var(--md-on-surface-variant) !important;
        margin: 2.5rem 0 0.75rem !important;
    }
    a, [data-testid="stMarkdown"] a {
        color: var(--md-primary);
        text-decoration: underline;
        text-underline-offset: 2px;
    }
    code {
        background: var(--md-surface-container-high);
        border-radius: var(--md-shape-xs);
        padding: 0.1rem 0.3rem;
    }

    /* Panel lateral: cajon de navegacion M3 con esquinas grandes al final. */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background: var(--md-surface-container);
        border-right: none;
        border-radius: 0 var(--md-shape-lg) var(--md-shape-lg) 0;
    }
    [data-testid="stSidebar"] { overflow: hidden; }
    [data-testid="stSidebar"] hr {
        border-color: var(--md-outline-variant);
        margin: 1.25rem 0;
    }
    [data-testid="stSidebarUserContent"] { padding-top: 1.5rem; }
    [data-testid="stSidebar"] h3 {
        font-size: var(--md-title-sm) !important;
        line-height: 1.25rem !important;
        font-weight: 500 !important;
        color: var(--md-on-surface-variant) !important;
        margin-bottom: 0.25rem !important;
    }
    [data-testid="stSidebar"] p { font-size: var(--md-body-md); line-height: 1.5; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: var(--md-label-lg);
        font-weight: 500;
        color: var(--md-on-surface-variant);
    }
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stDateInput { margin-bottom: 0.5rem; }

    /* Campos de texto y selectores: contenedor relleno M3. */
    [data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"],
    [data-testid="stDateInputField"] {
        background: var(--md-surface-container-highest);
        border: none;
        border-radius: var(--md-shape-sm);
        min-height: 48px;
        padding: 0 0.75rem;
        transition: background-color var(--md-dur-short) var(--md-ease-standard);
    }
    [data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"]:hover,
    [data-testid="stDateInputField"]:hover {
        background: color-mix(in srgb, var(--md-on-surface) 6%, var(--md-surface-container-highest));
    }
    [data-testid="stSelectbox"] .react-aria-ComboBox:focus-within > div[role="group"],
    [data-testid="stDateInput"] [data-testid="stDateInputField"]:focus-within {
        outline: 2px solid var(--md-primary);
        outline-offset: -2px;
    }
    [data-testid="stSelectbox"] input,
    [data-testid="stDateInput"] input {
        color: var(--md-on-surface);
        font-size: var(--md-body-lg);
    }
    [data-testid="stSelectbox"] button,
    [data-testid="stDateInput"] button { color: var(--md-on-surface-variant); }

    /* Control deslizante M3: pista de 4 px y pulgar de 20 px. */
    [data-testid="stSlider"] [role="group"] > div[style*="position: relative"] > div:first-child {
        height: 4px;
        border-radius: var(--md-shape-full);
    }
    [data-testid="stSlider"] [role="group"] > div[style*="position: relative"] > div[style*="position: absolute"] {
        width: 20px;
        height: 20px;
        border-radius: var(--md-shape-full);
        background: var(--md-primary);
        box-shadow: 0 0 0 0 transparent;
        transition: box-shadow var(--md-dur-short) var(--md-ease-standard);
    }
    [data-testid="stSlider"] [role="group"] > div[style*="position: relative"] > div[style*="position: absolute"]:hover {
        box-shadow: 0 0 0 8px var(--md-hover);
    }
    [data-testid="stSliderThumbValue"] p,
    [data-testid="stSliderTickBar"] p {
        font-size: var(--md-label-md);
        color: var(--md-on-surface-variant);
    }

    /* Interruptores M3: pista de 32 px, pulgar sobre primary. */
    [data-testid="stCheckbox"] label > div:first-of-type {
        background: var(--md-surface-container-highest) !important;
        border: 2px solid var(--md-outline) !important;
        transition: background-color var(--md-dur-short) var(--md-ease-standard),
                    border-color var(--md-dur-short) var(--md-ease-standard);
    }
    [data-testid="stCheckbox"] label > div:first-of-type > div {
        background: var(--md-outline) !important;
    }
    [data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type {
        background: var(--md-primary) !important;
        border-color: var(--md-primary) !important;
    }
    [data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type > div {
        background: var(--md-on-primary) !important;
    }
    [data-testid="stCheckbox"] label:has(input:focus-visible) > div:first-of-type {
        outline: 2px solid var(--md-primary);
        outline-offset: 2px;
    }
    [data-testid="stTooltipIcon"] button { color: var(--md-on-surface-variant); }

    /* Botones M3 rellenos tonales: forma completa y capa de estado. */
    [data-testid^="stBaseButton-"] {
        border: none !important;
        border-radius: var(--md-shape-full) !important;
        min-height: 40px;
        padding: 0 1.5rem !important;
        background: var(--md-secondary-container) !important;
        color: var(--md-on-secondary-container) !important;
        font-size: var(--md-label-lg) !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: background-color var(--md-dur-short) var(--md-ease-standard),
                    box-shadow var(--md-dur-short) var(--md-ease-standard);
    }
    [data-testid^="stBaseButton-"]:hover {
        background: color-mix(in srgb, var(--md-primary) 8%, var(--md-secondary-container)) !important;
        box-shadow: var(--md-elev-1) !important;
    }
    [data-testid^="stBaseButton-"]:active {
        background: color-mix(in srgb, var(--md-primary) 12%, var(--md-secondary-container)) !important;
    }
    [data-testid^="stBaseButton-"]:focus-visible {
        outline: 2px solid var(--md-primary);
        outline-offset: 2px;
    }

    /* Sugerencias: tarjetas M3 en lugar de botones centrados. */
    .st-key-suggestions [data-testid^="stBaseButton-"] {
        justify-content: flex-start !important;
        text-align: left !important;
        min-height: 64px;
        padding: 0.75rem 1rem !important;
        border-radius: var(--md-shape-md) !important;
        background: var(--md-surface-container-low) !important;
        color: var(--md-on-surface) !important;
        font-size: var(--md-body-md) !important;
        font-weight: 400 !important;
        line-height: 1.35;
    }
    .st-key-suggestions [data-testid^="stBaseButton-"]:hover {
        background: color-mix(in srgb, var(--md-primary) 8%, var(--md-surface-container-low)) !important;
        box-shadow: var(--md-elev-1) !important;
    }
    .st-key-suggestions [data-testid^="stBaseButton-"] > div,
    .st-key-suggestions [data-testid^="stBaseButton-"] > div > span {
        justify-content: flex-start !important;
        width: 100% !important;
    }
    .st-key-suggestions [data-testid="stMarkdownContainer"] p {
        text-align: left !important;
        white-space: normal !important;
    }

    /* Mensajes de chat: tarjetas M3 sobre surface-container-low. */
    [data-testid="stChatMessage"] {
        background: var(--md-surface-container-low);
        border: none;
        border-radius: var(--md-shape-lg);
        padding: 1rem 1.25rem;
        gap: 0.75rem;
    }
    [data-testid="stChatMessageContent"] { gap: 0.4rem; }
    [data-testid="stChatMessageAvatarCustom"] {
        background: var(--md-primary-container) !important;
        color: var(--md-on-primary-container) !important;
        border-radius: var(--md-shape-full);
    }
    .m3-card-title {
        font-size: var(--md-title-md) !important;
        line-height: 1.5rem !important;
        font-weight: 500 !important;
        color: var(--md-on-surface) !important;
        margin: 0.25rem 0 0 !important;
    }
    .st-key-appbar { margin-bottom: 1.25rem; }
    [data-testid="stChatMessage"]:has([class*="st-key-user-msg"]) {
        background: var(--md-secondary-container);
    }
    [data-testid="stChatMessage"]:has([class*="st-key-user-msg"]) p,
    [data-testid="stChatMessage"]:has([class*="st-key-user-msg"]) .stMarkdown {
        color: var(--md-on-secondary-container) !important;
        font-weight: 500;
    }
    [data-testid="stChatMessage"]:has([class*="st-key-user-msg"]) [data-testid="stChatMessageAvatarCustom"] {
        background: var(--md-primary) !important;
        color: var(--md-on-primary) !important;
    }
    [class*="st-key-answer"] { background: transparent; padding: 0; }
    [class*="st-key-answer"] [data-testid="stMarkdown"] p {
        font-size: var(--md-body-lg);
        line-height: 1.6;
    }

    /* Expansores M3: superficie contenedora, sin borde, con capa de estado. */
    [data-testid="stExpander"] { border: none !important; }
    [data-testid="stExpander"] details {
        background: var(--md-surface-container);
        border: none;
        border-radius: var(--md-shape-md);
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        padding: 0.75rem 1rem;
        font-size: var(--md-title-sm);
        font-weight: 500;
        color: var(--md-on-surface) !important;
        background: transparent !important;
        transition: background-color var(--md-dur-short) var(--md-ease-standard);
    }
    [data-testid="stExpander"] summary:hover {
        background: var(--md-hover) !important;
        color: var(--md-on-surface) !important;
    }
    [data-testid="stExpander"] summary svg { color: var(--md-on-surface-variant); }
    [data-testid="stExpanderDetails"] { padding: 0 1rem 1rem; }
    [data-testid="stExpanderDetails"] [data-testid="stMarkdown"] p { margin-bottom: 0.4rem; }
    [data-testid="stExpanderDetails"] [data-testid="stMarkdown"] p:last-child { margin-bottom: 0; }
    [class*="st-key-reasoning"],
    [class*="st-key-reasoning"] > div {
        border: none !important;
        box-shadow: none !important;
    }
    [class*="st-key-reasoning"] {
        background: var(--md-surface-container-low) !important;
        border-radius: var(--md-shape-md);
    }
    [class*="st-key-reasoning"] p {
        font-size: var(--md-body-md);
        line-height: 1.55;
    }

    /* Entrada de chat: barra de busqueda M3 con elevacion tonal. */
    [data-testid="stChatInput"] > div {
        background: var(--md-surface-container-high);
        border: none;
        border-radius: var(--md-shape-full);
        box-shadow: var(--md-elev-1);
        padding: 0.25rem 0.35rem 0.25rem 1rem;
        transition: box-shadow var(--md-dur-short) var(--md-ease-standard),
                    background-color var(--md-dur-short) var(--md-ease-standard);
    }
    [data-testid="stChatInput"] > div:focus-within {
        box-shadow: var(--md-elev-2);
        background: var(--md-surface-container-highest);
    }
    [data-testid="stChatInputTextArea"] {
        color: var(--md-on-surface);
        font-size: var(--md-body-lg);
        padding: 0.6rem 0.25rem;
    }
    [data-testid="stChatInputTextArea"]::placeholder { color: var(--md-on-surface-variant); }
    [data-testid="stChatInputSubmitButton"] {
        background: var(--md-primary);
        color: var(--md-on-primary);
        border-radius: var(--md-shape-full);
        width: 40px;
        height: 40px;
        transition: background-color var(--md-dur-short) var(--md-ease-standard);
    }
    [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
        background: color-mix(in srgb, var(--md-on-primary) 8%, var(--md-primary));
    }
    [data-testid="stChatInputSubmitButton"]:disabled {
        background: var(--md-surface-container-highest);
        color: color-mix(in srgb, var(--md-on-surface) 38%, transparent);
    }

    /* Chips M3 y avisos sobre contenedores semanticos. */
    .chip {
        display: inline-block;
        border-radius: var(--md-shape-sm);
        padding: 0.3rem 0.7rem;
        font-size: var(--md-label-md);
        font-weight: 500;
        margin: 0 0.35rem 0.35rem 0;
    }
    .chip-primary { background: var(--md-primary-container); color: var(--md-on-primary-container); }
    .chip-tertiary { background: var(--md-tertiary-container); color: var(--md-on-tertiary-container); }
    .chip-error { background: var(--md-error-container); color: var(--md-on-error-container); }
    .notice {
        background: var(--md-tertiary-container);
        color: var(--md-on-tertiary-container);
        border-radius: var(--md-shape-md);
        padding: 0.9rem 1rem;
        margin: 0.5rem 0;
        font-size: var(--md-body-md);
        line-height: 1.5;
    }
    .notice-error {
        background: var(--md-error-container);
        color: var(--md-on-error-container);
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: var(--md-shape-sm);
        padding: 0.25rem 0.6rem;
        font-size: var(--md-label-md);
        font-weight: 500;
    }
    .status-ok { background: var(--md-primary-container); color: var(--md-on-primary-container); }
    .status-off { background: var(--md-error-container); color: var(--md-on-error-container); }
    .status-pill .dot {
        width: 6px;
        height: 6px;
        border-radius: var(--md-shape-full);
        background: currentColor;
    }

    /* Menus y calendario: viven fuera del arbol de la app. */
    [data-testid="stSelectboxVirtualDropdown"] {
        background: var(--md-surface-container-high) !important;
        border-radius: var(--md-shape-sm);
        box-shadow: var(--md-elev-2);
    }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"] {
        color: var(--md-on-surface);
    }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover {
        background: var(--md-hover);
    }
    [data-testid="stDateInputCalendar"] {
        background: var(--md-surface-container-high) !important;
        border-radius: var(--md-shape-md);
        box-shadow: var(--md-elev-2);
    }
    [data-testid="stDateInputCalendar"] button,
    [data-testid="stDateInputCalendar"] [role="gridcell"] {
        color: var(--md-on-surface);
    }
    [data-testid="stDateInputCalendar"] select {
        background: var(--md-surface-container-highest) !important;
        color: var(--md-on-surface) !important;
        border: none !important;
        border-radius: var(--md-shape-sm);
    }
    [data-testid="stDateInputCalendar"] th {
        color: var(--md-on-surface-variant) !important;
    }

    /* Textos auxiliares y metricas. */
    [data-testid="stCaptionContainer"] p {
        font-size: var(--md-body-sm);
        line-height: 1.4;
        color: var(--md-on-surface-variant);
        font-variant-numeric: tabular-nums;
        margin: 0;
    }
    [data-testid="stMarkdown"] strong { font-weight: 500; }
    [data-testid="stSpinner"] p { color: var(--md-on-surface-variant); }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
        }
    }
    @media (max-width: 720px) {
        [data-testid="stMainBlockContainer"] { padding: 1.5rem 1rem 1rem; }
        [data-testid="stBottomBlockContainer"] { padding: 0 1rem 1rem; }
        [data-testid="stSidebar"] { border-radius: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if DARK:
    st.markdown(
        """
        <style>
        /* Material 3, esquema oscuro generado desde la misma semilla. */
        :root {
            --primary-color: #7fd8bb;
            --background-color: #101412;
            --secondary-background-color: #1c201d;
            --text-color: #e1e3de;
            --md-primary: #7fd8bb;
            --md-on-primary: #003828;
            --md-primary-container: #005140;
            --md-on-primary-container: #a6f2d9;
            --md-secondary: #b3ccc1;
            --md-on-secondary: #1e332b;
            --md-secondary-container: #354a42;
            --md-on-secondary-container: #cfe8dd;
            --md-tertiary: #ffb59e;
            --md-on-tertiary: #55200c;
            --md-tertiary-container: #6f2f18;
            --md-on-tertiary-container: #ffdbd0;
            --md-error: #ffb4ab;
            --md-on-error: #690005;
            --md-error-container: #93000a;
            --md-on-error-container: #ffdad6;
            --md-surface: #101412;
            --md-surface-container-lowest: #0b0f0d;
            --md-surface-container-low: #181c19;
            --md-surface-container: #1c201d;
            --md-surface-container-high: #262a27;
            --md-surface-container-highest: #313532;
            --md-on-surface: #e1e3de;
            --md-on-surface-variant: #bfc9c1;
            --md-outline: #89938b;
            --md-outline-variant: #3f4943;
            --md-inverse-surface: #e1e3de;
            --md-inverse-on-surface: #2e312e;
            --md-elev-1: 0 1px 2px rgba(0, 0, 0, 0.45), 0 1px 3px 1px rgba(0, 0, 0, 0.25);
            --md-elev-2: 0 1px 2px rgba(0, 0, 0, 0.45), 0 2px 6px 2px rgba(0, 0, 0, 0.30);
            --md-elev-3: 0 1px 3px rgba(0, 0, 0, 0.50), 0 4px 8px 3px rgba(0, 0, 0, 0.35);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_get(path: str, timeout: int = 10) -> dict | None:
    try:
        response = requests.get(f"{API}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


health = api_get("/health")
models_payload = api_get("/models")
models = [name for name in (models_payload or {}).get("models", []) if "embedding" not in name]
stats = (api_get("/stats") or {}).get("collections", {})

with st.sidebar:
    st.markdown("### Configuración")
    if health and health.get("status") == "ok":
        st.markdown(
            "<span class='status-pill status-ok'><span class='dot'></span>API conectada</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span class='status-pill status-off'><span class='dot'></span>API sin respuesta</span>",
            unsafe_allow_html=True,
        )
    st.toggle(
        "Modo noche",
        key="modo_noche",
        on_change=save_prefs,
        help="Cambia la interfaz a la paleta oscura de Material 3.",
    )
    if models:
        index = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
        model = st.selectbox("Modelo", models, index=index)
    else:
        st.error("No hay modelos disponibles en Ollama.")
        model = st.text_input("Modelo", value=DEFAULT_MODEL)
    k = st.slider("Fragmentos recuperados", 1, 20, 5)
    think = st.toggle(
        "Razonamiento",
        key="think",
        on_change=save_prefs,
        help="Muestra el razonamiento del modelo antes de la respuesta. Añade una llamada y unos 300-800 ms.",
    )
    strict = st.toggle(
        "Modo estricto",
        key="strict",
        on_change=save_prefs,
        help="Se abstiene si la pregunta no es del dominio del corpus. Usa un clasificador, no un juez.",
    )
    language = st.selectbox("Idioma del documento", ["", "es", "en"], format_func=lambda value: value or "cualquiera")
    date_from = st.date_input("Desde", value=None)
    date_to = st.date_input("Hasta", value=None)
    if stats:
        collection = st.selectbox("Colección", sorted(stats), index=0)
        st.caption(f"{stats.get(collection, 0):,} fragmentos")
    else:
        collection = st.text_input("Colección", value="")
    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.examples = random.sample(EXAMPLE_POOL, 3)
        st.rerun()
    st.divider()
    st.markdown("### Cómo leer la respuesta")
    st.markdown(
        "**Respuesta.** Se redacta solo con la evidencia recuperada.\n\n"
        "**Citas.** Cada afirmación lleva un número entre corchetes. Si no hay cita, el sistema se abstiene.\n\n"
        "**Abstención.** Si la evidencia no responde, no inventa: devuelve el aviso y el motivo."
    )

with st.container(key="appbar"):
    st.markdown("<h1 class='m3-headline'>Asistente RAG</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='m3-supporting'>Conversa con el corpus de prensa y Bluesky. Cada respuesta cita la evidencia recuperada.</p>",
        unsafe_allow_html=True,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "examples" not in st.session_state:
    st.session_state.examples = random.sample(EXAMPLE_POOL, 3)


def render_assistant(data: dict, position: int) -> None:
    reasoning = data.get("reasoning") or ""
    if reasoning:
        with st.expander("Razonamiento del modelo", expanded=True):
            if data.get("native_thinking"):
                st.caption("Razonamiento nativo del modelo. Puede estar en inglés. No sustituye la evidencia citada.")
            else:
                st.caption("Texto generado por el modelo antes de la respuesta. No sustituye la evidencia citada.")
            with st.container(height=320, key=f"reasoning-{position}"):
                st.markdown(reasoning.replace("\n", "  \n"))
            if data.get("reasoning_ms"):
                st.caption(f"{data['reasoning_ms']:.0f} ms")
    st.markdown("<h3 class='m3-card-title'>Respuesta</h3>", unsafe_allow_html=True)
    st.caption(f"Generada por {data.get('model')}")
    with st.container(key=f"answer-{position}"):
        st.markdown(data.get("answer", ""))
    chips = f"<span class='chip chip-primary'>Intención: {data.get('intent', 'dominio')}</span>"
    if data.get("guardrail"):
        chips += f"<span class='chip chip-tertiary'>Guardarraíl: {data['guardrail']}</span>"
    st.markdown(chips, unsafe_allow_html=True)
    if data.get("abstain_reason") == "fuera_dominio":
        st.markdown(
            "<div class='notice'>La pregunta no es del ámbito del corpus. Desactiva el Modo estricto para intentar responder con la evidencia.</div>",
            unsafe_allow_html=True,
        )
    elif data.get("abstained"):
        st.markdown(
            f"<div class='notice notice-error'>El sistema se abstuvo: {data.get('abstain_reason')}</div>",
            unsafe_allow_html=True,
        )
    if data.get("domain_probability") is not None:
        st.caption(f"Probabilidad de dominio: {data['domain_probability']:.3f}")
    metrics = [f"{data.get('retrieval_ms', 0):.0f} ms recuperación"]
    if data.get("reasoning_ms"):
        metrics.append(f"{data['reasoning_ms']:.0f} ms razonamiento")
    metrics.append(f"{data.get('generation_ms', 0):.0f} ms generación")
    metrics.append(f"{data.get('tokens_per_second') or 'n/d'} tok/s")
    metrics.append(f"{data.get('total_ms', 0):.0f} ms total")
    metrics.append(str(data.get("model")))
    st.caption(" · ".join(metrics))
    evidence = data.get("evidence", [])
    if evidence:
        with st.expander(f"Evidencias ({len(evidence)})"):
            for index, hit in enumerate(evidence, start=1):
                title = hit.get("title") or "Sin título"
                st.markdown(f"**[{index}] {title}** · score {hit.get('score', 0):.3f}")
                st.write(hit.get("text", ""))
                if hit.get("url"):
                    st.markdown(f"Fuente: [{hit['url']}]({hit['url']})")
                st.caption(
                    f"documento={hit.get('document_id')} · idioma={hit.get('language')} · "
                    f"fecha={hit.get('created_at')}"
                )


def request_answer(payload: dict) -> dict | None:
    """Envia la pregunta a la API y devuelve la respuesta, o None si falla."""
    try:
        response = requests.post(f"{API}/chat", json=payload, timeout=300)
    except requests.RequestException as error:
        st.error(f"No se pudo contactar la API: {error}")
        return None
    if response.status_code != 200:
        st.error(f"La API devolvió {response.status_code}. Revisa el servicio y la colección.")
        return None
    return response.json()


def resolve_pending() -> None:
    """Completa la pregunta pendiente cuando un rerun interrumpio la peticion.

    Cambiar un interruptor durante la generacion detiene el script de Streamlit.
    La pregunta queda guardada y se retoma en el siguiente rerun.
    """
    pending = st.session_state.get("pending")
    if not pending:
        return
    spinner = (
        "Recuperando evidencia, razonando y redactando..."
        if pending.get("think")
        else "Recuperando evidencia y redactando la respuesta..."
    )
    with st.spinner(spinner):
        data = request_answer(pending["payload"])
    st.session_state.pending = None
    if data is None:
        st.stop()
    st.session_state.messages.append(
        {"role": "assistant", "content": data.get("answer", ""), "data": data}
    )
    st.rerun()


def ask(query: str) -> None:
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ][-8:]
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.pending = {
        "payload": {
            "query": query,
            "model": model,
            "k": k,
            "temperature": 0.0,
            "language": language or None,
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
            "collection": collection or None,
            "history": history,
            "think": think,
            "strict": strict,
        },
        "think": think,
    }
    resolve_pending()


for position, message in enumerate(st.session_state.messages):
    avatar = ":material/person:" if message["role"] == "user" else ":material/menu_book:"
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "user":
            with st.container(key=f"user-msg-{position}"):
                st.markdown(message["content"])
        else:
            render_assistant(message.get("data", {}), position)

resolve_pending()

if not st.session_state.messages:
    st.markdown("<p class='hint'>Prueba con:</p>", unsafe_allow_html=True)
    with st.container(key="suggestions"):
        columns = st.columns(3)
        for column, example in zip(columns, st.session_state.examples):
            if column.button(example, use_container_width=True):
                ask(example)

prompt = st.chat_input("Escribe tu pregunta sobre el corpus")
if prompt:
    ask(prompt)
