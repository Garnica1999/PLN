"""Demo interactiva (Streamlit) del producto Juris-PLN.

Reutiliza el pipeline del CLI `scripts/10_evaluar_sentencia.py` (M0 validacion,
M4 area tematica, M7 articulos CP vinculados, M5 contexto jurisprudencial)
sin duplicar logica: carga el modulo por ruta y llama sus funciones.

Uso (desde cualquier directorio; la demo fija el cwd a SEGUNDA ENTREGA/):
    /home/carlos/venv/bin/python -m streamlit run scripts/demo_streamlit.py

Notas:
    - Todo corre en CPU; la primera evaluacion inicializa los componentes
      (lectura del corpus + indices) y puede tardar decenas de segundos.
    - El modo --rerank (v2.2: B2 + B1) es mas lento (~14 s/eval); el modo
      default (BM25 + prior de hubs) ronda 5-9 s.
    - Demo ilustrativa: no reemplaza al CLI ni al benchmark del proyecto.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]

CONFIG_DEMO = {
    "titulo": "Juris-PLN - demo del producto",
    "descripcion": (
        "Sistema de PLN sobre sentencias de la Corte Constitucional de Colombia. "
        "Valida si el texto es una providencia (M0), estima su area tematica (M4), "
        "identifica los articulos de la Constitucion vinculados (M7) y situa la "
        "sentencia en su contexto jurisprudencial (M5). Producto v2.2."
    ),
    "ids_ejemplo": ["2195", "293", "100"],
}


def cargar_modulo_producto():
    """Carga scripts/10_evaluar_sentencia.py como modulo importable.

    El nombre del archivo empieza por digito, por lo que no se puede importar
    con `import` normal; se usa importlib. Antes fija el cwd a SEGUNDA ENTREGA/
    (el producto usa rutas relativas: data/, modelos/) y agrega scripts/ al
    sys.path (el paquete lib/ vive alli, igual que al correr el CLI).

    Returns:
        Modulo del CLI del producto.
    """
    os.chdir(RAIZ)
    for ruta_syspath in (RAIZ / "scripts", RAIZ):
        if str(ruta_syspath) not in sys.path:
            sys.path.insert(0, str(ruta_syspath))
    ruta = RAIZ / "scripts" / "10_evaluar_sentencia.py"
    spec = importlib.util.spec_from_file_location("producto_juris_pln", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def inicializar(usar_reranker: bool):
    """Inicializa el modulo y los componentes del producto.

    Args:
        usar_reranker: True activa la ruta --rerank de v2.2 (B2 + B1).

    Returns:
        Tupla (modulo, componentes).
    """
    modulo = cargar_modulo_producto()
    return modulo, modulo.inicializar_componentes(usar_reranker=usar_reranker)


@st.cache_resource(show_spinner="Inicializando componentes del producto (una vez por modo)...")
def componentes_cache(usar_reranker: bool):
    """Version cacheada de inicializar (una vez por proceso y modo).

    Args:
        usar_reranker: True activa la ruta --rerank de v2.2 (B2 + B1).

    Returns:
        Tupla (modulo, componentes) cacheada.
    """
    return inicializar(usar_reranker)


def evaluar_entrada(modulo, componentes, modo: str, valor: str) -> dict:
    """Evalua una entrada (texto o id del corpus) con el pipeline del producto.

    Args:
        modulo: Modulo del producto (scripts/10_evaluar_sentencia.py).
        componentes: Componentes ya inicializados.
        modo: "Texto" o "ID del corpus".
        valor: Texto de la sentencia o id del corpus.

    Returns:
        Dict JSON-serializable del producto, con tiempo_segundos anadido.
    """
    inicio = time.time()
    if modo == "ID del corpus":
        texto_crudo, texto_limpio = modulo.texto_por_id(valor)
        id_entrada = str(valor)
    else:
        texto_crudo = valor
        texto_limpio = valor
        id_entrada = None
    salida = modulo.construir_salida(texto_crudo, texto_limpio, id_entrada, componentes)
    salida["tiempo_segundos"] = round(time.time() - inicio, 2)
    return salida


def main() -> None:
    """Construye la interfaz Streamlit de la demo."""
    st.set_page_config(page_title="Juris-PLN", layout="wide")
    st.title(CONFIG_DEMO["titulo"])
    st.caption(CONFIG_DEMO["descripcion"])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        usar_reranker = st.toggle(
            "Ruta --rerank (v2.2: B2 + B1; mas lenta)",
            value=False,
            help=(
                "Activa el retriever denso B2 y el reranker de dominio B1. "
                "Sin el toggle usa BM25 + prior de hubs (ruta default del producto)."
            ),
        )
    with col_b:
        modo = st.radio("Entrada", ["Texto", "ID del corpus"], horizontal=True)

    if modo == "Texto":
        valor = st.text_area(
            "Pega el texto de una sentencia (T-, C- o SU-)",
            height=240,
            placeholder="Sentencia T-...",
        )
    else:
        valor = st.text_input("ID del corpus (1-23742)", value=CONFIG_DEMO["ids_ejemplo"][0])
        st.caption("Ejemplos: " + ", ".join(CONFIG_DEMO["ids_ejemplo"]))

    if st.button("Evaluar", type="primary"):
        if not valor or not str(valor).strip():
            st.warning("La entrada esta vacia.")
        else:
            try:
                with st.spinner("Inicializando componentes y evaluando..."):
                    modulo, componentes = componentes_cache(usar_reranker)
                    salida = evaluar_entrada(modulo, componentes, modo, str(valor).strip())
            except SystemExit as exc:
                st.error(f"Error de entrada: {exc.code if exc.code else exc}")
            except Exception as exc:  # noqa: BLE001 - la demo muestra el traceback completo
                st.exception(exc)
            else:
                st.success(
                    f"Evaluacion completa en {salida.get('tiempo_segundos')} s | "
                    f"es_sentencia: {salida.get('es_sentencia')}"
                )
                st.json(salida)

    with st.expander("Como leer la salida"):
        st.markdown(
            "- `es_sentencia` / `score_validacion`: validador M0.\n"
            "- `area`: area tematica M4 (con probabilidades).\n"
            "- `articulos_cp`: articulos vinculados (NER M2 + recuperacion M7; "
            "`reranker_usado`, `hubs_usado`, `alpha_hubs`).\n"
            "- `sentencias_citadas`: contexto M5 (PageRank y area).\n"
            "- `resumen`: sintesis breve del resultado."
        )


if __name__ == "__main__":
    main()
