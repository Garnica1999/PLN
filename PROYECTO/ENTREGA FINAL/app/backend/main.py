"""main.py
Fecha: 2026-10-31
Descripcion: API HTTP del producto Juris-PLN (Entrega Final) sobre FastAPI.
    Expone salud y estado de componentes, catalogo de modelos de ollama,
    validacion M0, vinculo sentencia-articulos CP (M2+M7) con area M4,
    citadas M5 y resumen, detalle de un articulo, jurisprudencia por
    articulo (P2), evidencia por articulo (P1), sentencias similares (P5),
    la consulta conversacional con guardrails (/v1/ask) y un resumen de
    metricas clave del producto con la comparativa de rutas y los guardrails
    del agente leidos de los JSON congelados de la SEGUNDA ENTREGA. Los
    endpoints son sincronos: FastAPI los ejecuta en su threadpool, lo que
    evita bloquear el event loop con BM25, TF-IDF o con el LLM.
Estado del arte aplicado: contrato OpenAPI tipado, errores 400/404/422/500/503,
    CORS para el frontend y lectura defensiva de metricas congeladas.
"""

import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import agent
import core
import jurisprudencia
import similares
from schemas import (
    ArticleResponse,
    AskRequest,
    AskResponse,
    EvalSummaryResponse,
    EvidenciaRequest,
    EvidenciaResponse,
    HealthResponse,
    JurisprudenciaResponse,
    LinkRequest,
    LinkResponse,
    ModelsResponse,
    SimilarResponse,
    ValidateRequest,
    ValidateResponse,
)

VERSION = "1.0.0"
LOG = logging.getLogger("juris_pln.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="Juris-PLN API",
    version=VERSION,
    description=(
        "Producto de investigacion juridica sobre 26.072 sentencias de la "
        "Corte Constitucional de Colombia (1992-2026): validacion de "
        "providencias (M0), articulos de la Constitucion vinculados (M2+M7) "
        "y contexto jurisprudencial (M5), con un agente de tool calling y "
        "verificacion de grounding."
    ),
)
# Origenes permitidos: por defecto solo el frontend local. Si la demo se
# sirve desde otra direccion, se declara en JURIS_CORS_ORIGINS (separada por
# comas); nunca se abre a cualquier origen.
ORIGENES_CORS = [
    origen.strip()
    for origen in os.getenv(
        "JURIS_CORS_ORIGINS",
        "http://127.0.0.1:8502,http://localhost:8502,"
        "http://127.0.0.1:8501,http://localhost:8501",
    ).split(",")
    if origen.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENES_CORS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ===== SECCION 1: manejo de errores =====
@app.exception_handler(LookupError)
async def _manejar_no_encontrado(_request, exc: LookupError) -> JSONResponse:
    """Traduce una referencia inexistente a 404."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def _manejar_entrada_invalida(_request, exc: ValueError) -> JSONResponse:
    """Traduce una entrada invalida a 400."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _manejar_error_interno(request, exc: Exception) -> JSONResponse:
    """Registra el traceback y devuelve 500 sin filtrar detalles internos."""
    LOG.exception("Error no controlado en %s", request.url.path)
    return JSONResponse(
        status_code=500, content={"detail": "Error interno del servidor."}
    )


# ===== SECCION 2: salud y catalogo =====
@app.get("/health", response_model=HealthResponse, tags=["sistema"])
def health() -> dict:
    """Estado del servicio: artefactos, componentes cargados y ollama."""
    estado = core.estado()
    return {
        "status": "ok",
        "ready": estado["ready"],
        "cargados": estado["cargados"],
        "rutas": estado["rutas"],
        "rutas_faltantes": estado["rutas_faltantes"],
        "ollama": agent.ollama_disponible(),
        "modelo_defecto": agent.MODELO_DEFECTO,
        "version": VERSION,
    }


@app.get("/v1/models", response_model=ModelsResponse, tags=["sistema"])
def models() -> dict:
    """Modelos de ollama disponibles y su soporte estimado de tool calling."""
    return agent.listar_modelos()


# ===== SECCION 3: componentes del producto =====
@app.post("/v1/validate", response_model=ValidateResponse, tags=["producto"])
def validate(request: ValidateRequest) -> dict:
    """Valida si un texto (o un documento del corpus) es una providencia."""
    texto = request.texto
    if texto is None:
        texto = core.resolver_documento(request.id)["texto"]
    return core.validar(texto)


@app.post("/v1/link", response_model=LinkResponse, tags=["producto"])
def link(request: LinkRequest) -> dict:
    """Vincula una sentencia con articulos de la Constitucion Politica."""
    referencia = request.texto if request.texto is not None else request.id
    return core.vincular(referencia, k=request.k, rerank=request.rerank)


@app.get("/v1/article/{numero}", response_model=ArticleResponse, tags=["producto"])
def article(numero: str) -> dict:
    """Devuelve el texto recortado y el bloque tematico de un articulo CP."""
    return core.detalle_articulo(numero)


@app.get(
    "/v1/article/{numero}/jurisprudencia",
    response_model=JurisprudenciaResponse,
    tags=["producto"],
)
def article_jurisprudencia(
    numero: str,
    top: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Sentencias devueltas en el ranking por PageRank.",
    ),
) -> dict:
    """Sentencias del corpus que citan un articulo de la CP (P2)."""
    return jurisprudencia.por_articulo(numero, top=top)


@app.post("/v1/evidencia", response_model=EvidenciaResponse, tags=["producto"])
def evidencia(request: EvidenciaRequest) -> dict:
    """Fragmentos de la sentencia con mayor score hacia cada articulo (P1)."""
    referencia = request.texto if request.texto is not None else request.id
    return core.evidencia(
        referencia,
        request.articulos,
        rerank=request.rerank,
        max_fragmentos=request.max_fragmentos,
    )


@app.get("/v1/similar/{id}", response_model=SimilarResponse, tags=["producto"])
def similar(
    id: str,
    k: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Sentencias vecinas devueltas por similitud TF-IDF.",
    ),
) -> dict:
    """Sentencias del corpus mas parecidas a una dada (P5).

    La primera consulta puede construir el indice TF-IDF (hasta ~3 min); las
    siguientes salen de la cache en disco y responden en milisegundos.
    """
    try:
        return similares.buscar(id, k=k)
    except RuntimeError as exc:
        LOG.exception("Indice de similitud no disponible para el id %s", id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/ask", response_model=AskResponse, tags=["agente"])
def ask(request: AskRequest) -> dict:
    """Consulta conversacional con router, herramientas y grounding."""
    return agent.responder(
        request.pregunta,
        modelo=request.model,
        rerank=request.rerank,
        k=request.k,
        pensar=request.pensar,
    )


# ===== SECCION 4: metricas clave =====
RESULTADOS_DIR = core.SEGUNDA_ENTREGA_DIR / "resultados"
ARCHIVO_METRICAS = RESULTADOS_DIR / "metricas.json"
ARCHIVO_M0 = RESULTADOS_DIR / "m0_validacion.json"

# metrica -> (archivo, ruta con puntos). Se leen en modo defensivo: si una
# clave no existe se devuelve None, nunca se fabrica un valor.
EXTRACCIONES: dict[str, tuple[Path, str]] = {
    "m0_accuracy": (ARCHIVO_M0, "accuracy"),
    "m4_macro_f1_test": (
        ARCHIVO_METRICAS,
        "auditoria_etiquetas_m4.reproduccion_v1.test_macro_f1_referencia",
    ),
    "m4_accuracy_test": (
        ARCHIVO_METRICAS,
        "auditoria_etiquetas_m4.reproduccion_v1.test_accuracy_referencia",
    ),
    "m7_ndcg10_bm25_hubs": (
        ARCHIVO_METRICAS,
        (
            "ciclo_v2.w_faseW_a_iter2.full_90_alpha0_vs_alpha_elegido."
            "bm25.full_alpha_elegido.ndcg_at_10"
        ),
    ),
    "m7_ndcg10_rerank_v2_2": (
        ARCHIVO_METRICAS,
        "b2_retriever.evaluacion_v13.con_hubs.ndcg_at_10",
    ),
    "m7_ndcg10_gold_humano": (
        ARCHIVO_METRICAS,
        (
            "b3_gold_humano.evaluacion_gold_humano.rutas_completo."
            "b2_v13.ndcg_at_10.media"
        ),
    ),
    "c3_ndcg10_temporal": (
        ARCHIVO_METRICAS,
        "c3_deriva_temporal.metricas_c3.ndcg_at_10.media",
    ),
}

# Comparativa de rutas de recuperacion (P3). Los valores salen siempre del
# JSON congelado; aqui solo se declaran la etiqueta y la clave.
COMPARATIVA_RUTAS: tuple[tuple[str, str], ...] = (
    (
        "BM25 con prior de hubs (referencia)",
        "ciclo_v3.a1_diagnostico.ic_bootstrap_95_con_hubs.bm25.ndcg_at_10.media",
    ),
    (
        "v2.2, retriever B2 + reranker B1",
        "b2_retriever.evaluacion_v13.con_hubs.ndcg_at_10",
    ),
    (
        "v2.2 con gold humano (B3)",
        (
            "b3_gold_humano.evaluacion_gold_humano.rutas_completo."
            "b2_v13.ndcg_at_10.media"
        ),
    ),
    (
        "C3, corpus 2022-2026",
        "c3_deriva_temporal.metricas_c3.ndcg_at_10.media",
    ),
)

GUARDRAILS_TASAS: tuple[tuple[str, str, str], ...] = (
    (
        "ruteo_exacto",
        "Ruteo exacto de intención",
        "agente_guardrails_eval.metricas.ruteo_exacto",
    ),
    (
        "saludos_no_rechazados",
        "Saludos no rechazados",
        "agente_guardrails_eval.metricas.saludos_no_rechazados",
    ),
    (
        "rechazo_correcto_fuera_alcance",
        "Rechazo correcto fuera de alcance",
        "agente_guardrails_eval.metricas.rechazo_correcto_fuera_alcance",
    ),
    (
        "adversarios_contenidos",
        "Adversarios contenidos",
        "agente_guardrails_eval.metricas.adversarios_contenidos",
    ),
    (
        "tool_call_en_alcance",
        "Tool call en consultas de alcance",
        "agente_guardrails_eval.metricas.tool_call_en_alcance",
    ),
    (
        "grounding_ok_global",
        "Grounding verificado (global)",
        "agente_guardrails_eval.metricas.grounding_ok_global",
    ),
)

GUARDRAILS_LATENCIA: tuple[tuple[str, str], ...] = (
    ("p50", "Latencia p50"),
    ("p95", "Latencia p95"),
    ("max", "Latencia máxima"),
)

RUTA_LATENCIA_AGENTE = (
    "agente_guardrails_eval.metricas.latencia_ms_ruta_agente"
)


@lru_cache(maxsize=8)
def _leer_json_por_marca(ruta: str, _marca: int) -> dict:
    """Lee un JSON; la marca de modificacion invalida la entrada cacheada."""
    try:
        with open(ruta, encoding="utf-8") as archivo:
            datos = json.load(archivo)
    except (OSError, ValueError):
        return {}
    return datos if isinstance(datos, dict) else {}


def _leer_json(ruta: Path) -> dict:
    """Lee un JSON y devuelve {} si no existe o esta corrupto.

    El contenido se cachea por marca de modificacion: /v1/eval/summary no
    vuelve a parsear los 229 KB de metricas.json en cada peticion.
    """
    try:
        marca = ruta.stat().st_mtime_ns
    except OSError:
        return {}
    return _leer_json_por_marca(str(ruta), marca)


def _buscar(datos: dict, ruta: str) -> Any:
    """Recorre un dict por una ruta separada por puntos."""
    actual: Any = datos
    for clave in ruta.split("."):
        if not isinstance(actual, dict) or clave not in actual:
            return None
        actual = actual[clave]
    return actual


def _numero(valor: Any) -> float | None:
    """Convierte un valor numerico a float redondeado, o None."""
    if isinstance(valor, (int, float)) and not isinstance(valor, bool):
        return round(float(valor), 4)
    return None


def _procedencia(ruta: str) -> str:
    """Arma la procedencia legible de una clave del JSON de metricas."""
    archivo = str(ARCHIVO_METRICAS.relative_to(core.PROYECTO_DIR))
    return f"{archivo}:{ruta}"


def _comparativa(datos: dict) -> list[dict]:
    """Lee la tabla comparativa de rutas del JSON congelado."""
    return [
        {
            "ruta": etiqueta,
            "metrica": "nDCG@10",
            "valor": _numero(_buscar(datos, ruta)),
            "procedencia": _procedencia(ruta),
        }
        for etiqueta, ruta in COMPARATIVA_RUTAS
    ]


def _guardrails(datos: dict) -> dict:
    """Lee la evaluacion del agente con guardrails y su procedencia."""
    metricas = [
        {
            "clave": clave,
            "etiqueta": etiqueta,
            "valor": _numero(_buscar(datos, ruta)),
            "unidad": "tasa",
        }
        for clave, etiqueta, ruta in GUARDRAILS_TASAS
    ]
    latencias = _buscar(datos, RUTA_LATENCIA_AGENTE)
    latencia = [
        {
            "clave": clave,
            "etiqueta": etiqueta,
            "valor": _numero(
                latencias.get(clave) if isinstance(latencias, dict) else None
            ),
            "unidad": "ms",
        }
        for clave, etiqueta in GUARDRAILS_LATENCIA
    ]
    modelo = _buscar(datos, "agente_guardrails_eval.modelo")
    etiquetas = _buscar(datos, "agente_guardrails_eval.procedencia_etiquetas")
    aviso = (
        f"Evaluacion con {modelo}. " if modelo else "Evaluacion del agente. "
    )
    aviso += (
        str(etiquetas)
        if etiquetas
        else (
            "Las rutas esperadas son etiquetas automaticas del contrato; no "
            "son anotacion humana verificada."
        )
    )
    return {
        "metricas": metricas,
        "latencia": latencia,
        "procedencia": _procedencia("agente_guardrails_eval.metricas"),
        "aviso": aviso,
    }


def _avisos_procedencia(datos: dict) -> list[dict]:
    """Resume la procedencia de las etiquetas y de los gold sets."""
    avisos: list[dict] = []
    humano = _buscar(datos, "b3_gold_humano.anotacion_humana")
    if isinstance(humano, dict):
        avisos.append(
            {
                "tema": "Gold humano",
                "texto": (
                    f"B3: {humano.get('n_casos')} casos y "
                    f"{humano.get('n_grados')} candidatos calificados a mano "
                    f"por {humano.get('anotador')}. Es la unica anotacion "
                    "humana del benchmark M7."
                ),
            }
        )
    llm = _buscar(datos, "b3_gold_llm.preanotacion_llm")
    if isinstance(llm, dict):
        avisos.append(
            {
                "tema": "Gold-LLM (proxy)",
                "texto": (
                    f"B3: {llm.get('n_casos_muestra')} casos y "
                    f"{llm.get('n_candidatos_muestra')} candidatos "
                    f"pre-anotados por {llm.get('modelo')}. Es un proxy "
                    "automatico, no anotacion humana."
                ),
            }
        )
    agente = _buscar(datos, "agente_guardrails_eval.procedencia_etiquetas")
    if agente:
        avisos.append({"tema": "Etiquetas del agente", "texto": str(agente)})
    avisos.append(
        {
            "tema": "Etiquetas del producto",
            "texto": (
                "El area tematica (M4) y el area de las citadas (M5) son "
                "etiquetas automaticas sobre datos silver; no son una "
                "clasificacion verificada por humanos (ADR-19)."
            ),
        }
    )
    return avisos


@app.get("/v1/eval/summary", response_model=EvalSummaryResponse, tags=["sistema"])
def eval_summary() -> dict:
    """Metricas clave del producto con su procedencia exacta."""
    cache: dict[Path, dict] = {}
    metricas: dict[str, float | None] = {}
    procedencia: dict[str, str] = {}
    for nombre, (archivo, ruta) in EXTRACCIONES.items():
        if archivo not in cache:
            cache[archivo] = _leer_json(archivo)
        valor = _buscar(cache[archivo], ruta)
        metricas[nombre] = (
            round(float(valor), 4)
            if isinstance(valor, (int, float)) and not isinstance(valor, bool)
            else None
        )
        procedencia[nombre] = f"{archivo.relative_to(core.PROYECTO_DIR)}:{ruta}"
    existentes = [ruta for ruta in cache if ruta.exists()]
    actualizado = None
    if existentes:
        marca = max(ruta.stat().st_mtime for ruta in existentes)
        actualizado = (
            datetime.fromtimestamp(marca).astimezone().isoformat(timespec="seconds")
        )
    metricas_json = cache.get(ARCHIVO_METRICAS, {})
    latencias = _buscar(metricas_json, "qa_v2_2.latencias_s")
    return {
        "fuentes": [
            str(ruta.relative_to(core.PROYECTO_DIR)) for ruta in sorted(cache)
        ],
        "actualizado": actualizado,
        "metricas": metricas,
        "procedencia": procedencia,
        "latencias_s": latencias if isinstance(latencias, dict) else {},
        "comparativa_rutas": _comparativa(metricas_json),
        "agente_guardrails": _guardrails(metricas_json),
        "avisos_procedencia": _avisos_procedencia(metricas_json),
    }
