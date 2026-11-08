"""schemas.py
Fecha: 2026-10-31
Descripcion: Modelos Pydantic de request/response de la API Juris-PLN.
    Los requests validan la forma de la entrada (uno de texto/id, limites de
    k, longitudes) y los responses documentan el contrato que consume el
    frontend; los payloads internos se declaran con tipos explicitos para
    que OpenAPI sea util.
Estado del arte aplicado: contrato tipado con Pydantic v2 (validacion y
    serializacion), Literal para enumeraciones y Field para limites.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ===== SECCION 1: requests =====
class ValidateRequest(BaseModel):
    """Entrada de /v1/validate: texto literal o referencia del corpus."""

    model_config = ConfigDict(extra="forbid")

    texto: str | None = Field(
        default=None,
        min_length=1,
        max_length=200_000,
        description="Texto completo de la providencia a validar.",
    )
    id: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description=(
            "Id del corpus o canon de la sentencia (por ejemplo 2195 o "
            "T-293/97). Alternativo a 'texto'."
        ),
    )

    @model_validator(mode="after")
    def _exactamente_una_entrada(self) -> "ValidateRequest":
        """Exige exactamente uno de texto/id."""
        if bool(self.texto) == bool(self.id):
            raise ValueError("Indique exactamente uno de 'texto' o 'id'.")
        return self


class LinkRequest(BaseModel):
    """Entrada de /v1/link: referencia o texto y parametros de recuperacion."""

    model_config = ConfigDict(extra="forbid")

    texto: str | None = Field(
        default=None,
        min_length=3,
        max_length=200_000,
        description="Texto de la sentencia a analizar.",
    )
    id: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description="Id del corpus o canon de la sentencia (T-293/97).",
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Numero de articulos de la CP a recuperar.",
    )
    rerank: bool = Field(
        default=False,
        description=(
            "True activa la ruta hibrida B2+B1 (GPU, ~14 s). False usa BM25 "
            "con prior de hubs (CPU, ~5-9 s)."
        ),
    )

    @model_validator(mode="after")
    def _exactamente_una_entrada(self) -> "LinkRequest":
        """Exige exactamente uno de texto/id."""
        if bool(self.texto) == bool(self.id):
            raise ValueError("Indique exactamente uno de 'texto' o 'id'.")
        return self


class AskRequest(BaseModel):
    """Entrada de /v1/ask: pregunta en lenguaje natural."""

    model_config = ConfigDict(extra="forbid")

    pregunta: str = Field(
        min_length=2,
        max_length=4000,
        description="Pregunta del usuario sobre jurisprudencia constitucional.",
    )
    model: str | None = Field(
        default=None,
        max_length=120,
        description=(
            "Modelo de ollama a usar. Si se omite se usa el modelo por "
            "defecto del backend."
        ),
    )
    rerank: bool = Field(
        default=False,
        description="Activa la ruta de recuperacion B2+B1 en las herramientas.",
    )
    k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Articulos de la CP que recupera la herramienta de vinculo.",
    )
    pensar: bool = Field(
        default=False,
        description=(
            "Habilita el razonamiento nativo del modelo. Amplia el presupuesto "
            "de salida y devuelve el pensamiento en 'pensamiento'; la respuesta "
            "tarda mas y consume mas contexto."
        ),
    )


# ===== SECCION 2: respuestas de producto =====
class ReglasM0(BaseModel):
    """Detalle del cumplimiento de las tres reglas de la heuristica M0."""

    encabezado: bool
    marcador_corte: bool
    marcador_ponente: bool


class ValidateResponse(BaseModel):
    """Salida de /v1/validate."""

    es_sentencia: bool
    score_validacion: float = Field(ge=0.0, le=1.0)
    reglas: ReglasM0 | None = None
    canon_detectado: str | None = None
    id_detectado: str | None = None


class Articulo(BaseModel):
    """Articulo de la Constitucion Politica con su texto recortado."""

    numero: str
    score: float | None = None
    texto_150chars: str | None = None
    bloque: str | None = None


class CitaResuelta(BaseModel):
    """Sentencia citada resuelta contra la red de citas (M5)."""

    canon: str
    id: str
    resuelta: bool
    ano: int | None = None
    pagerank: float | None = None
    area: str | None = None


class AreaItem(BaseModel):
    """Area tematica del clasificador M4 con su probabilidad."""

    area: str
    prob: float


class AreaM4(BaseModel):
    """Top-3 del clasificador M4, rotulado como exploratorio (ADR-19)."""

    top3: list[AreaItem] = Field(default_factory=list)
    via: str
    exploratoria: bool = True


class LinkResponse(BaseModel):
    """Salida de /v1/link."""

    id: str | None = None
    canon_detectado: str | None = None
    origen: Literal["id", "canon", "texto"]
    area: AreaM4 | None = None
    articulos_extraidos: list[Articulo] = Field(default_factory=list)
    articulos_recuperados: list[Articulo] = Field(default_factory=list)
    sentencias_citadas: list[CitaResuelta] = Field(default_factory=list)
    resumen: str
    rerank: bool
    k: int
    ms: int


class ArticleResponse(BaseModel):
    """Salida de /v1/article/{numero}."""

    numero: str
    texto_150chars: str
    bloque: str | None = None


class ContextoResponse(BaseModel):
    """Salida del contexto jurisprudencial (usada por las herramientas)."""

    id: str | None = None
    canon_detectado: str | None = None
    citas: list[CitaResuelta] = Field(default_factory=list)
    n_citas: int
    n_resueltas: int


# ===== SECCION 3: jurisprudencia por articulo y evidencia =====
class JurisprudenciaTop(BaseModel):
    """Sentencia del corpus que cita un articulo, ordenada por PageRank."""

    id: str
    canon: str
    ano: int
    tipo: str
    pagerank: float | None = None
    area_silver: str | None = None


class JurisprudenciaResponse(BaseModel):
    """Salida de /v1/article/{numero}/jurisprudencia (P2)."""

    numero: str
    total: int
    por_ano: dict[str, int] = Field(default_factory=dict)
    top: list[JurisprudenciaTop] = Field(default_factory=list)
    aviso: str


class EvidenciaRequest(BaseModel):
    """Entrada de /v1/evidencia: sentencia y articulos a evidenciar (P1)."""

    model_config = ConfigDict(extra="forbid")

    texto: str | None = Field(
        default=None,
        min_length=3,
        max_length=200_000,
        description="Texto de la sentencia analizada. Alternativo a 'id'.",
    )
    id: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description="Id del corpus o canon de la sentencia (T-293/97).",
    )
    articulos: list[str] = Field(
        min_length=1,
        max_length=5,
        description='Articulos CP a evidenciar, por ejemplo ["86", "13"].',
    )
    rerank: bool = Field(
        default=False,
        description="Repuntua los fragmentos con la ruta B2+B1 (GPU).",
    )
    max_fragmentos: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Fragmentos devueltos por articulo.",
    )

    @model_validator(mode="after")
    def _exactamente_una_entrada(self) -> "EvidenciaRequest":
        """Exige exactamente uno de texto/id."""
        if bool(self.texto) == bool(self.id):
            raise ValueError("Indique exactamente uno de 'texto' o 'id'.")
        return self


class Fragmento(BaseModel):
    """Chunk de la sentencia con su score hacia un articulo de la CP."""

    texto: str
    score: float


class EvidenciaArticulo(BaseModel):
    """Fragmentos mejor puntuados de una sentencia para un articulo."""

    numero: str
    fragmentos: list[Fragmento] = Field(default_factory=list)


class EvidenciaResponse(BaseModel):
    """Salida de /v1/evidencia."""

    id: str | None = None
    canon_detectado: str | None = None
    articulos: list[EvidenciaArticulo] = Field(default_factory=list)
    rerank: bool
    ms: int


# ===== SECCION 4: modelos y salud =====
class ModelInfo(BaseModel):
    """Modelo de ollama disponible y su soporte estimado de tool calling."""

    nombre: str
    familia: str | None = None
    parametros: str | None = None
    tamano_gb: float | None = None
    tool_calling: bool
    defecto: bool


class ModelsResponse(BaseModel):
    """Salida de /v1/models."""

    disponible: bool
    defecto: str
    modelos: list[ModelInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Salida de /health."""

    status: str
    ready: bool
    cargados: dict[str, bool]
    rutas: dict[str, bool]
    rutas_faltantes: list[str] = Field(default_factory=list)
    ollama: bool
    modelo_defecto: str
    version: str


# ===== SECCION 5: agente =====
class ToolCallTraza(BaseModel):
    """Traza de una llamada a herramienta ejecutada por el agente."""

    herramienta: str
    argumentos: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    ms: int
    resumen: str


class Grounding(BaseModel):
    """Verificacion de que las citas de la respuesta existen en la evidencia."""

    ok: bool
    citas_verificadas: list[str] = Field(default_factory=list)
    citas_no_verificadas: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    """Salida de /v1/ask."""

    respuesta: str
    modo: Literal[
        "agente",
        "fallback_sin_tools",
        "sin_evidencia",
        "small_talk",
        "fuera_alcance",
        "aclaracion",
    ]
    intencion: str
    tool_calls: list[ToolCallTraza] = Field(default_factory=list)
    grounding: Grounding
    pensamiento: str | None = None
    modelo: str
    latencia_ms: int
    advertencia: str | None = None


# ===== SECCION 6: resumen de evaluacion =====
class MetricaGuardrail(BaseModel):
    """Tasa o latencia de la evaluacion del agente con su etiqueta."""

    clave: str
    etiqueta: str
    valor: float | None = None
    unidad: str


class AgenteGuardrails(BaseModel):
    """Seccion agente_guardrails de las metricas congeladas."""

    metricas: list[MetricaGuardrail] = Field(default_factory=list)
    latencia: list[MetricaGuardrail] = Field(default_factory=list)
    procedencia: str
    aviso: str


class RutaComparada(BaseModel):
    """Ruta de recuperacion con su nDCG@10 y la procedencia del valor."""

    ruta: str
    metrica: str
    valor: float | None = None
    procedencia: str


class AvisoProcedencia(BaseModel):
    """Aviso sobre el origen de una cifra o de una etiqueta."""

    tema: str
    texto: str


class EvalSummaryResponse(BaseModel):
    """Salida de /v1/eval/summary (metricas clave del producto)."""

    fuentes: list[str] = Field(default_factory=list)
    actualizado: str | None = None
    metricas: dict[str, float | None] = Field(default_factory=dict)
    procedencia: dict[str, str] = Field(default_factory=dict)
    latencias_s: dict[str, Any] = Field(default_factory=dict)
    comparativa_rutas: list[RutaComparada] = Field(default_factory=list)
    agente_guardrails: AgenteGuardrails | None = None
    avisos_procedencia: list[AvisoProcedencia] = Field(default_factory=list)


# ===== SECCION 7: sentencias similares (P5) =====
class SimilarItem(BaseModel):
    """Sentencia del corpus parecida a la consultada."""

    id: str
    canon: str
    ano: int
    tipo: str
    pagerank: float | None = None
    score: float


class SimilarResponse(BaseModel):
    """Salida de /v1/similar/{id}."""

    id: str
    canon: str | None = None
    similares: list[SimilarItem] = Field(default_factory=list)
    k: int
    ms: int
    aviso: str
