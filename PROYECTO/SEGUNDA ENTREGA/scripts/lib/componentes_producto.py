"""componentes_producto.py
Fecha: 2026-09-09
Descripcion: Componentes reutilizables del producto final Juris-PLN (Fase N).
    Cada clase envuelve un modulo ya entrenado/validado del proyecto sin
    duplicar su logica: ValidadorEntrada (M0, port de
    scripts/lib/validacion_entrada.py), ExtractorEntidades (M2, importlib de
    scripts/03_extraccion_entidades.py), ClasificadorArea (M4, TF-IDF de lemas
    enmascarados + LogReg de modelos/m4_baseline), VinculoCP (M7 v2, BM25 por
    pasajes con max-pooling de chunks, port de scripts/12_evaluar_vinculo_cp_v2.py
    con cache del indice en data/cache_bm25_cp/) y ContextoJurisprudencial
    (M5, citas resueltas al corpus + PageRank + area desde
    data/red_citas_resumen.parquet).
Estado del arte aplicado: heuristica lexica M0 (accuracy 1.0 en 52 casos),
    NER regex M2 (precision muestral >= 0.90), TF-IDF de lemas enmascarados
    con LogReg multinomial (macro-F1 0.287), BM25 por pasajes con bigramas
    LLR y stopwords juridicas (F1@5 0.218) y PageRank de la red de citas.

ciclo v2 - Fase S1 (2026-09-10, plan/PLAN_v2_ciclo_mejora.md): limpieza del
    indice CP en VinculoCP. El diagnostico de la Fase R
    (resultados/vinculo_cp_benchmark_v3.json) midio 24.4% de transitorios en
    top-5 y 152 IDs malformados en top-10; el indice BM25 se construye ahora
    por defecto SOLO sobre articulos permanentes (sin transitorios ni
    preambulo), con modo opcional incluir_transitorios=True para casos que
    citan disposiciones transitorias legitimas. recuperar() pasa de top-5 a
    top-10 y el cache del indice se versiona a v3
    (data/cache_bm25_cp/indice_bm25_cp_v3.joblib; el v2 del indice sucio NO
    se sobrescribe, ADR-11). El parquet fuente ya viene con IDs canonicos
    corregidos por 02_parseo_constitucion.py (misma Fase S1).

ciclo v2 - Fase S2 (2026-09-10, plan/PLAN_v2_ciclo_mejora.md): lectura
    selectiva del corpus de sentencias con pyarrow.dataset en
    ContextoJurisprudencial (mapa canon -> id construido desde las
    columnas ligeras id/tipo/numero/ano con particionado hive, sin
    materializar los textos completos). La lectura de la fila del id
    pedido por el CLI se optimiza en scripts/10_evaluar_sentencia.py
    (texto_por_id con filter de pyarrow). Motivo: la auditoria de
    rendimiento v1 (resultados/qa_perf/AUDITORIA_RENDIMIENTO.md) midio
    que materializar el corpus completo con pandas era el cuello de
    botella (~75% del tiempo de inicializacion, ~8 GB de RAM); con
    filters y column pruning la RAM baja a ~2 GB. La red de citas
    (data/red_citas_resumen.parquet) es pequena y sigue cargandose
    completa. La salida JSON del producto no cambia.

ciclo v2 - Fase T (2026-09-10, plan/PLAN_v2_ciclo_mejora.md): modo
    reranker OPCIONAL en VinculoCP (usar_reranker=False por defecto: el
    producto base NO cambia su comportamiento). Cuando se activa, tras
    el BM25 top-20 el cross-encoder BAAI/bge-reranker-v2-m3 (scripts/
    lib/reranker_bge.py, misma logica que el benchmark) puntua cada par
    (query, pasaje) y reordena: recuperar() devuelve el top-10
    rerankeado en el MISMO campo recuperados_bm25 y el JSON del producto
    anade reranker_usado (true/false). La carga del reranker es lazy
    (solo con usar_reranker=True) para no penalizar el init del producto
    base; GPU con fp16 si hay >= 6 GiB de VRAM libre, CPU batch 4 si no
    (WARNING documentado, no bloquea).

ciclo v2 - Fase X (producto v2, 2026-09-11, plan/PLAN_v2_ciclo_mejora.md):
    el producto adopta las mejoras GANADORAS del ciclo v2 con evidencia en
    el benchmark v8 (resultados/vinculo_cp_benchmark_v8.json,
    v8-ciclo2-faseW-a-iter2):
    (1) prior de hubs CP (Fase W-a iter2): al ranking base se le suma
        alpha * prior(canon) sobre scores min-max normalizados; alpha=0.75
        en la ruta BM25 por defecto (F1@5 0.233 -> 0.396) y alpha=1.5 en
        la ruta reranker (F1@5 0.340 -> 0.425 de la ganadora). El prior es
        un archivo estatico de frecuencia global de citacion
        (data/prior_hubs_cp.json, no-leakage por diseno); su carga es
        tolerante: si falta, se desactiva con WARNING documentado.
    (2) ruta reranker = hibrido_rerank_chunks_top50 (ganadora v6/v8):
        candidatos top-50 del hibrido 0.5*minmax(BM25) + 0.5*minmax(E5)
        sobre TODOS los articulos permanentes; E5 =
        intfloat/multilingual-e5-base (scripts/12_evaluar_vinculo_cp_v2.py)
        con embeddings de pasajes de la CP cacheados en
        data/cache_e5_cp/ (nueva cache) y embedding de chunks de query en
        runtime; rerank por chunks (mejor pasaje BM25 + top-3 chunks ->
        3 pares al cross-encoder BAAI/bge-reranker-v2-m3 -> max-pooling)
        y prior alpha=1.5 -> top-10. La carga de E5 y del reranker es LAZY
        (solo con usar_reranker=True); GPU si torch.cuda.is_available()
        con >= 2 GiB libres para E5 y >= 6 GiB para el reranker, CPU con
        WARNING si no (documentado). FALLBACK documentado: si E5 falla o
        no esta disponible, los candidatos son el top-50 BM25 (con
        WARNING) y el resto del pipeline no cambia.
    ClasificadorArea (M4) queda SIN cambios (gates de Fase U/V no
    cumplidos): LogReg v1 + TF-IDF de lemas enmascarados del joblib.
"""

import importlib.util
import itertools
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from rank_bm25 import BM25Okapi

CONFIG = {
    "ventana_cp": 60,
    "zona_encabezado_chars": 3000,
    "k_bm25_default": 10,
    "chunk_bm25_chars": 800,
    "solape_bm25_chars": 200,
    "pasaje_max_palabras": 120,
    "modelo_spacy": "es_core_news_md",
    "spacy_max_length": 5_000_000,
    "ruta_validacion": "scripts/lib/validacion_entrada.py",
    "ruta_script_m2": "scripts/03_extraccion_entidades.py",
    "ruta_constitucion": "data/constitucion_articulos.parquet",
    "ruta_red_citas": "data/red_citas_resumen.parquet",
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_tfidf": "modelos/m4_baseline/tfidf_enmascarado.joblib",
    "ruta_logreg": "modelos/m4_baseline/logreg_enmascarado.joblib",
    "dir_cache_bm25": "data/cache_bm25_cp",
    "archivo_cache_bm25": "indice_bm25_cp_v3.joblib",
    "version_cache": 3,
    # --- ciclo v2 - Fase X (producto v2): prior de hubs de la CP ---
    "ruta_prior_hubs": "data/prior_hubs_cp.json",
    "usar_hubs_default": True,
    "alpha_hubs_base": 0.75,
    "alpha_hubs_rerank": 1.5,
    # --- ciclo v2 - Fase X (producto v2): reranker ganador v8 ---
    "k_candidatos_rerank_top50": 50,
    "k_chunks_rerank": 3,
    "pesos_hibrido": {"bm25": 0.5, "e5": 0.5},
    # --- ciclo v2 - Fase X (producto v2): E5 hibrido (lazy) ---
    "modelo_e5": "intfloat/multilingual-e5-base",
    "tamano_chunk": 512,
    "solape_chunk": 64,
    "reserva_prefijo_tokens": 4,
    "prefijo_query": "query: ",
    "prefijo_pasaje": "passage: ",
    "batch_embedding": 32,
    "batch_embedding_oom": 16,
    "vram_minima_gb_e5": 2.0,
    "dir_cache_e5": "data/cache_e5_cp",
    "archivo_cache_e5": "pasajes_e5_base.joblib",
}

ROOT = Path(__file__).resolve().parent.parent.parent

# Stopwords juridicas (port de 12_evaluar_vinculo_cp_v2.py): terminos
# procesales presentes en casi todas las sentencias que no discriminan entre
# articulos de la CP. Los bigramas compuestos (con "_") nunca se filtran.
STOPWORDS_JURIDICAS = frozenset(
    {
        "articulo", "articulos", "ley", "leyes", "numeral", "numerales",
        "inciso", "incisos", "parrafo", "parrafos", "literal", "literales",
        "caso", "casos", "sentencia", "sentencias", "expediente", "bogota",
        "honorable", "magistrado", "magistrados", "ponente", "secretaria",
        "general", "demanda", "demandas", "demandado", "demandados",
        "demandada", "demandadas", "demandante", "demandantes", "dos",
        "mil", "senor", "senora", "senores",
    }
)

# Bigramas LLR ganadores del EDA de PRIMERA ENTREGA (port de 12): top-20 de
# resultados/eda_extendido/colocaciones.csv con exclusiones documentadas
# (fechas, conectivos procesales y nombres propios).
BIGRAMAS_LLR = frozenset(
    {
        ("derechos", "fundamentales"),
        ("corte", "constitucional"),
        ("seguridad", "social"),
        ("constitucion", "politica"),
        ("debido", "proceso"),
        ("derecho", "fundamental"),
        ("minimo", "vital"),
        ("primera", "instancia"),
        ("perjuicio", "irremediable"),
        ("cosa", "juzgada"),
        ("derechos", "humanos"),
        ("corte", "suprema"),
        ("tribunal", "superior"),
        ("segunda", "instancia"),
    }
)

# ===== Enmascarado de lemas (replica de 06_baseline_clasico.py, Fase L2) =====
# Lemas nucleo de citas normativas tras la lematizacion (formas con y sin
# tilde; "politica" excluido por no ser nucleo de cita).
LEMAS_CITA = {
    "articulo", "articulos", "ley", "leyes", "decreto", "decretos",
    "acto", "legislativo", "sentencia", "sentencias", "auto",
    "numeral", "numerales", "inciso", "incisos", "literal", "literales",
    "paragrafo", "constitucion",
}
TOKEN_CITA = "CITA"
# Romanos REALES (regex estricta; evita falsos como civil/mi/mil).
RE_ROMANO = re.compile(
    r"^(?=.{1,7}$)m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)
RE_ARABIGO = re.compile(r"^\d{1,4}$")
# Prefijos de sentencia citada ya lematizados (t, c, su + numero).
RE_SENT_CITADA = re.compile(r"^(?:t|c|su)$", re.IGNORECASE)


# ===== SECCION 1: utilidades base =====
def _cargar_modulo(ruta: Path, nombre: str) -> object:
    """Carga un modulo Python por ruta con importlib.

    Los scripts del proyecto empiezan por digito (03_...) o viven en lib/
    sin paquete, por lo que no admiten import estandar; se cargan con spec
    para reutilizar sus funciones sin duplicar logica.

    Args:
        ruta: Ruta absoluta al archivo .py a cargar.
        nombre: Nombre con el que se registra el modulo.

    Returns:
        Modulo Python ejecutado (solo definiciones si tiene main guard).

    Raises:
        RuntimeError: Si la ruta no existe o el spec no se puede crear.
    """
    if not ruta.is_file():
        raise RuntimeError(f"No se encontro el modulo a cargar: {ruta}")
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo crear spec de importlib para: {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def normalizar_sin_tildes(texto: str) -> str:
    """Convierte un texto a minusculas y sin diacriticos.

    Args:
        texto: Texto original.

    Returns:
        Texto normalizado (minusculas, sin tildes).
    """
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto) if not unicodedata.combining(c)
    )
    return sin_tildes.lower()


def tokenizar(texto: str) -> list[str]:
    """Tokeniza para BM25: unigramas sin stopwords juridicas + bigramas LLR.

    Port de 12_evaluar_vinculo_cp_v2.py: los bigramas se detectan sobre la
    secuencia completa de unigramas y se anaden como tokens "a_b"; despues
    se filtran los unigramas procesales.

    Args:
        texto: Texto a tokenizar.

    Returns:
        Lista de tokens (unigramas utiles + bigramas compuestos).
    """
    tokens = re.findall(r"[a-z0-9]+", normalizar_sin_tildes(texto))
    bigramas = [
        f"{a}_{b}"
        for a, b in itertools.pairwise(tokens)
        if (a, b) in BIGRAMAS_LLR
    ]
    unigramas = [t for t in tokens if t not in STOPWORDS_JURIDICAS]
    return unigramas + bigramas


def dividir_pasajes(texto: str, max_palabras: int) -> list[str]:
    """Segmenta un articulo en pasajes de oraciones completas.

    Port de 12_evaluar_vinculo_cp_v2.py: split tras punto agrupando
    oraciones hasta max_palabras; una oracion mas larga que el limite queda
    como pasaje propio.

    Args:
        texto: Texto del articulo.
        max_palabras: Limite de palabras por pasaje.

    Returns:
        Lista de pasajes (>= 1 para todo texto no vacio).
    """
    oraciones = [o.strip() for o in re.split(r"(?<=\.)\s+", texto) if o.strip()]
    pasajes: list[str] = []
    actual: list[str] = []
    palabras = 0
    for oracion in oraciones:
        largo = len(oracion.split())
        if palabras + largo > max_palabras and actual:
            pasajes.append(" ".join(actual))
            actual, palabras = [], 0
        actual.append(oracion)
        palabras += largo
    if actual:
        pasajes.append(" ".join(actual))
    return pasajes if pasajes else ([texto.strip()] if texto.strip() else [])


def es_numero_cita(token: str) -> bool:
    """Determina si un token de lemas puede ser el numero de una cita.

    Args:
        token: Token (posible arabigo o romano).

    Returns:
        True si es arabigo de 1-4 digitos o romano real.
    """
    return bool(RE_ARABIGO.match(token) or RE_ROMANO.match(token))


def normalizar_lemas(texto: str) -> str:
    """Normaliza lemas multi-palabra a un solo token con guion bajo.

    Replica de 06_baseline_clasico.py: spaCy emite lemas con espacio
    ("hacer el") que deben unirse como "hacer_el" antes de vectorizar.

    Args:
        texto: Secuencia de lemas separados por espacio.

    Returns:
        Secuencia con lemas multi-palabra unidos por "_".
    """
    return " ".join(tok.replace(" ", "_") for tok in texto.split())


def enmascarar_citas_lemas(texto: str) -> str:
    """Enmascara secuencias de lemas tipicas de citas normativas.

    Replica de 06_baseline_clasico.py (version auditada L2): nucleo de cita
    (con/sin tilde, singular/plural) + numero arabigo o romano estricto,
    series con conectores ("articulo 11 y 13") y prefijos de sentencia
    citada (t/c/su + numero [+ de + ano]) se reemplazan por el token CITA.

    Args:
        texto: Secuencia de lemas normalizados (normalizar_lemas previo).

    Returns:
        Secuencia con las citas reemplazadas por CITA.
    """
    tokens = texto.split()
    salida: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        base = tok.split("_")[0]
        if base in LEMAS_CITA:
            j = i + 1
            consumio_numero = False
            while j < len(tokens):
                sig = tokens[j]
                if es_numero_cita(sig):
                    consumio_numero = True
                    j += 1
                elif consumio_numero and sig in {
                    "de", "del", "y", "la", "el", "las", "los",
                }:
                    j += 1
                    if j < len(tokens) and es_numero_cita(tokens[j]):
                        continue
                    j -= 1
                    break
                else:
                    break
            if consumio_numero:
                salida.append(TOKEN_CITA)
                i = j
            else:
                salida.append(tok)
                i += 1
        elif (
            RE_SENT_CITADA.match(base)
            and i + 1 < len(tokens)
            and es_numero_cita(tokens[i + 1])
        ):
            j = i + 2
            if (
                j < len(tokens)
                and tokens[j] == "de"
                and j + 1 < len(tokens)
                and es_numero_cita(tokens[j + 1])
            ):
                j += 2
            salida.append(TOKEN_CITA)
            i = j
        else:
            salida.append(tok)
            i += 1
    return " ".join(salida)


# ===== SECCION 2: M0 validacion de entrada =====
class ValidadorEntrada:
    """M0: decide si un texto es providencia de la Corte Constitucional.

    Envuelve scripts/lib/validacion_entrada.py (heuristica del anexo 25:
    encabezado de sentencia, marcador "corte constitucional" y marcador
    "magistrado ponente"; umbral 2/3 reglas, accuracy 1.0 en 52 casos).
    """

    def __init__(self) -> None:
        """Carga el modulo M0 por importlib (port sin duplicacion)."""
        self._m0 = _cargar_modulo(
            ROOT / CONFIG["ruta_validacion"], "validacion_entrada_producto"
        )

    def validar(self, texto: str) -> dict:
        """Evalua las tres reglas M0 sobre un texto.

        Args:
            texto: Texto completo del documento a validar (crudo).

        Returns:
            Dict con es_sentencia (bool, umbral 2/3), score_validacion
            (reglas cumplidas / total, en [0, 1]) y reglas (detalle del
            cumplimiento por regla; None si el texto es vacio).
        """
        if not isinstance(texto, str) or not texto.strip():
            return {
                "es_sentencia": False,
                "score_validacion": 0.0,
                "reglas": None,
            }
        res = self._m0.evaluar_reglas(texto)
        return {
            "es_sentencia": bool(res["es_sentencia"]),
            "score_validacion": res["reglas_cumplidas"] / res["total_reglas"],
            "reglas": {
                "encabezado": res["encabezado"],
                "marcador_corte": res["marcador_corte"],
                "marcador_ponente": res["marcador_ponente"],
            },
        }


# ===== SECCION 3: M2 extraccion de entidades =====
class ExtractorEntidades:
    """M2: extraccion de entidades juridicas por regex (NER del proyecto).

    Envuelve scripts/03_extraccion_entidades.py via importlib: articulos de
    la CP citados (ventana de constitucionalidad), sentencias citadas en
    canon TIPO-NUM-AAAA y magistrado ponente.
    """

    def __init__(self, ventana_cp: int | None = None) -> None:
        """Carga el modulo M2 y fija la ventana de constitucionalidad.

        Args:
            ventana_cp: Radio en caracteres para vincular articulo y
                constitucion (default CONFIG["ventana_cp"] = 60).
        """
        self._m2 = _cargar_modulo(
            ROOT / CONFIG["ruta_script_m2"], "extraccion_m2_producto"
        )
        self._ventana = ventana_cp or CONFIG["ventana_cp"]

    def extraer(self, texto: str) -> dict:
        """Extrae articulos CP, sentencias citadas y ponente de un texto.

        Args:
            texto: Texto crudo de la sentencia.

        Returns:
            Dict con arts_cp (lista de canones "13"/"22A" unicos en orden),
            sents_citadas (lista de canones "T-293-1997" unicos en orden) y
            ponente (nombre o None).
        """
        contadores: Counter = Counter()
        menciones: Counter = Counter()
        return {
            "arts_cp": self._m2.extraer_arts_cp(texto, self._ventana),
            "sents_citadas": self._m2.extraer_sents(texto, contadores, menciones),
            "ponente": self._m2.extraer_ponente(texto),
        }

    def detectar_canon_encabezado(self, texto: str) -> str | None:
        """Detecta el canon TIPO-NUM-AAAA propio del encabezado.

        Recorre las citas del M2 sobre la zona inicial del texto y devuelve
        la primera valida (T/C/SU con ano en rango): en una providencia el
        canon propio aparece antes que cualquier cita del cuerpo.

        Args:
            texto: Texto crudo de la sentencia.

        Returns:
            Canon "T-293-1997" o None si el encabezado no casa.
        """
        zona = texto[: CONFIG["zona_encabezado_chars"]]
        for coinc in self._m2.RE_SENT.finditer(zona):
            base = coinc.group(1).upper()
            tipo = "SU" if base == "S" and coinc.group(2) else base
            if tipo not in {"T", "C", "SU"}:
                continue
            ano = self._m2.normalizar_ano_cita(coinc.group(4))
            if ano is None:
                continue
            return f"{tipo}-{int(coinc.group(3))}-{ano}"
        return None


# ===== SECCION 4: M4 clasificador de area tematica =====
class ClasificadorArea:
    """M4: clasifica el area tematica constitucional de la sentencia.

    Pipeline heredado de 06_baseline_clasico.py (macro-F1 0.287): lemas
    spaCy del texto crudo (is_alpha, minusculas) -> normalizacion de lemas
    multi-palabra con "_" -> enmascarado de citas -> TF-IDF -> LogReg
    multinomial; devuelve el top-3 de areas con probabilidad.
    """

    def __init__(self) -> None:
        """Carga el vectorizador TF-IDF y el LogReg enmascarados (joblib)."""
        self._vectorizador = joblib.load(ROOT / CONFIG["ruta_tfidf"])
        self._modelo = joblib.load(ROOT / CONFIG["ruta_logreg"])
        self._nlp = None

    def _cargar_spacy(self) -> None:
        """Carga el pipeline spaCy (lazy, sin parser ni NER).

        Solo se necesita el lematizador; excluir parser/NER reduce el tiempo
        de carga y de procesamiento en CPU moderada.
        """
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(
                CONFIG["modelo_spacy"], exclude=["parser", "ner"]
            )
            self._nlp.max_length = CONFIG["spacy_max_length"]

    def clasificar(self, texto: str) -> dict:
        """Clasifica el area tematica y devuelve el top-3 con probabilidad.

        Args:
            texto: Texto crudo de la sentencia.

        Returns:
            Dict con top3 (lista de {area, prob} ordenada por probabilidad
            descendente) y via (identificador del metodo del producto).

        Raises:
            ValueError: Si el texto es vacio o no es cadena.
        """
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError("El clasificador M4 requiere un texto no vacio.")
        self._cargar_spacy()
        doc = self._nlp(texto)
        lemas = " ".join(
            tok.lemma_.lower().strip() for tok in doc if tok.is_alpha
        )
        mascarada = enmascarar_citas_lemas(normalizar_lemas(lemas))
        matriz = self._vectorizador.transform([mascarada])
        probs = self._modelo.predict_proba(matriz)[0]
        pares = sorted(
            zip(self._modelo.classes_, probs, strict=True),
            key=lambda par: -par[1],
        )
        top3 = [
            {"area": str(area), "prob": round(float(prob), 4)}
            for area, prob in pares[:3]
        ]
        return {"top3": top3, "via": "clasificador_m4_logreg_enmascarado"}


# ===== SECCION 5: M7 v2 vinculo con la Constitucion (BM25) =====
class VinculoCP:
    """M7 v2: recupera articulos de la CP vinculados a la sentencia.

    Port de la indexacion de 12_evaluar_vinculo_cp_v2.py: indice BM25 sobre
    PASAJES de los articulos (oraciones agrupadas hasta 120 palabras),
    query en chunks de 800 chars con solape 200 y score del articulo =
    maximo sobre (chunks x pasajes). El indice se construye una sola vez y
    se cachea en data/cache_bm25_cp/ para no reindexar en cada llamada.

    ciclo v2 - Fase S1: el indice se construye por defecto SOLO sobre
    articulos PERMANENTES (excluye transitorios y preambulo) porque el
    diagnostico de la Fase R midio 24.4% de transitorios en top-5
    contaminando el ranking (las sentencias citan el articulado
    permanente; los transitorios legitimos se recuperan con
    incluir_transitorios=True). recuperar() devuelve top-10 (antes top-5)
    y el cache se versiona a v3 sobre el parquet con IDs corregidos.

    ciclo v2 - Fase T: modo reranker OPCIONAL (usar_reranker=False por
    defecto; el producto base no cambia). Con usar_reranker=True,
    recuperar() toma el top-20 BM25, lo puntua con el cross-encoder
    bge-reranker-v2-m3 (carga lazy en el primer recuperar; GPU fp16 si
    hay VRAM libre, CPU batch 4 si no) y devuelve el top-10 rerankeado
    con los scores del reranker en el mismo formato.

    ciclo v2 - Fase X (producto v2): el producto adopta las mejoras
    ganadoras del ciclo v2 con defaults fijados por el benchmark v8.
    (1) Prior de hubs CP (usar_hubs=True): score_final =
        min-max(score_base) + alpha * prior_hubs(canon), con alpha=0.75 en
        la ruta BM25 por defecto y alpha=1.5 en la ruta reranker; el prior
        se carga de CONFIG['ruta_prior_hubs'] (data/prior_hubs_cp.json) y
        si falta se desactiva con WARNING (degradacion documentada, no
        bloqueante).
    (2) Ruta reranker = hibrido_rerank_chunks_top50 (ganadora v6/v8):
        candidatos top-50 del hibrido 0.5*minmax(BM25) + 0.5*minmax(E5)
        (E5 = intfloat/multilingual-e5-base con pasajes de la CP cacheados
        en data/cache_e5_cp/ y chunks de query en runtime); para cada
        candidato, mejor pasaje BM25 + top-3 chunks de la query -> 3 pares
        al cross-encoder -> max-pooling; despues prior alpha=1.5 -> top-10.
        E5 y reranker se cargan LAZY (solo con usar_reranker=True).
        FALLBACK: si E5 falla o no esta disponible, los candidatos pasan a
        ser el top-50 BM25 con WARNING documentado.
    """

    def __init__(
        self,
        usar_cache: bool = True,
        incluir_transitorios: bool = False,
        usar_reranker: bool = False,
        usar_hubs: bool = True,
        alpha_hubs_base: float = 0.75,
        alpha_hubs_rerank: float = 1.5,
    ) -> None:
        """Carga el indice BM25 desde cache o lo construye y persiste.

        Args:
            usar_cache: Si True (default) reutiliza el joblib del cache
                cuando existe y es valido; si False reconstruye siempre.
            incluir_transitorios: Si False (default) el indice cubre solo
                articulos permanentes (sin transitorios ni preambulo,
                Fase S1); si True indexa los 497 articulos completos para
                casos que citan disposiciones transitorias legitimas.
            usar_reranker: Si False (default) recuperar() es la ruta BM25
                por defecto; si True (Fase X) se activa la estrategia
                ganadora del v8 (top-50 hibrido BM25/E5 -> rerank por
                chunks -> prior alpha=alpha_hubs_rerank -> top-10). La
                carga de E5 y del reranker es lazy (no penaliza este init).
            usar_hubs: Si True (default) aplica el prior de hubs CP a los
                scores base (Fase W-a iter2); si False, ranking sin prior.
            alpha_hubs_base: Peso del prior en la ruta BM25 default
                (default 0.75, optimo full-90 del benchmark v8).
            alpha_hubs_rerank: Peso del prior en la ruta reranker
                (default 1.5, optimo full-90 del benchmark v8).
        """
        self._usar_cache = usar_cache
        self._incluir_transitorios = incluir_transitorios
        self._usar_reranker = usar_reranker
        self._alpha_hubs_base = float(alpha_hubs_base)
        self._alpha_hubs_rerank = float(alpha_hubs_rerank)
        self._reranker = None
        self._e5: tuple | None = None
        self._prior_hubs = self._cargar_prior_hubs() if usar_hubs else None
        self._usar_hubs = self._prior_hubs is not None
        datos = self._cargar_o_construir()
        self._canones: list[str] = datos["canones"]
        self._grupos: list[np.ndarray] = datos["grupos_indices"]
        self._indice: BM25Okapi = datos["indice"]
        self._articulos: dict = datos["articulos"]
        self._textos_pasajes: list[str] = [
            fila["texto"] for fila in self._filas_pasajes()
        ]

    def _ruta_cache(self) -> Path:
        """Devuelve la ruta absoluta del archivo de cache del indice.

        La version v3 (Fase S1) separa los dos modos del indice: el cache
        de permanentes usa el nombre base y el de transitorios el sufijo
        _full, para que ambos puedan coexistir sin sobrescribirse.
        """
        nombre = CONFIG["archivo_cache_bm25"]
        if self._incluir_transitorios:
            nombre = nombre.replace(".joblib", "_full.joblib")
        return ROOT / CONFIG["dir_cache_bm25"] / nombre

    def _filas_pasajes(self) -> list[dict]:
        """Reconstruye los pasajes indexados (canon, texto) del parquet.

        ciclo v2 - Fase T: el reranker necesita el TEXTO de cada pasaje
        (el mismo que indexa BM25). El cache v3 no guarda los textos
        (solo tokens) y NO se reescribe (ADR-11), por lo que los pasajes
        se reconstruyen deterministamente desde el parquet con la misma
        segmentacion de _construir_indice (dividir_pasajes, 120
        palabras): el orden canon->pasajes es identico al de grupos_indices
        porque ambos derivan del mismo parquet y del mismo iterador.

        Returns:
            Lista de dicts {canon, texto} en el orden de indexacion (un
            pasaje por fila, alineada con grupos_indices).
        """
        tabla = pd.read_parquet(ROOT / CONFIG["ruta_constitucion"])
        es_transitorio = tabla["numero"].astype(str).str.startswith(
            "Transitorio", na=False
        )
        es_preambulo = tabla["numero"].astype(str) == "Preambulo"
        indexables = tabla[~es_transitorio & ~es_preambulo]
        if self._incluir_transitorios:
            indexables = tabla
        filas: list[dict] = []
        for canon, texto in zip(
            indexables["numero"].astype(str),
            indexables["texto"].fillna("").astype(str),
            strict=True,
        ):
            for trozo in dividir_pasajes(texto, CONFIG["pasaje_max_palabras"]):
                filas.append({"canon": canon, "texto": trozo})
        return filas

    def _cargar_o_construir(self) -> dict:
        """Carga el cache del indice o lo construye desde la CP.

        El cache es valido si la version coincide y canones/grupos_indices
        son paralelos (un grupo de pasajes por canon indexable). Nota: el
        dict articulos conserva TODOS los articulos del parquet (497) aunque
        el indice solo cubra los permanentes (384), por lo que NO puede
        usarse len(articulos) como chequeo de validez del modo permanentes
        (haria el cache invalido siempre y se reescribiria en cada arranque).

        Returns:
            Dict con version, canones, grupos_indices, indice (BM25Okapi)
            y articulos (dict numero -> {texto, bloque}).
        """
        ruta = self._ruta_cache()
        if self._usar_cache and ruta.is_file():
            try:
                datos = joblib.load(ruta)
            except Exception:  # noqa: BLE001 - cache corrupto: reconstruir
                datos = None
            if (
                datos is not None
                and datos.get("version") == CONFIG["version_cache"]
                and len(datos["canones"]) == len(datos["grupos_indices"])
            ):
                return datos
        datos = self._construir_indice()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(datos, ruta)
        return datos

    def _construir_indice(self) -> dict:
        """Indexa los pasajes de la Constitucion con la tokenizacion M7 v2.

        ciclo v2 - Fase S1: con incluir_transitorios=False (default) se
        excluyen del indice los transitorios y el preambulo; los articulos
        excluidos se conservan en el dict articulos para que
        detalle_articulo siga resolviendolos (solo dejan de rankearse).

        Returns:
            Dict con version, canones (384 permanentes o 497 completos),
            grupos_indices (posiciones de pasajes por articulo), indice
            BM25Okapi y articulos (dict con texto completo y bloque
            tematico de TODOS los articulos del parquet).
        """
        tabla = pd.read_parquet(ROOT / CONFIG["ruta_constitucion"])
        es_transitorio = tabla["numero"].astype(str).str.startswith(
            "Transitorio", na=False
        )
        es_preambulo = tabla["numero"].astype(str) == "Preambulo"
        indexables = tabla[~es_transitorio & ~es_preambulo]
        if self._incluir_transitorios:
            indexables = tabla
        canones = indexables["numero"].astype(str).tolist()
        textos = indexables["texto"].fillna("").astype(str).tolist()
        filas: list[dict] = []
        grupos: list[np.ndarray] = []
        posicion = 0
        for canon, texto in zip(canones, textos, strict=True):
            indices: list[int] = []
            for trozo in dividir_pasajes(texto, CONFIG["pasaje_max_palabras"]):
                filas.append({"canon": canon, "texto": trozo})
                indices.append(posicion)
                posicion += 1
            grupos.append(np.asarray(indices, dtype=int))
        tokens = [tokenizar(fila["texto"]) for fila in filas]
        indice = BM25Okapi(tokens)
        articulos = {
            canon: {"texto": texto, "bloque": bloque}
            for canon, texto, bloque in zip(
                tabla["numero"].astype(str),
                tabla["texto"].fillna("").astype(str),
                tabla["bloque_tematico"],
                strict=True,
            )
        }
        return {
            "version": CONFIG["version_cache"],
            "canones": canones,
            "grupos_indices": grupos,
            "indice": indice,
            "articulos": articulos,
        }

    def _chunks_query(self, texto: str) -> list[str]:
        """Divide la query en chunks de caracteres con solape (cobertura total).

        Port de 12_evaluar_vinculo_cp_v2.py: la evidencia de calibracion
        refuto la ventana inicial corta (las citas aparecen a lo largo de
        todo el texto), por lo que la query es el texto completo.

        Args:
            texto: Texto de la sentencia (limpio si viene del corpus).

        Returns:
            Lista de chunks (>= 1 si el texto no esta vacio).
        """
        chunks: list[str] = []
        inicio = 0
        while inicio < len(texto):
            fin = min(inicio + CONFIG["chunk_bm25_chars"], len(texto))
            chunks.append(texto[inicio:fin])
            if fin >= len(texto):
                break
            inicio = fin - CONFIG["solape_bm25_chars"]
        return chunks

    @property
    def usar_reranker(self) -> bool:
        """Indica si la ruta reranker (ganadora v8) esta activa."""
        return self._usar_reranker

    @property
    def usar_hubs(self) -> bool:
        """Indica si el prior de hubs CP se aplica (activo y cargado)."""
        return self._usar_hubs

    @property
    def alpha_hubs(self) -> float:
        """Peso del prior que corresponde a la ruta activa (base o rerank)."""
        return (
            self._alpha_hubs_rerank
            if self._usar_reranker
            else self._alpha_hubs_base
        )

    def _cargar_prior_hubs(self) -> dict[str, float] | None:
        """Carga el prior de hubs CP desde CONFIG['ruta_prior_hubs'].

        Port de cargar_prior_hubs de 12_evaluar_vinculo_cp_v2.py: el
        archivo data/prior_hubs_cp.json contiene la frecuencia global de
        citacion por articulo CP normalizada a [0, 1]. La carga es
        tolerante: si el archivo no existe se emite un WARNING y el prior
        queda desactivado (el producto sigue funcionando sin la senal).

        Returns:
            Dict {canon: valor} o None si el archivo no esta disponible.
        """
        ruta = ROOT / CONFIG["ruta_prior_hubs"]
        if not ruta.is_file():
            print(
                "[aviso] prior de hubs no encontrado en "
                f"{ruta}; producto v2 sin prior (documentado)"
            )
            return None
        with ruta.open("r", encoding="utf-8") as manejador:
            crudo = json.load(manejador)
        return {str(canon): float(valor) for canon, valor in crudo.items()}

    def _cargar_reranker(self) -> object:
        """Carga el cross-encoder bge-reranker-v2-m3 (lazy, Fase X).

        Solo se invoca cuando usar_reranker=True y una sola vez por
        instancia; la ruta BM25 por defecto nunca paga este costo.

        Returns:
            RerankerBGE ya cargado (scripts/lib/reranker_bge.py) en el
            dispositivo elegido por su guard de VRAM.
        """
        if self._reranker is None:
            from lib.reranker_bge import RerankerBGE

            self._reranker = RerankerBGE()
        return self._reranker

    # --- ciclo v2 - Fase X: scores BM25 por pasajes y matriz del caso ---
    def _scores_bm25_caso(
        self, chunks: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scores BM25 por articulo y matriz (chunk, pasaje) del caso.

        Port exacto de scores_bm25_caso de 12_evaluar_vinculo_cp_v2.py:
        por cada chunk de query se puntuan todos los pasajes; el score del
        pasaje es el maximo entre chunks y el del articulo el maximo entre
        sus pasajes. La matriz completa (n_chunks, n_pasajes) se devuelve
        ademas para que la ruta reranker derive, para cada candidato, su
        mejor pasaje y los top-3 chunks contra ese pasaje.

        Args:
            chunks: Chunks de la ventana de query (texto completo).

        Returns:
            Tupla (scores, matriz): scores (n_articulos,) alineado con
            self._canones y matriz (n_chunks, n_pasajes) de scores BM25.
        """
        n_pasajes = self._indice.corpus_size
        matriz = np.zeros((len(chunks), n_pasajes), dtype=np.float64)
        max_pasajes = np.zeros(n_pasajes, dtype=np.float64)
        for fila, chunk in enumerate(chunks):
            tokens = tokenizar(chunk)
            if not tokens:
                continue
            scores_chunk = self._indice.get_scores(tokens)
            matriz[fila] = scores_chunk
            np.maximum(max_pasajes, scores_chunk, out=max_pasajes)
        scores = np.zeros(len(self._canones), dtype=np.float64)
        for i, indices in enumerate(self._grupos):
            scores[i] = float(max_pasajes[indices].max()) if len(indices) else 0.0
        return scores, matriz

    # --- ciclo v2 - Fase X: hibrido BM25/E5 para los candidatos del rerank ---
    @staticmethod
    def _normalizar_min_max(scores: np.ndarray) -> np.ndarray:
        """Normaliza min-max un vector de scores (port del benchmark).

        Si todos los scores son iguales devuelve ceros: esa estrategia no
        discrimina en el caso y no aporta al hibrido.

        Args:
            scores: Array de scores de una estrategia para un caso.

        Returns:
            Array normalizado a [0, 1].
        """
        minimo, maximo = float(scores.min()), float(scores.max())
        if maximo <= minimo:
            return np.zeros_like(scores)
        return (scores - minimo) / (maximo - minimo)

    def _scores_hibridos(
        self, scores_bm25: np.ndarray, scores_e5: np.ndarray
    ) -> np.ndarray:
        """Combina BM25 y E5 con pesos 0.5/0.5 (min-max por caso).

        Port de scores_hibridos de 12_evaluar_vinculo_cp_v2.py (pesos
        pre-registrados CONFIG['pesos_hibrido']).

        Args:
            scores_bm25: Scores BM25 del caso (n_articulos,).
            scores_e5: Scores E5 del caso (n_articulos,).

        Returns:
            Array (n_articulos,) de scores hibridos.
        """
        pesos = CONFIG["pesos_hibrido"]
        return pesos["bm25"] * self._normalizar_min_max(
            scores_bm25
        ) + pesos["e5"] * self._normalizar_min_max(scores_e5)

    @staticmethod
    def _chunkear_texto(texto: str, tokenizer: object) -> list[str]:
        """Divide un texto en chunks de <=512 tokens con solape (ADR-3).

        Port de chunkear_texto de 12_evaluar_vinculo_cp_v2.py: reserva
        CONFIG['reserva_prefijo_tokens'] tokens para el prefijo
        "query: "/"passage: " que se anade al codificar.

        Args:
            texto: Texto crudo largo (ventana de query o pasaje).
            tokenizer: Tokenizer del modelo E5.

        Returns:
            Lista de chunks decodificados (sin el prefijo).
        """
        ids = tokenizer.encode(texto, add_special_tokens=False)
        maximo = CONFIG["tamano_chunk"] - CONFIG["reserva_prefijo_tokens"]
        solape = CONFIG["solape_chunk"]
        chunks: list[str] = []
        inicio = 0
        total = len(ids)
        while inicio < total:
            fin = min(inicio + maximo, total)
            chunks.append(
                tokenizer.decode(ids[inicio:fin], skip_special_tokens=True)
            )
            if fin == total:
                break
            inicio = fin - solape
        return chunks

    @staticmethod
    def _codificar_con_fallback(modelo: object, textos: list[str]) -> np.ndarray:
        """Codifica textos con E5 reintentando con batch menor ante OOM.

        Args:
            modelo: SentenceTransformer cargado en el dispositivo elegido.
            textos: Textos con prefijo ya incluido.

        Returns:
            Matriz de embeddings normalizados (n, dim).
        """
        try:
            return np.asarray(
                modelo.encode(
                    textos,
                    batch_size=CONFIG["batch_embedding"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            )
        except RuntimeError:
            print(
                "[aviso] OOM en E5: se re-codifica con batch "
                f"{CONFIG['batch_embedding_oom']} (documentado)"
            )
            return np.asarray(
                modelo.encode(
                    textos,
                    batch_size=CONFIG["batch_embedding_oom"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            )

    @staticmethod
    def _media_por_grupos(
        matriz: np.ndarray, longitudes: list[int]
    ) -> np.ndarray:
        """Agrega por media (chunk-mean de ADR-3) y renormaliza cada grupo.

        Args:
            matriz: Embeddings planos (sum(longitudes), dim).
            longitudes: Numero de chunks de cada grupo, en orden.

        Returns:
            Matriz (len(longitudes), dim) con un vector unitario por grupo.
        """
        vectores = []
        inicio = 0
        for n_chunks in longitudes:
            bloque = matriz[inicio : inicio + n_chunks]
            inicio += n_chunks
            media = bloque.mean(axis=0)
            norma = float(np.linalg.norm(media))
            vectores.append(media / norma if norma > 0 else media)
        return np.vstack(vectores)

    @staticmethod
    def _scores_e5_caso(
        vectores_query: np.ndarray,
        matriz_pasajes: np.ndarray,
        grupos_indices: list[np.ndarray],
    ) -> np.ndarray:
        """Scores E5 por articulo: max coseno entre chunks de query y pasajes.

        Port de scores_e5_caso de 12_evaluar_vinculo_cp_v2.py: max-pooling
        doble (entre chunks de query y entre pasajes del articulo).

        Args:
            vectores_query: Matriz (n_chunks, dim) de embeddings unitarios.
            matriz_pasajes: Matriz (n_pasajes, dim) de embeddings unitarios.
            grupos_indices: Arrays de posiciones de pasajes por articulo.

        Returns:
            Array (n_articulos,) de similitudes maximas por articulo.
        """
        similitudes = matriz_pasajes @ vectores_query.T
        maximos_pasajes = similitudes.max(axis=1)
        scores = np.zeros(len(grupos_indices), dtype=np.float64)
        for i, indices in enumerate(grupos_indices):
            scores[i] = maximos_pasajes[indices].max()
        return scores

    @staticmethod
    def _elegir_dispositivo_e5() -> str:
        """Elige GPU para E5 solo si hay VRAM libre suficiente (>= 2 GiB).

        Port de elegir_dispositivo de 12_evaluar_vinculo_cp_v2.py: si CUDA
        no esta disponible o la VRAM libre es menor que
        CONFIG['vram_minima_gb_e5'] se degrada a CPU con WARNING
        documentado (no bloquea la ejecucion).

        Returns:
            "cuda" si hay VRAM suficiente; si no "cpu".
        """
        import torch

        if not torch.cuda.is_available():
            print(
                "[aviso] E5: CUDA no disponible; se codifica en CPU "
                "(documentado)"
            )
            return "cpu"
        libre, _ = torch.cuda.mem_get_info()
        minimo = int(CONFIG["vram_minima_gb_e5"] * 1024**3)
        if libre < minimo:
            print(
                "[aviso] E5: VRAM libre insuficiente "
                f"({libre / 1024**3:.1f} GiB < "
                f"{CONFIG['vram_minima_gb_e5']} GiB); se codifica en CPU "
                "(documentado)"
            )
            return "cpu"
        return "cuda"

    def _cargar_e5(self) -> tuple:
        """Carga E5 y la matriz de pasajes de la CP (lazy, Fase X).

        Port del pipeline E5 de 12_evaluar_vinculo_cp_v2.py: backbone
        intfloat/multilingual-e5-base con max_seq_length=512 y pooling
        mean; los pasajes de la CP se trocean/agregan una sola vez y se
        cachean en data/cache_e5_cp/ (nueva cache de la Fase X). Solo se
        invoca con usar_reranker=True.

        Returns:
            Tupla (modelo SentenceTransformer, tokenizer, matriz_pasajes).
        """
        if self._e5 is None:
            dispositivo = self._elegir_dispositivo_e5()
            from sentence_transformers import SentenceTransformer
            from sentence_transformers import models as st_models

            backbone = st_models.Transformer(
                CONFIG["modelo_e5"], max_seq_length=CONFIG["tamano_chunk"]
            )
            pooling = st_models.Pooling(
                backbone.get_embedding_dimension(), pooling_mode="mean"
            )
            modelo = SentenceTransformer(
                modules=[backbone, pooling], device=dispositivo
            )
            tokenizer = modelo.tokenizer
            matriz_pasajes = self._matriz_pasajes_e5(modelo, tokenizer)
            self._e5 = (modelo, tokenizer, matriz_pasajes)
            print(
                f"[info] E5 {CONFIG['modelo_e5']} cargado en {dispositivo} "
                f"(pasajes {matriz_pasajes.shape})"
            )
        return self._e5

    def _ruta_cache_e5(self) -> Path:
        """Devuelve la ruta absoluta de la cache de embeddings de pasajes."""
        return ROOT / CONFIG["dir_cache_e5"] / CONFIG["archivo_cache_e5"]

    def _matriz_pasajes_e5(
        self, modelo: object, tokenizer: object
    ) -> np.ndarray:
        """Carga o construye la matriz de embeddings E5 de los pasajes.

        La cache guarda la matriz AGREGADA (un vector por pasaje, media de
        sus chunks renormalizada) junto con la configuracion que la valida
        (modelo, chunking y numero de pasajes). El orden de filas es el de
        self._textos_pasajes (identico al del indice BM25).

        Args:
            modelo: SentenceTransformer cargado.
            tokenizer: Tokenizer del modelo E5.

        Returns:
            Matriz (n_pasajes, dim) de embeddings unitarios por pasaje.
        """
        ruta = self._ruta_cache_e5()
        if self._usar_cache and ruta.is_file():
            try:
                datos = joblib.load(ruta)
            except Exception:  # noqa: BLE001 - cache corrupto: reconstruir
                datos = None
            if (
                datos is not None
                and datos.get("modelo") == CONFIG["modelo_e5"]
                and datos.get("tamano_chunk") == CONFIG["tamano_chunk"]
                and datos.get("solape_chunk") == CONFIG["solape_chunk"]
                and datos.get("n_pasajes") == len(self._textos_pasajes)
                and np.asarray(datos["matriz"]).shape[0]
                == len(self._textos_pasajes)
            ):
                print(f"[info] E5 pasajes: cache cargada de {ruta}")
                return np.asarray(datos["matriz"])
        matriz = self._construir_matriz_pasajes(modelo, tokenizer)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "modelo": CONFIG["modelo_e5"],
                "tamano_chunk": CONFIG["tamano_chunk"],
                "solape_chunk": CONFIG["solape_chunk"],
                "n_pasajes": len(self._textos_pasajes),
                "matriz": matriz,
            },
            ruta,
        )
        print(f"[info] E5 pasajes: cache nueva escrita en {ruta}")
        return matriz

    def _construir_matriz_pasajes(
        self, modelo: object, tokenizer: object
    ) -> np.ndarray:
        """Codifica los pasajes de la CP con chunk-mean (port del benchmark).

        Args:
            modelo: SentenceTransformer cargado.
            tokenizer: Tokenizer del modelo E5.

        Returns:
            Matriz (n_pasajes, dim) de embeddings unitarios por pasaje.
        """
        grupos = [
            self._chunkear_texto(texto, tokenizer) or [""]
            for texto in self._textos_pasajes
        ]
        longitudes = [len(grupo) for grupo in grupos]
        textos = [
            CONFIG["prefijo_pasaje"] + chunk
            for grupo in grupos
            for chunk in grupo
        ]
        embeddings = self._codificar_con_fallback(modelo, textos)
        return self._media_por_grupos(embeddings, longitudes)

    def _scores_query_e5(self, texto: str) -> np.ndarray:
        """Codifica los chunks de la query y devuelve los scores E5.

        Args:
            texto: Texto de la sentencia (texto_limpio del corpus).

        Returns:
            Array (n_articulos,) de similitudes maximas por articulo.

        Raises:
            Exception: Propaga el fallo de E5 para que el llamador active
                el fallback documentado a BM25.
        """
        modelo, tokenizer, matriz_pasajes = self._cargar_e5()
        grupos = self._chunkear_texto(texto, tokenizer) or [""]
        textos = [CONFIG["prefijo_query"] + chunk for chunk in grupos]
        embeddings = self._codificar_con_fallback(modelo, textos)
        return self._scores_e5_caso(embeddings, matriz_pasajes, self._grupos)

    def _scores_candidatos_rerank(
        self, scores_bm25: np.ndarray, texto: str
    ) -> np.ndarray:
        """Scores de la primera etapa de la ruta reranker (hibrido 0.5/0.5).

        Intenta el hibrido BM25/E5 (ganadora v8); si E5 falla o no esta
        disponible activa el FALLBACK documentado: candidatos = top-50
        BM25, con WARNING (el resto del pipeline no cambia).

        Args:
            scores_bm25: Scores BM25 del caso (n_articulos,).
            texto: Texto de la sentencia (query de E5).

        Returns:
            Array (n_articulos,) que ordena a los candidatos del rerank.
        """
        try:
            scores_e5 = self._scores_query_e5(texto)
            return self._scores_hibridos(scores_bm25, scores_e5)
        except Exception as exc:  # noqa: BLE001 - fallback documentado
            print(
                f"[aviso] E5 no disponible ({exc}); fallback a candidatos "
                "top-50 BM25 + reranker con prior alpha="
                f"{self._alpha_hubs_rerank} (documentado)"
            )
            return scores_bm25.copy()

    def _rerankear_chunks(
        self,
        scores_candidatos: np.ndarray,
        matriz_bm25: np.ndarray,
        chunks: list[str],
        texto: str,
    ) -> np.ndarray:
        """Rerankea el top-50 de candidatos por pares (chunk, pasaje).

        Port de ranking_rerank_chunks de 12_evaluar_vinculo_cp_v2.py
        (estrategia ganadora hibrido_rerank_chunks_top50): para cada
        candidato del top-50 (ordenado por scores_candidatos) se toma su
        mejor pasaje por score BM25 y los top-3 chunks de la query contra
        ese pasaje; el cross-encoder puntua los 3 pares y el score del
        articulo es el maximo (max-pooling). El parametro texto se
        conserva por paridad de firma con el benchmark (los pares usan los
        chunks, no la query completa).

        Args:
            scores_candidatos: Scores que ordenan a los candidatos
                (hibrido BM25/E5 o BM25 en el fallback), alineados con
                self._canones.
            matriz_bm25: Matriz (n_chunks, n_pasajes) del caso.
            chunks: Chunks de la ventana de query (filas de la matriz).
            texto: Texto de la sentencia (no usado; paridad de firma).

        Returns:
            Array (n_articulos,) con el score del reranker en los
            candidatos y -inf en el resto.
        """
        del texto  # firma espejo del benchmark; los pares usan chunks
        reranker = self._cargar_reranker()
        orden = np.argsort(-scores_candidatos, kind="stable")
        candidatos = orden[: CONFIG["k_candidatos_rerank_top50"]]
        maximos_pasajes = matriz_bm25.max(axis=0)
        queries: list[str] = []
        pasajes_lote: list[str] = []
        mapa_pares: list[tuple[int, int]] = []
        for pos_articulo, indice in enumerate(candidatos):
            indices_pasaje = self._grupos[indice]
            mejor_pasaje = int(
                indices_pasaje[np.argmax(maximos_pasajes[indices_pasaje])]
            )
            scores_chunks = matriz_bm25[:, mejor_pasaje]
            top_chunks = np.argsort(-scores_chunks, kind="stable")[
                : CONFIG["k_chunks_rerank"]
            ]
            for fila_chunk in top_chunks:
                queries.append(chunks[fila_chunk])
                pasajes_lote.append(self._textos_pasajes[mejor_pasaje])
                mapa_pares.append((pos_articulo, fila_chunk))
        scores_pares = reranker.puntuar(queries, pasajes_lote)
        scores_rerank = np.full(len(self._canones), -np.inf, dtype=np.float64)
        maximos = np.full(len(candidatos), -np.inf, dtype=np.float64)
        for (pos_articulo, _), score in zip(
            mapa_pares, scores_pares, strict=True
        ):
            maximos[pos_articulo] = max(maximos[pos_articulo], float(score))
        for pos_articulo, indice in enumerate(candidatos):
            scores_rerank[indice] = maximos[pos_articulo]
        return scores_rerank

    def _aplicar_prior_scores(
        self, scores: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Suma el prior de hubs a un vector de scores (port del benchmark).

        Port de aplicar_prior_scores + aplicar_prior_hubs de
        12_evaluar_vinculo_cp_v2.py: score_final = min-max(score) +
        alpha * prior; los elementos no finitos (p. ej. -inf del reranker
        para los articulos fuera del top-50) se preservan tal cual y la
        normalizacion min-max ocurre solo entre los finitos, de modo que
        el prior NO puede resucitar articulos fuera del recall de la
        primera etapa. Con alpha=0 devuelve una copia intacta.

        Args:
            scores: Array (n_articulos,) de scores base.
            alpha: Peso del prior (>= 0).

        Returns:
            Array nuevo (n_articulos,) con los scores priorizados.
        """
        if (
            alpha == 0.0
            or self._prior_hubs is None
            or not len(scores)
        ):
            return scores.copy()
        finitos = np.flatnonzero(np.isfinite(scores))
        if finitos.size == 0:
            return scores.copy()
        dicc = {self._canones[i]: float(scores[i]) for i in finitos}
        valores = np.asarray(list(dicc.values()), dtype=np.float64)
        minimo, maximo = float(valores.min()), float(valores.max())
        if maximo > minimo:
            valores = (valores - minimo) / (maximo - minimo)
        else:
            valores = np.zeros_like(valores)
        priorizado = {
            canon: float(base + alpha * float(self._prior_hubs.get(canon, 0.0)))
            for (canon, _), base in zip(dicc.items(), valores, strict=True)
        }
        resultado = scores.copy()
        for i in finitos:
            resultado[i] = priorizado[self._canones[i]]
        return resultado

    def recuperar(self, texto: str, k: int | None = None) -> list[dict]:
        """Recupera los k articulos de la CP mas vinculados al texto.

        Score del pasaje = maximo entre chunks de la query; score del
        articulo = maximo entre sus pasajes (max-pooling doble, ADR-9).
        ciclo v2 - Fase S1: la profundidad por defecto pasa de 5 a 10
        (CONFIG['k_bm25_default']) y el ranking solo contiene articulos
        del indice (permanentes por defecto; transitorios solo con
        incluir_transitorios=True).
        ciclo v2 - Fase X (producto v2): dos rutas segun usar_reranker,
        ambas con prior de hubs cuando usar_hubs=True:
        - default (usar_reranker=False): BM25 por pasajes -> prior
          alpha=CONFIG['alpha_hubs_base'] (0.75) -> top-k;
        - reranker (usar_reranker=True): candidatos top-50 del hibrido
          BM25/E5 -> rerank por chunks con bge-reranker-v2-m3 -> prior
          alpha=CONFIG['alpha_hubs_rerank'] (1.5) -> top-k. Los scores
          devueltos son los finales (priorizados).

        Args:
            texto: Texto de la sentencia (texto_limpio del corpus cuando
                este disponible; el tokenizador es robusto al crudo).
            k: Profundidad del ranking (default CONFIG["k_bm25_default"]).

        Returns:
            Lista de {numero, score} ordenada por score descendente.
        """
        profundidad = k or CONFIG["k_bm25_default"]
        if not isinstance(texto, str) or not texto.strip():
            return []
        if self._indice.corpus_size == 0:
            return []
        chunks = self._chunks_query(texto)
        scores_bm25, matriz_bm25 = self._scores_bm25_caso(chunks)
        if self._usar_reranker:
            scores_candidatos = self._scores_candidatos_rerank(
                scores_bm25, texto
            )
            scores = self._rerankear_chunks(
                scores_candidatos, matriz_bm25, chunks, texto
            )
            alpha = self._alpha_hubs_rerank
        else:
            scores = scores_bm25
            alpha = self._alpha_hubs_base
        if self._usar_hubs and alpha > 0.0:
            scores = self._aplicar_prior_scores(scores, alpha)
        orden = np.argsort(-scores, kind="stable")[:profundidad]
        return [
            {"numero": self._canones[i], "score": round(float(scores[i]), 4)}
            for i in orden
        ]

    def detalle_articulo(self, numero: str) -> dict | None:
        """Devuelve texto recortado y bloque tematico de un articulo CP.

        ciclo v2 - Fase S1: el dict articulos conserva TODOS los
        articulos del parquet (incluidos transitorios y preambulo) aunque
        el indice solo rankee permanentes, para que el detalle siga
        resolviendo cualquier canon del parquet.

        Args:
            numero: Canon del articulo ("13", "22A", "Transitorio 68").

        Returns:
            Dict con texto_150chars y bloque, o None si el articulo no
            existe en la Constitucion parseada.
        """
        info = self._articulos.get(numero)
        if info is None:
            return None
        return {
            "texto_150chars": info["texto"][:150],
            "bloque": info["bloque"],
        }


# ===== SECCION 6: M5 contexto jurisprudencial =====
class ContextoJurisprudencial:
    """M5: resuelve citas a sentencias y agrega su contexto de la red.

    Carga data/red_citas_resumen.parquet (PageRank, ano y area por
    sentencia del corpus; archivo pequeno, se carga completo) y el marco
    ligero del corpus (canon -> id) para resolver cada cita canon
    TIPO-NUM-AAAA del M2.

    ciclo v2 - Fase S2: el marco canon -> id se lee con pyarrow.dataset
    (columnas ligeras id/tipo/numero/ano, particionado hive) en vez de
    pandas.read_parquet del corpus completo; los textos pesados no se
    materializan (RAM 8 -> 2 GB, AUDITORIA_RENDIMIENTO.md).
    """

    def __init__(self) -> None:
        """Carga la red de citas y el mapa canon -> id del corpus."""
        red = pd.read_parquet(ROOT / CONFIG["ruta_red_citas"])
        self._por_id = {
            str(fila.id): {
                "ano": int(fila.ano),
                "pagerank": float(fila.pagerank),
                "area": str(fila.area),
            }
            for fila in red.itertuples(index=False)
        }
        # ciclo v2 - Fase S2: lectura selectiva de columnas ligeras con
        # pyarrow (sin textos); el mapa canon -> id es identico al que
        # producia pandas (verificado: 23.742 entradas, sin duplicados).
        corpus = ds.dataset(
            ROOT / CONFIG["ruta_corpus"], format="parquet", partitioning="hive"
        ).to_table(columns=["id", "tipo", "numero", "ano"])
        self._canon_a_id = {
            f"{tipo}-{int(numero)}-{int(ano)}": str(id_doc)
            for tipo, numero, ano, id_doc in zip(
                corpus["tipo"].to_pylist(),
                corpus["numero"].to_pylist(),
                corpus["ano"].to_pylist(),
                corpus["id"].to_pylist(),
                strict=True,
            )
        }

    def id_de_canon(self, canon: str) -> str | None:
        """Resuelve un canon TIPO-NUM-AAAA al id del corpus.

        Args:
            canon: Canon de sentencia ("T-293-1997").

        Returns:
            Id del documento citado o None si no esta en el corpus.
        """
        return self._canon_a_id.get(canon)

    @staticmethod
    def _ano_de_canon(canon: str) -> int | None:
        """Extrae el ano de un canon TIPO-NUM-AAAA.

        Args:
            canon: Canon de sentencia.

        Returns:
            Ano en 4 digitos o None si el canon no tiene el formato esperado.
        """
        try:
            return int(canon.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return None

    def resolver(self, citas: list[str]) -> list[dict]:
        """Resuelve cada cita canon a su contexto en la red de citas.

        Args:
            citas: Lista de canones TIPO-NUM-AAAA extraidos por M2 (orden de
                aparicion en el texto).

        Returns:
            Lista alineada con citas: para citas resueltas, id del corpus,
            resuelta=True, ano, pagerank (redondeado a 6 decimales) y area
            de la sentencia citada; para no resueltas, id=canon,
            resuelta=False, ano del canon y pagerank/area None.
        """
        salida: list[dict] = []
        for canon in citas:
            id_doc = self._canon_a_id.get(canon)
            if id_doc is None:
                salida.append(
                    {
                        "id": canon,
                        "resuelta": False,
                        "ano": self._ano_de_canon(canon),
                        "pagerank": None,
                        "area": None,
                    }
                )
                continue
            info = self._por_id.get(id_doc)
            salida.append(
                {
                    "id": id_doc,
                    "resuelta": True,
                    "ano": info["ano"] if info else None,
                    "pagerank": round(info["pagerank"], 6) if info else None,
                    "area": info["area"] if info else None,
                }
            )
        return salida
