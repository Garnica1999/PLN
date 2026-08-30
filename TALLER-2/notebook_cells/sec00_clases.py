# %% [markdown]
# # Sec00 — Núcleo POO: patrones de diseño, SOLID y clean code
#
# Esta sección define las clases reutilizables que usan todas las secciones
# posteriores del notebook. Aplica patrones de diseño clásicos:
#
# | Patrón | Clase | Responsabilidad |
# |---|---|---|
# | **Repository** | `TesauroRepositorio` | Acceso único y cacheado a las fuentes de datos (Excel, JSON) |
# | **Singleton + Facade** | `ServicioOntologia` | Una sola instancia por ontología; construye, guarda, carga, valida y consulta la OWL |
# | **Abstract Factory + Registry** | `FabricaExtractores` | Crea el extractor de WordNet según el tipo de relación; añadir un tipo nuevo no modifica código existente (Open/Closed) |
# | **Strategy** | `ExtractorRelacion` + 9 implementaciones | Una clase por tipo de relación semántica, intercambiables |
# | **Strategy** | `NormalizadorTermino` + variantes | Normalización de términos con variantes (base, plural, sin acentos) |
# | **Facade** | `MotorRelacionesWordNet`, `SparqlServicio`, `GrafoOntologia` | APIs simples que orquestan NLTK, rdflib y networkx |
#
# SOLID: cada clase tiene una única responsabilidad (S); el registro de
# extractores permite extender sin modificar (O); las clases comparten
# interfaces abstractas y son sustituibles (L, I); las dependencias se
# inyectan por constructor (D).
#
# El módulo es **importable de forma independiente** (los tests de `tests/`
# lo importan directamente); por eso repite los imports de sec00, que en el
# notebook son operaciones sin coste (Python cachea los módulos ya cargados).

# %%
"""Imports del núcleo POO (idempotentes; ver explicación en el markdown anterior)."""
import json
import re
import types
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rdflib
from nltk.corpus import wordnet as wn
from owlready2 import (
    AnnotationProperty,
    ObjectProperty,
    Restriction,
    SymmetricProperty,
    Thing,
    get_ontology,
)

# =====================================================================
# Constantes del dominio (parámetros de los algoritmos de construcción
# y extracción). Las constantes del PROYECTO (rutas, colores, umbrales de
# EDA/selección, nombres de grupos) viven en sec00_setup.
# =====================================================================

IRI_BASE = "http://www.semanticweb.org/cev/tesauro#"

SLUG_RAIZ: dict[str, str] = {
    "Actores armados y otros participantes en las dinámicas de la guerra": "Actores_armados",
    "Impactos": "Impactos",
    "Armas": "Armas",
    "Poblaciones": "Poblaciones",
    "Hechos victimizantes": "Hechos_victimizantes",
    "Afrontamientos y resistencias no violentas - cotidianas": "Afrontamientos_resistencias_no_violentas",
}

EJE_POR_GRUPO: dict[str, str] = {
    "Actores armados y otros participantes en las dinámicas de la guerra": "agentes",
    "Impactos": "estados_efectos",
    "Armas": "instrumentos",
    "Poblaciones": "poblaciones",
    "Hechos victimizantes": "eventos",
    "Afrontamientos y resistencias no violentas - cotidianas": "afrontamientos_resistencias",
}

MAX_RELACIONES_AFINES = 45
MAX_PERPETRADOR = 15

# Modelo semántico (Fase 6): relaciones de dominio entre grupos y clases
# definidas con restricciones. Pares (grupo_origen, propiedad, grupo_destino).
RELACIONES_ENTRE_GRUPOS: tuple[tuple[str, str, str], ...] = (
    ("Hechos victimizantes", "perpetradoPor", "Actores armados y otros participantes en las dinámicas de la guerra"),
    ("Hechos victimizantes", "afectaA", "Poblaciones"),
    ("Hechos victimizantes", "generaImpacto", "Impactos"),
    ("Poblaciones", "experimentaImpacto", "Impactos"),
    ("Poblaciones", "desarrollaAfrontamiento", "Afrontamientos y resistencias no violentas - cotidianas"),
    ("Afrontamientos y resistencias no violentas - cotidianas", "respondeA", "Hechos victimizantes"),
    ("Actores armados y otros participantes en las dinámicas de la guerra", "empleaArma", "Armas"),
)

# Clases definidas: (nombre, clase_base, propiedad, clase_destino)
CLASES_DEFINIDAS: tuple[tuple[str, str, str, str], ...] = (
    ("HechoPerpetradoPorActor", "Hechos_victimizantes", "perpetradoPor", "Actores_armados"),
    ("PoblacionAfectadaPorHechos", "Poblaciones", "esAfectadaPor", "Hechos_victimizantes"),
    ("PoblacionConImpacto", "Poblaciones", "experimentaImpacto", "Impactos"),
    ("PoblacionConAfrontamiento", "Poblaciones", "desarrollaAfrontamiento", "Afrontamientos_resistencias_no_violentas"),
    ("HechoQueGeneraImpacto", "Hechos_victimizantes", "generaImpacto", "Impactos"),
    ("AfrontamientoAsociadoAHecho", "Afrontamientos_resistencias_no_violentas", "respondeA", "Hechos_victimizantes"),
)
UMBRAL_DF_AFINES = 10
UMBRAL_DF_PERPETRADOR = 8

TIPOS_RELACION: tuple[str, ...] = (
    "sinonimia",
    "antonimia",
    "hiponimia",
    "hiperonimia",
    "meronimia",
    "holonimia",
    "troponimia",
    "coordinados",
    "derivacion",
)

STOPWORDS: frozenset[str] = frozenset({
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por", "segun",
    "sin", "so", "sobre", "tras", "el", "la", "los", "las", "un", "una",
    "unos", "unas", "y", "o", "e", "u", "al", "del", "lo", "su", "sus",
})

TOKENS_GENERICOS: frozenset[str] = frozenset({
    "actores", "armados", "grupos", "conflicto", "armado", "territorio",
    "civiles", "formas", "revictimizacion", "asociadas", "parte", "nuevos",
    "cambios", "organizacion", "dinamicas", "espaciales", "percepciones",
    "ideas", "mismos", "aliados", "enemigos", "reincidencia", "control",
    "orden", "poblacion", "civil", "regulacion", "comunitaria", "contexto",
    "previo", "vinculacion", "combatientes", "experiencia",
    "posdesmovilizacion", "intrafilas", "cotidiano", "interior",
    "individual", "responsabilidad", "procesos", "formacion",
    "entrenamiento", "servicios", "salud", "acceso", "limitado",
    "condiciones", "sana", "otros", "vida", "familiar", "comunitario",
    "excombatientes", "sociales",
})

# %%
# =====================================================================
# Dominio: estructuras de datos del tesauro
# =====================================================================


@dataclass(frozen=True)
class Termino:
    """Descriptor del tesauro con su jerarquía ya parseada.

    Attributes:
        grupo: Grupo raíz (núcleo/dominio/directorio) al que pertenece.
        termino: Path completo original, p. ej. "Armas - Instrumentos - Daga".
        segmentos: Tupla de segmentos del path.
        es_hoja: True si ningún otro término del grupo extiende este path.
    """

    grupo: str
    termino: str
    segmentos: tuple[str, ...]
    es_hoja: bool = True

    @property
    def niveles(self) -> int:
        """Número de niveles jerárquicos del path."""
        return len(self.segmentos)

    @property
    def concepto(self) -> str:
        """Segmento final (el concepto en sí, sin la jerarquía)."""
        return self.segmentos[-1] if self.segmentos else ""


@dataclass
class ResultadoRelaciones:
    """Resultado de la extracción de relaciones WordNet para un concepto.

    Attributes:
        concepto: Concepto consultado (segmento final normalizado).
        encontrado: True si se hallaron synsets en español (omw-1.4).
        variante_usada: Variante del término que produjo los synsets.
        relaciones: Dict tipo de relación -> lista de nombres relacionados.
        synsets_consulta: Nombres de los synsets consultados.
    """

    concepto: str
    encontrado: bool
    variante_usada: str
    relaciones: dict[str, list[str]] = field(default_factory=dict)
    synsets_consulta: list[str] = field(default_factory=list)

    def conteos(self) -> dict[str, int]:
        """Conteo de relaciones por tipo (0 para los tipos ausentes)."""
        return {tipo: len(self.relaciones.get(tipo, [])) for tipo in TIPOS_RELACION}

    def total(self) -> int:
        """Número total de relaciones encontradas."""
        return sum(self.conteos().values())

    def a_json(self) -> dict[str, Any]:
        """Serialización compacta para CSV/JSON de trazabilidad."""
        return {
            "concepto": self.concepto,
            "encontrado": self.encontrado,
            "variante_usada": self.variante_usada,
            "conteos": self.conteos(),
            "ejemplos": {
                tipo: lista[:3] for tipo, lista in self.relaciones.items() if lista
            },
        }


# %%
# =====================================================================
# Strategy: normalización de términos
# =====================================================================


def _quitar_acentos(texto: str) -> str:
    """Elimina los diacríticos (acentos y virgulilla) de un texto."""
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


class NormalizadorTermino(ABC):
    """Strategy: normaliza un término del tesauro para consultarlo en WordNet."""

    @abstractmethod
    def normalizar(self, termino: str) -> str:
        """Retorna la variante normalizada del término."""


class NormalizadorBase(NormalizadorTermino):
    """Normalización base: último segmento del path, minúsculas, sin paréntesis."""

    def normalizar(self, termino: str) -> str:
        if not isinstance(termino, str) or not termino.strip():
            return ""
        base = termino.split(" - ")[-1].strip().lower()
        base = re.sub(r"\([^)]*\)", " ", base)
        base = re.sub(r"[^a-z0-9\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1\s-]+", " ", base)
        return re.sub(r"\s+", " ", base).strip()


class NormalizadorPlural(NormalizadorBase):
    """Variante singular/plural (heurística simple del español)."""

    @staticmethod
    def _singular(termino: str) -> str:
        if termino.endswith("es") and len(termino) > 3:
            return termino[:-2]
        if termino.endswith("s") and len(termino) > 2:
            return termino[:-1]
        return termino

    @staticmethod
    def _plural(termino: str) -> str:
        if termino.endswith("s") or not termino:
            return termino
        if termino[-1] in "aeiou" and termino[-1] not in "\u00e1\u00e9\u00ed\u00f3\u00fa":
            return termino + "s"
        return termino + "es"

    def normalizar(self, termino: str) -> str:
        base = super().normalizar(termino)
        if base.endswith("s"):
            return self._singular(base)
        return self._plural(base)


class NormalizadorAcentos(NormalizadorBase):
    """Variante sin acentos (fallback cuando la forma acentuada no existe)."""

    def normalizar(self, termino: str) -> str:
        return _quitar_acentos(super().normalizar(termino))


class CadenaNormalizadores:
    """Genera la lista ordenada de variantes de búsqueda para un término.

    Cada normalizador produce una variante; se deduplican conservando el
    orden. Añade además las variantes multiword con guiones bajos y los
    tokens extremos de los compuestos (núcleos léxicos del tesauro).
    """

    def __init__(self, normalizadores: list[NormalizadorTermino] | None = None) -> None:
        self._normalizadores = normalizadores or [
            NormalizadorBase(),
            NormalizadorPlural(),
            NormalizadorAcentos(),
        ]

    def candidatos(self, termino: str) -> list[str]:
        """Lista de variantes únicas, en orden de prioridad."""
        base = self._normalizadores[0].normalizar(termino)
        if not base:
            return []
        candidatos = [normalizador.normalizar(termino) for normalizador in self._normalizadores]
        candidatos.append(base.replace(" ", "_"))
        tokens = base.split()
        if len(tokens) > 1:
            candidatos.extend([tokens[-1], tokens[0]])
        vistos: set[str] = set()
        unicos: list[str] = []
        for candidato in candidatos:
            if candidato and candidato not in vistos:
                vistos.add(candidato)
                unicos.append(candidato)
        return unicos


# %%
# =====================================================================
# Repository: acceso a los datos del tesauro
# =====================================================================


class TesauroRepositorio:
    """Repository: acceso único y cacheado a las fuentes del tesauro.

    Encapsula la lectura del Excel (hojas "Tesauro" y "Listas") y del JSON
    de grupos seleccionados, de modo que ninguna otra sección conozca los
    detalles de formato. Los DataFrames se cachean (DRY y rendimiento).

    Attributes:
        ruta_excel: Ruta al archivo tesauro_orginal_cev.xlsx.
        ruta_json: Ruta a grupos_seleccionados.json (opcional).
    """

    COLUMNA_GRUPO = 1
    COLUMNA_TERMINO = 2

    def __init__(self, ruta_excel: Path, ruta_json: Path | None = None) -> None:
        self.ruta_excel = Path(ruta_excel)
        self.ruta_json = Path(ruta_json) if ruta_json else None
        self._cache_tesauro: pd.DataFrame | None = None
        self._cache_listas: pd.DataFrame | None = None
        self._cache_grupos: dict[str, Any] | None = None

    def cargar_tesauro(self) -> pd.DataFrame:
        """Carga la hoja "Tesauro" (12.341 términos) limpia y cacheada.

        La hoja no tiene encabezados: col1 = grupo, col2 = término con la
        jerarquía en el propio texto (separador " - ").

        Returns:
            DataFrame con columnas `grupo` y `termino`.

        Raises:
            FileNotFoundError: Si el Excel no existe.
        """
        if self._cache_tesauro is not None:
            return self._cache_tesauro
        if not self.ruta_excel.exists():
            raise FileNotFoundError(f"Excel no encontrado: {self.ruta_excel}")
        df = pd.read_excel(
            self.ruta_excel,
            sheet_name="Tesauro",
            header=None,
            skiprows=2,
            usecols=[self.COLUMNA_GRUPO, self.COLUMNA_TERMINO],
        )
        df.columns = ["grupo", "termino"]
        df["grupo"] = df["grupo"].astype(str).str.strip()
        df["termino"] = df["termino"].astype(str).str.strip()
        self._cache_tesauro = df.dropna(subset=["grupo", "termino"]).reset_index(drop=True)
        return self._cache_tesauro

    def cargar_listas(self) -> pd.DataFrame:
        """Carga la hoja "Listas" (6.880 términos por núcleo) cacheada."""
        if self._cache_listas is not None:
            return self._cache_listas
        if not self.ruta_excel.exists():
            raise FileNotFoundError(f"Excel no encontrado: {self.ruta_excel}")
        df = pd.read_excel(
            self.ruta_excel, sheet_name="Listas", header=None, skiprows=2, usecols=[0, 1]
        )
        df.columns = ["agrupacion", "termino"]
        df = df.dropna().reset_index(drop=True)
        df["termino"] = df["termino"].astype(str).str.strip()
        self._cache_listas = df
        return self._cache_listas

    def grupos_seleccionados(self) -> dict[str, Any]:
        """Carga (y cachea) el JSON con los 5 grupos seleccionados de sec02.

        Returns:
            Dict completo del JSON (grupos, justificaciones, términos).

        Raises:
            FileNotFoundError: Si el JSON no existe.
        """
        if self._cache_grupos is not None:
            return self._cache_grupos
        if self.ruta_json is None or not self.ruta_json.exists():
            raise FileNotFoundError(f"JSON no encontrado: {self.ruta_json}")
        with open(self.ruta_json, encoding="utf-8") as archivo:
            self._cache_grupos = json.load(archivo)
        return self._cache_grupos

    def terminos_por_grupo(self) -> dict[str, list[str]]:
        """Paths completos de los 5 grupos seleccionados (para la ontología)."""
        return self.grupos_seleccionados()["terminos_por_grupo"]

    @staticmethod
    def con_estructura(df: pd.DataFrame) -> pd.DataFrame:
        """Añade columnas derivadas del path: segmentos, niveles y hoja."""
        df = df.copy()
        df["segmentos"] = df["termino"].str.split(" - ")
        df["niveles"] = df["segmentos"].str.len()
        df["hoja"] = df["segmentos"].str[-1].str.strip()
        return df

    @staticmethod
    def marcar_hojas(df: pd.DataFrame) -> pd.DataFrame:
        """Marca nodos hoja: un nodo es hoja si su path no es prefijo de otro."""
        df = df.copy()
        df["es_hoja"] = ~df["termino"].apply(
            lambda t: bool((df["termino"].str.startswith(t + " - ")).sum() > 0)
        )
        return df

    @staticmethod
    def detectar_anomalias(df: pd.DataFrame) -> pd.DataFrame:
        """Detecta patrones anómalos por grupo (duplicados, vacíos, notas)."""
        t = df["termino"]
        anomalias = pd.DataFrame({
            "duplicados": t.duplicated(keep=False),
            "vacios": t.str.len() == 0,
            "separador_final": t.str.endswith(" - ") | t.str.endswith("-"),
            "con_parentesis": t.str.contains(r"\(|\)", regex=True),
            "con_comillas": t.str.contains(r'["\u201c\u201d]', regex=True),
            "acronimo": t.str.contains(
                r"^(?=.*[A-Z])[A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00d1 ]{3,}$", regex=True
            ),
        })
        anomalias["grupo"] = df["grupo"]
        return anomalias.groupby("grupo").sum()


# %%
# =====================================================================
# Strategy: extractores de relaciones WordNet
# =====================================================================


def _nombre_espanol_synset(synset) -> str:
    """Nombre en español de un synset (omw-1.4), con fallback PWN marcado.

    Args:
        synset: Synset de NLTK WordNet.

    Returns:
        Primer lemma en español; si no existe, nombre PWN con marca "(en)".
    """
    lemmas_spa = synset.lemma_names("spa")
    return lemmas_spa[0] if lemmas_spa else f"{synset.name()} (en)"


class ExtractorRelacion(ABC):
    """Strategy: extrae un tipo de relación semántica de un synset WordNet."""

    nombre: ClassVar[str] = ""

    @abstractmethod
    def extraer(self, synset, profundidad: int = 1) -> list[tuple[str, str]]:
        """Extrae las entidades relacionadas con el synset.

        Args:
            synset: Synset de referencia.
            profundidad: Niveles de recursión (hiponimia/hiperonimia).

        Returns:
            Lista de tuplas (nombre_legible, synset_destino).
        """


class SinonimiaExtractor(ExtractorRelacion):
    """Sinónimos: lemmas en español del propio synset."""

    nombre = "sinonimia"

    def extraer(self, synset, profundidad: int = 1) -> list[tuple[str, str]]:
        return [
            (lemma, synset.name()) for lemma in synset.lemma_names("spa") if lemma
        ]


class AntonimiaExtractor(ExtractorRelacion):
    """Antónimos: enlaces de antonimia de los lemmas ingleses (PWN), retraducidos."""

    nombre = "antonimia"

    def extraer(self, synset, profundidad: int = 1) -> list[tuple[str, str]]:
        resultados = []
        for lemma in synset.lemmas():
            for antonimo in lemma.antonyms():
                resultados.append(
                    (_nombre_espanol_synset(antonimo.synset()), antonimo.synset().name())
                )
        return resultados


class DerivacionExtractor(ExtractorRelacion):
    """Formas derivadas morfológicamente (lemmas ingleses, retraducidas)."""

    nombre = "derivacion"

    def extraer(self, synset, profundidad: int = 1) -> list[tuple[str, str]]:
        resultados = []
        for lemma in synset.lemmas():
            for derivado in lemma.derivationally_related_forms():
                resultados.append(
                    (_nombre_espanol_synset(derivado.synset()), derivado.synset().name())
                )
        return resultados


class HiponimiaExtractor(ExtractorRelacion):
    """Hipónimos: synsets más específicos, con recursión de profundidad."""

    nombre = "hiponimia"

    def extraer(self, synset, profundidad: int = 1) -> list[tuple[str, str]]:
        resultados: list[tuple[str, str]] = []
        visitados: set[str] = set()

        def explorar(synset_actual, nivel: int) -> None:
            if nivel < 0 or synset_actual.name() in visitados:
                return
            visitados.add(synset_actual.name())
            for hijo in synset_actual.hyponyms():
                resultados.append((_nombre_espanol_synset(hijo), hijo.name()))
                explorar(hijo, nivel - 1)

        explorar(synset, profundidad)
        return resultados


class HiperonimiaExtractor(ExtractorRelacion):
    """Hiperónimos: synsets más generales, con recursión de profundidad."""

    nombre = "hiperonimia"

    def extraer(self, synset, profundidad: int = 1) -> list[tuple[str, str]]:
        resultados: list[tuple[str, str]] = []
        visitados: set[str] = set()

        def explorar(synset_actual, nivel: int) -> None:
            if nivel < 0 or synset_actual.name() in visitados:
                return
            visitados.add(synset_actual.name())
            for padre in synset_actual.hypernyms():
                resultados.append((_nombre_espanol_synset(padre), padre.name()))
                explorar(padre, nivel - 1)

        explorar(synset, profundidad)
        return resultados


class MeronimiaExtractor(ExtractorRelacion):
    """Merónimos: partes/sustancias/miembros del concepto."""

    nombre = "meronimia"

    def extraer(self, synset, profundidad: int = 1) -> list[tuple[str, str]]:
        meronimos = (
            synset.part_meronyms()
            + synset.substance_meronyms()
            + synset.member_meronyms()
        )
        return [(_nombre_espanol_synset(m), m.name()) for m in meronimos]


class HolonimiaExtractor(ExtractorRelacion):
    """Holónimos: conceptos de los que el synset es parte/sustancia/miembro."""

    nombre = "holonimia"

    def extraer(self, synset, profundidad: int = 1) -> list[tuple[str, str]]:
        holonimos = (
            synset.part_holonyms()
            + synset.substance_holonyms()
            + synset.member_holonyms()
        )
        return [(_nombre_espanol_synset(h), h.name()) for h in holonimos]


class TroponimiaExtractor(ExtractorRelacion):
    """Tropónimos: hipónimos de un verbo (maneras específicas de hacer).

    NLTK 3.9 no expone `Synset.troponyms()`; en el PWN los tropónimos son
    los hipónimos de un synset verbal.
    """

    nombre = "troponimia"

    def extraer(self, synset, profundidad: int = 1) -> list[tuple[str, str]]:
        if synset.pos() != "v":
            return []
        return [(_nombre_espanol_synset(t), t.name()) for t in synset.hyponyms()]


class CoordinadosExtractor(ExtractorRelacion):
    """Términos coordinados: hermanos bajo el mismo hiperónimo directo."""

    nombre = "coordinados"

    def extraer(self, synset, profundidad: int = 1) -> list[tuple[str, str]]:
        resultados = []
        for hiperonimo in synset.hypernyms():
            for hermano in hiperonimo.hyponyms():
                if hermano.name() != synset.name():
                    resultados.append((_nombre_espanol_synset(hermano), hermano.name()))
        return resultados


# %%
# =====================================================================
# Abstract Factory + Registry: fábrica de extractores
# =====================================================================


class FabricaExtractores:
    """Abstract Factory + Registry de extractores de relaciones WordNet.

    Cada tipo de relación se registra con su clase; `crear(tipo)` devuelve
    una instancia. Añadir una relación nueva implica registrar una clase
    nueva, sin modificar código existente (principio Open/Closed).
    """

    _registro: ClassVar[dict[str, type[ExtractorRelacion]]] = {}

    @classmethod
    def registrar(cls, tipo: str, clase: type[ExtractorRelacion]) -> None:
        """Registra una clase extractora bajo un nombre de tipo."""
        cls._registro[tipo] = clase

    @classmethod
    def crear(cls, tipo: str) -> ExtractorRelacion:
        """Crea una instancia del extractor del tipo pedido.

        Raises:
            KeyError: Si el tipo no está registrado.
        """
        if tipo not in cls._registro:
            raise KeyError(f"Tipo de relación no registrado: {tipo}")
        return cls._registro[tipo]()

    @classmethod
    def crear_todos(cls) -> list[ExtractorRelacion]:
        """Crea una instancia de cada extractor registrado (en orden de tipo)."""
        return [cls.crear(tipo) for tipo in TIPOS_RELACION]

    @classmethod
    def tipos_registrados(cls) -> list[str]:
        """Tipos de relación disponibles (orden canónico)."""
        return [tipo for tipo in TIPOS_RELACION if tipo in cls._registro]


# Registro de las 9 relaciones mínimas exigidas por el taller.
FabricaExtractores.registrar("sinonimia", SinonimiaExtractor)
FabricaExtractores.registrar("antonimia", AntonimiaExtractor)
FabricaExtractores.registrar("hiponimia", HiponimiaExtractor)
FabricaExtractores.registrar("hiperonimia", HiperonimiaExtractor)
FabricaExtractores.registrar("meronimia", MeronimiaExtractor)
FabricaExtractores.registrar("holonimia", HolonimiaExtractor)
FabricaExtractores.registrar("troponimia", TroponimiaExtractor)
FabricaExtractores.registrar("coordinados", CoordinadosExtractor)
FabricaExtractores.registrar("derivacion", DerivacionExtractor)


# %%
# =====================================================================
# Facade: motor de relaciones WordNet
# =====================================================================


class MotorRelacionesWordNet:
    """Facade que orquesta normalización + fábrica de extractores + NLTK.

    Recibe un término del tesauro (path jerárquico o concepto) y devuelve
    un `ResultadoRelaciones` con las 9 relaciones encontradas en WordNet
    español (omw-1.4), con trazabilidad de la variante usada.

    Attributes:
        cadena: CadenaNormalizadores que genera las variantes de búsqueda.
        extractores: Lista de extractores a aplicar.
    """

    def __init__(
        self,
        cadena: CadenaNormalizadores | None = None,
        extractores: list[ExtractorRelacion] | None = None,
    ) -> None:
        self.cadena = cadena or CadenaNormalizadores()
        self.extractores = extractores or FabricaExtractores.crear_todos()
        self._cache: dict[str, ResultadoRelaciones] = {}

    def synsets_de(self, termino: str) -> tuple[list[Any], str]:
        """Busca los synsets en español del término usando sus variantes.

        Returns:
            Tupla (synsets_encontrados, variante_usada); vacíos si no existe.
        """
        for variante in self.cadena.candidatos(termino):
            synsets = wn.synsets(variante, lang="spa")
            if synsets:
                return synsets, variante
        return [], ""

    def extraer_relaciones(self, termino: str, profundidad: int = 1) -> ResultadoRelaciones:
        """Extrae las 9 relaciones de WordNet para un término del tesauro.

        El resultado se cachea por término (mismo concepto repetido en
        varias consultas del análisis masivo).

        Args:
            termino: Término del tesauro (path "A - B - C" o concepto).
            profundidad: Niveles de recursión para hiponimia/hiperonimia.

        Returns:
            ResultadoRelaciones con conteos y ejemplos por tipo.
        """
        if termino in self._cache:
            return self._cache[termino]
        synsets, variante = self.synsets_de(termino)
        resultado = ResultadoRelaciones(
            concepto=self.cadena.candidatos(termino)[0] if self.cadena.candidatos(termino) else "",
            encontrado=bool(synsets),
            variante_usada=variante,
        )
        if not synsets:
            self._cache[termino] = resultado
            return resultado
        for synset in synsets:
            resultado.synsets_consulta.append(synset.name())
            for extractor in self.extractores:
                pares = extractor.extraer(synset, profundidad=profundidad)
                nombres = dict.fromkeys(nombre for nombre, _ in pares)
                resultado.relaciones.setdefault(extractor.nombre, []).extend(nombres)
        self._cache[termino] = resultado
        return resultado

    def analizar_conceptos(self, conceptos: pd.DataFrame) -> pd.DataFrame:
        """Aplica la extracción a un DataFrame de conceptos y lo enriquece.

        El DataFrame de entrada debe tener columnas `grupo`, `termino`
        (path completo) y `concepto` (segmento final). Se añaden columnas
        con los conteos por tipo de relación, la variante usada y un JSON
        compacto de ejemplos.

        Returns:
            Nuevo DataFrame con las columnas de conteos y trazabilidad.
        """
        filas = []
        for fila in conceptos.itertuples(index=False):
            resultado = self.extraer_relaciones(fila.termino)
            conteos = resultado.conteos()
            filas.append({
                "grupo": fila.grupo,
                "termino": fila.termino,
                "concepto": fila.concepto,
                "encontrado": resultado.encontrado,
                "variante_usada": resultado.variante_usada,
                **conteos,
                "total_relaciones": resultado.total(),
                "ejemplos": json.dumps(resultado.a_json()["ejemplos"], ensure_ascii=False),
            })
        return pd.DataFrame(filas)


# %%
# =====================================================================
# Singleton + Facade: servicio de la ontología
# =====================================================================


class ServicioOntologia:
    """Singleton (por ruta) + Facade sobre owlready2.

    Centraliza la construcción, guardado, carga, validación y consulta de
    la ontología del tesauro. Todas las secciones (sec03, sec04, sec05)
    usan la misma instancia: la OWL se carga una sola vez (rendimiento y
    consistencia).

    Attributes:
        ruta_owl: Ruta del archivo .owl (RDF/XML).
        ruta_json: Ruta del JSON con los grupos seleccionados (construcción).
        ruta_trazabilidad: Ruta del .txt de trazabilidad de relaciones TR.
    """

    _instancias: ClassVar[dict[str, "ServicioOntologia"]] = {}

    def __init__(
        self,
        ruta_owl: Path,
        ruta_json: Path | None = None,
        ruta_trazabilidad: Path | None = None,
    ) -> None:
        self.ruta_owl = Path(ruta_owl)
        self.ruta_json = Path(ruta_json) if ruta_json else None
        self.ruta_trazabilidad = (
            Path(ruta_trazabilidad) if ruta_trazabilidad else self.ruta_owl.parent / "relaciones_afines.txt"
        )
        self.onto = None
        self.resumen_construccion: dict[str, Any] = {}

    @classmethod
    def obtener(
        cls,
        ruta_owl: Path,
        ruta_json: Path | None = None,
        ruta_trazabilidad: Path | None = None,
    ) -> "ServicioOntologia":
        """Devuelve la instancia única para una ruta de OWL (Singleton).

        Si la instancia ya existe se reutiliza; los parámetros extra solo
        se aplican en la primera creación.
        """
        clave = str(Path(ruta_owl).resolve())
        if clave not in cls._instancias:
            cls._instancias[clave] = cls(ruta_owl, ruta_json, ruta_trazabilidad)
        return cls._instancias[clave]

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    def construir_desde_grupos(self) -> dict[str, Any]:
        """Construye la ontología completa desde el JSON de grupos.

        Reglas:
        1. Cinco clases raíz (una por grupo) subclases de ConceptoDelTesauro.
        2. Un nodo = una clase; superclase = segmento anterior del path.
        3. Nombres locales únicos (colisión resuelta con sufijo del grupo).
        4. Anotaciones: rdfs:label, rdfs:comment (path original), cev:grupo,
           cev:eje.
        5. Un individuo por término hoja (entidad consultable en SPARQL).
        6. Object properties: esParteDe/parteDe (inversas, axioma por edge),
           relacionadoCon (simétrica, heurística IDF + raíces fijas CEV) y
           perpetradoPor/esVictimaDe (inversas, co-ocurrencia específica).

        Returns:
            Resumen con conteos de la construcción.

        Raises:
            FileNotFoundError: Si no hay JSON y la OWL no existe.
        """
        if self.ruta_json is None or not self.ruta_json.exists():
            raise FileNotFoundError(
                f"JSON de grupos no encontrado para construir: {self.ruta_json}"
            )
        terminos = json.loads(self.ruta_json.read_text(encoding="utf-8"))["terminos_por_grupo"]
        nodos = self._construir_nodos(terminos)
        self._marcar_hojas(nodos)

        onto = get_ontology(IRI_BASE)
        with onto:
            onto.comment = [
                (
                    "Ontología del Tesauro de la Comisión de la Verdad (CEV). "
                    "TALLER-2: 586 términos seleccionados, 5 grupos, jerarquía "
                    "rdfs:subClassOf, individuos por término hoja y object "
                    "properties esParteDe/parteDe, relacionadoCon y "
                    "perpetradoPor/esVictimaDe."
                )
            ]
            self._definir_anotaciones(onto)
            clases_por_nodo = self._crear_clases_y_jerarquia(onto, nodos)
            individuos = self._crear_individuos_hoja(onto, nodos, clases_por_nodo)
            propiedades = self._definir_propiedades(onto)
            axiomas_parte = self._agregar_axiomas_es_parte_de(onto, nodos, clases_por_nodo)
            afines, raiz = self._agregar_relaciones_afines(onto, nodos, clases_por_nodo)
            perpetradores = self._calcular_perpetradores(onto, nodos, clases_por_nodo)
            dominio = self._agregar_relaciones_entre_grupos(onto)
            definidas = self._definir_clases_definidas(onto)

        self.onto = onto
        self.resumen_construccion = {
            "clases": len(list(onto.classes())),
            "individuos": individuos,
            "nodos": len(nodos),
            "hojas": sum(1 for n in nodos if n["es_hoja"]),
            "axiomas_parte": axiomas_parte,
            "relaciones_afines": afines + raiz,
            "perpetradores": perpetradores,
            "relaciones_dominio": dominio,
            "clases_definidas": definidas,
            "object_properties": len(propiedades),
            "annotation_properties": 2,
        }
        return self.resumen_construccion

    def _agregar_relaciones_entre_grupos(self, onto) -> int:
        """Declara axiomas raíz del modelo semántico (ciclo del conflicto).

        Para cada par (grupo_origen, propiedad, grupo_destino) de
        RELACIONES_ENTRE_GRUPOS se añade el axioma de clase:
        `GrupoOrigen subClassOf (propiedad some GrupoDestino)`.
        Con ello el razonador puede clasificar las clases definidas
        (ver _definir_clases_definidas) y responder consultas del ciclo
        actor -> hecho -> población -> impacto -> afrontamiento.

        Returns:
            Número de axiomas de dominio creados.
        """
        clases_raiz = {grupo: onto[SLUG_RAIZ[grupo]] for grupo in SLUG_RAIZ}
        creados = 0
        for grupo_origen, propiedad, grupo_destino in RELACIONES_ENTRE_GRUPOS:
            clases_raiz[grupo_origen].is_a.append(
                onto[propiedad].some(clases_raiz[grupo_destino])
            )
            creados += 1
        return creados

    def _definir_clases_definidas(self, onto) -> int:
        """Crea las clases definidas con restricciones (equivalentClass).

        Cada clase definida es una subclase de su grupo base y equivale a
        la intersección `Base AND (propiedad SOME Destino)`. HermiT puede
        clasificar automáticamente cualquier clase/individuo que satisfaga
        la restricción (p. ej. PoblacionAfectadaPorHechos).

        Returns:
            Número de clases definidas creadas.
        """
        creados = 0
        for nombre, clase_base, propiedad, clase_destino in CLASES_DEFINIDAS:
            base = onto[clase_base]
            definida = types.new_class(nombre, (base,))
            definida.label = [nombre.replace("_", " ")]
            definida.comment = [
                (
                    f"Clase definida: {clase_base.replace('_', ' ')} AND "
                    f"({propiedad} SOME {clase_destino.replace('_', ' ')})"
                )
            ]
            definida.equivalent_to = [
                base & onto[propiedad].some(onto[clase_destino])
            ]
            creados += 1
        return creados

    @staticmethod
    def _construir_nodos(terminos_por_grupo: dict[str, list[str]]) -> list[dict]:
        """Convierte paths "A - B - C" en nodos únicos (con sintéticos)."""
        nodos: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        for grupo, paths in terminos_por_grupo.items():
            for path in paths:
                segmentos = tuple(s.strip() for s in path.split(" - "))
                for indice in range(1, len(segmentos) + 1):
                    clave = (grupo, segmentos[:indice])
                    if clave not in nodos:
                        nodos[clave] = {
                            "grupo": grupo,
                            "segs": segmentos[:indice],
                            "label": segmentos[indice - 1],
                            "nivel": indice,
                            "path_original": " - ".join(segmentos[:indice]),
                            "es_termino": False,
                            "tokens": set(),
                        }
                    if indice == len(segmentos):
                        nodos[clave]["es_termino"] = True
        return list(nodos.values())

    @staticmethod
    def _marcar_hojas(nodos: list[dict]) -> None:
        """Marca in-place los nodos hoja (sin hijos en su grupo)."""
        segs_por_grupo: dict[str, set[tuple[str, ...]]] = {}
        for nodo in nodos:
            segs_por_grupo.setdefault(nodo["grupo"], set()).add(nodo["segs"])
        for nodo in nodos:
            posibles = segs_por_grupo.get(nodo["grupo"], set())
            nodo["es_hoja"] = not any(
                len(segs_hijo) == nodo["nivel"] + 1
                and segs_hijo[: nodo["nivel"]] == nodo["segs"]
                for segs_hijo in posibles
            )

    @staticmethod
    def _definir_anotaciones(onto) -> None:
        """Crea las annotation properties cev:grupo y cev:eje."""
        grupo = types.new_class("grupo", (AnnotationProperty,))
        eje = types.new_class("eje", (AnnotationProperty,))
        grupo.label = ["grupo raiz"]
        grupo.comment = ["Grupo raiz del tesauro al que pertenece el concepto."]
        eje.label = ["eje conceptual"]
        eje.comment = ["Eje conceptual del grupo (filtro de seleccion)."]

    @staticmethod
    def _crear_clase_unica(onto, slug_deseado: str, grupo: str, superclase):
        """Crea una clase con nombre local único (colisión -> sufijo de grupo)."""
        candidatos = [slug_deseado, f"{slug_deseado}_{SLUG_RAIZ[grupo]}"]
        nombre = None
        for candidato in candidatos:
            if candidato not in onto.classes():
                nombre = candidato
                break
        if nombre is None:
            contador = 2
            while f"{slug_deseado}_{contador}" in onto.classes():
                contador += 1
            nombre = f"{slug_deseado}_{contador}"
        return types.new_class(nombre, (superclase,))

    @staticmethod
    def _anotar_concepto(clase, nodo: dict) -> None:
        """Anota label, comment (path original), grupo raíz y eje conceptual."""
        clase.label = [nodo["label"]]
        clase.comment = [nodo["path_original"]]
        clase.grupo = [nodo["grupo"]]
        clase.eje = [EJE_POR_GRUPO[nodo["grupo"]]]

    def _crear_clases_y_jerarquia(self, onto, nodos: list[dict]) -> dict:
        """Crea todas las clases con su jerarquía rdfs:subClassOf."""
        concepto = types.new_class("ConceptoDelTesauro", (Thing,))
        concepto.label = ["Concepto del tesauro de la Comision de la Verdad"]
        clases_por_nodo: dict[tuple[str, tuple[str, ...]], Any] = {}
        for nodo in nodos:
            grupo, segmentos = nodo["grupo"], nodo["segs"]
            superclase = concepto if nodo["nivel"] == 1 else clases_por_nodo[(grupo, segmentos[:-1])]
            slug = SLUG_RAIZ[grupo] if nodo["nivel"] == 1 else self._slug(nodo["label"])
            clase = self._crear_clase_unica(onto, slug, grupo, superclase)
            self._anotar_concepto(clase, nodo)
            clases_por_nodo[(grupo, segmentos)] = clase
        return clases_por_nodo

    def _crear_individuos_hoja(self, onto, nodos: list[dict], clases_por_nodo: dict) -> int:
        """Crea un individuo (NamedIndividual) por cada término hoja del JSON."""
        creados = 0
        for nodo in nodos:
            if not (nodo["es_hoja"] and nodo["es_termino"]):
                continue
            clase = clases_por_nodo[(nodo["grupo"], nodo["segs"])]
            individuo = clase(f"instancia_{self._slug(nodo['label'])}")
            self._anotar_concepto(individuo, nodo)
            creados += 1
        return creados

    def _definir_propiedades(self, onto) -> dict[str, ObjectProperty]:
        """Crea las 17 object properties con dominio/rango ConceptoDelTesauro.

        Dos familias:
        1. Estructurales: esParteDe/parteDe (meronimia del tesauro),
           relacionadoCon (TR simétrica) y perpetradoPor/esVictimaDe.
        2. Semánticas de dominio (modelo fuerte): pares inversos que
           formalizan el ciclo del conflicto: afectaA/esAfectadaPor
           (hecho->población), generaImpacto/esGeneradoPor (hecho->impacto),
           experimentaImpacto/impactaA (población<->impacto),
           desarrollaAfrontamiento/esDesarrolladoPor (población->afrontamiento),
           respondeA/esRespondidoMediante (afrontamiento->hecho/impacto) y
           empleaArma/esEmpleadaEn (actor<->armas).
        """
        concepto = onto.ConceptoDelTesauro
        es_parte_de = types.new_class("esParteDe", (ObjectProperty,))
        parte_de = types.new_class("parteDe", (ObjectProperty,))
        relacionado_con = types.new_class("relacionadoCon", (ObjectProperty, SymmetricProperty))
        perpetrado_por = types.new_class("perpetradoPor", (ObjectProperty,))
        es_victima_de = types.new_class("esVictimaDe", (ObjectProperty,))
        afecta_a = types.new_class("afectaA", (ObjectProperty,))
        es_afectada_por = types.new_class("esAfectadaPor", (ObjectProperty,))
        genera_impacto = types.new_class("generaImpacto", (ObjectProperty,))
        es_generado_por = types.new_class("esGeneradoPor", (ObjectProperty,))
        experimenta_impacto = types.new_class("experimentaImpacto", (ObjectProperty,))
        impacta_a = types.new_class("impactaA", (ObjectProperty,))
        desarrolla_afrontamiento = types.new_class("desarrollaAfrontamiento", (ObjectProperty,))
        es_desarrollado_por = types.new_class("esDesarrolladoPor", (ObjectProperty,))
        responde_a = types.new_class("respondeA", (ObjectProperty,))
        es_respondido_mediante = types.new_class("esRespondidoMediante", (ObjectProperty,))
        emplea_arma = types.new_class("empleaArma", (ObjectProperty,))
        es_empleada_en = types.new_class("esEmpleadaEn", (ObjectProperty,))

        todas = (
            es_parte_de, parte_de, relacionado_con, perpetrado_por, es_victima_de,
            afecta_a, es_afectada_por, genera_impacto, es_generado_por,
            experimenta_impacto, impacta_a, desarrolla_afrontamiento,
            es_desarrollado_por, responde_a, es_respondido_mediante,
            emplea_arma, es_empleada_en,
        )
        for propiedad in todas:
            propiedad.domain = [concepto]
            propiedad.range = [concepto]

        es_parte_de.label = ["es parte de"]
        es_parte_de.comment = [
            (
                "Meronimia del tesauro: el concepto hijo es parte del concepto padre. "
                "Se declara por axioma de clase para cada edge de la jerarquia."
            )
        ]
        parte_de.label = ["parte de"]
        parte_de.comment = ["Propiedad inversa de cev:esParteDe."]
        relacionado_con.label = ["relacionado con"]
        relacionado_con.comment = [
            (
                "Termino relacionado (TR) del tesauro: afinidad terminologica entre "
                "conceptos de grupos distintos. Simetrica. Ver criterio en relaciones_afines.txt."
            )
        ]
        perpetrado_por.label = ["perpetrado por"]
        perpetrado_por.comment = [
            (
                "Vinculo hecho victimizante - actor armado por co-ocurrencia "
                "terminologica especifica en el marco del conflicto (criterio en relaciones_afines.txt)."
            )
        ]
        es_victima_de.label = ["es victima de"]
        es_victima_de.comment = ["Propiedad inversa de cev:perpetradoPor."]

        # Pares inversos del modelo semántico de dominio
        afecta_a.label = ["afecta a"]
        afecta_a.comment = [
            "Un hecho victimizante afecta a una poblacion (modelo del ciclo del conflicto)."
        ]
        es_afectada_por.label = ["es afectada por"]
        es_afectada_por.comment = ["Propiedad inversa de cev:afectaA."]
        genera_impacto.label = ["genera impacto"]
        genera_impacto.comment = ["Un hecho victimizante genera impactos en poblaciones y territorios."]
        es_generado_por.label = ["es generado por"]
        es_generado_por.comment = ["Propiedad inversa de cev:generaImpacto."]
        experimenta_impacto.label = ["experimenta impacto"]
        experimenta_impacto.comment = ["Una poblacion experimenta los impactos del conflicto."]
        impacta_a.label = ["impacta a"]
        impacta_a.comment = ["Propiedad inversa de cev:experimentaImpacto."]
        desarrolla_afrontamiento.label = ["desarrolla afrontamiento"]
        desarrolla_afrontamiento.comment = [
            "Una poblacion desarrolla afrontamientos y resistencias no violentas."
        ]
        es_desarrollado_por.label = ["es desarrollado por"]
        es_desarrollado_por.comment = ["Propiedad inversa de cev:desarrollaAfrontamiento."]
        responde_a.label = ["responde a"]
        responde_a.comment = [
            "Un afrontamiento o resistencia responde a un hecho victimizante o un impacto."
        ]
        es_respondido_mediante.label = ["es respondido mediante"]
        es_respondido_mediante.comment = ["Propiedad inversa de cev:respondeA."]
        emplea_arma.label = ["emplea arma"]
        emplea_arma.comment = ["Un actor armado emplea armas en el conflicto."]
        es_empleada_en.label = ["es empleada en"]
        es_empleada_en.comment = ["Propiedad inversa de cev:empleaArma."]

        es_parte_de.inverse_property = parte_de
        perpetrado_por.inverse_property = es_victima_de
        afecta_a.inverse_property = es_afectada_por
        genera_impacto.inverse_property = es_generado_por
        experimenta_impacto.inverse_property = impacta_a
        desarrolla_afrontamiento.inverse_property = es_desarrollado_por
        responde_a.inverse_property = es_respondido_mediante
        emplea_arma.inverse_property = es_empleada_en

        return {
            "es_parte_de": es_parte_de,
            "parte_de": parte_de,
            "relacionado_con": relacionado_con,
            "perpetrado_por": perpetrado_por,
            "es_victima_de": es_victima_de,
            "afecta_a": afecta_a,
            "es_afectada_por": es_afectada_por,
            "genera_impacto": genera_impacto,
            "es_generado_por": es_generado_por,
            "experimenta_impacto": experimenta_impacto,
            "impacta_a": impacta_a,
            "desarrolla_afrontamiento": desarrolla_afrontamiento,
            "es_desarrollado_por": es_desarrollado_por,
            "responde_a": responde_a,
            "es_respondido_mediante": es_respondido_mediante,
            "emplea_arma": emplea_arma,
            "es_empleada_en": es_empleada_en,
        }

    @staticmethod
    def _agregar_axiomas_es_parte_de(onto, nodos: list[dict], clases_por_nodo: dict) -> int:
        """Declara `hijo subClassOf (esParteDe some padre)` por cada edge."""
        axiomas = 0
        for nodo in nodos:
            if nodo["nivel"] == 1:
                continue
            hijo = clases_por_nodo[(nodo["grupo"], nodo["segs"])]
            padre = clases_por_nodo[(nodo["grupo"], nodo["segs"][:-1])]
            hijo.is_a.append(onto.esParteDe.some(padre))
            axiomas += 1
        return axiomas

    def _agregar_relaciones_afines(self, onto, nodos: list[dict], clases_por_nodo: dict) -> tuple[int, int]:
        """Crea cev:relacionadoCon (heurística IDF + 5 raíces fijas CEV)."""
        relaciones_raiz = [
            ("Hechos victimizantes", "Actores armados y otros participantes en las dinámicas de la guerra"),
            ("Armas", "Actores armados y otros participantes en las dinámicas de la guerra"),
            ("Armas", "Hechos victimizantes"),
            ("Impactos", "Hechos victimizantes"),
            ("Impactos", "Poblaciones"),
        ]
        pares = self._calcular_relaciones_afines(nodos)
        clases_raiz = {
            nodo["grupo"]: clases_por_nodo[(nodo["grupo"], nodo["segs"])]
            for nodo in nodos
            if nodo["nivel"] == 1
        }
        creadas_heuristica = 0
        for par in pares:
            clase_a = clases_por_nodo[(par["nodo_a"]["grupo"], par["nodo_a"]["segs"])]
            clase_b = clases_por_nodo[(par["nodo_b"]["grupo"], par["nodo_b"]["segs"])]
            clase_a.is_a.append(onto.relacionadoCon.some(clase_b))
            creadas_heuristica += 1
        creadas_raiz = 0
        for grupo_a, grupo_b in relaciones_raiz:
            clases_raiz[grupo_a].is_a.append(onto.relacionadoCon.some(clases_raiz[grupo_b]))
            creadas_raiz += 1
        self._escribir_trazabilidad(pares, relaciones_raiz)
        return creadas_heuristica, creadas_raiz

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        plegado = unicodedata.normalize("NFKD", texto)
        return "".join(c for c in plegado if not unicodedata.combining(c)).lower()

    @staticmethod
    def _slug(segmento: str) -> str:
        palabras = [
            p
            for p in re.split(r"[^0-9A-Za-z]+", ServicioOntologia._normalizar_texto(segmento).title())
            if p
        ]
        return "".join(palabras) or "Concepto"

    @staticmethod
    def _tokens_significativos(segmentos: tuple[str, ...]) -> set[str]:
        tokens = set()
        for segmento in segmentos[1:]:
            for palabra in re.findall(r"[a-z]{5,}", ServicioOntologia._normalizar_texto(segmento)):
                if palabra not in STOPWORDS and palabra not in TOKENS_GENERICOS:
                    tokens.add(palabra)
        return tokens

    @staticmethod
    def _frecuencia_documento(nodos: list[dict]) -> Counter:
        conteo: Counter = Counter()
        for nodo in nodos:
            for token in nodo["tokens"]:
                conteo[token] += 1
        return conteo

    def _calcular_relaciones_afines(self, nodos: list[dict]) -> list[dict]:
        """Pares de afinidad (TR) entre grupos con heurística IDF."""
        for nodo in nodos:
            nodo["tokens"] = self._tokens_significativos(nodo["segs"])
        df = self._frecuencia_documento(nodos)
        depurados = [
            {**n, "tokens": {t for t in n["tokens"] if df[t] <= UMBRAL_DF_AFINES}}
            for n in nodos
            if n["nivel"] >= 2 and n["tokens"]
        ]
        por_grupo: dict[str, list[dict]] = defaultdict(list)
        for nodo in depurados:
            por_grupo[nodo["grupo"]].append(nodo)
        grupos = list(por_grupo)
        pares: list[dict] = []
        for i in range(len(grupos)):
            for j in range(i + 1, len(grupos)):
                for nodo_a in por_grupo[grupos[i]]:
                    for nodo_b in por_grupo[grupos[j]]:
                        compartidos = nodo_a["tokens"] & nodo_b["tokens"]
                        if not compartidos:
                            continue
                        score = sum(len(nodos) / (1 + df[t]) for t in compartidos)
                        pares.append({
                            "nodo_a": nodo_a,
                            "nodo_b": nodo_b,
                            "score": score,
                            "tokens": sorted(compartidos),
                        })
        pares.sort(key=lambda p: (-p["score"], -len(p["tokens"])))
        return pares[:MAX_RELACIONES_AFINES]

    def _escribir_trazabilidad(self, pares: list[dict], relaciones_raiz: list[tuple[str, str]]) -> None:
        """Documenta el criterio de afinidad y los pares en un .txt."""
        lineas = [
            "CRITERIO DE RELACIONES DE AFINIDAD (cev:relacionadoCon)",
            "=======================================================",
            "1. Nodos de nivel >= 2 (conceptos especificos, sin raices).",
            "2. Tokens: segmentos del path salvo la raiz, ASCII, sin stopwords",
            "   ni tokens genericos del dominio.",
            f"3. Especificidad: frecuencia de documento <= {UMBRAL_DF_AFINES} (IDF alto).",
            "4. Puntuacion: suma de IDF de los tokens compartidos.",
            f"5. Se conservan los {MAX_RELACIONES_AFINES} mejores pares entre grupos distintos.",
            "",
            "RELACIONES RAIZ FIJAS (modelo conceptual CEV):",
        ]
        for grupo_a, grupo_b in relaciones_raiz:
            lineas.append(f"- {grupo_a} <-> {grupo_b}")
        lineas.append("")
        lineas.append("PARES HEURISTICOS (nodo A <-> nodo B | score | tokens):")
        for par in pares:
            lineas.append(
                f"- [{par['score']:.1f}] {par['nodo_a']['path_original']} <-> "
                f"{par['nodo_b']['path_original']} | {', '.join(par['tokens'])}"
            )
        self.ruta_trazabilidad.parent.mkdir(parents=True, exist_ok=True)
        self.ruta_trazabilidad.write_text("\n".join(lineas), encoding="utf-8")

    def _calcular_perpetradores(self, onto, nodos: list[dict], clases_por_nodo: dict) -> int:
        """Declara cev:perpetradoPor entre Hechos victimizantes y Actores armados."""
        for nodo in nodos:
            nodo["tokens"] = self._tokens_significativos(nodo["segs"])
        df = self._frecuencia_documento(nodos)
        hechos = [
            {**n, "tokens": {t for t in n["tokens"] if df[t] <= UMBRAL_DF_PERPETRADOR}}
            for n in nodos
            if n["grupo"] == "Hechos victimizantes" and n["nivel"] <= 3
        ]
        actores = [
            {**n, "tokens": {t for t in n["tokens"] if df[t] <= UMBRAL_DF_PERPETRADOR}}
            for n in nodos
            if n["grupo"].startswith("Actores") and n["nivel"] <= 3
        ]
        pares: list[tuple[dict, dict, float]] = []
        for hecho in hechos:
            for actor in actores:
                compartidos = hecho["tokens"] & actor["tokens"]
                if not compartidos:
                    continue
                score = sum(len(nodos) / (1 + df[t]) for t in compartidos)
                pares.append((hecho, actor, score))
        pares.sort(key=lambda p: -p[2])
        creados = 0
        for hecho, actor, _ in pares[:MAX_PERPETRADOR]:
            clase_hecho = clases_por_nodo[(hecho["grupo"], hecho["segs"])]
            clase_actor = clases_por_nodo[(actor["grupo"], actor["segs"])]
            clase_hecho.is_a.append(onto.perpetradoPor.some(clase_actor))
            creados += 1
        return creados

    # ------------------------------------------------------------------
    # Persistencia y validación
    # ------------------------------------------------------------------

    def guardar(self, ruta: Path | None = None) -> Path:
        """Guarda la ontología en RDF/XML y devuelve la ruta escrita."""
        if self.onto is None:
            raise RuntimeError("No hay ontología construida o cargada para guardar.")
        destino = Path(ruta) if ruta else self.ruta_owl
        destino.parent.mkdir(parents=True, exist_ok=True)
        self.onto.save(str(destino), format="rdfxml")
        return destino

    def cargar(self) -> bool:
        """Carga la OWL existente (o la construye desde JSON si no existe).

        Returns:
            True si se construyó desde JSON; False si se cargó del archivo.
        """
        if self.ruta_owl.exists():
            self.onto = get_ontology(f"file://{self.ruta_owl.resolve()}").load()
            return False
        self.construir_desde_grupos()
        self.guardar()
        return True

    def metricas(self) -> dict[str, Any]:
        """Conteos estructurales: clases, individuos y propiedades."""
        if self.onto is None:
            raise RuntimeError("Ontología no cargada: ejecute construir/cargar primero.")
        clases = list(self.onto.classes())
        subclases_por_raiz: dict[str, int] = {}
        for clase in clases:
            if clase.name == "ConceptoDelTesauro":
                continue
            ancestros = clase.ancestors(include_self=False)
            raiz = next(
                (a.name for a in ancestros if a.name in SLUG_RAIZ.values()),
                "ConceptoDelTesauro",
            )
            subclases_por_raiz[raiz] = subclases_por_raiz.get(raiz, 0) + 1
        return {
            "clases": len(clases),
            "individuos": len(list(self.onto.individuals())),
            "object_properties": sum(1 for _ in self.onto.object_properties()),
            "annotation_properties": sum(1 for _ in self.onto.annotation_properties()),
            "subclases_por_raiz": subclases_por_raiz,
        }

    def validar_rdflib(self) -> dict[str, int]:
        """Parsea el .owl con rdflib y cuenta triples estructurales."""
        grafo = rdflib.Graph()
        grafo.parse(str(self.ruta_owl), format="xml")
        subclases = sum(
            1
            for _, objeto in grafo.subject_objects(rdflib.RDFS.subClassOf)
            if isinstance(objeto, rdflib.URIRef)
        )
        restricciones = sum(
            1
            for _, objeto in grafo.subject_objects(rdflib.RDFS.subClassOf)
            if not isinstance(objeto, rdflib.URIRef)
        )
        return {"triples": len(grafo), "subclassof_nombradas": subclases, "subclassof_restricciones": restricciones}

    def validar_hermit(self) -> bool:
        """Ejecuta el reasoner HermiT si está disponible (consistencia)."""
        if self.onto is None:
            return False
        try:
            from owlready2 import sync_reasoner

            sync_reasoner(self.onto, infer_property_values=False)
            return True
        except Exception as error:  # noqa: BLE001 - falla por entorno, el flujo sigue
            print(f"HermiT no disponible o fallo: {type(error).__name__}: {error}")
            return False

    def conceptos(self) -> pd.DataFrame:
        """DataFrame de conceptos de la ontología (grupo, path, concepto).

        Usa la anotación rdfs:comment (path completo original) y cev:grupo.
        Incluye todas las clases cuyo comment es un path del tesauro
        (contiene " - "), excluyendo las clases definidas (su comment es la
        fórmula OWL) y la superclase técnica.
        """
        filas = []
        for clase in self.onto.classes():
            if clase.name == "ConceptoDelTesauro":
                continue
            comentario = getattr(clase, "comment", [])
            if not comentario or " - " not in comentario[0]:
                continue
            path = comentario[0]
            grupo = getattr(clase, "grupo", [None])[0] or ""
            filas.append({
                "grupo": grupo,
                "termino": path,
                "concepto": path.split(" - ")[-1].strip(),
            })
        return pd.DataFrame(filas)

    def grafo(self) -> nx.DiGraph:
        """Grafo dirigido de la ontología para visualización.

        Nodos: clases y individuos. Aristas: subClassOf (nombradas y
        restricciones some de esParteDe/relacionadoCon/perpetradoPor) y
        rdf:type (individuo -> clase). Los nodos llevan atributos `tipo`
        ("clase"/"individuo") y `grupo` (grupo raíz, "" si no aplica).
        """
        if self.onto is None:
            raise RuntimeError("Ontología no cargada.")
        grafo = nx.DiGraph()

        def agregar_arista(origen: str, destino: str, tipo: str) -> None:
            """Añade una arista acumulando el tipo si el par ya existe.

            La jerarquía rdfs:subClassOf y la restricción esParteDe unen los
            mismos pares (padre estructural = padre meronímico); se conserva
            el conjunto de tipos para no perder información.
            """
            if grafo.has_edge(origen, destino):
                grafo[origen][destino]["tipo"].add(tipo)
            else:
                grafo.add_edge(origen, destino, tipo={tipo})

        for clase in self.onto.classes():
            nombre = clase.name
            grupo = (getattr(clase, "grupo", [None]) or [None])[0] or ""
            grafo.add_node(nombre, tipo="clase", grupo=grupo)
            for axioma in clase.is_a:
                if isinstance(axioma, Restriction):
                    propiedad = axioma.property.name
                    destino = axioma.value.name if axioma.value is not None else None
                    if destino and propiedad in ("esParteDe", "relacionadoCon", "perpetradoPor"):
                        agregar_arista(nombre, destino, propiedad)
                elif isinstance(axioma, type):  # superclase nombrada (clase owlready2)
                    agregar_arista(nombre, axioma.name, "subclassof")
        for individuo in self.onto.individuals():
            nombre = individuo.name
            grupo = (getattr(individuo, "grupo", [None]) or [None])[0] or ""
            grafo.add_node(nombre, tipo="individuo", grupo=grupo)
            for clase in individuo.is_a:
                grafo.add_edge(nombre, clase.name, tipo={"instancia"})
        return grafo



# %%
# =====================================================================
# Facade: consultas SPARQL
# =====================================================================


class SparqlServicio:
    """Facade que ejecuta consultas SPARQL sobre la ontología con rdflib.

    Encapsula la carga del grafo RDF y el formateo de resultados como
    DataFrame de pandas, con límite de filas y medición de tiempo.
    """

    def __init__(self, ruta_owl: Path) -> None:
        self.ruta_owl = Path(ruta_owl)
        self.grafo = rdflib.Graph()
        self.grafo.parse(str(self.ruta_owl), format="xml")

    def ejecutar(
        self,
        consulta: str,
        columnas: list[str],
        limite: int = 15,
    ) -> tuple[pd.DataFrame, int, float]:
        """Ejecuta una consulta SELECT y devuelve (df, filas_totales, ms).

        Args:
            consulta: Texto SPARQL (SELECT).
            columnas: Nombres de las columnas del resultado.
            limite: Límite de filas a mostrar (el resultado completo se
                computa para reportar el total real).
            ms: Tiempo de ejecución en milisegundos.

        Returns:
            Tupla (DataFrame truncado, total de filas, milisegundos).
        """
        import time

        inicio = time.perf_counter()
        resultados = self.grafo.query(consulta)
        filas = [[str(valor) for valor in fila] for fila in resultados]
        total = len(filas)
        ms = (time.perf_counter() - inicio) * 1000
        df = pd.DataFrame(filas[:limite], columns=columnas)
        return df, total, ms

    @staticmethod
    def mostrar(df: pd.DataFrame, total: int, ms: float) -> None:
        """Imprime el resultado truncado con el total real y el tiempo."""
        if df.empty:
            print("(sin resultados)")
            return
        with pd.option_context("display.max_colwidth", 70, "display.width", 140):
            print(df.to_string(index=False))
        print(f"-> Filas totales: {total} | Tiempo: {ms:.0f} ms")


# %%
# =====================================================================
# Facade: grafos de la ontología para visualización
# =====================================================================


class GrafoOntologia:
    """Facade que produce figuras de la ontología (matplotlib/networkx).

    Dos vistas:
    - `grafo_completo`: todos los nodos (clases + individuos) con sus
      aristas, coloreados por grupo raíz (imagen estática de alta
      resolución, guardada en disco).
    - `grafo_por_grupos`: vista contraída con los 5 grupos y la
      superclase, mostrando las relaciones entre grupos.

    Attributes:
        servicio: Instancia de ServicioOntologia (grafo fuente).
        colores_grupo: Dict grupo -> color hex (se inyecta desde sec00).
    """

    def __init__(self, servicio: ServicioOntologia, colores_grupo: dict[str, str]) -> None:
        self.servicio = servicio
        self.colores_grupo = colores_grupo
        self.color_superclase = "#999999"

    def grafo_completo(self) -> nx.DiGraph:
        """Grafo completo de clases, individuos y relaciones."""
        return self.servicio.grafo()

    def grafo_por_grupos(self) -> nx.DiGraph:
        """Grafo contraído: 5 grupos + superclase con aristas agregadas."""
        completo = self.servicio.grafo()
        grupo_de: dict[str, str] = {}
        for nodo, atributos in completo.nodes(data=True):
            if atributos.get("grupo"):
                grupo_de[nodo] = atributos["grupo"]
        contraido = nx.DiGraph()
        for grupo in self.colores_grupo:
            contraido.add_node(grupo, tipo="grupo")
        contraido.add_node("ConceptoDelTesauro", tipo="superclase")
        for origen, destino, atributos in completo.edges(data=True):
            grupo_origen = grupo_de.get(origen)
            grupo_destino = grupo_de.get(destino)
            if not grupo_origen or not grupo_destino or grupo_origen == grupo_destino:
                continue
            tipos = atributos.get("tipo", {"relacion"})
            if contraido.has_edge(grupo_origen, grupo_destino):
                contraido[grupo_origen][grupo_destino]["tipo"].update(tipos)
            else:
                contraido.add_edge(grupo_origen, grupo_destino, tipo=set(tipos))
        return contraido

    def guardar_figura_completa(self, ruta: Path, dpi: int = 200, mostrar: bool = True) -> Path:
        """Genera y guarda la figura del grafo completo de la ontología.

        Los nodos se colorean por grupo raíz; los individuos se dibujan más
        pequeños y con borde. Sin etiquetas a esa escala (ilegibles en
        1.100 nodos); la leyenda identifica los grupos.

        Returns:
            Ruta del PNG guardado.
        """
        grafo = self.grafo_completo()
        posicion = nx.spring_layout(grafo, seed=42, k=0.32, iterations=60)
        figura, eje = plt.subplots(figsize=(24, 20))
        colores: list[str] = []
        tamanos: list[int] = []
        for nodo, atributos in grafo.nodes(data=True):
            grupo = atributos.get("grupo", "")
            colores.append(self.colores_grupo.get(grupo, self.color_superclase))
            tamanos.append(28 if atributos.get("tipo") == "individuo" else 65)
        nx.draw_networkx_edges(
            grafo, posicion, ax=eje, alpha=0.16, width=0.4, arrows=True,
            arrowsize=6, edge_color="#666666",
        )
        nx.draw_networkx_nodes(
            grafo, posicion, ax=eje, node_color=colores, node_size=tamanos,
            edgecolors="#ffffff", linewidths=0.25, alpha=0.9,
        )
        leyenda = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markersize=11, label=grupo)
            for grupo, color in self.colores_grupo.items()
        ]
        leyenda.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=self.color_superclase,
                       markersize=11, label="ConceptoDelTesauro")
        )
        eje.legend(handles=leyenda, loc="upper left", fontsize=11, framealpha=0.9)
        eje.set_title(
            "Grafo completo de la ontología del Tesauro CEV "
            f"({grafo.number_of_nodes()} nodos, {grafo.number_of_edges()} aristas)",
            fontsize=16,
        )
        eje.axis("off")
        figura.tight_layout()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(str(ruta), dpi=dpi, bbox_inches="tight")
        if mostrar:
            plt.show()
        plt.close(figura)
        return ruta

    def guardar_figura_grupos(self, ruta: Path, dpi: int = 200, mostrar: bool = True) -> Path:
        """Genera y guarda la figura contraída entre los 6 grupos."""
        grafo = self.grafo_por_grupos()
        posicion = nx.circular_layout(grafo)
        figura, eje = plt.subplots(figsize=(15, 13))
        colores = [
            self.colores_grupo.get(nodo, self.color_superclase)
            for nodo in grafo.nodes()
        ]
        nx.draw_networkx_edges(grafo, posicion, ax=eje, alpha=0.55, width=1.6,
                               arrows=True, arrowsize=14, edge_color="#555555")
        nx.draw_networkx_nodes(grafo, posicion, ax=eje, node_color=colores,
                               node_size=3200, edgecolors="#ffffff", linewidths=1.6)
        # Etiquetas FUERA del nodo: el texto completo se desplaza hacia el
        # exterior del circulo segun el angulo del layout, con caja blanca
        # para evitar cruces con las aristas (evita el truncado del texto).
        for nodo, (x, y) in posicion.items():
            angulo = np.arctan2(y, x)
            desp_x = np.cos(angulo) * 0.30
            desp_y = np.sin(angulo) * 0.30
            alineacion = "right" if desp_x < 0 else "left"
            eje.text(
                x + desp_x, y + desp_y, nodo,
                ha=alineacion, va="center", fontsize=10.5, color="#1a1a1a",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": "#cccccc", "alpha": 0.95},
            )
        eje.set_xlim(-1.45, 1.45)
        eje.set_ylim(-1.45, 1.45)
        eje.set_title(
            "Vista contraída: relaciones entre los 6 grupos de tesauros "
            f"({grafo.number_of_edges()} aristas inter-grupo)",
            fontsize=15,
        )
        eje.axis("off")
        figura.tight_layout()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(str(ruta), dpi=dpi, bbox_inches="tight")
        if mostrar:
            plt.show()
        plt.close(figura)
        return ruta

# %%
"""Registro del módulo `sec00_clases` dentro del notebook (autocontenido).

Las secciones sec01-sec06 usan `import sec00_clases as sc`. Para que el
notebook se ejecute en CUALQUIER máquina sin depender del archivo
notebook_cells/sec00_clases.py (p. ej. en Colab con solo el notebook), esta
celda registra en sys.modules un módulo que expone las clases, constantes y
dependencias del núcleo definidas en las celdas anteriores. En modo script
standalone (tests) el import real del archivo sigue funcionando igual: el
módulo registrado contiene exactamente los mismos nombres.
"""
import sys

MODULO_NUCLEO = types.ModuleType("sec00_clases")
NOMBRES_NUCLEO = [
    # Clases del dominio
    "Termino", "ResultadoRelaciones",
    # Strategy: normalización
    "NormalizadorTermino", "NormalizadorBase", "NormalizadorPlural",
    "NormalizadorAcentos", "CadenaNormalizadores",
    # Repository
    "TesauroRepositorio",
    # Strategy: extractores + Abstract Factory + Registry
    "ExtractorRelacion", "SinonimiaExtractor", "AntonimiaExtractor",
    "DerivacionExtractor", "HiponimiaExtractor", "HiperonimiaExtractor",
    "MeronimiaExtractor", "HolonimiaExtractor", "TroponimiaExtractor",
    "CoordinadosExtractor", "FabricaExtractores",
    # Facades
    "MotorRelacionesWordNet", "ServicioOntologia", "SparqlServicio",
    "GrafoOntologia",
    # Constantes del dominio
    "IRI_BASE", "SLUG_RAIZ", "EJE_POR_GRUPO", "TIPOS_RELACION",
    "RELACIONES_ENTRE_GRUPOS", "CLASES_DEFINIDAS",
    "MAX_RELACIONES_AFINES", "MAX_PERPETRADOR",
    "UMBRAL_DF_AFINES", "UMBRAL_DF_PERPETRADOR",
    "STOPWORDS", "TOKENS_GENERICOS",
    # Dependencias a las que las secciones acceden a través de sc
    "plt", "nx", "np", "pd", "rdflib", "wn", "Restriction",
]
for _nombre in NOMBRES_NUCLEO:
    if _nombre in globals():
        setattr(MODULO_NUCLEO, _nombre, globals()[_nombre])
sys.modules["sec00_clases"] = MODULO_NUCLEO
print("Nucleo POO registrado como modulo 'sec00_clases' (notebook autocontenido)")
