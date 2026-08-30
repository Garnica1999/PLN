# %% [markdown]
# # SECCION 5: 22 consultas SPARQL sobre la ontologia del Tesauro CEV
#
# **Objetivo:** documentar y ejecutar 22 consultas SPARQL (3 por cada uno de
# los 6 grupos de tesauros seleccionados, mas 4 consultas del modelo semantico
# del ciclo del conflicto) contra la ontologia OWL `../ontologia/tesauro_cev.owl`,
# explicando que informacion busca cada una y como se relaciona con los
# tesauros (relaciones TG/TE, TR y USE-UP) y con los axiomas de la ontologia
# creada en la seccion 3.
#
# **Como leer esta seccion:** cada consulta tiene un bloque de documentacion
# (grupo, objetivo, relacion con el tesauro y la ontologia, consulta SPARQL,
# resultado ejecutado). Las 5 familias de relaciones del tesauro quedan
# mapeadas a axiomas OWL de la siguiente forma:
#
# - **TG/TE (jerarquia):** axiomas de clase `C rdfs:subClassOf
#   [owl:Restriction esParteDe some D]`, es decir, cada termino C "es parte de"
#   (es subconcepto de) su termino padre D. La propiedad `cev:esParteDe` tiene
#   inversa `cev:parteDe`.
# - **TR (terminos relacionados):** axiomas `C rdfs:subClassOf
#   [owl:Restriction relacionadoCon some D]` con `cev:relacionadoCon`
#   declarada simetrica (50 axiomas).
# - **USE-UP / relacion hecho-actor:** axiomas `C rdfs:subClassOf
#   [owl:Restriction perpetradoPor some D]` (13 axiomas: 12 por co-ocurrencia
#   mas el axioma raiz del modelo semantico); `cev:esVictimaDe` es la inversa
#   declarada de `cev:perpetradoPor`.
# - **Modelo semantico (ciclo del conflicto):** 7 axiomas raiz entre los 6
#   grupos (perpetradoPor, afectaA, generaImpacto, experimentaImpacto,
#   desarrollaAfrontamiento, respondeA, empleaArma) y 6 clases definidas con
#   `owl:equivalentClass` (base AND restriccion) que HermiT clasifica
#   automaticamente.
# - **Anotaciones de trazabilidad:** `cev:grupo` (grupo raiz del tesauro) y
#   `cev:eje` (eje conceptual) sobre los individuos `instancia_<slug>`.
#
# **Implementacion (refactor POO):** la ejecucion usa el facade
# `sc.SparqlServicio` de la seccion 00 (carga del grafo rdflib, `.ejecutar`
# con limite y medicion de tiempo, y `.mostrar` estatico). Las 22 consultas
# viven como **constantes estructuradas** (lista de dicts con id, grupo,
# titulo, objetivo, relacion con el tesauro, SPARQL y columnas) y un unico
# helper `_ejecutar` las corre sin repetir el patron de 22 bloques (DRY).
# Los resultados se resuelven a labels del tesauro para la presentacion.

# %% [markdown]
# ## 5.0 Imports, carga de la ontologia y helpers de ejecucion
#
# La seccion usa la OWL de `../ontologia/tesauro_cev.owl`, que la seccion 3
# deja construida con el modelo semantico extendido (623 clases, 534
# individuos, 17 object properties, 6 clases definidas). Si por algun motivo
# el archivo existente es la version anterior (593 clases), NO se modifica:
# se construye la version nueva en un directorio temporal para poder validar
# las consultas contra el modelo completo, y se reporta la diferencia.

# %%
"""Imports, bloque standalone, carga del grafo SPARQL y consultas estructuradas."""
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import RDFS

# Rutas relativas al notebook (las celdas viven en notebook_cells/).
try:
    RUTA_BASE = Path(__file__).resolve().parent.parent  # modo script
except NameError:
    RUTA_BASE = Path(os.getcwd()).parent  # modo notebook (notebooks/)

# Nucleo POO de sec00: en el notebook ya se ejecuto como celdas; en modo
# script se importa desde notebook_cells/ para que la seccion sea standalone.
sys.path.insert(0, str(RUTA_BASE / "notebook_cells"))
import sec00_clases as sc

# Constantes globales del proyecto: en el notebook las define sec00_setup; el
# import solo se ejecuta en modo script standalone (archivo presente).
if "IRI_BASE" not in globals():
    from sec00_setup import (
        IRI_BASE,
        RUTA_JSON_GRUPOS,
        RUTA_ONTOLOGIA,
        RUTA_TRAZABILIDAD,
    )

PREFIJOS = f"""
PREFIX cev: <{IRI_BASE}>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

# Carga la OWL del proyecto y verifica que sea la version con el modelo
# semantico completo (623 clases). Si el archivo es la version anterior, se
# construye la nueva en /tmp (sin tocar la del proyecto) y se avisa.
servicio_ontologia = sc.ServicioOntologia.obtener(RUTA_ONTOLOGIA)
servicio_ontologia.cargar()
metricas_ontologia = servicio_ontologia.metricas()
RUTA_OWL_USAR = RUTA_ONTOLOGIA
if metricas_ontologia["clases"] != 623:
    ruta_tmp = Path(tempfile.gettempdir()) / "tesauro_cev_nueva.owl"
    servicio_tmp = sc.ServicioOntologia.obtener(
        ruta_tmp, RUTA_JSON_GRUPOS, RUTA_TRAZABILIDAD
    )
    servicio_tmp.construir_desde_grupos()
    servicio_tmp.guardar()
    metricas_ontologia = servicio_tmp.metricas()
    RUTA_OWL_USAR = ruta_tmp
    print(
        f"[aviso] la OWL de ontologia/ es la version anterior "
        f"(sin el grupo 6); se valida contra la construida en {ruta_tmp}"
    )

# Facade de consultas SPARQL (sec00): carga el grafo rdflib de la OWL.
sparql = sc.SparqlServicio(RUTA_OWL_USAR)
print(f"Ontologia cargada: {RUTA_OWL_USAR}")
print(f"Clases: {metricas_ontologia['clases']} | "
      f"Individuos: {metricas_ontologia['individuos']} | "
      f"Object properties: {metricas_ontologia['object_properties']}")

# Indice label -> texto para resolver entidades del grafo (clases e individuos).
LABELS = {
    str(sujeto): str(label)
    for sujeto, label in sparql.grafo.subject_objects(RDFS.label)
}


def _resolver_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Resuelve URIs a labels del tesauro para la presentacion.

    Los literales (labels, ejes, conteos) se conservan tal cual; las URIs de
    clases e individuos se resuelven con su rdfs:label del grafo, o con su
    nombre local cuando no hay label.

    Args:
        df: DataFrame con los resultados (valores convertidos a texto).

    Returns:
        DataFrame con las URIs resueltas a labels legibles.
    """
    def _a_label(valor: str) -> str:
        if valor.startswith(("http://", "https://")):
            return LABELS.get(valor, valor.split("#")[-1])
        return valor

    return df.map(_a_label)


def _ejecutar(consulta_id: str) -> None:
    """Ejecuta una consulta de CONSULTAS y muestra su tabla con labels.

    Args:
        consulta_id: Identificador de la consulta en CONSULTAS_POR_ID.
    """
    definicion = CONSULTAS_POR_ID[consulta_id]
    df, total, ms = sparql.ejecutar(
        PREFIJOS + definicion["consulta"],
        definicion["columnas"],
        limite=15,
    )
    sparql.mostrar(_resolver_labels(df), total, ms)


# Consultas estructuradas (DRY): una definicion por consulta con su
# documentacion programatica (grupo, objetivo, relacion con el tesauro).
CONSULTAS: list[dict[str, Any]] = [
    {
        "id": "1.1",
        "grupo": "Actores armados y otros participantes en las dinamicas de la guerra",
        "titulo": "Subcategorias de Actores armados (jerarquia TG/TE)",
        "objetivo": "Listar las subcategorias directas (TE) de la raiz de actores con su label.",
        "relacion_tesauro": "TG/TE: el grupo es el termino general y cada subcategoria un TE, via esParteDe.",
        "consulta": """
SELECT DISTINCT ?clase ?label
WHERE {
  ?clase rdfs:subClassOf ?restriccion .
  ?restriccion a owl:Restriction ;
               owl:onProperty cev:esParteDe ;
               owl:someValuesFrom cev:Actores_armados .
  OPTIONAL { ?clase rdfs:label ?label }
}
ORDER BY ?label
""",
        "columnas": ["Subcategoria (clase)", "Label del tesauro"],
    },
    {
        "id": "1.2",
        "grupo": "Actores armados y otros participantes en las dinamicas de la guerra",
        "titulo": "Individuos (terminos hoja) del grupo Actores armados",
        "objetivo": "Recuperar todos los individuos instancia_<slug> del grupo con su label y eje.",
        "relacion_tesauro": "Terminos hoja materializados como individuos anotados con cev:grupo y cev:eje.",
        "consulta": """
SELECT DISTINCT ?individuo ?label ?eje
WHERE {
  ?individuo a ?clase .
  ?clase rdfs:subClassOf* cev:Actores_armados .
  FILTER(CONTAINS(STR(?individuo), "instancia_"))
  ?individuo rdfs:label ?label .
  OPTIONAL { ?individuo cev:eje ?eje }
}
ORDER BY ?label
""",
        "columnas": ["Individuo", "Label", "Eje"],
    },
    {
        "id": "1.3",
        "grupo": "Actores armados y otros participantes en las dinamicas de la guerra",
        "titulo": "Hechos perpetrados por actores armados (perpetradoPor)",
        "objetivo": "Identificar los hechos victimizantes perpetrados por actores armados.",
        "relacion_tesauro": "USE-UP: relacion hecho-actor modelada con los 13 axiomas perpetradoPor.",
        "consulta": """
SELECT DISTINCT ?hecho ?perpetrador
WHERE {
  ?hecho rdfs:subClassOf ?restriccion .
  ?restriccion a owl:Restriction ;
               owl:onProperty cev:perpetradoPor ;
               owl:someValuesFrom ?perpetrador .
  ?perpetrador rdfs:subClassOf* cev:Actores_armados .
}
ORDER BY ?perpetrador
""",
        "columnas": ["Hecho victimizante", "Perpetrador (grupo actores)"],
    },
    {
        "id": "2.1",
        "grupo": "Armas",
        "titulo": "Jerarquia TG/TE completa del grupo Armas",
        "objetivo": "Recuperar todos los pares (hijo, padre) del arbol de armas.",
        "relacion_tesauro": "TG/TE: cada triple hijo esParteDe padre es un salto de la jerarquia.",
        "consulta": """
SELECT DISTINCT ?hijo ?padre
WHERE {
  ?hijo rdfs:subClassOf ?restriccion .
  ?restriccion a owl:Restriction ;
               owl:onProperty cev:esParteDe ;
               owl:someValuesFrom ?padre .
  ?hijo rdfs:subClassOf* cev:Armas .
}
ORDER BY ?padre ?hijo
""",
        "columnas": ["Hijo (TE)", "Padre (TG)"],
    },
    {
        "id": "2.2",
        "grupo": "Armas",
        "titulo": "Relaciones TR (afines) entre Armas y otros grupos",
        "objetivo": "Listar los terminos de Armas con afinidad terminologica (TR) con otros grupos.",
        "relacion_tesauro": "TR: los 50 axiomas de relacionadoCon (simetrica) cruzando grupos.",
        "consulta": """
SELECT DISTINCT ?terminoArmas ?afin ?grupoAfiliado
WHERE {
  { ?terminoArmas rdfs:subClassOf ?r1 .
    ?r1 a owl:Restriction ; owl:onProperty cev:relacionadoCon ; owl:someValuesFrom ?afin .
    ?terminoArmas rdfs:subClassOf* cev:Armas . }
  UNION
  { ?afin rdfs:subClassOf ?r1 .
    ?r1 a owl:Restriction ; owl:onProperty cev:relacionadoCon ; owl:someValuesFrom ?terminoArmas .
    ?terminoArmas rdfs:subClassOf* cev:Armas . }
  ?afin rdfs:subClassOf* ?raiz .
  FILTER(?raiz IN (cev:Actores_armados, cev:Impactos, cev:Hechos_victimizantes, cev:Poblaciones))
  ?raiz rdfs:label ?grupoAfiliado .
}
ORDER BY ?afin
""",
        "columnas": ["Termino de Armas", "Termino afin (TR)", "Grupo del afin"],
    },
    {
        "id": "2.3",
        "grupo": "Armas",
        "titulo": "Busqueda por keyword en labels del grupo Armas",
        "objetivo": "Buscar terminos de Armas cuyo label contenga arma, fuego, explos o blanca.",
        "relacion_tesauro": "Recuperacion de informacion sobre los labels rdfs:label del tesauro.",
        "consulta": """
SELECT DISTINCT ?termino ?label ?tipo
WHERE {
  { ?termino rdfs:subClassOf* cev:Armas . ?termino a owl:Class . BIND("clase" AS ?tipo) }
  UNION
  { ?termino a ?clase . ?clase rdfs:subClassOf* cev:Armas .
    FILTER(CONTAINS(STR(?termino), "instancia_")) . BIND("individuo" AS ?tipo) }
  ?termino rdfs:label ?label .
  FILTER(
    CONTAINS(LCASE(?label), "arma") ||
    CONTAINS(LCASE(?label), "fuego") ||
    CONTAINS(LCASE(?label), "explos") ||
    CONTAINS(LCASE(?label), "blanca"))
}
ORDER BY ?tipo ?label
""",
        "columnas": ["Termino", "Label", "Tipo"],
    },
    {
        "id": "3.1",
        "grupo": "Hechos victimizantes",
        "titulo": "Subcategorias de hechos victimizantes con conteo de terminos",
        "objetivo": "Listar las subcategorias directas de hechos ordenadas por terminos hoja.",
        "relacion_tesauro": "TG/TE con agregacion: densidad terminologica por TE (COUNT de individuos).",
        "consulta": """
SELECT ?clase ?label (COUNT(DISTINCT ?individuo) AS ?numTerminos)
WHERE {
  ?clase rdfs:subClassOf ?restriccion .
  ?restriccion a owl:Restriction ;
               owl:onProperty cev:esParteDe ;
               owl:someValuesFrom cev:Hechos_victimizantes .
  OPTIONAL { ?clase rdfs:label ?label }
  ?individuo a ?sub .
  ?sub rdfs:subClassOf* ?clase .
  FILTER(CONTAINS(STR(?individuo), "instancia_"))
}
GROUP BY ?clase ?label
ORDER BY DESC(?numTerminos) ?label
""",
        "columnas": ["Subcategoria", "Label", "Num. terminos hoja"],
    },
    {
        "id": "3.2",
        "grupo": "Hechos victimizantes",
        "titulo": "Relaciones de perpetracion y su lectura inversa (esVictimaDe)",
        "objetivo": "Listar los pares (hecho, perpetrador) modelados en la ontologia.",
        "relacion_tesauro": "USE-UP: perpetradoPor materializa la relacion hecho-actor del tesauro.",
        "consulta": """
SELECT DISTINCT ?hecho ?perpetrador
WHERE {
  ?hecho rdfs:subClassOf ?restriccion .
  ?restriccion a owl:Restriction ;
               owl:onProperty cev:perpetradoPor ;
               owl:someValuesFrom ?perpetrador .
}
ORDER BY ?perpetrador ?hecho
""",
        "columnas": ["Hecho victimizante", "Perpetrador (esVictimaDe: inversa)"],
    },
    {
        "id": "3.2_inv",
        "grupo": "Hechos victimizantes",
        "titulo": "Lectura inversa de la perpetracion (esVictimaDe)",
        "objetivo": "Cuantos hechos distintos senalan a cada perpetrador (agregacion inversa).",
        "relacion_tesauro": "USE-UP: ejercicio de la semantica inversa esVictimaDe declarada en la OWL.",
        "consulta": """
SELECT ?perpetrador (COUNT(DISTINCT ?hecho) AS ?numHechos)
WHERE {
  ?hecho rdfs:subClassOf ?restriccion .
  ?restriccion a owl:Restriction ;
               owl:onProperty cev:perpetradoPor ;
               owl:someValuesFrom ?perpetrador .
}
GROUP BY ?perpetrador
ORDER BY DESC(?numHechos)
""",
        "columnas": ["Perpetrador (lectura esVictimaDe)", "Num. hechos de los que es victima"],
    },
    {
        "id": "3.3",
        "grupo": "Hechos victimizantes",
        "titulo": "Interseccion TR entre Poblaciones y Hechos victimizantes",
        "objetivo": "Encontrar los pares de afinidad (TR) que cruzan Poblaciones y Hechos.",
        "relacion_tesauro": "TR como interseccion entre grupos: cruces que el tesauro plano no muestra.",
        "consulta": """
SELECT DISTINCT ?poblacion ?hecho
WHERE {
  { ?poblacion rdfs:subClassOf ?r1 .
    ?r1 a owl:Restriction ; owl:onProperty cev:relacionadoCon ; owl:someValuesFrom ?hecho .
    ?poblacion rdfs:subClassOf* cev:Poblaciones .
    ?hecho rdfs:subClassOf* cev:Hechos_victimizantes . }
  UNION
  { ?hecho rdfs:subClassOf ?r1 .
    ?r1 a owl:Restriction ; owl:onProperty cev:relacionadoCon ; owl:someValuesFrom ?poblacion .
    ?poblacion rdfs:subClassOf* cev:Poblaciones .
    ?hecho rdfs:subClassOf* cev:Hechos_victimizantes . }
}
ORDER BY ?poblacion ?hecho
""",
        "columnas": ["Poblacion", "Hecho victimizante (TR)"],
    },
    {
        "id": "4.1",
        "grupo": "Impactos",
        "titulo": "Subcategorias de Impactos con conteo de terminos",
        "objetivo": "Listar las 8 subcategorias directas de impactos con sus terminos hoja.",
        "relacion_tesauro": "TG/TE con agregacion aplicada al grupo de estados_efectos.",
        "consulta": """
SELECT ?clase ?label (COUNT(DISTINCT ?individuo) AS ?numTerminos)
WHERE {
  ?clase rdfs:subClassOf ?restriccion .
  ?restriccion a owl:Restriction ;
               owl:onProperty cev:esParteDe ;
               owl:someValuesFrom cev:Impactos .
  OPTIONAL { ?clase rdfs:label ?label }
  ?individuo a ?sub .
  ?sub rdfs:subClassOf* ?clase .
  FILTER(CONTAINS(STR(?individuo), "instancia_"))
}
GROUP BY ?clase ?label
ORDER BY DESC(?numTerminos) ?label
""",
        "columnas": ["Subcategoria", "Label", "Num. terminos hoja"],
    },
    {
        "id": "4.2",
        "grupo": "Impactos",
        "titulo": "Caminos de 2 niveles dentro de Impactos (TG-TE-TE)",
        "objetivo": "Recorrer la jerarquia de impactos a dos niveles de profundidad.",
        "relacion_tesauro": "TG/TE encadenado: dos saltos de la jerarquia (nivel1 y nivel2).",
        "consulta": """
SELECT DISTINCT ?nivel1 ?nivel2
WHERE {
  ?nivel1 rdfs:subClassOf ?r1 .
  ?r1 a owl:Restriction ; owl:onProperty cev:esParteDe ; owl:someValuesFrom cev:Impactos .
  ?nivel2 rdfs:subClassOf ?r2 .
  ?r2 a owl:Restriction ; owl:onProperty cev:esParteDe ; owl:someValuesFrom ?nivel1 .
}
ORDER BY ?nivel1 ?nivel2
""",
        "columnas": ["Nivel 1 (TE de Impactos)", "Nivel 2 (TE del nivel 1)"],
    },
    {
        "id": "4.3",
        "grupo": "Impactos",
        "titulo": "Auditoria de anotaciones: entidades por (grupo, eje)",
        "objetivo": "Agrupar todas las entidades anotadas segun cev:grupo y cev:eje.",
        "relacion_tesauro": "Anotaciones de trazabilidad agregadas en la fase 2 (auditoria de coherencia).",
        "consulta": """
SELECT ?grupo ?eje (COUNT(DISTINCT ?entidad) AS ?numEntidades)
WHERE {
  ?entidad cev:grupo ?grupo .
  ?entidad cev:eje ?eje .
}
GROUP BY ?grupo ?eje
ORDER BY DESC(?numEntidades)
""",
        "columnas": ["Grupo del tesauro", "Eje", "Entidades anotadas"],
    },
    {
        "id": "5.1",
        "grupo": "Poblaciones",
        "titulo": "Top 10 subcategorias de Poblaciones con mas terminos",
        "objetivo": "Rankear las subcategorias de poblaciones por numero de terminos hoja.",
        "relacion_tesauro": "TG/TE con agregacion y ordenamiento descendente (top-k).",
        "consulta": """
SELECT ?clase ?label (COUNT(DISTINCT ?individuo) AS ?numTerminos)
WHERE {
  ?clase rdfs:subClassOf ?restriccion .
  ?restriccion a owl:Restriction ;
               owl:onProperty cev:esParteDe ;
               owl:someValuesFrom cev:Poblaciones .
  OPTIONAL { ?clase rdfs:label ?label }
  ?individuo a ?sub .
  ?sub rdfs:subClassOf* ?clase .
  FILTER(CONTAINS(STR(?individuo), "instancia_"))
}
GROUP BY ?clase ?label
ORDER BY DESC(?numTerminos) ?label
LIMIT 10
""",
        "columnas": ["Subcategoria", "Label", "Num. terminos hoja"],
    },
    {
        "id": "5.2",
        "grupo": "Poblaciones",
        "titulo": "Busqueda por keyword sobre poblaciones (indigena, afro, campesin, joven, mujer)",
        "objetivo": "Recuperar clases e individuos de Poblaciones con vocabulario diferencial.",
        "relacion_tesauro": "RI sobre labels con FILTER(CONTAINS(LCASE(...))) case-insensitive.",
        "consulta": """
SELECT DISTINCT ?termino ?label ?tipo
WHERE {
  { ?termino rdfs:subClassOf* cev:Poblaciones . ?termino a owl:Class . BIND("clase" AS ?tipo) }
  UNION
  { ?termino a ?clase . ?clase rdfs:subClassOf* cev:Poblaciones .
    FILTER(CONTAINS(STR(?termino), "instancia_")) . BIND("individuo" AS ?tipo) }
  ?termino rdfs:label ?label .
  FILTER(
    CONTAINS(LCASE(?label), "indigena") ||
    CONTAINS(LCASE(?label), "afro") ||
    CONTAINS(LCASE(?label), "campesin") ||
    CONTAINS(LCASE(?label), "joven") ||
    CONTAINS(LCASE(?label), "nino") ||
    CONTAINS(LCASE(?label), "nina") ||
    CONTAINS(LCASE(?label), "mujer"))
}
ORDER BY ?tipo ?label
""",
        "columnas": ["Termino", "Label", "Tipo"],
    },
    {
        "id": "5.3",
        "grupo": "Poblaciones",
        "titulo": "Pueblos indigenas: sus partes (esParteDe) y su parte (parteDe)",
        "objetivo": "Listar los subconceptos directos de Pueblos indigenas y su termino general.",
        "relacion_tesauro": "Propiedades inversas: hijo esParteDe PueblosIndigenas (TE) y parteDe (TG).",
        "consulta": """
SELECT ?termino ?relacion
WHERE {
  { ?termino rdfs:subClassOf ?r1 .
    ?r1 a owl:Restriction ; owl:onProperty cev:esParteDe ; owl:someValuesFrom cev:PueblosIndigenas .
    BIND("TE: es parte de Pueblos indigenas (esParteDe)" AS ?relacion) }
  UNION
  { cev:PueblosIndigenas rdfs:subClassOf ?r2 .
    ?r2 a owl:Restriction ; owl:onProperty cev:esParteDe ; owl:someValuesFrom ?termino .
    BIND("TG: Pueblos indigenas es parte de (parteDe)" AS ?relacion) }
}
ORDER BY ?relacion ?termino
""",
        "columnas": ["Termino", "Relacion con Pueblos indigenas"],
    },
    {
        "id": "6.1",
        "grupo": "Afrontamientos y resistencias no violentas - cotidianas",
        "titulo": "Arbol TG/TE completo del grupo Afrontamientos",
        "objetivo": "Recuperar todos los pares (hijo, padre) del arbol de afrontamientos y resistencias.",
        "relacion_tesauro": "TG/TE: cada triple hijo esParteDe padre es un salto de la jerarquia del grupo.",
        "consulta": """
SELECT DISTINCT ?hijo ?padre
WHERE {
  ?hijo rdfs:subClassOf ?restriccion .
  ?restriccion a owl:Restriction ;
               owl:onProperty cev:esParteDe ;
               owl:someValuesFrom ?padre .
  ?hijo rdfs:subClassOf* cev:Afrontamientos_resistencias_no_violentas .
}
ORDER BY ?padre ?hijo
""",
        "columnas": ["Hijo (TE)", "Padre (TG)"],
    },
    {
        "id": "6.2",
        "grupo": "Afrontamientos y resistencias no violentas - cotidianas",
        "titulo": "Busqueda por keyword en labels del grupo (participacion, apoyo, familia)",
        "objetivo": "Buscar terminos del grupo cuyo label mencione participacion, apoyo o familia.",
        "relacion_tesauro": "RI sobre labels: el vocabulario de la respuesta social al conflicto.",
        "consulta": """
SELECT DISTINCT ?termino ?label ?tipo
WHERE {
  { ?termino rdfs:subClassOf* cev:Afrontamientos_resistencias_no_violentas . ?termino a owl:Class . BIND("clase" AS ?tipo) }
  UNION
  { ?termino a ?clase . ?clase rdfs:subClassOf* cev:Afrontamientos_resistencias_no_violentas .
    FILTER(CONTAINS(STR(?termino), "instancia_")) . BIND("individuo" AS ?tipo) }
  ?termino rdfs:label ?label .
  FILTER(
    CONTAINS(LCASE(?label), "participaci") ||
    CONTAINS(LCASE(?label), "apoyo") ||
    CONTAINS(LCASE(?label), "familia"))
}
ORDER BY ?tipo ?label
""",
        "columnas": ["Termino", "Label", "Tipo"],
    },
    {
        "id": "6.3",
        "grupo": "Afrontamientos y resistencias no violentas - cotidianas",
        "titulo": "El grupo en el modelo semantico (respondeA / desarrollaAfrontamiento)",
        "objetivo": "Listar los axiomas del ciclo del conflicto cuyo origen o destino es el grupo.",
        "relacion_tesauro": "Modelo semantico: los afrontamientos responden a hechos y las poblaciones los desarrollan.",
        "consulta": """
SELECT DISTINCT ?origen ?propiedad ?destino
WHERE {
  { ?origen rdfs:subClassOf ?r .
    ?r a owl:Restriction ; owl:onProperty ?propiedad ; owl:someValuesFrom ?destino .
    ?origen rdfs:subClassOf* cev:Afrontamientos_resistencias_no_violentas .
    FILTER(?propiedad IN (cev:respondeA, cev:desarrollaAfrontamiento,
                          cev:esRespondidoMediante, cev:esDesarrolladoPor)) }
  UNION
  { ?origen rdfs:subClassOf ?r .
    ?r a owl:Restriction ; owl:onProperty ?propiedad ; owl:someValuesFrom ?destino .
    ?destino rdfs:subClassOf* cev:Afrontamientos_resistencias_no_violentas .
    FILTER(?propiedad IN (cev:respondeA, cev:desarrollaAfrontamiento,
                          cev:esRespondidoMediante, cev:esDesarrolladoPor)) }
}
ORDER BY ?origen
""",
        "columnas": ["Clase origen", "Propiedad", "Clase destino"],
    },
    {
        "id": "M.1",
        "grupo": "Modelo semantico (ciclo del conflicto)",
        "titulo": "Clases definidas y su formula (owl:equivalentClass)",
        "objetivo": "Listar las 6 clases definidas con su interseccion base AND (propiedad SOME destino).",
        "relacion_tesauro": "Modelo semantico: clasificacion automatica con HermiT a partir de restricciones.",
        "consulta": """
SELECT DISTINCT ?clase ?base ?propiedad ?destino
WHERE {
  ?clase owl:equivalentClass ?inter .
  ?inter owl:intersectionOf ?lista .
  ?lista rdf:rest*/rdf:first ?miembro .
  ?miembro a owl:Restriction ; owl:onProperty ?propiedad ; owl:someValuesFrom ?destino .
  ?inter owl:intersectionOf ?lista2 .
  ?lista2 rdf:rest*/rdf:first ?base .
  ?base a owl:Class .
}
ORDER BY ?clase
""",
        "columnas": ["Clase definida", "Clase base", "Propiedad (SOME)", "Clase destino"],
    },
    {
        "id": "M.2",
        "grupo": "Modelo semantico (ciclo del conflicto)",
        "titulo": "Pares inversos del modelo: poblaciones afectadas por hechos",
        "objetivo": "Listar los 7 axiomas raiz del ciclo con su lectura inversa (owl:inverseOf).",
        "relacion_tesauro": "Modelo semantico: afectaA/esAfectadaPor y el resto de pares inversos entre grupos.",
        "consulta": """
SELECT DISTINCT ?origen ?propiedad ?destino ?inversa
WHERE {
  ?origen rdfs:subClassOf ?r .
  ?r a owl:Restriction ; owl:onProperty ?propiedad ; owl:someValuesFrom ?destino .
  ?propiedad owl:inverseOf ?inversa .
  FILTER(?propiedad IN (cev:perpetradoPor, cev:afectaA, cev:generaImpacto,
                        cev:experimentaImpacto, cev:desarrollaAfrontamiento,
                        cev:respondeA, cev:empleaArma))
  FILTER(?destino IN (cev:Actores_armados, cev:Impactos, cev:Armas, cev:Poblaciones,
                      cev:Hechos_victimizantes, cev:Afrontamientos_resistencias_no_violentas))
}
ORDER BY ?origen ?propiedad
""",
        "columnas": ["Clase origen", "Propiedad", "Clase destino", "Lectura inversa"],
    },
    {
        "id": "M.3",
        "grupo": "Modelo semantico (ciclo del conflicto)",
        "titulo": "Ciclo completo del conflicto como cadena de 4 saltos",
        "objetivo": "Materializar el camino Actor - Hecho - Poblacion - Impacto - Afrontamiento.",
        "relacion_tesauro": "Modelo semantico: los 4 axiomas raiz encadenados recorren el ciclo causal CEV.",
        "consulta": """
SELECT DISTINCT ?actor ?hecho ?poblacion ?impacto ?afrontamiento
WHERE {
  ?hecho rdfs:subClassOf ?r1 .
  ?r1 a owl:Restriction ; owl:onProperty cev:perpetradoPor ; owl:someValuesFrom ?actor .
  ?hecho rdfs:subClassOf ?r2 .
  ?r2 a owl:Restriction ; owl:onProperty cev:afectaA ; owl:someValuesFrom ?poblacion .
  ?poblacion rdfs:subClassOf ?r3 .
  ?r3 a owl:Restriction ; owl:onProperty cev:experimentaImpacto ; owl:someValuesFrom ?impacto .
  ?poblacion rdfs:subClassOf ?r4 .
  ?r4 a owl:Restriction ; owl:onProperty cev:desarrollaAfrontamiento ; owl:someValuesFrom ?afrontamiento .
}
""",
        "columnas": ["Actor", "Hecho", "Poblacion", "Impacto", "Afrontamiento"],
    },
    {
        "id": "M.4",
        "grupo": "Modelo semantico (ciclo del conflicto)",
        "titulo": "Usabilidad: conteo de individuos por grupo",
        "objetivo": "Distribucion de los 534 individuos (terminos hoja) entre los 6 grupos.",
        "relacion_tesauro": "Anotaciones cev:grupo agregadas: tamano real de cada dominio en la ontologia.",
        "consulta": """
SELECT ?grupo (COUNT(DISTINCT ?individuo) AS ?numIndividuos)
WHERE {
  ?individuo cev:grupo ?grupo .
  FILTER(CONTAINS(STR(?individuo), "instancia_"))
}
GROUP BY ?grupo
ORDER BY DESC(?numIndividuos)
""",
        "columnas": ["Grupo del tesauro", "Individuos (terminos hoja)"],
    },
]

CONSULTAS_POR_ID: dict[str, dict[str, Any]] = {c["id"]: c for c in CONSULTAS}
ids_principales = [c["id"] for c in CONSULTAS if not c["id"].endswith("_inv")]
print(f"{len(ids_principales)} consultas SPARQL definidas: "
      f"{', '.join(ids_principales)} (mas la lectura inversa 3.2_inv)")

# %% [markdown]
# # GRUPO 1: "Actores armados y otros participantes en las dinamicas de la guerra"
#
# Clase raiz `cev:Actores_armados` (eje `agentes`): 30 subcategorias directas y
# 73 individuos. Relaciones del tesauro involucradas: TG/TE (estructura de
# actores armados), TR y USE-UP (relacion hecho-actor via `perpetradoPor`).

# %% [markdown]
# ## Consulta 1.1 - Subcategorias de Actores armados (jerarquia TG/TE)
#
# **Objetivo:** listar las subcategorias directas (TE) de la raiz
# "Actores armados y otros participantes en las dinamicas de la guerra" con su
# label del tesauro original. Busca responder: ?cual es la estructura
# inmediata de actores contemplada por el tesauro?
#
# **Relacion con el tesauro y la ontologia:** representa la relacion TG/TE del
# tesauro: el grupo es el termino general (TG) y cada subcategoria es un
# termino especifico (TE). En la ontologia se modela con el patron de
# restricciones `?clase rdfs:subClassOf [esParteDe some cev:Actores_armados]`,
# consultado via la propiedad `cev:esParteDe` (inversa `cev:parteDe`).

# %%
_ejecutar("1.1")

# %% [markdown]
# ## Consulta 1.2 - Individuos (terminos hoja) del grupo Actores armados
#
# **Objetivo:** recuperar todos los individuos `instancia_<slug>` (terminos
# hoja del tesauro) que pertenecen al grupo de actores armados, con su label.
# Busca responder: ?cuantos terminos concretos de actores (reclutamiento,
# acciones belicas, etc.) quedaron instanciados y como se llaman?
#
# **Relacion con el tesauro y la ontologia:** en la ontologia, cada termino
# hoja del tesauro se materializo como un individuo anotado con su label,
# `cev:grupo` y `cev:eje`. La consulta usa la jerarquia transitiva
# `rdfs:subClassOf*` para atrapar individuos de cualquier profundidad bajo la
# raiz `cev:Actores_armados`, y distingue individuos de clases por el prefijo
# `instancia_` del IRI.

# %%
_ejecutar("1.2")

# %% [markdown]
# ## Consulta 1.3 - Hechos perpetrados por actores armados (perpetradoPor)
#
# **Objetivo:** identificar que hechos victimizantes tienen como perpetrador a
# un termino del grupo "Actores armados" (p. ej. el reclutamiento como actor
# del conflicto). Busca responder: ?que hechos del tesauro estan vinculados
# causalmente con actores armados?
#
# **Relacion con el tesauro y la ontologia:** representa la relacion USE-UP /
# de perpetracion del dominio CEV, modelada con los 13 axiomas
# `?hecho rdfs:subClassOf [perpetradoPor some ?perpetrador]` (12 por
# co-ocurrencia de la fase 2 mas el axioma raiz del modelo semantico). La
# propiedad `cev:perpetradoPor` tiene inversa `cev:esVictimaDe`; aqui se filtra
# que el perpetrador pertenezca (transitivamente) al grupo de actores armados.

# %%
_ejecutar("1.3")

# %% [markdown]
# # GRUPO 2: "Armas"
#
# Clase raiz `cev:Armas` (eje `instrumentos`): 2 subcategorias directas
# (Instrumentos y Medios) y 74 individuos. Relaciones del tesauro
# involucradas: TG/TE (jerarquia de armas), TR (afinidad de armas con otros
# grupos) y busqueda por keyword en labels.

# %% [markdown]
# ## Consulta 2.1 - Jerarquia TG/TE completa del grupo Armas
#
# **Objetivo:** recuperar todos los pares (hijo, padre) de la jerarquia del
# grupo Armas, es decir, el arbol TG/TE completo de instrumentos y medios
# (p. ej. "Armas de fuego" es parte de "Instrumentos"). Busca responder:
# ?como se organiza el arbol de conceptos de armas?
#
# **Relacion con el tesauro y la ontologia:** es la consulta canonica de la
# relacion TG/TE: cada triple `hijo esParteDe padre` (restriccion OWL) es un
# salto de la jerarquia del tesauro. Como la ontologia es un arbol cerrado por
# grupo, basta restringir el hijo con `rdfs:subClassOf* cev:Armas` para que el
# padre quede automaticamente dentro del grupo; se muestran los labels
# originales.

# %%
_ejecutar("2.1")

# %% [markdown]
# ## Consulta 2.2 - Relaciones TR (afines) entre Armas y otros grupos
#
# **Objetivo:** listar los terminos del grupo Armas que tienen relaciones de
# afinidad terminologica (TR) con terminos de otros grupos, mostrando a que
# grupo pertenece el termino afiliado. Busca responder: ?con que otros
# dominios del tesauro (actores, hechos) se cruza el dominio de las armas?
#
# **Relacion con el tesauro y la ontologia:** representa la relacion TR del
# tesauro, modelada con los 50 axiomas de `cev:relacionadoCon` (propiedad
# simetrica). La consulta une las dos direcciones de la simetria y, para el
# termino no-armas, deriva su grupo raiz recorriendo la jerarquia, lo que
# revela las intersecciones entre grupos de la ontologia.

# %%
_ejecutar("2.2")

# %% [markdown]
# ## Consulta 2.3 - Busqueda por keyword en labels del grupo Armas
#
# **Objetivo:** buscar terminos del grupo Armas cuyo label contenga las
# palabras "arma", "fuego", "explos" o "blanca", distinguiendo si el resultado
# es una clase o un individuo. Busca responder: ?que conceptos del tesauro
# mencionan explicitamente la nocion de arma y sus tipos?
#
# **Relacion con el tesauro y la ontologia:** ejercicio de recuperacion de
# informacion (RI) sobre los labels `rdfs:label` que conservan el texto
# original del tesauro. Combina la jerarquia transitiva (`subClassOf*`) con
# `FILTER(CONTAINS(LCASE(?label), ...))`, mostrando que los labels son la
# puerta de entrada textual a la ontologia.

# %%
_ejecutar("2.3")

# %% [markdown]
# # GRUPO 3: "Hechos victimizantes"
#
# Clase raiz `cev:Hechos_victimizantes` (eje `eventos`): 22 subcategorias
# directas y 77 individuos. Relaciones del tesauro involucradas: TG/TE,
# USE-UP (perpetracion/victimizacion) y TR con el grupo de poblaciones.

# %% [markdown]
# ## Consulta 3.1 - Subcategorias de hechos victimizantes con conteo de terminos
#
# **Objetivo:** listar las 22 subcategorias directas de hechos victimizantes
# ordenadas por el numero de terminos hoja (individuos) que albergan. Busca
# responder: ?cuales son los grandes bloques de hechos del tesauro y cual es
# el mas desarrollado terminologicamente?
#
# **Relacion con el tesauro y la ontologia:** combina la jerarquia TG/TE
# (restriccion `esParteDe some Hechos_victimizantes`) con una agregacion
# `COUNT(DISTINCT ?individuo)` sobre todas las instancias del sub-arbol de
# cada clase (`?sub rdfs:subClassOf* ?clase`), mostrando la densidad
# terminologica del tesauro por TE.

# %%
_ejecutar("3.1")

# %% [markdown]
# ## Consulta 3.2 - Relaciones de perpetracion y su lectura inversa (esVictimaDe)
#
# **Objetivo:** listar los 13 pares (hecho, perpetrador) modelados en la
# ontologia y, en una segunda agregacion, leerlos en la direccion inversa:
# ?de cuantos hechos distintos es victima (esVictimaDe) cada perpetrador?
# Busca responder: ?quien perpetra que en la ontologia CEV?
#
# **Relacion con el tesauro y la ontologia:** la propiedad `cev:perpetradoPor`
# (con inversa declarada `cev:esVictimaDe`) materializa la relacion
# USE-UP hecho-actor del tesauro. Los 13 axiomas son los 12 de co-ocurrencia
# de la fase 2 mas el axioma raiz `Hechos_victimizantes perpetradoPor
# Actores_armados` declarado por el modelo semantico. La consulta principal
# extrae los axiomas de restriccion; la segunda reutiliza el mismo patron pero
# agrega por perpetrador, ejerciendo la semantica inversa documentada en la
# ontologia.

# %%
_ejecutar("3.2")

print("Lectura inversa (esVictimaDe): cuantos hechos senalan a cada perpetrador")
_ejecutar("3.2_inv")

# %% [markdown]
# ## Consulta 3.3 - Interseccion TR entre Poblaciones y Hechos victimizantes
#
# **Objetivo:** encontrar los pares de afinidad terminologica (TR) que cruzan
# los grupos "Poblaciones" y "Hechos victimizantes" (p. ej. ninos, ninas y
# adolescentes afines al reclutamiento forzado). Busca responder: ?que
# poblaciones del tesauro estan semanticamente ligadas a que hechos?
#
# **Relacion con el tesauro y la ontologia:** segunda consulta de TR, ahora
# como interseccion entre grupos: se recorren los axiomas de
# `cev:relacionadoCon` (simetrica) y se exige que un extremo pertenezca
# (transitivamente) a Poblaciones y el otro a Hechos victimizantes. Muestra el
# poder de la ontologia para descubrir cruces que el tesauro plano no hace
# explicitos en una sola vista.

# %%
_ejecutar("3.3")

# %% [markdown]
# # GRUPO 4: "Impactos"
#
# Clase raiz `cev:Impactos` (eje `estados_efectos`): 8 subcategorias directas
# y 93 individuos. Relaciones del tesauro involucradas: TG/TE (incluida la
# profundidad de 2 niveles) y anotaciones de eje conceptual.

# %% [markdown]
# ## Consulta 4.1 - Subcategorias de Impactos con conteo de terminos
#
# **Objetivo:** listar las 8 subcategorias directas de impactos con el numero
# de terminos hoja de cada una. Busca responder: ?sobre que dimensiones
# (individuales, familiares, colectivas, ambientales...) impacta la guerra
# segun el tesauro y cual tiene mas desarrollo?
#
# **Relacion con el tesauro y la ontologia:** misma familia de consultas
# TG/TE con agregacion que la 3.1, aplicada al grupo de estados_efectos:
# restriccion `esParteDe some cev:Impactos` + `COUNT(DISTINCT ?individuo)`
# contando transitivamente todo el sub-arbol de cada subcategoria.

# %%
_ejecutar("4.1")

# %% [markdown]
# ## Consulta 4.2 - Caminos de 2 niveles dentro de Impactos (TG-TE-TE)
#
# **Objetivo:** recorrer la jerarquia de impactos a dos niveles de
# profundidad: pares (nivel1, nivel2) donde nivel2 es subconcepto de nivel1 y
# nivel1 es subconcepto de Impactos (p. ej. Impactos individuales >
# Impactos en el proyecto de vida personal). Busca responder: ?como se
# profundiza el arbol de impactos?
#
# **Relacion con el tesauro y la ontologia:** aplica dos veces el patron TG/TE
# (restricciones `esParteDe` encadenadas), equivalente a recorrer dos saltos
# de la jerarquia del tesauro. Es la consulta de "caminos de 2 niveles" que
# evidencia que la ontologia conserva la profundidad multi-nivel del tesauro
# original (74% de los terminos tiene entre 3 y 5 niveles segun el EDA).

# %%
_ejecutar("4.2")

# %% [markdown]
# ## Consulta 4.3 - Auditoria de anotaciones: entidades por (grupo, eje)
#
# **Objetivo:** agrupar TODAS las entidades anotadas de la ontologia (clases e
# individuos) segun el par de anotaciones de trazabilidad `cev:grupo` y
# `cev:eje`. Busca responder: ?cuantas entidades tiene cada grupo, y es
# consistente la anotacion de eje dentro de cada grupo?
#
# **Relacion con el tesauro y la ontologia:** explota las anotaciones
# agregadas en la fase 2: cada clase e individuo lleva su grupo raiz del
# tesauro y su eje conceptual. Esta consulta es una auditoria de coherencia:
# si el eje fuera inconsistente, aparecerian pares (grupo, eje) mixtos. La
# suma de las entidades anotadas (1.150) debe igualar las clases del tesauro
# (616; las 6 clases definidas y la superclase tecnica no llevan path) mas los
# 534 individuos de la ontologia.

# %%
_ejecutar("4.3")

# %% [markdown]
# # GRUPO 5: "Poblaciones"
#
# Clase raiz `cev:Poblaciones` (eje `poblaciones`): 44 subcategorias directas
# y 199 individuos (el grupo mas grande). Relaciones del tesauro
# involucradas: TG/TE (con ranking), busqueda por keyword y propiedades
# inversas `esParteDe`/`parteDe`.

# %% [markdown]
# ## Consulta 5.1 - Top 10 subcategorias de Poblaciones con mas terminos
#
# **Objetivo:** rankear las subcategorias directas de poblaciones por numero
# de terminos hoja y quedarse con las 10 mas desarrolladas. Busca responder:
# ?cuales son las poblaciones mas finamente descritas por el tesauro (p. ej.
# pueblos indigenas, victimas del conflicto, mujeres)?
#
# **Relacion con el tesauro y la ontologia:** consulta TG/TE con agregacion y
# ordenamiento descendente (top-k). Evidencia que la ontologia conserva la
# granularidad del tesauro: 199 individuos en 44 TE directos, con
# concentracion en unas pocas subcategorias.

# %%
_ejecutar("5.1")

# %% [markdown]
# ## Consulta 5.2 - Busqueda por keyword sobre poblaciones (indigena, afro, campesin, joven, mujer)
#
# **Objetivo:** recuperar las clases e individuos del grupo Poblaciones cuyo
# label menciona a los grupos etarios/etnicos mas relevantes del conflicto
# colombiano: indigena, afro, campesin, joven, nino, nina, mujer. Busca
# responder: ?como aparece representada la poblacion diferencial en el
# tesauro?
#
# **Relacion con el tesauro y la ontologia:** consulta de RI sobre labels con
# `FILTER(CONTAINS(LCASE(?label), ...))` (case-insensitive), combinada con la
# jerarquia transitiva del grupo. Complementa la 2.3 pero sobre el grupo
# poblacional y con un vocabulario de busqueda distinto, demostrando que los
# labels son consultables como indice textual del tesauro.

# %%
_ejecutar("5.2")

# %% [markdown]
# ## Consulta 5.3 - Pueblos indigenas: sus partes (esParteDe) y su parte (parteDe)
#
# **Objetivo:** para la clase "Pueblos indigenas" (la subcategoria con mas
# terminos del grupo), listar por un lado sus subconceptos directos (los 100
# pueblos: Achagua, Arhuaco, ...) via `esParteDe`, y por otro su termino
# general "Poblaciones" via la propiedad inversa `parteDe`. Busca responder:
# ?que pueblos concretos componen el TE y a que TG pertenece?
#
# **Relacion con el tesauro y la ontologia:** ejercicio de las propiedades
# inversas: `hijo esParteDe PueblosIndigenas` (TE) y la lectura inversa
# `PueblosIndigenas parteDe Poblaciones` (TG). En SPARQL se modela con una
# UNION de los dos sentidos sobre el mismo patron de restriccion, y la
# columna "Relacion" indica la direccion leida. Muestra la riqueza del
# sub-arbol etnico del tesauro (USE-UP del termino "pueblos").

# %%
_ejecutar("5.3")

# %% [markdown]
# # GRUPO 6: "Afrontamientos y resistencias no violentas - cotidianas"
#
# Clase raiz `cev:Afrontamientos_resistencias_no_violentas` (eje
# `afrontamientos_resistencias`): el grupo mas pequeno (23 terminos) y el unico
# con un TE intermedio unico ("cotidianas", que da nombre al grupo en el
# tesauro original): 1 subcategoria directa de la raiz, 16 TE de segundo nivel
# y 6 de tercer nivel (23 individuos). Relaciones del tesauro involucradas:
# TG/TE (arbol completo), busqueda por keyword del vocabulario de la respuesta
# social al conflicto y participacion en el modelo semantico (las poblaciones
# desarrollan afrontamientos que responden a hechos).

# %% [markdown]
# ## Consulta 6.1 - Arbol TG/TE completo del grupo Afrontamientos
#
# **Objetivo:** recuperar todos los pares (hijo, padre) del arbol del grupo:
# la raiz tiene un TE directo ("cotidianas"), que despliega 16 TE (Actividad
# fisica, deporte y recreacion; Afrontamientos familiares; Memoria colectiva;
# Redes o expresiones de apoyo; ...) y estos a su vez 6 TE de tercer nivel
# (Movilizacion emocional, Proyecto artistico, Convivio, ...). Busca
# responder: ?como se organiza la respuesta social al conflicto segun el
# tesauro?
#
# **Relacion con el tesauro y la ontologia:** patron canonico TG/TE (igual que
# la 2.1): cada triple `hijo esParteDe padre` es un salto de la jerarquia,
# con el hijo restringido transitivamente bajo la raiz del grupo. La consulta
# evidencia que el arbol de afrontamientos es poco profundo (3-4 niveles) y
# tematicamente variado, dominado por practicas cotidianas y comunitarias.

# %%
_ejecutar("6.1")

# %% [markdown]
# ## Consulta 6.2 - Busqueda por keyword en labels del grupo (participacion, apoyo, familia)
#
# **Objetivo:** recuperar las clases e individuos del grupo cuyo label
# menciona "participacion" (consejos comunitarios, juntas de accion comunal,
# partidos politicos), "apoyo" (redes de apoyo, apoyo institucional) o
# "familia" (afrontamientos familiares). Se eligen estos terminos porque son
# el nucleo lexico de la respuesta social al conflicto: la participacion
# comunitaria y las redes de apoyo distinguen a este grupo de los otros cinco
# y no aparecen en el vocabulario de hechos ni impactos.
#
# **Relacion con el tesauro y la ontologia:** consulta de RI sobre labels con
# `FILTER(CONTAINS(LCASE(?label), ...))`. Se usa la raiz "participaci" (sin
# tilde) para cubrir tanto "participacion" como "participación" en una sola
# condicion case-insensitive, demostrando que los labels son un indice textual
# consultable incluso con variaciones ortograficas del tesauro original.

# %%
_ejecutar("6.2")

# %% [markdown]
# ## Consulta 6.3 - El grupo en el modelo semantico (respondeA / desarrollaAfrontamiento)
#
# **Objetivo:** listar los axiomas del ciclo del conflicto cuyo origen o
# destino es el grupo de afrontamientos: la raiz del grupo declara que los
# afrontamientos responden a hechos (`respondeA some Hechos_victimizantes`,
# lectura inversa `esRespondidoMediante`) y el grupo Poblaciones declara que
# las poblaciones los desarrollan (`desarrollaAfrontamiento some
# Afrontamientos...`, lectura inversa `esDesarrolladoPor`). Busca responder:
# ?que papel juega este grupo en el modelo semantico CEV?
#
# **Relacion con el tesauro y la ontologia:** el modelo semantico cierra el
# ciclo del conflicto con los afrontamientos como respuesta: las clases
# definidas `AfrontamientoAsociadoAHecho` (afrontamientos que responden a
# hechos) y `PoblacionConAfrontamiento` (poblaciones que los desarrollan)
# materializan estas dos direcciones y permiten a HermiT clasificar
# automaticamente los individuos que las satisfacen.

# %%
_ejecutar("6.3")

# %% [markdown]
# # CONSULTAS DEL MODELO SEMANTICO (CICLO DEL CONFLICTO)
#
# Las 4 consultas siguientes explotan el modelo fuerte agregado a la
# ontologia (seccion 3): 7 axiomas raiz entre los 6 grupos y 6 clases
# definidas con `owl:equivalentClass`. No sustituyen las 3 consultas por
# grupo: son un nivel adicional que demuestra que la ontologia no solo
# organiza terminos sino que permite inferir y consultar ciclos causales.

# %% [markdown]
# ## Consulta M.1 - Clases definidas y su formula (owl:equivalentClass)
#
# **Objetivo:** listar las 6 clases definidas con su formula completa: la
# interseccion de una clase base del tesauro con una restriccion
# `propiedad SOME destino` (p. ej. `HechoPerpetradoPorActor` equivale a
# `Hechos_victimizantes AND (perpetradoPor SOME Actores_armados)`). Busca
# responder: ?cuales son los conceptos derivados del modelo y como se definen?
#
# **Relacion con el tesauro y la ontologia:** las clases definidas son la
# pieza de razonamiento: al declarar la equivalencia, HermiT clasifica
# automaticamente como miembros de `PoblacionAfectadaPorHechos` a las
# poblaciones que satisfacen la restriccion, sin necesidad de declararlo a
# mano. La consulta navega `owl:equivalentClass`, `owl:intersectionOf`
# (listas rdf:rest) y las restricciones anidadas para reconstruir la formula
# legible.

# %%
_ejecutar("M.1")

# %% [markdown]
# ## Consulta M.2 - Pares inversos del modelo: poblaciones afectadas por hechos
#
# **Objetivo:** listar los 7 axiomas raiz del ciclo del conflicto con su
# lectura inversa declarada (`owl:inverseOf`), con foco en el par
# `Hechos_victimizantes afectaA Poblaciones` / `Poblaciones esAfectadaPor
# Hechos_victimizantes`: las poblaciones afectadas por hechos son consultables
# en ambas direcciones sin duplicar axiomas. Busca responder: ?que relaciones
# entre grupos declara el modelo y como se leen en sentido inverso?
#
# **Relacion con el tesauro y la ontologia:** ejercicio de semantica inversa
# sobre el modelo: `perpetradoPor/esVictimaDe`, `afectaA/esAfectadaPor`,
# `generaImpacto/esGeneradoPor`, `experimentaImpacto/impactaA`,
# `desarrollaAfrontamiento/esDesarrolladoPor`, `respondeA/esRespondidoMediante`
# y `empleaArma/esEmpleadaEn`. El filtro doble (propiedad del modelo y destino
# raiz) aisla exactamente los 7 axiomas raiz, excluyendo los 610 axiomas
# estructurales `esParteDe`.

# %%
_ejecutar("M.2")

# %% [markdown]
# ## Consulta M.3 - Ciclo completo del conflicto como cadena de 4 saltos
#
# **Objetivo:** materializar en una sola consulta el ciclo causal del
# conflicto: Actor -[perpetra]-> Hecho -[afecta]-> Poblacion -[experimenta]->
# Impacto y Poblacion -[desarrolla]-> Afrontamiento. Busca responder: ?existe
# un camino completo que una los 5 dominios del modelo?
#
# **Relacion con el tesauro y la ontologia:** encadena 4 patrones de
# restriccion `someValuesFrom` (un JOIN conceptual de 4 saltos): con solo los
# axiomas raiz declarados, la consulta devuelve el ciclo cerrado
# `Actores_armados -> Hechos_victimizantes -> Poblaciones -> Impactos ->
# Afrontamientos_resistencias_no_violentas`. Es la demostracion de que el
# modelo permite consultar trayectorias causales completas, algo imposible
# sobre el tesauro plano en Excel o sobre un analisis puramente lexico.

# %%
_ejecutar("M.3")

# %% [markdown]
# ## Consulta M.4 - Usabilidad: conteo de individuos por grupo
#
# **Objetivo:** distribucion de los 534 individuos (terminos hoja) entre los 6
# grupos a partir de la anotacion `cev:grupo`. Busca responder: ?de que tamano
# es cada dominio consultable de la ontologia?
#
# **Relacion con el tesauro y la ontologia:** agregacion sobre las anotaciones
# de trazabilidad (igual patron que la 4.3 pero solo con individuos): Poblaciones
# 199, Impactos 93, Hechos victimizantes 77, Armas 74, Actores armados 73 y
# Afrontamientos 18. La suma (534) coincide con el total de individuos de las
# metricas de la seccion 3, validando la coherencia de la anotacion.

# %%
_ejecutar("M.4")

# %% [markdown]
# # Sintesis de la seccion
#
# **Que revelan las 22 consultas sobre la ontologia y el tesauro:**
#
# 1. **La ontologia conserva fielmente la jerarquia TG/TE del tesauro:** las
#    consultas de jerarquia (1.1, 2.1, 3.1, 4.1, 4.2, 5.1, 5.3, 6.1)
#    recuperan el arbol multi-nivel de los 6 grupos sin perdida de
#    profundidad: se verifican 82 clases bajo Actores armados, 96 pares
#    TG/TE bajo Armas, 82 pares de 2 niveles bajo Impactos, 100 pueblos bajo
#    Pueblos indigenas y 23 pares en el arbol de afrontamientos.
# 2. **Las relaciones TR (afinidad) cruzan grupos y solo existen donde el
#    tesauro las declara:** 4 afinidades de Armas (con actores y hechos) y 9
#    cruces Poblaciones-Hechos victimizantes (concentrados en ninos, ninas y
#    adolescentes y reclutamiento), lo que confirma que la heuristica IDF de
#    la fase 2 no produjo ruido.
# 3. **Las relaciones de perpetracion son escasas y semanticamente
#    precisas:** 13 axiomas `perpetradoPor` (12 por co-ocurrencia mas el
#    axioma raiz del modelo semantico, con su inversa `esVictimaDe`)
#    concentrados en reclutamiento y utilizacion de menores, coherentes con el
#    dominio de la Comision de la Verdad.
# 4. **Las anotaciones `grupo` y `eje` permiten auditar la trazabilidad:**
#    la consulta 4.3 confirma que las 1.150 entidades anotadas (616 clases +
#    534 individuos) llevan su grupo raiz y que el eje es consistente dentro
#    de cada grupo (un solo par grupo-eje por fila, incluido el nuevo grupo de
#    afrontamientos), de modo que los filtros por `cev:grupo` +
#    `subClassOf*` segmentan la ontologia por dominio en cualquier consulta.
# 5. **Los labels son un indice textual consultable:** las busquedas por
#    keyword (2.3, 5.2, 6.2) responden preguntas de recuperacion de
#    informacion directamente sobre la ontologia, algo que el tesauro plano en
#    Excel no permite hacer de forma integrada con la estructura.
# 6. **El grupo de afrontamientos cierra el ciclo con la respuesta social:**
#    la 6.1 muestra un arbol tematicamente variado (practicas cotidianas,
#    comunitarias, familiares y espirituales) y la 6.3 lo ubica en el modelo:
#    las poblaciones desarrollan afrontamientos que responden a hechos.
# 7. **El modelo semantico convierte la ontologia en una base de
#    conocimiento inferible:** las consultas M.1-M.3 muestran las 6 clases
#    definidas (clasificables por HermiT), los 7 pares inversos del ciclo y la
#    cadena causal completa Actor-Hecho-Poblacion-Impacto-Afrontamiento en una
#    sola consulta; M.4 valida los 534 individuos distribuidos en 6 dominios.
