# %% [markdown]
# # SECCION 3: Ontologia OWL del Tesauro de la Comision de la Verdad
#
# **Objetivo:** transformar los 609 terminos (paths jerarquicos "A - B - C")
# seleccionados en la seccion 2 en una ontologia OWL (RDF/XML) consultable con
# SPARQL, con clases, individuos, anotaciones de trazabilidad y object
# properties que reflejan la estructura del tesauro y el ciclo del conflicto.
#
# **Entregables de esta seccion:**
# - `../ontologia/tesauro_cev.owl` (RDF/XML, entregable principal).
# - `../ontologia/relaciones_afines.txt` (trazabilidad de las relaciones TR).
# - `../imagenes/arbol_ontologia.png` (jerarquia visual).
#
# **Diseno en una linea:** 6 clases raiz (una por grupo) bajo una superclase
# tecnica `ConceptoDelTesauro`; cada path se convierte en una cadena de
# `rdfs:subClassOf`; cada termino hoja recibe un individuo; 17 object
# properties (5 estructurales + 6 pares inversos del ciclo del conflicto)
# modelan la estructura meronimica, la afinidad terminologica (TR), la
# relacion hecho-actor y el ciclo Actor -> Hecho -> Poblacion -> Impacto ->
# Afrontamiento, con 6 clases definidas (equivalentClass) y 7 axiomas raiz
# de dominio que el razonador explota para clasificacion automatica.
#
# **Implementacion:** la construccion, persistencia y validacion viven en el
# nucleo POO `ServicioOntologia` (Singleton + Facade, seccion 00); esta
# seccion invoca su API (`obtener`, `cargar`, `guardar`, `metricas`,
# `validar_rdflib`, `validar_hermit`) y conserva la figura del arbol
# jerarquico como clase propia de visualizacion.

# %% [markdown]
# ## 3.1 Imports, API del servicio y rutas (Singleton + Facade)
#
# El refactor de esta seccion sustituye las funciones sueltas de construccion
# y validacion (nodos, jerarquia, anotaciones, TR, perpetradores, dominio,
# clases definidas, rdflib, HermiT) por la API unica de `ServicioOntologia`:
#
# 1. `obtener(ruta_owl, ruta_json, ruta_trazabilidad)`: instancia unica por
#    ruta (Singleton); todas las secciones comparten la misma OWL en memoria.
# 2. `cargar()`: carga el .owl existente o construye desde el JSON y guarda.
# 3. `guardar()`: persiste en RDF/XML (solo si se reconstruyo).
# 4. `metricas()`, `validar_rdflib()`, `validar_hermit()`: validacion.
#
# En el notebook las constantes y el nucleo POO ya existen (sec00_setup +
# sec00_clases); en ejecucion standalone se definen e importan localmente.

# %%
# Imports y compatibilidad: el nucleo POO (sec00_clases) y las constantes del
# proyecto (sec00_setup) ya viven en el namespace del notebook; en modo script
# se importan y definen aqui mismo (mismo patron que el resto de secciones).
import sys
from pathlib import Path

import matplotlib

# En ejecucion headless (script) se fuerza el backend Agg; en el notebook se
# conserva el backend inline activado en sec00 (figuras embebidas).
try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")

try:
    _DIR_CELDAS = Path(__file__).resolve().parent  # modo script
except NameError:
    _DIR_CELDAS = Path.cwd()  # modo notebook: el kernel corre en notebooks/

# El modulo sec00_clases vive en notebook_cells/ (script) o en
# notebooks/../notebook_cells (notebook abierto desde notebooks/).
_DIR_MODULO = (
    _DIR_CELDAS
    if (_DIR_CELDAS / "sec00_clases.py").exists()
    else _DIR_CELDAS.parent / "notebook_cells"
)
sys.path.insert(0, str(_DIR_MODULO))

if "sc" not in globals():
    import sec00_clases as sc

# Constantes del proyecto: en el notebook las define sec00_setup; en modo
# script se replican localmente (sin duplicar las que ya viven en sc).
if "FIGURAS" not in globals():
    RUTA_BASE = _DIR_CELDAS.parent
    RUTA_DATOS = RUTA_BASE / "data"
    RUTA_IMAGENES = RUTA_BASE / "imagenes"
    RUTA_ONTOLOGIA_DIR = RUTA_BASE / "ontologia"
    RUTA_JSON_GRUPOS = RUTA_DATOS / "grupos_seleccionados.json"
    RUTA_ONTOLOGIA = RUTA_ONTOLOGIA_DIR / "tesauro_cev.owl"
    RUTA_TRAZABILIDAD = RUTA_ONTOLOGIA_DIR / "relaciones_afines.txt"
    FIGURAS = {
        "arbol_ontologia": RUTA_IMAGENES / "arbol_ontologia.png",
        "grafo_completo": RUTA_IMAGENES / "grafo_ontologia_completo.png",
        "grafo_grupos": RUTA_IMAGENES / "grafo_ontologia_grupos.png",
    }
    COLORES_GRUPO = {
        "Actores armados y otros participantes en las dinámicas de la guerra": "#d62728",
        "Impactos": "#1f77b4",
        "Armas": "#2ca02c",
        "Poblaciones": "#ff7f0e",
        "Hechos victimizantes": "#9467bd",
        "Afrontamientos y resistencias no violentas - cotidianas": "#8c564b",
    }

# Constantes locales de la seccion (visualizacion jerarquica).
GRUPO_ARBOL = "Armas"
NIVELES_PANEL_RAICES = 2

print("Nucleo POO importado:", sc.__name__)
print(f"Ruta ontologia: {RUTA_ONTOLOGIA}")

# %% [markdown]
# ## 3.2 Construccion o carga de la ontologia
#
# En Colab el flujo es: si existe `../ontologia/tesauro_cev.owl` se carga
# directamente (rapido); si no, se reconstruye desde el JSON de la seccion 2.
# El guardado solo ocurre cuando la ontologia fue reconstruida. Reglas de
# construccion (implementadas en `ServicioOntologia.construir_desde_grupos`):
#
# 1. Seis clases raiz (una por grupo) como subclases de `ConceptoDelTesauro`.
# 2. Cada nodo es una clase; su superclase es la clase del segmento anterior
#    del path (rdfs:subClassOf); los prefijos ausentes del JSON se crean como
#    nodos sinteticos (siete en esta seleccion).
# 3. Nombres locales unicos: si un slug colisiona entre grupos se anade el
#    sufijo del grupo (mecanismo defensivo; no ocurre en esta seleccion).
# 4. Anotaciones en toda clase: rdfs:label (descriptor legible), rdfs:comment
#    (path completo original), cev:grupo (grupo raiz) y cev:eje (eje).
# 5. Un individuo (NamedIndividual) por cada termino HOJA del JSON (534).
# 6. Object properties (17): esParteDe/parteDe (inversas, axioma por edge),
#    relacionadoCon (simetrica, heuristica IDF + 5 raices fijas CEV),
#    perpetradoPor/esVictimaDe (inversas, co-ocurrencia especifica) y los 6
#    pares inversos del ciclo (afectaA/esAfectadaPor, generaImpacto/
#    esGeneradoPor, experimentaImpacto/impactaA, desarrollaAfrontamiento/
#    esDesarrolladoPor, respondeA/esRespondidoMediante, empleaArma/
#    esEmpleadaEn).
# 7. Modelo semantico de dominio: 7 axiomas raiz del ciclo (grupo subClassOf
#    propiedad some grupo) y 6 clases definidas con equivalentClass (ver
#    seccion 3.6).

# %%
# El Singleton garantiza una sola instancia por ruta de OWL: si otra seccion
# ya cargo la ontologia, esta llamada la reutiliza (misma ontologia en
# memoria, consistencia entre secciones).
servicio = sc.ServicioOntologia.obtener(
    RUTA_ONTOLOGIA, RUTA_JSON_GRUPOS, RUTA_TRAZABILIDAD
)
construido = servicio.cargar()

if construido:
    servicio.guardar()
    print(f"Ontologia construida desde JSON y guardada en {RUTA_ONTOLOGIA}")
else:
    print(f"Ontologia cargada desde {RUTA_ONTOLOGIA} (no se reescribio)")

resumen = servicio.resumen_construccion
print("=== Resumen de la construccion ===")
if resumen:
    for clave, valor in resumen.items():
        print(f"- {clave}: {valor}")
else:
    print("- Ontologia existente: no se reconstruyo; metricas en 3.7")

# %% [markdown]
# ## 3.5 Object properties y relaciones semanticas
#
# Cinco familias de propiedades (17 en total, definidas y declaradas por el
# servicio; todas con dominio/rango `ConceptoDelTesauro`):
#
# 1. **cev:esParteDe / cev:parteDe** (inversas): estructura meronimica del
#    tesauro. Para cada edge del arbol (hijo -> padre) se declara el axioma
#    `hijo subClassOf (cev:esParteDe some padre)`: el concepto hijo es parte
#    del concepto padre. Genera 610 axiomas (uno por edge: 616 nodos menos
#    las 6 raices).
#
# 2. **cev:relacionadoCon** (simetrica, termino relacionado / TR del tesauro):
#    afinidad entre grupos. Criterio documentado en 3.5.2: co-ocurrencia de
#    tokens especificos (lemas normalizados con IDF alto) en nodos de grupos
#    distintos, ranqueada por suma de IDF, acotada a 45 pares; se suman 5
#    relaciones raiz fijas justificadas por el modelo conceptual de la CEV
#    (total 50).
#
# 3. **cev:perpetradoPor / cev:esVictimaDe** (inversas): vinculo hecho
#    victimizante - actor armado. Criterio: co-ocurrencia de tokens muy
#    especificos (IDF alto, nivel <= 3) entre clases de Hechos victimizantes
#    y de Actores armados. Solo se conservan pares con sentido en el dominio
#    CEV (reclutamiento, acciones belicas, inteligencia, secuestro politico);
#    total 12.
#
# 4. **Modelo semantico del ciclo del conflicto** (6 pares inversos):
#    formaliza las relaciones de dominio entre los 6 grupos
#    (Actor -> Hecho -> Poblacion -> Impacto -> Afrontamiento); la tabla
#    siguiente resume propiedad, par inverso y significado.
#
# | Propiedad | Par inverso | Significado |
# |-----------|-------------|-------------|
# | cev:afectaA | cev:esAfectadaPor | un hecho victimizante afecta a una poblacion |
# | cev:generaImpacto | cev:esGeneradoPor | un hecho victimizante genera impactos |
# | cev:experimentaImpacto | cev:impactaA | una poblacion experimenta los impactos |
# | cev:desarrollaAfrontamiento | cev:esDesarrolladoPor | una poblacion desarrolla afrontamientos no violentos |
# | cev:respondeA | cev:esRespondidoMediante | un afrontamiento responde a un hecho victimizante |
# | cev:empleaArma | cev:esEmpleadaEn | un actor armado emplea armas en el conflicto |
# | cev:perpetradoPor | cev:esVictimaDe | un hecho victimizante es perpetrado por un actor armado |
#
# Todas las propiedades tienen dominio y rango `ConceptoDelTesauro`: operan
# solo sobre conceptos de este vocabulario.

# %% [markdown]
# ### 3.5.2 Relaciones de afinidad (TR): heuristica documentada
#
# **Criterio (documentado en `../ontologia/relaciones_afines.txt`):**
# 1. Se consideran nodos de nivel >= 2 (conceptos especificos; se excluyen las
#    raices y los subgrupos de primer nivel demasiado genericos).
# 2. Los tokens se extraen de todos los segmentos del path salvo la raiz,
#    normalizados a ASCII, sin stopwords ni tokens genericos del dominio.
# 3. Solo cuentan tokens especificos: frecuencia de documento <= 10 (IDF alto).
# 4. Un par (nodo1, nodo2) de grupos distintos es candidato si comparte al
#    menos un token especifico; su puntuacion es la suma de IDF de los tokens
#    compartidos.
# 5. Se conservan los 45 pares mejor puntuados (maximo 50 con las 5 raiz).
#
# **Relaciones raiz fijas (justificadas por el modelo CEV):**
# - Hechos victimizantes ~ Actores armados (los hechos son cometidos por
#   actores armados).
# - Armas ~ Actores armados (los actores emplean armas).
# - Armas ~ Hechos victimizantes (las armas son instrumentos de los hechos).
# - Impactos ~ Hechos victimizantes (los hechos generan impactos).
# - Impactos ~ Poblaciones (los impactos recaen sobre las poblaciones).

# %% [markdown]
# ### 3.5.3 Relacion hecho - actor: cev:perpetradoPor / cev:esVictimaDe
#
# **Justificacion:** en el marco de la CEV los hechos victimizantes son
# perpetrados por actores armados. La relacion se materializa SOLO cuando la
# terminologia lo respalda: co-ocurrencia de tokens muy especificos (frecuencia
# de documento <= 8) entre una clase de Hechos victimizantes (nivel <= 3) y
# una de Actores armados (nivel <= 3). Ejemplos de pares obtenidos:
# "Reclutamiento, uso y utilizacion de ninos" <-> "Reclutamiento" (actores);
# "Utilizacion en acciones belicas" <-> "Acciones belicas";
# "Secuestro politico" <-> "Conflicto politico armado".
# Se omite esta propiedad si la co-ocurrencia no produce pares de calidad.

# %% [markdown]
# ### 3.6 Modelo semantico de dominio: axiomas raiz y clases definidas
#
# **Axiomas raiz del ciclo (7):** se declara sobre las clases raiz
# `GrupoOrigen subClassOf (propiedad some GrupoDestino)` para cada par de
# RELACIONES_ENTRE_GRUPOS; con ello el razonador puede clasificar las clases
# definidas y responder consultas del ciclo completo:
#
# - Hechos_victimizantes subClassOf (perpetradoPor some Actores_armados)
# - Hechos_victimizantes subClassOf (afectaA some Poblaciones)
# - Hechos_victimizantes subClassOf (generaImpacto some Impactos)
# - Poblaciones subClassOf (experimentaImpacto some Impactos)
# - Poblaciones subClassOf (desarrollaAfrontamiento some
#   Afrontamientos_resistencias_no_violentas)
# - Afrontamientos_resistencias_no_violentas subClassOf (respondeA some
#   Hechos_victimizantes)
# - Actores_armados subClassOf (empleaArma some Armas)
#
# **Clases definidas (6) con equivalentClass:** cada una equivale a la
# interseccion `Base AND (propiedad SOME Destino)`; HermiT las usa para
# reparenting automatico (ver seccion 3.8 Usabilidad):
#
# - HechoPerpetradoPorActor ≡ Hechos_victimizantes AND
#   (perpetradoPor SOME Actores_armados)
# - PoblacionAfectadaPorHechos ≡ Poblaciones AND
#   (esAfectadaPor SOME Hechos_victimizantes)
# - PoblacionConImpacto ≡ Poblaciones AND
#   (experimentaImpacto SOME Impactos)
# - PoblacionConAfrontamiento ≡ Poblaciones AND
#   (desarrollaAfrontamiento SOME Afrontamientos_resistencias_no_violentas)
# - HechoQueGeneraImpacto ≡ Hechos_victimizantes AND
#   (generaImpacto SOME Impactos)
# - AfrontamientoAsociadoAHecho ≡ Afrontamientos_resistencias_no_violentas
#   AND (respondeA SOME Hechos_victimizantes)

# %% [markdown]
# ## 3.7 Validacion
#
# Se verifica el entregable por cuatro vias (API del servicio):
# 1. `metricas()`: conteos estructurales en memoria (clases, individuos,
#    propiedades y subclases por raiz).
# 2. Conteo del modelo semantico: axiomas esParteDe, relaciones
#    relacionadoCon (TR), perpetradoPor, axiomas raiz del ciclo y clases
#    definidas (restricciones some sobre la ontologia cargada).
# 3. `validar_rdflib()`: parse con rdflib (RDF/XML) y conteo de triples,
#    subClassOf y restricciones.
# 4. `validar_hermit()`: reasoner HermiT (incluido en el paquete owlready2):
#    comprobacion de consistencia; el pip install hermit externo no es
#    necesario.
#
# Cifras verificadas del entregable: 623 clases, 534 individuos, 17 object
# properties, 2 annotation, 610 axiomas esParteDe, 50 relacionadoCon, 12
# perpetradoPor, 7 relaciones de dominio, 6 clases definidas, rdflib ~9.812
# triples y HermiT consistente.

# %%
def _conteos_semanticos(servicio: sc.ServicioOntologia) -> dict[str, int]:
    """Cuenta los axiomas del modelo semantico de la ontologia cargada.

    Recorre las restricciones some (is_a de cada clase) por propiedad y las
    clases con equivalentClass; los axiomas raiz del ciclo se cuentan solo
    sobre las clases raiz (SLUG_RAIZ) para no mezclarlos con las 12
    relaciones heuristicas de perpetradoPor.

    Args:
        servicio: ServicioOntologia con la ontologia cargada.

    Returns:
        Dict con axiomas_es_parte_de, relacionado_con, perpetrado_por,
        relaciones_dominio y clases_definidas.
    """
    onto = servicio.onto
    props_modelo = {
        "perpetradoPor", "afectaA", "generaImpacto", "experimentaImpacto",
        "desarrollaAfrontamiento", "respondeA", "empleaArma",
    }
    raices = set(sc.SLUG_RAIZ.values())

    def contar(propiedad: str, excluir_raices: bool = False) -> int:
        total = 0
        for clase in onto.classes():
            if excluir_raices and clase.name in raices:
                continue
            for axioma in clase.is_a:
                if (
                    isinstance(axioma, sc.Restriction)
                    and getattr(axioma.property, "name", None) == propiedad
                ):
                    total += 1
        return total

    relaciones_dominio = 0
    for nombre_raiz in raices:
        for axioma in onto[nombre_raiz].is_a:
            if (
                isinstance(axioma, sc.Restriction)
                and getattr(axioma.property, "name", None) in props_modelo
            ):
                relaciones_dominio += 1
    return {
        "axiomas_es_parte_de": contar("esParteDe"),
        "relacionado_con": contar("relacionadoCon"),
        # Solo pares heuristicos (excluye el axioma raiz perpetradoPor, que
        # ya se reporta dentro de relaciones_dominio): coincide con el
        # resumen de construccion del nucleo (12).
        "perpetrado_por": contar("perpetradoPor", excluir_raices=True),
        "relaciones_dominio": relaciones_dominio,
        "clases_definidas": sum(1 for c in onto.classes() if c.equivalent_to),
    }


metricas = servicio.metricas()
print("=== Validacion estructural (ServicioOntologia.metricas) ===")
print(f"- Clases: {metricas['clases']}")
print(f"- Individuos: {metricas['individuos']}")
print(f"- Object properties: {metricas['object_properties']}")
print(f"- Annotation properties: {metricas['annotation_properties']}")
for raiz, total in sorted(metricas["subclases_por_raiz"].items()):
    print(f"  * {raiz}: {total} clases")

semanticos = _conteos_semanticos(servicio)
print("=== Modelo semantico (axiomas sobre la ontologia) ===")
print(f"- Axiomas esParteDe (some): {semanticos['axiomas_es_parte_de']}")
print(f"- Relaciones relacionadoCon (TR): {semanticos['relacionado_con']}")
print(f"- Relaciones perpetradoPor: {semanticos['perpetrado_por']}")
print(f"- Axiomas raiz del ciclo (dominio): {semanticos['relaciones_dominio']}")
print(f"- Clases definidas (equivalentClass): {semanticos['clases_definidas']}")

rdflib_resumen = servicio.validar_rdflib()
print("=== Validacion de parse (rdflib, RDF/XML) ===")
print(f"- Triples totales: {rdflib_resumen['triples']}")
print(f"- subClassOf entre clases nombradas: {rdflib_resumen['subclassof_nombradas']}")
print(f"- subClassOf con restricciones some: {rdflib_resumen['subclassof_restricciones']}")
print("- Parse rdflib: OK")
print("=== Reasoner ===")
if servicio.validar_hermit():
    print("- HermiT: consistencia verificada, sin errores")
else:
    print("- HermiT: omitido (ver advertencia en el reporte final)")

# %% [markdown]
# ### 3.7.2 Ejemplo de consulta SPARQL sobre el .owl
#
# Nota 1: owlready2 serializa los literales de anotacion con
# `datatype="xsd:string"` explicito; en SPARQL conviene comparar con
# `STR(?o) = "..."` o usar `"..."^^xsd:string` para no perder coincidencias.
# Nota 2: en OWL DL las relaciones entre CLASES se declaran como
# restricciones (subClassOf + owl:someValuesFrom); owlready2 transpila
# automaticamente toda object property clase-clase a esa forma. Por eso
# las consultas navegan la restriccion; un endpoint con reasoner (HermiT,
# Pellet) infiere ademas las relaciones entre individuos.

# %%
def consultar_ejemplos(ruta_owl: Path) -> None:
    """Ejecuta consultas SPARQL de demostracion sobre la ontologia guardada.

    Args:
        ruta_owl: Ruta del archivo RDF/XML.
    """
    grafo = sc.rdflib.Graph()
    grafo.parse(str(ruta_owl), format="xml")
    prefijos = (
        "PREFIX cev: <http://www.semanticweb.org/cev/tesauro#>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    )
    consultas = {
        "Conceptos del grupo Armas": (
            prefijos
            + "SELECT ?label WHERE { ?c rdfs:label ?label . "
            + "?c cev:grupo ?g . FILTER(STR(?g) = \"Armas\") } LIMIT 4"
        ),
        "Relaciones de afinidad de ninos, ninas y adolescentes (via restriccion)": (
            prefijos
            + "SELECT ?label WHERE { cev:NinosNinasYAdolescentes "
            + "rdfs:subClassOf ?r . ?r rdf:type owl:Restriction ; "
            + "owl:onProperty cev:relacionadoCon ; owl:someValuesFrom ?c . "
            + "?c rdfs:label ?label } LIMIT 6"
        ),
        "Hechos perpetrados por Reclutamiento (via restriccion)": (
            prefijos
            + "SELECT ?label WHERE { ?c rdfs:subClassOf ?r . "
            + "?r rdf:type owl:Restriction ; owl:onProperty cev:perpetradoPor ; "
            + "owl:someValuesFrom cev:Reclutamiento . ?c rdfs:label ?label }"
        ),
    }
    for nombre, consulta in consultas.items():
        filas = list(grafo.query(consulta))
        print(f"[{nombre}] ({len(filas)} resultados)")
        for fila in filas[:4]:
            print("   -", fila[0])


consultar_ejemplos(RUTA_ONTOLOGIA)

# %% [markdown]
# ## 3.8 Usabilidad de la ontología: ¿para qué se puede usar?
#
# **(a) Consultas formales del ciclo del conflicto (SPARQL).** La ontologia
# no es un diagrama: es una base de conocimiento consultable. Sobre el
# RDF/XML guardado se pueden responder preguntas del ciclo completo:
#
# - **Que actores perpetraron que hechos**: `SELECT ?hecho ?actor WHERE {
#   ?hecho cev:perpetradoPor ?actor }` (o navegando la restriccion
#   `perpetradoPor some` sobre Hechos victimizantes).
# - **Que poblaciones fueron afectadas**: `SELECT ?poblacion WHERE {
#   ?poblacion cev:esAfectadaPor ?hecho }`, con `Poblaciones` como rango
#   garantizado por el axioma raiz `Hechos_victimizantes subClassOf
#   (afectaA some Poblaciones)`.
# - **Que impactos generaron los hechos**: `SELECT ?hecho ?impacto WHERE {
#   ?hecho cev:generaImpacto ?impacto }` (inverso cev:esGeneradoPor).
# - **Que afrontamientos desarrollaron las poblaciones**: `SELECT ?poblacion
#   ?afrontamiento WHERE { ?poblacion cev:desarrollaAfrontamiento
#   ?afrontamiento }`, posible gracias al grupo incorporado en la seccion 2
#   y a su par inverso cev:esDesarrolladoPor.
#
# **(b) Clasificacion automatica con razonador (HermiT).** Las 6 clases
# definidas (seccion 3.6) y los 7 axiomas raiz permiten inferencia: HermiT
# reclasifica la jerarquia sin tocar el archivo (reparenting). Ejemplos
# reales verificados en la validacion 3.7:
#
# - `DesplazamientoForzado` (de Hechos victimizantes) se infiere subclase de
#   `HechoPerpetradoPorActor` porque el axioma de dominio lo conecta con
#   Actores_armados; lo mismo ocurre con Homicidio, DesaparicionForzada,
#   ViolenciaSexual, Extorsion, Reclutamiento, Tortura y los demas hechos
#   con restriccion perpetradoPor.
# - Los axiomas raiz hacen equivalente `Poblaciones` con
#   `PoblacionConAfrontamiento` (y con `PoblacionConImpacto`): el razonador
#   deduce que toda poblacion desarrolla afrontamientos y reparentea las
#   subclases de Poblaciones (Mujeres, PueblosIndigenas,
#   NinosNinasYAdolescentes, DefensoresDeDerechosHumanos, LideresSociales,
#   ComunidadesCampesinas...) bajo `PoblacionConAfrontamiento`. La raiz
#   `Afrontamientos_resistencias_no_violentas` queda equivalente a
#   `AfrontamientoAsociadoAHecho` (las cotidianas se reparentean a esta
#   clase definida).
#
# **(c) Interoperabilidad y reutilizacion.** El entregable es OWL 2 DL
# estandar (RDF/XML): se abre en Protege, Pellet, HermiT o cualquier
# endpoint SPARQL sin dependencias del taller. El IRI base estable
# (`http://www.semanticweb.org/cev/tesauro#`) permite referenciar y
# enlazar los conceptos desde otros vocabularios, y las anotaciones
# rdfs:label / cev:grupo / cev:eje hacen las consultas legibles.
#
# **(d) Extension para la catalogacion documental de la CEV.** El modelo se
# diseno para crecer: los 609 terminos son la semilla; se pueden anadir mas
# grupos del tesauro (Resistencias organizadas, Exilio, Despojo) con el
# mismo patron de 3.6 (clase raiz + axiomas del ciclo) y poblar los
# individuos con los expedientes reales (testimonios, sentencias, informes)
# usando las object properties como predicados de metadatos documentales.

# %% [markdown]
# ## 3.9 Grafico jerarquico
#
# Dos paneles: (a) las 6 raices con 2 niveles de profundidad (vision global
# del tesauro) y (b) el subarbol completo del grupo mas profundo ("Armas",
# 6 niveles). Layout jerarquico por niveles, colores por grupo. Esta figura
# es parte del analisis ya entregado y se conserva como clase POO propia de
# la seccion: `VisualizadorJerarquia` recibe el servicio (o su ontologia) y
# la ruta de salida.

# %%
class VisualizadorJerarquia:
    """Genera el PNG del arbol jerarquico de la ontologia (dos paneles).

    Panel izquierdo: las 6 raices con `niveles_raices` niveles de profundidad
    (vision global del tesauro). Panel derecho: subarbol completo del grupo
    `grupo_arbol` (el mas profundo de la seleccion: "Armas", 6 niveles).

    Attributes:
        servicio: ServicioOntologia con la ontologia cargada.
        ruta_salida: Ruta del PNG a generar.
        grupo_arbol: Grupo cuyo subarbol completo se dibuja.
        niveles_raices: Profundidad del panel de raices.
    """

    def __init__(
        self,
        servicio: sc.ServicioOntologia,
        ruta_salida: Path,
        grupo_arbol: str = GRUPO_ARBOL,
        niveles_raices: int = NIVELES_PANEL_RAICES,
    ) -> None:
        self.servicio = servicio
        self.ruta_salida = Path(ruta_salida)
        self.grupo_arbol = grupo_arbol
        self.niveles_raices = niveles_raices

    @property
    def ontologia(self) -> object:
        """Ontologia owlready2 cargada en el servicio."""
        return self.servicio.onto

    def layout_jerarquico(self, grafo, raices: list[str], horizontal: bool = False) -> dict:
        """Calcula posiciones de layout jerarquico (raices arriba o a la izquierda).

        Con horizontal=False las raices quedan arriba y los hijos se despliegan
        hacia abajo (adecuado para arboles profundos). Con horizontal=True las
        raices quedan a la izquierda y los niveles avanzan a la derecha
        (adecuado para arboles anchos con muchos nodos por nivel).

        Args:
            grafo: Digrafo dirigido de padres a hijos.
            raices: Nodos sin padre.
            horizontal: True para layout izquierda-a-derecha.

        Returns:
            Dict nodo -> (x, y).
        """
        niveles = {raiz: 0 for raiz in raices}
        orden = []
        pila = list(raices)
        visitados = set()
        while pila:
            nodo = pila.pop(0)
            if nodo in visitados:
                continue
            visitados.add(nodo)
            orden.append(nodo)
            for hijo in grafo.successors(nodo):
                niveles[hijo] = niveles[nodo] + 1
                pila.append(hijo)
        por_nivel = [[] for _ in range(max(niveles.values()) + 1)]
        for nodo in orden:
            por_nivel[niveles[nodo]].append(nodo)
        posiciones = {}
        for nivel, nodos_nivel in enumerate(por_nivel):
            ancho = max(1, len(nodos_nivel) - 1)
            for i, nodo in enumerate(nodos_nivel):
                avance, lateral = (i / ancho, -nivel) if not horizontal else (-nivel, i / ancho)
                posiciones[nodo] = (lateral, avance)
        return posiciones

    def subgrafo_desde_raices(self, raices: list[str], profundidad: int) -> tuple:
        """Construye el digrafo jerarquico de la ontologia hasta una profundidad.

        Args:
            raices: Nombres de las clases raiz a incluir.
            profundidad: Maxima profundidad de descendencia (1 = solo raices).

        Returns:
            Tupla (grafo, dict nodo -> etiqueta legible).
        """
        grafo = sc.nx.DiGraph()
        etiquetas = {}
        visitados = set()

        def expandir(clase, nivel: int) -> None:
            if clase.name in visitados or nivel > profundidad:
                return
            visitados.add(clase.name)
            etiquetas[clase.name] = clase.label[0] if clase.label else clase.name
            for hijo in clase.subclasses():
                grafo.add_edge(clase.name, hijo.name)
                expandir(hijo, nivel + 1)

        for nombre in raices:
            expandir(self.ontologia[nombre], 1)
        return grafo, etiquetas

    def _dibujar_panel(
        self,
        grafo,
        etiquetas: dict,
        eje,
        color_por_nodo: dict,
        fuente: int,
        horizontal: bool = False,
        max_label: int = 0,
    ) -> None:
        """Dibuja un panel jerarquico del grafo sobre un eje de matplotlib.

        Args:
            grafo: Digrafo padres -> hijos.
            etiquetas: Dict nodo -> etiqueta legible.
            eje: Eje matplotlib.
            color_por_nodo: Dict nodo -> color.
            fuente: Tamano de fuente para las etiquetas.
            horizontal: True para layout izquierda-a-derecha.
            max_label: Si > 0, trunca etiquetas a esa cantidad de caracteres.
        """
        raices = [n for n in grafo.nodes() if grafo.in_degree(n) == 0]
        posiciones = self.layout_jerarquico(grafo, raices, horizontal)
        etiquetas_finales = {
            nodo: (texto[:max_label] + "..." if max_label and len(texto) > max_label else texto)
            for nodo, texto in etiquetas.items()
        }
        sc.nx.draw_networkx_edges(grafo, posiciones, ax=eje, arrows=False, alpha=0.35, width=0.7)
        for nodo in grafo.nodes():
            color = color_por_nodo.get(nodo, "#888888")
            sc.nx.draw_networkx_nodes(
                grafo, posiciones, nodelist=[nodo], ax=eje, node_size=90,
                node_color=color, edgecolors="white", linewidths=0.4,
            )
        sc.nx.draw_networkx_labels(
            grafo, posiciones, labels=etiquetas_finales, ax=eje, font_size=fuente,
            font_family="DejaVu Sans",
        )
        eje.axis("off")
        eje.margins(0.06)

    def _colores_por_nodo(self) -> dict:
        """Asigna a cada clase de la ontologia el color de su grupo raiz."""
        color_por_nodo = {}
        for nodo in self.ontologia.classes():
            if nodo.name == "ConceptoDelTesauro":
                continue
            grupo = nodo.grupo[0] if nodo.grupo else None
            color_por_nodo[nodo.name] = COLORES_GRUPO.get(grupo, "#888888")
        return color_por_nodo

    def generar(self) -> Path:
        """Genera el PNG de dos paneles, lo guarda y devuelve la ruta."""
        raices = [sc.SLUG_RAIZ[g] for g in sc.SLUG_RAIZ]
        color_por_nodo = self._colores_por_nodo()

        grafo_raices, etiquetas_raices = self.subgrafo_desde_raices(
            raices, self.niveles_raices
        )
        grafo_arbol, etiquetas_arbol = self.subgrafo_desde_raices(
            [sc.SLUG_RAIZ[self.grupo_arbol]], 99
        )

        figura, (ax_izq, ax_der) = sc.plt.subplots(
            1, 2, figsize=(30, 18), gridspec_kw={"width_ratios": [1, 1]}
        )
        ax_izq.set_title(
            "Tesauro CEV: 6 grupos raiz y 2 niveles de profundidad\n"
            "(raices a la izquierda, niveles a la derecha)",
            fontsize=13,
        )
        self._dibujar_panel(
            grafo_raices, etiquetas_raices, ax_izq, color_por_nodo,
            fuente=5, horizontal=True, max_label=30,
        )
        ax_der.set_title(f"Subarbol completo del grupo: {self.grupo_arbol}", fontsize=13)
        self._dibujar_panel(
            grafo_arbol, etiquetas_arbol, ax_der, color_por_nodo, fuente=5
        )

        leyendas = [
            sc.plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=9, label=g)
            for g, c in COLORES_GRUPO.items()
        ]
        figura.legend(handles=leyendas, loc="lower center", ncol=6, fontsize=10, frameon=False)
        self.ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(str(self.ruta_salida), dpi=150, bbox_inches="tight")
        sc.plt.close(figura)
        print(
            f"Grafico guardado en {self.ruta_salida} "
            f"(nodos raices: {grafo_raices.number_of_nodes()}, "
            f"nodos arbol {self.grupo_arbol}: {grafo_arbol.number_of_nodes()})"
        )
        return self.ruta_salida


visualizador = VisualizadorJerarquia(servicio, FIGURAS["arbol_ontologia"])
visualizador.generar()

# %% [markdown]
# ## 3.10 Decisiones de diseno (registro)
#
# - **IRI base**: `http://www.semanticweb.org/cev/tesauro#` (namespace `cev`).
# - **Clases**: 623 (609 terminos + 7 nodos intermedios sinteticos por
#   prefijos ausentes en el JSON + 6 clases definidas del modelo semantico +
#   `ConceptoDelTesauro`, superclase tecnica que da dominio/rango a las
#   object properties sin mezclar con vocabularios externos). Las 6 raices
#   conservan el nombre de su grupo; el resto usa CamelCase ASCII generado
#   desde el descriptor (los acentos viven en `rdfs:label`, nunca en el
#   QName).
# - **Desambiguacion**: los paths se deduplican por (grupo, tupla de
#   segmentos); si un slug colisionara entre grupos se anadiria el sufijo del
#   grupo (mecanismo defensivo; en esta seleccion no hay colisiones).
# - **Anotaciones**: `rdfs:label` (descriptor legible), `rdfs:comment` (path
#   original completo, trazabilidad), `cev:grupo` y `cev:eje` (consultables
#   por SPARQL, por ejemplo para filtrar conceptos de un grupo).
# - **Individuos**: uno por termino hoja (534), con IRI
#   `instancia_<slug>` y las mismas anotaciones. Materializan las entidades
#   consultables del tesauro.
# - **cev:esParteDe / cev:parteDe**: axiomas de clase `hijo subClassOf
#   (esParteDe some padre)` para los 610 edges; captura la meronimia del
#   tesauro y es razonable con HermiT.
# - **cev:relacionadoCon**: 50 relaciones (45 heuristicas por IDF + 5 raiz).
#   Criterio completo en `relaciones_afines.txt`. Se acoto a 50 para
#   mantener calidad y justificabilidad. En OWL DL las relaciones entre
#   clases se declaran como restricciones some (subClassOf + someValuesFrom)
#   y se consultan navegando la restriccion o con un reasoner; la simetria
#   queda declarada en la propiedad. No se usan triples directos
#   clase-clase para no degradar la ontologia a OWL Full (punning).
# - **cev:perpetradoPor / cev:esVictimaDe**: 12 pares hecho-actor con
#   co-ocurrencia terminologica especifica (reclutamiento, acciones belicas,
#   inteligencia, secuestro politico); ver trazabilidad en
#   `relaciones_afines.txt`. Se declaran como restricciones some igual que
#   relacionadoCon.
# - **Modelo semantico del ciclo**: 6 pares inversos adicionales
#   (afectaA/esAfectadaPor, generaImpacto/esGeneradoPor,
#   experimentaImpacto/impactaA, desarrollaAfrontamiento/esDesarrolladoPor,
#   respondeA/esRespondidoMediante, empleaArma/esEmpleadaEn), 7 axiomas raiz
#   `GrupoOrigen subClassOf (propiedad some GrupoDestino)` y 6 clases
#   definidas con equivalentClass (tabla 3.6). Total: 17 object properties
#   y 2 annotation properties.
# - **Literales xsd:string**: owlready2 serializa los literales de anotacion
#   con datatype explicito; en SPARQL se compara con `STR(?o) = "..."` (ver
#   celda 3.7.2 de ejemplo).
# - **Advertencias**: HermiT (incluido en owlready2) verifico la
#   consistencia de la ontologia sin errores y clasifica (reparenting de las
#   clases definidas, ver 3.8). El pip install hermit externo falla en
#   Python 3.14 (javabridge/Pillow sin wheel) pero no es necesario. El token
#   "muse" de un par heuristico proviene de la terminologia de municiones
#   del grupo Armas; quedo fuera del top-45 por puntuacion.
# - **Arquitectura (refactor)**: toda la construccion y validacion vive en
#   `ServicioOntologia` (sec00_clases, Singleton por ruta + Facade); esta
#   seccion solo invoca la API y conserva la figura del arbol jerarquico en
#   `VisualizadorJerarquia`.
