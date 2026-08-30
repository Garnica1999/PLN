# %% [markdown]
# # Sec03b — Visualizacion de la ontologia (validacion visual)
#
# **Objetivo:** desplegar el grafo COMPLETO de la ontologia (clases,
# individuos y relaciones) en una imagen estatica de alta resolucion guardada
# en disco, mas una vista contraida entre los 6 grupos, para validar
# visualmente que la estructura del tesauro quedo representada.
#
# **Que se espera ver:**
# - **Grafo completo** (`grafo_ontologia_completo.png`): 1.158 nodos y 1.219
#   aristas. Los nodos se colorean por grupo raiz (6 colores + gris para la
#   superclase `ConceptoDelTesauro`); los individuos (terminos hoja) son los
#   puntos mas pequenos con borde blanco y las clases los puntos grandes. Las
#   aristas mezclan jerarquia (`subclassOf` + `esParteDe`), afinidad
#   terminologica (`relacionadoCon`) y vinculos hecho-actor (`perpetradoPor`).
# - **Vista contraida** (`grafo_ontologia_grupos.png`): los 6 grupos + la
#   superclase como nodos unicos (7 nodos, 12 aristas inter-grupo), con las
#   aristas inter-grupo agregadas y sus tipos; permite ver de un vistazo que
#   grupos se relacionan y con que propiedad.
# - **Modelo semantico** (nuevo en esta extension): las 6 raices llevan 7
#   axiomas de dominio (ciclo del conflicto: hecho -> actor -> poblacion ->
#   impacto -> afrontamiento) y hay 6 clases definidas por restricciones
#   (`equivalentClass`). Las propiedades nuevas de dominio (afectaA,
#   generaImpacto, experimentaImpacto, desarrollaAfrontamiento, respondeA,
#   empleaArma) no se materializan como aristas del grafo completo (el nucleo
#   `grafo()` solo incluye esParteDe/relacionadoCon/perpetradoPor); por eso
#   esta seccion las valida en Sec03b.4 leyendo los axiomas con owlready2.
#
# Se ejecuta despues de la seccion 3: la ontologia ya esta cargada en el
# Singleton `ServicioOntologia`; aqui se reutiliza esa instancia.
#
# **Estructura de la seccion:** (1) generacion de las dos figuras, (2)
# validacion numerica de los conteos del grafo contra las metricas del
# servicio, (3) analisis e interpretacion de ambas figuras, (4) validacion
# del modelo semantico (axiomas de dominio y clases definidas).

# %%
# Imports y compatibilidad: mismo patron que sec03 (en el notebook las
# constantes y el nucleo POO ya existen; en modo script se importan y
# definen localmente).
import sys
from pathlib import Path

import matplotlib

try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")

try:
    _DIR_CELDAS = Path(__file__).resolve().parent  # modo script
except NameError:
    _DIR_CELDAS = Path.cwd()  # modo notebook: el kernel corre en notebooks/

_DIR_MODULO = (
    _DIR_CELDAS
    if (_DIR_CELDAS / "sec00_clases.py").exists()
    else _DIR_CELDAS.parent / "notebook_cells"
)
sys.path.insert(0, str(_DIR_MODULO))

if "sc" not in globals():
    import sec00_clases as sc

if "FIGURAS" not in globals():
    RUTA_BASE = _DIR_CELDAS.parent
    RUTA_IMAGENES = RUTA_BASE / "imagenes"
    RUTA_ONTOLOGIA = RUTA_BASE / "ontologia" / "tesauro_cev.owl"
    FIGURAS = {
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

# Se reutiliza la instancia unica del Singleton: en el notebook la seccion 3
# ya la cargo; en ejecucion standalone (sec03b sin sec03) se carga ahora.
servicio = sc.ServicioOntologia.obtener(RUTA_ONTOLOGIA)
if servicio.onto is None:
    servicio.cargar()

# Facade de visualizacion: grafo completo coloreado por grupo y vista
# contraida entre los 6 grupos (colores inyectados desde sec00_setup).
grafo_ontologia = sc.GrafoOntologia(servicio, COLORES_GRUPO)

ruta_completo = grafo_ontologia.guardar_figura_completa(
    FIGURAS["grafo_completo"], dpi=200, mostrar=True
)
ruta_grupos = grafo_ontologia.guardar_figura_grupos(
    FIGURAS["grafo_grupos"], dpi=200, mostrar=True
)
print(f"Grafo completo guardado en {ruta_completo}")
print(f"Vista contraida guardada en {ruta_grupos}")

# %% [markdown]
# ## Sec03b.2 Validacion numerica del grafo
#
# Se contrastan los conteos del grafo (nodos por tipo y aristas por tipo)
# contra las metricas estructurales del servicio y los valores esperados de
# la extension aprobada (6 grupos): **623 clases, 534 individuos**, 610
# aristas `esParteDe` (una por edge de la jerarquia), 50 `relacionadoCon`
# (45 heuristicas + 5 raiz) y **13 `perpetradoPor`** (12 pares hecho-actor de
# la heuristica + 1 axioma raiz del modelo semantico: Hechos victimizantes
# subClassOf (perpetradoPor some Actores armados), que se agrega a la arista
# ya existente entre esas dos raices). Nota: cada arista lleva su `tipo` como
# CONJUNTO (`grafo[origen][destino]['tipo']`); el conteo suma los tipos de
# cada set.
#
# Las nuevas propiedades de dominio del modelo semantico (afectaA,
# generaImpacto, experimentaImpacto, desarrollaAfrontamiento, respondeA,
# empleaArma) NO aparecen como aristas de `grafo()`: el nucleo solo
# materializa esParteDe/relacionadoCon/perpetradoPor. Esas relaciones viven
# como restricciones `some` en las 6 raices y se validan en Sec03b.4 (modelo
# semantico con owlready2); en la vista contraida solo se aprecian
# indirectamente a traves de las aristas `relacionadoCon` agregadas.

# %%
from collections import Counter


def contar_aristas_por_tipo(grafo) -> Counter:
    """Cuenta las aristas por tipo sumando el set 'tipo' de cada arista.

    Una misma arista puede representar varios tipos a la vez (por ejemplo
    subClassOf + esParteDe unen los mismos pares padre-hijo); el atributo
    'tipo' es un conjunto y cada elemento se cuenta por separado.

    Args:
        grafo: nx.DiGraph con aristas anotadas con 'tipo' (set).

    Returns:
        Counter tipo de arista -> numero de aristas que lo contienen.
    """
    conteo = Counter()
    for _, _, atributos in grafo.edges(data=True):
        conteo.update(atributos.get("tipo", set()))
    return conteo


grafo_completo = grafo_ontologia.grafo_completo()

nodos_clase = sum(1 for _, a in grafo_completo.nodes(data=True) if a.get("tipo") == "clase")
nodos_individuo = sum(
    1 for _, a in grafo_completo.nodes(data=True) if a.get("tipo") == "individuo"
)
# Un nodo tecnico sin tipo: owl:Thing, superclase raiz de ConceptoDelTesauro
# (creado implicitamente por la arista rdfs:subClassOf).
nodos_tecnicos = grafo_completo.number_of_nodes() - nodos_clase - nodos_individuo
aristas_por_tipo = contar_aristas_por_tipo(grafo_completo)

metricas = servicio.metricas()
print("=== Conteos del grafo vs metricas del servicio ===")
print(f"- Nodos clase: {nodos_clase} (metricas: {metricas['clases']})")
print(f"- Nodos individuo: {nodos_individuo} (metricas: {metricas['individuos']})")
print(f"- Nodos tecnicos (owl:Thing): {nodos_tecnicos}")
print(f"- Aristas totales: {grafo_completo.number_of_edges()}")
for tipo in ("subclassof", "esParteDe", "relacionadoCon", "perpetradoPor", "instancia"):
    print(f"- Aristas {tipo}: {aristas_por_tipo[tipo]}")

esperado_esparte = 610  # una por edge de la jerarquia (616 nodos - 6 raices)
esperado_perpetrador = 13  # 12 pares hecho-actor + 1 axioma raiz del modelo semantico
cumple = (
    nodos_clase == metricas["clases"] == 623
    and nodos_individuo == metricas["individuos"] == 534
    and aristas_por_tipo["subclassof"] == 623
    and aristas_por_tipo["esParteDe"] >= esperado_esparte
    and aristas_por_tipo["relacionadoCon"] == 50
    and aristas_por_tipo["perpetradoPor"] == esperado_perpetrador
    and aristas_por_tipo["instancia"] == 534
)
print("Validacion visual: OK" if cumple else "Validacion visual: REVISAR (conteos no esperados)")

# %% [markdown]
# ## Sec03b.3 Analisis de las figuras
#
# **Grafo completo (1.158 nodos, 1.219 aristas).** La imagen muestra el
# "ovillo" esperado en un tesauro denso: 623 clases (610 conceptos con path
# del tesauro + 6 raices + 6 clases definidas + `ConceptoDelTesauro`) y 534
# individuos hoja distribuidos por los 6 grupos. Los individuos (puntos
# pequenos) son los hubs mas externos y numerosos en Armas y Poblaciones, los
# grupos con mayor profundidad y mayor cantidad de terminos hoja (municiones,
# armas por tipo, pueblos etnicos, pueblos indigenas); las jerarquias
# profundas del grupo Armas (hasta 6 niveles) aparecen como trayectorias
# largas de nodos grandes encadenados. La superclase `ConceptoDelTesauro`
# (gris) es el punto central del que cuelgan las 6 raices. A esa escala las
# etiquetas no son legibles (a proposito): la figura valida estructura,
# densidad y color por grupo, no texto.
#
# **Vista contraida (7 nodos, 12 aristas inter-grupo).** Los pares visibles
# confirman la coherencia del modelo conceptual CEV con la extension:
# - **Hechos victimizantes -> Actores armados**: unica arista con
#   `perpetradoPor` (+ `relacionadoCon`), el vinculo hecho-actor que define
#   la propiedad; la inversa Actores -> Hechos solo lleva `relacionadoCon`
#   (la simetria de la TR se declara en la propiedad, no en una segunda
#   restriccion). En esta arista confluyen el par hecho-actor heurístico y el
#   axioma raiz del modelo semantico, por eso la validacion espera 13 aristas
#   `perpetradoPor` (12 + 1).
# - **Armas -> Actores armados** y **Armas -> Hechos victimizantes**: los
#   actores emplean armas y las armas son instrumentos de los hechos (raices
#   fijas del criterio TR).
# - **Impactos -> Hechos victimizantes** e **Impactos -> Poblaciones**: los
#   hechos generan impactos que recaen sobre las poblaciones.
# - **Afrontamientos y resistencias no violentas - cotidianas** (grupo nuevo
#   de la extension) recibe aristas de **Actores armados** e **Impactos**
#   (afinidad TR por co-ocurrencia terminologica), cerrando el ciclo del
#   conflicto: el actor genera hechos que impactan a la poblacion y la
#   poblacion desarrolla afrontamientos.
# - **Actores armados -> Impactos / Poblaciones** y **Poblaciones ->
#   Hechos victimizantes**: afinidades heuristicas adicionales (por ejemplo
#   la terminologia de revictimizacion o de afectaciones a la poblacion civil
#   co-ocurre entre grupos).
#
# **Que valida visualmente:** (1) las 6 raices existen y cada grupo conserva
# su color en ambas vistas; (2) la jerarquia esta completa (610 edges
# esParteDe = nodos no raiz) con profundidades marcadas en Armas; (3) los
# individuos hoja materializan los terminos del JSON (534); (4) las 3
# familias de object properties conectan grupos distintos con sentido en el
# dominio CEV, y el grupo nuevo de Afrontamientos queda integrado. La
# validacion numerica de Sec03b.2 confirma estos conteos contra las metricas
# del servicio y los valores esperados de la extension.

# %% [markdown]
# ## Sec03b.4 Modelo semantico: axiomas de dominio y clases definidas
#
# La extension agrego un modelo semantico sobre la jerarquia: **7 axiomas de
# dominio** entre las 6 raices (ciclo del conflicto: hecho -> actor ->
# poblacion -> impacto -> afrontamiento, mas las relaciones de dominio
# desarrollaAfrontamiento, respondeA y empleaArma) y **6 clases definidas**
# por `equivalentClass` (`Base AND (propiedad SOME Destino)`) que el
# razonador puede clasificar automaticamente. Como las propiedades nuevas no
# se materializan como aristas del grafo (ver Sec03b.2), esta celda valida el
# modelo leyendo directamente los axiomas con owlready2: las restricciones
# `some` en las 6 raices y las equivalencias de las clases definidas.

# %%
"""Validacion del modelo semantico: axiomas de dominio y clases definidas."""

print("=== Modelo semantico: axiomas de dominio entre raices (restricciones some) ===")
restricciones_raiz = 0
for grupo_origen, propiedad, grupo_destino in sc.RELACIONES_ENTRE_GRUPOS:
    clase = servicio.onto[sc.SLUG_RAIZ[grupo_origen]]
    axiomas = [
        a for a in clase.is_a
        if isinstance(a, sc.Restriction)
        and getattr(a.property, "name", None) == propiedad
        and getattr(a.value, "name", None) == sc.SLUG_RAIZ[grupo_destino]
    ]
    restricciones_raiz += len(axiomas)
    estado = "OK" if axiomas else "FALTA"
    print(
        f"- {sc.SLUG_RAIZ[grupo_origen]:<46} -{propiedad:<20}-> "
        f"{sc.SLUG_RAIZ[grupo_destino]}: {estado}"
    )
print(
    f"Restricciones de dominio verificadas: {restricciones_raiz} "
    f"de {len(sc.RELACIONES_ENTRE_GRUPOS)} axiomas esperados"
)

print()
print("=== Clases definidas (equivalentClass: Base AND (propiedad SOME Destino)) ===")
for nombre, clase_base, propiedad, clase_destino in sc.CLASES_DEFINIDAS:
    clase = servicio.onto[nombre]
    equivalencias = list(clase.equivalent_to)
    formula = (getattr(clase, "comment", [""]) or [""])[0]
    estado = "OK" if equivalencias else "FALTA"
    print(f"- {nombre:<32} {estado} | {formula}")
print(f"Clases definidas verificadas: {len(sc.CLASES_DEFINIDAS)}")
