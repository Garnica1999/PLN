"""test_ontologia.py — Tests de ServicioOntologia y GrafoOntologia.

Cubre la construcción completa de la OWL desde el JSON de grupos (6
grupos, extensión aprobada), el guardado, la recarga, la validación con
rdflib, las consultas conceptos()/grafo() y el Singleton. La ontología se
construye una sola vez por sesión (fixture `ontologia` en conftest.py):
~623 clases, ~534 individuos y ~9.810 triples RDF, con 17 object
properties, 7 axiomas de dominio entre grupos y 6 clases definidas.

Tests adicionales del modelo semántico fuerte (Fase 6):
    - Las 6 clases definidas existen y su equivalent_to es una
      restricción SOME de la propiedad esperada.
    - Los 7 axiomas raíz de RELACIONES_ENTRE_GRUPOS están declarados en
      la clase raíz del grupo origen.
    - HermiT clasifica: un hecho (DesplazamientoForzado) hereda de
      HechoPerpetradoPorActor tras sync_reasoner.
    - El conteo de object properties es exactamente 17.
"""

import contextlib
import io
from collections import Counter

import pytest

CLASES_ESPERADAS = 623
INDIVIDUOS_ESPERADOS = 534
AXIOMAS_PARTE_ESPERADOS = 610
TRIPLES_RDFLIB_MIN = 9700
CONCEPTOS_ESPERADOS = 610


# ----------------------------------------------------------------------
# Construcción y resumen
# ----------------------------------------------------------------------


def test_construccion_resumen(ontologia):
    """El resumen de construcción tiene los conteos estructurales esperados."""
    resumen = ontologia["resumen"]
    assert resumen["clases"] == CLASES_ESPERADAS
    assert resumen["individuos"] == INDIVIDUOS_ESPERADOS
    assert resumen["nodos"] == 616
    assert resumen["hojas"] == INDIVIDUOS_ESPERADOS
    assert resumen["axiomas_parte"] >= AXIOMAS_PARTE_ESPERADOS
    assert resumen["relaciones_afines"] == 50
    assert resumen["perpetradores"] >= 12
    assert resumen["relaciones_dominio"] == 7
    assert resumen["clases_definidas"] == 6
    assert resumen["object_properties"] == 17
    assert resumen["annotation_properties"] == 2


def test_guardar_crea_archivo_owl(ontologia):
    """guardar() escribe el archivo RDF/XML en disco."""
    ruta_owl = ontologia["ruta_owl"]
    assert ruta_owl.exists()
    assert ruta_owl.stat().st_size > 0


def test_trazabilidad_escrita(ontologia):
    """La construcción documenta el criterio de afinidad en el .txt."""
    ruta_rel = ontologia["ruta_rel"]
    assert ruta_rel.exists()
    contenido = ruta_rel.read_text(encoding="utf-8")
    assert "CRITERIO" in contenido
    assert "relacionadoCon" in contenido


# ----------------------------------------------------------------------
# Carga desde el .owl y métricas
# ----------------------------------------------------------------------


def test_nueva_instancia_carga_owl(ontologia, modulo):
    """Una instancia nueva sobre el .owl guardado carga y reporta 623/534."""
    servicio = modulo.ServicioOntologia(
        ontologia["ruta_owl"], ontologia["ruta_json"], ontologia["ruta_rel"]
    )
    assert servicio.cargar() is False  # False = cargado desde archivo, no construido
    metricas = servicio.metricas()
    assert metricas["clases"] == CLASES_ESPERADAS
    assert metricas["individuos"] == INDIVIDUOS_ESPERADOS
    assert metricas["object_properties"] == 17
    assert metricas["annotation_properties"] == 2


def test_metricas_subclases_por_raiz(ontologia):
    """Cada grupo raíz (6) tiene su bloque de subclases en las métricas."""
    metricas = ontologia["servicio"].metricas()
    raices = metricas["subclases_por_raiz"]
    assert set(raices) == {
        "ConceptoDelTesauro",
        "Actores_armados",
        "Impactos",
        "Armas",
        "Poblaciones",
        "Hechos_victimizantes",
        "Afrontamientos_resistencias_no_violentas",
    }
    # 622 = 623 clases menos ConceptoDelTesauro, que se excluye del conteo.
    assert sum(raices.values()) == CLASES_ESPERADAS - 1
    # El grupo nuevo aporta al menos sus 23 términos más su clase definida.
    assert raices["Afrontamientos_resistencias_no_violentas"] >= 23


def test_metricas_sin_cargar_lanza(ontologia, modulo, tmp_path):
    """metricas() sin ontología cargada lanza RuntimeError."""
    servicio = modulo.ServicioOntologia(tmp_path / "vacia.owl")
    with pytest.raises(RuntimeError):
        servicio.metricas()


def test_guardar_sin_ontologia_lanza(ontologia, modulo, tmp_path):
    """guardar() sin ontología construida o cargada lanza RuntimeError."""
    servicio = modulo.ServicioOntologia(tmp_path / "vacia.owl")
    with pytest.raises(RuntimeError):
        servicio.guardar()


# ----------------------------------------------------------------------
# Modelo semántico de dominio (Fase 6)
# ----------------------------------------------------------------------


def test_clases_definidas_en_owl(ontologia, modulo):
    """Las 6 clases definidas existen y equivalen a una restricción SOME."""
    onto = ontologia["servicio"].onto
    for nombre, base, propiedad, destino in modulo.CLASES_DEFINIDAS:
        clase = onto[nombre]
        assert clase is not None, f"Clase definida ausente: {nombre}"
        assert clase.equivalent_to, f"{nombre} sin equivalent_to"
        equivalencia = clase.equivalent_to[0]
        restricciones = [
            operando
            for operando in equivalencia.Classes
            if isinstance(operando, modulo.Restriction)
        ]
        assert len(restricciones) == 1, f"{nombre} sin restricción en equivalent_to"
        assert restricciones[0].property.name == propiedad
        assert restricciones[0].value.name == destino
        assert onto[base] in clase.ancestors(include_self=False)


def test_axiomas_raiz_de_dominio(ontologia, modulo):
    """Los 7 axiomas de RELACIONES_ENTRE_GRUPOS están en la raíz del grupo origen."""
    onto = ontologia["servicio"].onto
    for grupo_origen, propiedad, grupo_destino in modulo.RELACIONES_ENTRE_GRUPOS:
        raiz = onto[modulo.SLUG_RAIZ[grupo_origen]]
        restricciones = [
            axioma
            for axioma in raiz.is_a
            if isinstance(axioma, modulo.Restriction)
            and axioma.property.name == propiedad
        ]
        assert len(restricciones) == 1, (
            f"Falta axioma {grupo_origen} -> {propiedad} -> {grupo_destino}"
        )
        assert restricciones[0].value.name == modulo.SLUG_RAIZ[grupo_destino]


def test_hermit_clasifica_hechos_perpetrados(ontologia, modulo):
    """HermiT clasifica: DesplazamientoForzado hereda de HechoPerpetradoPorActor.

    Se razona sobre una instancia fresca cargada del .owl para no mutar la
    ontología compartida de la sesión (sync_reasoner re-parenta clases).
    """
    servicio = modulo.ServicioOntologia(
        ontologia["ruta_owl"], ontologia["ruta_json"], ontologia["ruta_rel"]
    )
    assert servicio.cargar() is False
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        assert servicio.validar_hermit() is True
    clase = servicio.onto["DesplazamientoForzado"]
    ancestros = {a.name for a in clase.ancestors(include_self=False)}
    assert "HechoPerpetradoPorActor" in ancestros
    assert "Hechos_victimizantes" in ancestros


def test_object_properties_diecisiete(ontologia):
    """Hay 17 object properties: 5 estructurales + 12 del ciclo del conflicto."""
    onto = ontologia["servicio"].onto
    nombres = {p.name for p in onto.object_properties()}
    assert len(nombres) == 17
    estructurales = {
        "esParteDe",
        "parteDe",
        "relacionadoCon",
        "perpetradoPor",
        "esVictimaDe",
    }
    ciclo = {
        "afectaA",
        "esAfectadaPor",
        "generaImpacto",
        "esGeneradoPor",
        "experimentaImpacto",
        "impactaA",
        "desarrollaAfrontamiento",
        "esDesarrolladoPor",
        "respondeA",
        "esRespondidoMediante",
        "empleaArma",
        "esEmpleadaEn",
    }
    assert estructurales | ciclo == nombres


# ----------------------------------------------------------------------
# Validación y consultas
# ----------------------------------------------------------------------


def test_validar_rdflib(ontologia):
    """El .owl parseado con rdflib tiene más de 9.700 triples."""
    validacion = ontologia["servicio"].validar_rdflib()
    assert validacion["triples"] >= TRIPLES_RDFLIB_MIN
    assert validacion["subclassof_nombradas"] >= CLASES_ESPERADAS
    assert validacion["subclassof_restricciones"] > 0


def test_conceptos(ontologia):
    """conceptos() lista ~610 conceptos con grupo, path y concepto.

    610 = 623 clases - superclase - 6 clases definidas - 6 raíces de grupo
    (estas últimas tienen comment sin separador " - ").
    """
    conceptos = ontologia["servicio"].conceptos()
    assert len(conceptos) == CONCEPTOS_ESPERADOS
    assert list(conceptos.columns) == ["grupo", "termino", "concepto"]
    assert conceptos["grupo"].nunique() == 6
    assert conceptos["concepto"].notna().all()


# ----------------------------------------------------------------------
# Grafo de la ontología
# ----------------------------------------------------------------------


def test_grafo_conteos_globales(ontologia):
    """El grafo completo tiene >=1.120 nodos y >=1.100 aristas."""
    grafo = ontologia["servicio"].grafo()
    assert grafo.number_of_nodes() >= 1120
    assert grafo.number_of_edges() >= 1100


def test_grafo_conteos_por_tipo(ontologia):
    """Las aristas se clasifican: subclassof, esParteDe, relacionadoCon, perpetradoPor e instancia."""
    grafo = ontologia["servicio"].grafo()
    conteos: Counter = Counter()
    for _, _, atributos in grafo.edges(data=True):
        conteos.update(atributos.get("tipo", set()))
    assert conteos["subclassof"] >= CLASES_ESPERADAS
    assert conteos["esParteDe"] >= AXIOMAS_PARTE_ESPERADOS
    assert conteos["relacionadoCon"] == 50
    assert conteos["perpetradoPor"] >= 12
    assert conteos["instancia"] >= INDIVIDUOS_ESPERADOS


def test_grafo_nodos_tipo_y_grupo(ontologia):
    """Los nodos llevan atributos tipo (clase/individuo) y grupo."""
    grafo = ontologia["servicio"].grafo()
    tipos = {atributos.get("tipo") for _, atributos in grafo.nodes(data=True)}
    assert {"clase", "individuo"} <= tipos
    nodos_con_grupo = sum(
        1 for _, atributos in grafo.nodes(data=True) if atributos.get("grupo")
    )
    assert nodos_con_grupo > 0


# ----------------------------------------------------------------------
# GrafoOntologia (vista contraída)
# ----------------------------------------------------------------------


@pytest.fixture()
def colores_grupo():
    """Paleta inyectable de colores por grupo (constante del taller)."""
    return {
        "Actores armados y otros participantes en las dinámicas de la guerra": "#000000",
        "Impactos": "#111111",
        "Armas": "#222222",
        "Poblaciones": "#333333",
        "Hechos victimizantes": "#444444",
        "Afrontamientos y resistencias no violentas - cotidianas": "#555555",
    }


def test_grafo_por_grupos(ontologia, modulo, colores_grupo):
    """La vista contraída tiene 7 nodos (6 grupos + superclase) y >=10 aristas."""
    grafo_grupos = modulo.GrafoOntologia(
        ontologia["servicio"], colores_grupo
    ).grafo_por_grupos()
    assert grafo_grupos.number_of_nodes() == 7
    assert grafo_grupos.number_of_edges() >= 10
    assert "ConceptoDelTesauro" in grafo_grupos.nodes()
    grupos = [n for n in grafo_grupos.nodes() if n != "ConceptoDelTesauro"]
    assert len(grupos) == 6


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------


def test_singleton_misma_ruta_misma_instancia(ontologia, modulo):
    """obtener() con la misma ruta devuelve la misma instancia (is)."""
    primera = modulo.ServicioOntologia.obtener(
        ontologia["ruta_owl"], ontologia["ruta_json"], ontologia["ruta_rel"]
    )
    segunda = modulo.ServicioOntologia.obtener(ontologia["ruta_owl"])
    assert primera is segunda
    assert primera is ontologia["servicio"]


def test_singleton_rutas_distintas_instancias_distintas(modulo, tmp_path):
    """obtener() con rutas distintas crea instancias distintas."""
    primera = modulo.ServicioOntologia.obtener(tmp_path / "a.owl")
    segunda = modulo.ServicioOntologia.obtener(tmp_path / "b.owl")
    assert primera is not segunda
