"""test_sparql.py — Tests de SparqlServicio sobre la OWL construida.

Verifica que la ontología guardada en el directorio temporal de sesión
es consultable por SPARQL (rdflib): conteo de clases y de individuos.
"""


def test_sparql_conteo_de_clases(ontologia, modulo):
    """El conteo de clases owl:Class es > 0 (623 reales)."""
    sparql = modulo.SparqlServicio(ontologia["ruta_owl"])
    df, total, ms = sparql.ejecutar(
        "SELECT (COUNT(?c) AS ?n) WHERE { ?c rdf:type owl:Class }",
        columnas=["n"],
        limite=5,
    )
    assert total == 1
    assert not df.empty
    assert int(df.iloc[0, 0]) > 0
    assert ms >= 0


def test_sparql_conteo_de_individuos(ontologia, modulo):
    """El conteo de individuos (owl:NamedIndividual) es > 0 (534 reales)."""
    sparql = modulo.SparqlServicio(ontologia["ruta_owl"])
    df, total, _ = sparql.ejecutar(
        "SELECT (COUNT(?i) AS ?n) WHERE { ?i rdf:type owl:NamedIndividual }",
        columnas=["n"],
        limite=5,
    )
    assert total == 1
    assert int(df.iloc[0, 0]) > 0


def test_sparql_lista_clases_con_limite(ontologia, modulo):
    """Una consulta de listado respeta el límite y reporta el total real."""
    sparql = modulo.SparqlServicio(ontologia["ruta_owl"])
    df, total, _ = sparql.ejecutar(
        "SELECT ?c WHERE { ?c rdf:type owl:Class }",
        columnas=["clase"],
        limite=3,
    )
    assert total >= 623
    assert len(df) == 3


def test_sparql_consulta_sin_resultados(ontologia, modulo):
    """Una consulta que no encuentra nada devuelve DataFrame vacío."""
    sparql = modulo.SparqlServicio(ontologia["ruta_owl"])
    df, total, _ = sparql.ejecutar(
        "SELECT ?x WHERE { ?x rdf:type <http://www.semanticweb.org/cev/tesauro#ClaseInexistente> }",
        columnas=["x"],
        limite=5,
    )
    assert total == 0
    assert df.empty
