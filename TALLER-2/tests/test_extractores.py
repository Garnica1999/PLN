"""test_extractores.py — Tests de la fábrica de extractores y del motor WordNet.

Cubre FabricaExtractores (Abstract Factory + Registry), los 9 extractores
registrados y MotorRelacionesWordNet contra términos reales del tesauro
(omw-1.4 en español). Las aserciones sobre WordNet se verificaron contra
los synsets reales: "mujer" -> antonimia "hombre"; "verdad" -> sinonimia
"realidad"; "desplazamiento" -> derivación "desplazarse"/"trasladar";
"ordenar" -> troponimia no vacía.
"""

import pytest

# Tipos de los que dependen las aserciones de la fábrica y el motor.
TIPOS_RELACION = (
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


# ----------------------------------------------------------------------
# FabricaExtractores (Abstract Factory + Registry)
# ----------------------------------------------------------------------


def test_fabrica_tipos_registrados(modulo):
    """Los 9 tipos registrados coinciden con el orden canónico TIPOS_RELACION."""
    assert modulo.FabricaExtractores.tipos_registrados() == list(TIPOS_RELACION)
    assert len(modulo.FabricaExtractores.tipos_registrados()) == 9


def test_fabrica_crear_sinonimia(modulo):
    """crear("sinonimia") devuelve una instancia de SinonimiaExtractor."""
    extractor = modulo.FabricaExtractores.crear("sinonimia")
    assert isinstance(extractor, modulo.SinonimiaExtractor)
    assert isinstance(extractor, modulo.ExtractorRelacion)
    assert extractor.nombre == "sinonimia"


def test_fabrica_crear_todos_los_tipos(modulo):
    """crear_todos() instancia un extractor por cada tipo registrado."""
    extractores = modulo.FabricaExtractores.crear_todos()
    assert len(extractores) == 9
    assert [e.nombre for e in extractores] == list(TIPOS_RELACION)


def test_fabrica_crear_tipo_no_registrado(modulo):
    """Un tipo no registrado lanza KeyError."""
    with pytest.raises(KeyError):
        modulo.FabricaExtractores.crear("no_existe")


def test_fabrica_extractor_es_abstracto(modulo):
    """ExtractorRelacion es un ABC y no se puede instanciar."""
    with pytest.raises(TypeError):
        modulo.ExtractorRelacion()


# ----------------------------------------------------------------------
# MotorRelacionesWordNet sobre términos reales
# ----------------------------------------------------------------------


@pytest.fixture()
def motor(modulo):
    """Motor con cadena por defecto y los 9 extractores registrados."""
    return modulo.MotorRelacionesWordNet()


def test_motor_mujer_antonimia(motor):
    """ "mujer" tiene antónimo "hombre" (o "masculino") en español."""
    resultado = motor.extraer_relaciones("mujer")
    assert resultado.encontrado is True
    antonimos = resultado.relaciones["antonimia"]
    assert any("hombre" in a or "masculino" in a for a in antonimos)


def test_motor_verdad_sinonimia(motor):
    """La sinonimia de "verdad" incluye "realidad" o "evidencia"."""
    resultado = motor.extraer_relaciones("verdad")
    assert resultado.encontrado is True
    sinonimos = resultado.relaciones["sinonimia"]
    assert "realidad" in sinonimos or "evidencia" in sinonimos


def test_motor_desplazamiento_derivacion_e_hiperonimia(motor):
    """ "desplazamiento" deriva en "desplazarse"/"trasladar" y tiene hiperónimos."""
    resultado = motor.extraer_relaciones("desplazamiento")
    assert resultado.encontrado is True
    derivados = resultado.relaciones["derivacion"]
    assert any("desplazarse" in d or "trasladar" in d for d in derivados)
    assert resultado.relaciones["hiperonimia"]


def test_motor_verbo_troponimia(motor):
    """El verbo "ordenar" tiene tropónimos (hipónimos verbales) en español."""
    resultado = motor.extraer_relaciones("ordenar")
    assert resultado.encontrado is True
    assert resultado.relaciones["troponimia"]


def test_motor_path_jerarquico_del_tesauro(motor):
    """Un path real del tesauro se normaliza en cadena hasta hallar synsets."""
    resultado = motor.extraer_relaciones("Poblaciones - Mujeres")
    assert resultado.encontrado is True
    assert resultado.variante_usada == "mujer"
    assert resultado.relaciones["sinonimia"]


def test_motor_termino_inexistente(motor):
    """Un término inexistente da encontrado=False y relaciones vacías."""
    resultado = motor.extraer_relaciones("zzzqqq")
    assert resultado.encontrado is False
    assert resultado.variante_usada == ""
    assert resultado.relaciones == {}
    assert resultado.total() == 0


def test_motor_termino_grupo_afrontamientos_sin_synsets(motor):
    """Término del grupo nuevo (extensión) sin synsets: se procesa sin error.

    "Afrontamientos familiares" no tiene synsets en omw-1.4: el motor debe
    responder encontrado=False de forma consistente, sin lanzar excepciones.
    """
    resultado = motor.extraer_relaciones(
        "Afrontamientos y resistencias no violentas - cotidianas"
        " - Afrontamientos familiares"
    )
    assert resultado.encontrado is False
    assert resultado.variante_usada == ""
    assert resultado.relaciones == {}
    assert resultado.total() == 0


def test_motor_termino_grupo_afrontamientos_con_cobertura(motor):
    """Término del grupo nuevo con synsets: "Alimentación" se resuelve en español."""
    resultado = motor.extraer_relaciones(
        "Afrontamientos y resistencias no violentas - cotidianas - Alimentación"
    )
    assert resultado.encontrado is True
    assert resultado.variante_usada == "alimentación"
    assert "nutrición" in resultado.relaciones["sinonimia"]


# ----------------------------------------------------------------------
# ResultadoRelaciones
# ----------------------------------------------------------------------


def test_resultado_conteos_nueve_claves(motor):
    """conteos() expone las 9 claves de TIPOS_RELACION, con 0 si ausentes."""
    resultado = motor.extraer_relaciones("zzzqqq")
    conteos = resultado.conteos()
    assert list(conteos.keys()) == list(TIPOS_RELACION)
    assert all(v == 0 for v in conteos.values())
    assert resultado.total() == 0


def test_resultado_conteos_y_total(motor):
    """El total es la suma de los conteos por tipo."""
    resultado = motor.extraer_relaciones("mujer")
    conteos = resultado.conteos()
    assert sum(conteos.values()) == resultado.total()
    assert resultado.total() > 0


def test_resultado_a_json(motor):
    """a_json() serializa conteos y hasta 3 ejemplos por tipo."""
    resultado = motor.extraer_relaciones("verdad")
    serializado = resultado.a_json()
    assert serializado["concepto"] == resultado.concepto
    assert serializado["encontrado"] is True
    assert list(serializado["conteos"].keys()) == list(TIPOS_RELACION)
    for ejemplos in serializado["ejemplos"].values():
        assert len(ejemplos) <= 3


def test_motor_cachea_resultados(motor):
    """El mismo término devuelve la misma instancia de ResultadoRelaciones."""
    primero = motor.extraer_relaciones("mujer")
    segundo = motor.extraer_relaciones("mujer")
    assert primero is segundo


def test_motor_synsets_de_variantes(motor):
    """synsets_de() reporta la variante usada que sí tiene synsets en español."""
    synsets, variante = motor.synsets_de("Poblaciones - Mujeres")
    assert variante == "mujer"
    assert synsets
    assert all(s.pos() in ("n", "v", "a", "r", "s") for s in synsets)
