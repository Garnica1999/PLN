"""test_normalizacion.py — Tests de la cadena de normalización de términos.

Cubre NormalizadorBase (último segmento, paréntesis, símbolos), las
variantes Plural y Acentos, y el orden/dedupe de CadenaNormalizadores.
Los valores esperados reflejan el comportamiento real verificado de las
heurísticas implementadas en sec00_clases.
"""

import pytest


def test_normalizador_base_es_estrategia_abstracta(modulo):
    """NormalizadorTermino es un ABC y no se puede instanciar."""
    with pytest.raises(TypeError):
        modulo.NormalizadorTermino()


def test_normalizador_base_ultimo_segmento(modulo):
    """Toma el último segmento del path, en minúsculas y sin espacios extra."""
    normalizador = modulo.NormalizadorBase()
    assert normalizador.normalizar("A - B - C") == "c"
    assert normalizador.normalizar("Armas - Instrumentos - Daga") == "daga"
    assert (
        normalizador.normalizar("Hechos victimizantes - Desaparición forzada")
        == "desaparición forzada"
    )


def test_normalizador_base_quita_parentesis(modulo):
    """El contenido entre paréntesis se elimina."""
    normalizador = modulo.NormalizadorBase()
    assert normalizador.normalizar("A - B (nota)") == "b"
    assert normalizador.normalizar("Armas - Fusil (mecanismo)") == "fusil"


def test_normalizador_base_simbolos_no_alfanumericos(modulo):
    """Símbolos como / o + se convierten en espacios y se colapsan."""
    normalizador = modulo.NormalizadorBase()
    assert normalizador.normalizar("A - B/C") == "b c"
    assert normalizador.normalizar("A - B + C") == "b c"


def test_normalizador_base_vacio_o_no_texto(modulo):
    """Entradas vacías o no textuales devuelven cadena vacía."""
    normalizador = modulo.NormalizadorBase()
    assert normalizador.normalizar("") == ""
    assert normalizador.normalizar("   ") == ""
    assert normalizador.normalizar(123) == ""
    assert normalizador.normalizar(None) == ""


def test_normalizador_plural_singular(modulo):
    """Singulariza sustantivos terminados en s (heurística simple)."""
    normalizador = modulo.NormalizadorPlural()
    assert normalizador.normalizar("tierras") == "tierra"
    # Heurística documentada: el sufijo "es" se quita completo ("clases" -> "clas").
    assert normalizador.normalizar("clases") == "clas"


def test_normalizador_plural_singular_a_plural(modulo):
    """Pluraliza sustantivos en singular (vocal -> +s, resto -> +es)."""
    normalizador = modulo.NormalizadorPlural()
    assert normalizador.normalizar("mujer") == "mujeres"
    assert normalizador.normalizar("casa") == "casas"
    assert normalizador.normalizar("clase") == "clases"


def test_normalizador_plural_termina_en_s(modulo):
    """Toda s final se elimina salvo el sufijo es completo (heurística simple)."""
    normalizador = modulo.NormalizadorPlural()
    assert normalizador.normalizar("crisis") == "crisi"
    assert normalizador.normalizar("lunes") == "lun"


def test_normalizador_acentos(modulo):
    """Elimina diacríticos manteniendo el resto de la normalización base."""
    normalizador = modulo.NormalizadorAcentos()
    assert normalizador.normalizar("bélicas") == "belicas"
    assert normalizador.normalizar("Desaparición") == "desaparicion"
    assert (
        normalizador.normalizar("Niños - Reclutamiento ilícito")
        == "reclutamiento ilicito"
    )


def test_cadena_candidatos_orden_y_dedupe(modulo):
    """Variantes en orden de prioridad, sin duplicados, con underscore y tokens extremos."""
    cadena = modulo.CadenaNormalizadores()
    assert cadena.candidatos("A - B C") == ["b c", "b ces", "b_c", "c", "b"]


def test_cadena_candidatos_multiword_real(modulo):
    """Término real multiword: base, plural, underscore y núcleos extremos."""
    cadena = modulo.CadenaNormalizadores()
    assert cadena.candidatos("Armas - Fusil de asalto") == [
        "fusil de asalto",
        "fusil de asaltos",
        "fusil_de_asalto",
        "asalto",
        "fusil",
    ]


def test_cadena_candidatos_sin_duplicados_ni_vacios(modulo):
    """Todos los candidatos son no vacíos y únicos."""
    cadena = modulo.CadenaNormalizadores()
    candidatos = cadena.candidatos("Poblaciones - Mujeres - Lideresas comunitarias")
    assert candidatos
    assert len(candidatos) == len(set(candidatos))
    assert all(isinstance(c, str) and c for c in candidatos)
    # La variante base siempre es la primera (prioridad máxima).
    assert candidatos[0] == "lideresas comunitarias"


def test_cadena_candidatos_vacio(modulo):
    """Término vacío produce lista de candidatos vacía."""
    cadena = modulo.CadenaNormalizadores()
    assert cadena.candidatos("") == []
    assert cadena.candidatos("   ") == []


def test_cadena_normalizadores_por_defecto(modulo):
    """La cadena por defecto encadena base, plural y acentos."""
    cadena = modulo.CadenaNormalizadores()
    assert len(cadena._normalizadores) == 3
    assert isinstance(cadena._normalizadores[0], modulo.NormalizadorBase)
    assert isinstance(cadena._normalizadores[1], modulo.NormalizadorPlural)
    assert isinstance(cadena._normalizadores[2], modulo.NormalizadorAcentos)
