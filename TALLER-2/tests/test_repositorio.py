"""test_repositorio.py — Tests de TesauroRepositorio (patrón Repository).

Verifica la carga cacheada del Excel (hojas Tesauro y Listas), el JSON
de grupos seleccionados y las transformaciones estructurales
(con_estructura, marcar_hojas, detectar_anomalias).
"""

import pandas as pd
import pytest

GRUPOS_ESPERADOS = [
    "Actores armados y otros participantes en las dinámicas de la guerra",
    "Impactos",
    "Armas",
    "Poblaciones",
    "Hechos victimizantes",
    "Afrontamientos y resistencias no violentas - cotidianas",
]

GRUPO_AFRONTAMIENTOS = "Afrontamientos y resistencias no violentas - cotidianas"
TERMINOS_AFRONTAMIENTOS = 23

COLUMNAS_ANOMALIAS = [
    "duplicados",
    "vacios",
    "separador_final",
    "con_parentesis",
    "con_comillas",
    "acronimo",
]


def test_cargar_tesauro_shape_y_columnas(repositorio):
    """La hoja Tesauro tiene ~12.341 filas, columnas grupo/termino y sin nulos."""
    df = repositorio.cargar_tesauro()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (12341, 2)
    assert list(df.columns) == ["grupo", "termino"]
    assert df["grupo"].notna().all()
    assert df["termino"].notna().all()
    assert not df.empty


def test_cargar_tesauro_cache(repositorio):
    """La segunda llamada devuelve el mismo objeto (cache)."""
    primera = repositorio.cargar_tesauro()
    segunda = repositorio.cargar_tesauro()
    assert id(primera) == id(segunda)


def test_cargar_listas(repositorio):
    """La hoja Listas tiene columnas agrupacion/termino y no está vacía."""
    df = repositorio.cargar_listas()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["agrupacion", "termino"]
    assert not df.empty
    assert df["termino"].notna().all()


def test_con_estructura(repositorio):
    """Añade segmentos (lista), niveles (int) y hoja (último segmento)."""
    df = repositorio.cargar_tesauro().head(50)
    estructurado = repositorio.con_estructura(df)
    assert list(estructurado.columns) == [
        "grupo",
        "termino",
        "segmentos",
        "niveles",
        "hoja",
    ]
    assert all(isinstance(s, list) and s for s in estructurado["segmentos"])
    assert (estructurado["niveles"] >= 1).all()
    assert (estructurado["hoja"] == estructurado["segmentos"].str[-1].str.strip()).all()


def test_marcar_hojas(repositorio):
    """Marca es_hoja: los paths que son prefijo de otro no son hoja."""
    df = repositorio.con_estructura(repositorio.cargar_tesauro())
    marcado = repositorio.marcar_hojas(df)
    assert "es_hoja" in marcado.columns
    assert marcado["es_hoja"].dtype == bool
    assert marcado["es_hoja"].sum() > 0
    assert (~marcado["es_hoja"]).sum() > 0


def test_detectar_anomalias(repositorio):
    """Agrupa por grupo los conteos de 6 patrones anómalos."""
    df = repositorio.cargar_tesauro()
    anomalias = repositorio.detectar_anomalias(df)
    assert anomalias.index.name == "grupo"
    assert list(anomalias.columns) == COLUMNAS_ANOMALIAS
    assert len(anomalias) == df["grupo"].nunique()
    assert (anomalias >= 0).all().all()


def test_grupos_seleccionados_estructura(repositorio):
    """El JSON tiene total_terminos==609 y las 6 claves de grupos."""
    grupos = repositorio.grupos_seleccionados()
    assert grupos["total_terminos"] == 609
    terminos = grupos["terminos_por_grupo"]
    assert list(terminos.keys()) == GRUPOS_ESPERADOS
    assert sum(len(v) for v in terminos.values()) == 609


def test_grupo_afrontamientos_presente_con_veintitres_terminos(repositorio):
    """El grupo nuevo (extensión aprobada) tiene 23 términos en el JSON."""
    terminos = repositorio.terminos_por_grupo()
    assert GRUPO_AFRONTAMIENTOS in terminos
    assert len(terminos[GRUPO_AFRONTAMIENTOS]) == TERMINOS_AFRONTAMIENTOS
    assert all(
        t.startswith(GRUPO_AFRONTAMIENTOS) for t in terminos[GRUPO_AFRONTAMIENTOS]
    )


def test_grupos_seleccionados_cache(repositorio):
    """La segunda llamada al JSON devuelve el mismo objeto (cache)."""
    primera = repositorio.grupos_seleccionados()
    segunda = repositorio.grupos_seleccionados()
    assert id(primera) == id(segunda)


def test_terminos_por_grupo(repositorio):
    """terminos_por_grupo expone solo los paths de los 6 grupos."""
    terminos = repositorio.terminos_por_grupo()
    assert list(terminos.keys()) == GRUPOS_ESPERADOS
    assert all(isinstance(v, list) and v for v in terminos.values())


def test_excel_inexistente_lanza_file_not_found(modulo, tmp_path):
    """Un Excel inexistente lanza FileNotFoundError."""
    repositorio = modulo.TesauroRepositorio(tmp_path / "no_existe.xlsx")
    with pytest.raises(FileNotFoundError):
        repositorio.cargar_tesauro()


def test_json_inexistente_lanza_file_not_found(modulo, ruta_excel, tmp_path):
    """Un JSON inexistente lanza FileNotFoundError."""
    repositorio = modulo.TesauroRepositorio(ruta_excel, tmp_path / "no_existe.json")
    with pytest.raises(FileNotFoundError):
        repositorio.grupos_seleccionados()
