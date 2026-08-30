"""conftest.py — Configuración compartida de los tests del TALLER-2.

Añade el directorio `notebook_cells/` al sys.path (el núcleo POO
`sec00_clases.py` se importa sin efectos colaterales) y expone las rutas
a las fuentes de datos del taller. Las fixtures de ontología son
session-scoped: la construcción de la OWL tarda unos segundos y se
comparte entre todos los tests (requisito: suite < 3 minutos).

Fixtures principales:
    - `modulo`: Módulo sec00_clases ya importado (fuente de verdad).
    - `repositorio`: TesauroRepositorio sobre el Excel y el JSON reales.
    - `ontologia`: ServicioOntologia construido y guardado en un
      directorio temporal de sesión (OWL + trazabilidad).
"""

import sys
from pathlib import Path

import pytest

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_NOTEBOOK_CELLS = RUTA_PROYECTO / "notebook_cells"
RUTA_DATA = RUTA_PROYECTO / "data"

RUTA_EXCEL = RUTA_DATA / "tesauro_orginal_cev.xlsx"
RUTA_JSON = RUTA_DATA / "grupos_seleccionados.json"

sys.path.insert(0, str(RUTA_NOTEBOOK_CELLS))


@pytest.fixture(scope="session")
def modulo():
    """Módulo sec00_clases importado (cacheado por sys.modules)."""
    import sec00_clases

    return sec00_clases


@pytest.fixture(scope="session")
def ruta_excel() -> Path:
    """Ruta al Excel del tesauro (hojas Tesauro y Listas)."""
    return RUTA_EXCEL


@pytest.fixture(scope="session")
def ruta_json() -> Path:
    """Ruta al JSON de grupos seleccionados de sec02."""
    return RUTA_JSON


@pytest.fixture(scope="session")
def repositorio(modulo, ruta_excel, ruta_json):
    """TesauroRepositorio sobre las fuentes reales del taller."""
    return modulo.TesauroRepositorio(ruta_excel, ruta_json)


@pytest.fixture(scope="session")
def ontologia(modulo, ruta_json, tmp_path_factory) -> dict:
    """Ontología construida y guardada en un directorio temporal.

    Construye la OWL completa desde el JSON (una sola vez por sesión),
    la guarda en RDF/XML y devuelve un dict con el servicio, las rutas y
    el resumen de construcción. Reutilizada por test_ontologia y
    test_sparql.

    Returns:
        Dict con claves: servicio, ruta_owl, ruta_json, ruta_rel, resumen.
    """
    directorio = tmp_path_factory.mktemp("ontologia_tests")
    ruta_owl = directorio / "tesauro_cev.owl"
    ruta_rel = directorio / "relaciones_afines.txt"
    servicio = modulo.ServicioOntologia.obtener(ruta_owl, ruta_json, ruta_rel)
    resumen = servicio.construir_desde_grupos()
    servicio.guardar()
    return {
        "servicio": servicio,
        "ruta_owl": ruta_owl,
        "ruta_json": ruta_json,
        "ruta_rel": ruta_rel,
        "resumen": resumen,
    }
