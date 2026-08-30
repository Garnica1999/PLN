# ADR-005 — Tests unitarios independientes fuera del entregable

- **Status**: Aceptado (2026-08-30)
- **Date**: 2026-08-30
- **Context**: El usuario pidió incluir tests unitarios, pero recordó que el
  código del taller debe ser solo el notebook; los tests van aparte como
  código independiente y NO dentro del taller.
- **Decision**: Tests pytest en `tests/` (test_normalizacion, test_repositorio,
  test_extractores, test_ontologia, test_sparql; 60 tests) que importan el
  mismo archivo fuente embebido en el notebook (`notebook_cells/sec00_clases.py`)
  vía sys.path. Una sola fuente de verdad: lo que se testea es exactamente lo
  que se entrega en el notebook. Los tests no se incluyen en la entrega de
  CampusVirtual.
- **Alternatives Considered**: tests dentro del notebook (contradice la
  instrucción); duplicar el código en un paquete de tests (dos fuentes de
  verdad).
- **Consequences**: evidencia de calidad verificable en el repo (rúbrica
  criterio 4) sin ensuciar el notebook; el módulo de celdas debe mantenerse
  sin efectos colaterales al importar (restricción de diseño respetada).
