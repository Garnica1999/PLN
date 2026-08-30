# ADR-004 — Núcleo POO con patrones de diseño dentro del notebook

- **Status**: Aceptado (2026-08-30)
- **Date**: 2026-08-30
- **Context**: El usuario pidió código lo más reutilizable posible con
  implementación de clases y patrones de diseño (Singleton, Abstract Factory,
  entre otros), cumpliendo SOLID y clean code. Restricción: todo el código del
  taller debe vivir dentro del notebook entregable (nada de paquetes externos).
- **Decision**: Crear un núcleo POO embebido en el notebook (celdas fuente en
  `notebook_cells/sec00_clases.py`) con los patrones: Repository
  (`TesauroRepositorio`), Singleton + Facade (`ServicioOntologia.obtener()`),
  Abstract Factory + Registry (`FabricaExtractores` con las 9 relaciones),
  Strategy (`ExtractorRelacion` + 9 implementaciones y `CadenaNormalizadores`
  con variantes base/plural/acentos) y Facade (`MotorRelacionesWordNet`,
  `SparqlServicio`, `GrafoOntologia`). Las secciones sec01-sec06 consumen el
  núcleo; las constantes e imports son únicos (sec00). El módulo es importable
  sin efectos colaterales, lo que permite tests externos.
- **Alternatives Considered**: paquete Python externo `src/` (más reutilizable
  pero viola la restricción "todo el código en el notebook"); solo funciones
  sueltas (no reutilizable, duplicación detectada).
- **Consequences**: notebook más largo en la sección 00b pero con DRY real;
  curva de lectura mayor; tests unitarios posibles importando el módulo desde
  `tests/` (ADR-005).
