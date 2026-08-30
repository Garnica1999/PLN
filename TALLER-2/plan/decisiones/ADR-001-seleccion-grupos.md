# ADR-001 — Selección automática de los 5 grupos de tesauros

- **Status**: Aceptado (2026-08-30)
- **Date**: 2026-08-30
- **Context**: El taller pide seleccionar 5 grupos de tesauros de la Comisión de
  la Verdad y justificar la elección. El Excel contiene 41 grupos y 12.341
  términos; incluir todos es inviable para una ontología didáctica.
- **Decision**: Selección automática con criterios objetivos calculados con
  pandas: (1) tamaño manejable (50-450 términos), (2) riqueza jerárquica
  (niveles de profundidad), (3) cobertura preliminar en WordNet español
  (omw-1.4) >= 1%, (4) relevancia temática para la CEV, (5) diversidad de ejes
  conceptuales (agentes, eventos, estados, instrumentos, poblaciones).
  Resultado: Actores armados (82), Impactos (107), Armas (94), Poblaciones
  (213), Hechos victimizantes (90) = 586 términos.
- **Alternatives Considered**: los 5 grupos más grandes (Organizaciones 6.405 +
  resto = 10.340 términos, dominados por entidades nombradas sin cobertura
  WordNet); selección manual (sin justificación cuantitativa).
- **Consequences**: ontología de tamaño razonable (593 clases) con 5 ejes
  distintos; la cobertura WordNet de Poblaciones es baja (16,7%) y se reporta
  honestamente como limitación del recurso léxico, no del modelo.
