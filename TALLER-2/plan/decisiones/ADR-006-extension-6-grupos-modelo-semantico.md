# ADR-006 — Extensión a 6 grupos y modelo semántico fuerte

- **Status**: Aceptado (2026-08-30)
- **Date**: 2026-08-30
- **Context**: Tras revisar el taller resuelto por otro equipo (Taller_ontologia.rar)
  se identificaron dos aportes valiosos: (1) el grupo "Afrontamientos y
  resistencias no violentas - cotidianas" (23 términos), que completa el ciclo
  conceptual Actor→Hecho→Población→Impacto→Afrontamiento; y (2) un modelo
  semántico rico (19 object properties, clases definidas con restricciones).
  Carlos pidió integrar ambos: unión de grupos (6, no 10: 4 estaban duplicados)
  y un modelo semántico más fuerte, corrigiendo los errores del taller ajeno.
- **Decision**:
  1. **6 grupos** (609 términos): los 5 previos + Afrontamientos y resistencias
     no violentas - cotidianas (eje afrontamientos_resistencias; cobertura
     WordNet exacta 4,3% / 73,9% con fallback de segmentos). `grupos_seleccionados.json`
     regenerado byte-idéntico al esperado.
  2. **Modelo semántico fuerte**: 17 object properties (5 estructurales + 12
     nuevas en 6 pares inversos: afectaA/esAfectadaPor, generaImpacto/esGeneradoPor,
     experimentaImpacto/impactaA, desarrollaAfrontamiento/esDesarrolladoPor,
     respondeA/esRespondidoMediante, empleaArma/esEmpleadaEn), 7 axiomas raíz
     del ciclo y 6 clases definidas con restricciones (HechoPerpetradoPorActor,
     PoblacionAfectadaPorHechos, PoblacionConImpacto, PoblacionConAfrontamiento,
     HechoQueGeneraImpacto, AfrontamientoAsociadoAHecho) que el razón HermiT
     clasifica automáticamente.
  3. **Correcciones del taller ajeno aplicadas por diseño**: IRI cev
     personalizado (no example.org), sin doble etiquetado, propiedades con
     dominio/rango concretos (no owl:Thing), paths fieles al tesauro original
     (los 5 errores ajenos —OSIGD con jerarquía rota, MUSE duplicado, etc.—
     no pueden ocurrir: la construcción deriva de los paths exactos del JSON).
  4. **22 consultas SPARQL** (3 × 6 grupos + 4 del modelo semántico).
- **Alternatives Considered**: 10 grupos únicos (rechazado: 4 duplicados; el
  usuario eligió la unión de 6); mantener 5 grupos (rechazado: no incorpora el
  aporte ajeno).
- **Consequences**: ontología más grande (623 clases, 534 individuos, 9.812
  triples) pero consistente (HermiT) y clasificable; WordNet sobre 610
  conceptos (cobertura 50,5%, 54.130 relaciones); el taller ajeno queda
  superado en alcance, modelado e inferencia.
