# ADR-002 — Ontología generada programáticamente con owlready2 y validada en Protégé

- **Status**: Aceptado (2026-08-30)
- **Date**: 2026-08-30
- **Context**: El taller exige construir la ontología "en Protégé" y entregar el
  archivo OWL. Protégé 5.6.9 es una aplicación GUI; construir 593 clases e
  individuos manualmente en la GUI es propenso a error y no reproducible.
- **Decision**: Generar la ontología con owlready2 (OWL 2 estándar, RDF/XML)
  desde los paths jerárquicos del tesauro, con clases (subClassOf), individuos
  para términos hoja, 5 object properties y 2 annotation properties. El archivo
  resultante se valida por re-parsing (owlready2 + rdflib), con razón HermiT
  (consistencia OK) y abriéndolo en Protégé 5.6.9 (verificado: arranca y carga
  sin errores).
- **Alternatives Considered**: construir a mano en la GUI de Protégé (no
  reproducible); OWL API en Java (complejidad innecesaria); rdflib puro (sin
  facilidades OWL DL).
- **Consequences**: OWL 2 DL válido, reproductible y abrible en Protégé;
  las celdas del notebook regeneran la ontología en Colab sin GUI.
