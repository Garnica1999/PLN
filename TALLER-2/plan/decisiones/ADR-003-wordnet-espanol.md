# ADR-003 — WordNet en español vía NLTK + omw-1.4 con reporte honesto de cobertura

- **Status**: Aceptado (2026-08-30)
- **Date**: 2026-08-30
- **Context**: La función del taller debe usar WordNet (NLTK) sobre conceptos
  de la ontología (español). El WordNet de NLTK es inglés; omw-1.4 aporta
  synsets en español (~33% de los synsets de Princeton están traducidos).
- **Decision**: Usar `wn.synsets(termino, lang="spa")` con normalización
  robusta (segmento final del path, minúsculas, sin paréntesis, fallback
  singular/plural, con/sin acentos). La antonimia y la derivación morfológica
  viven en los lemmas ingleses: se resuelven ahí y se retraducen al español.
  La cobertura (49,7% global; 6,1% coincidencia exacta) se reporta como métrica
  de análisis y limitación del recurso.
- **Alternatives Considered**: traducir términos al inglés (pierde fidelidad);
  spaCy/WordNet español externo (dependencia extra no pedida por el taller).
- **Consequences**: honestidad metodológica; las figuras y el análisis
  documentan qué grupos tienen vocabulario general (Hechos victimizantes 78,9%)
  y cuáles no (Poblaciones 16,7%).
