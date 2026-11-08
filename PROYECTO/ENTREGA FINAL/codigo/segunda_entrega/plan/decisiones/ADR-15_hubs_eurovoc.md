# ADR-15: Prior de hubs ADOPTADO en retrieval; EuroVoc auxiliar NO adoptado

- **Status:** Aceptado
- **Date:** 2026-09-11
- **Context:** Fase W del plan v2 incluía (a) prior de co-citación de artículos CP
  como señal de ranking (motivado por UA-StatuteRetrieval 2026, arXiv:2607.11109:
  los artículos hub resisten el decay temporal) y (b) EuroVoc-es como señal
  auxiliar (features de léxico o taxonomía) en clasificación. Ambos debían
  adoptarse SOLO con evidencia de mejora, documentando el negativo en caso contrario.
- **(a) Prior de hubs — DECISIÓN: ADOPTAR.** Prior estático global = frecuencia de
  citación por artículo CP sobre las 23.742 sentencias de M2, normalizada [0,1]
  (`data/prior_hubs_cp.json`; top-5: 241, 86, 13, 2, 1; determinista, md5
  80ef66a2…). No es información por caso (estadística de corpus, tipo IDF).
  Aplicación: score_final = minmax(score_base) + alpha·prior (respeta -inf del
  rerank). Calibración honesta A/B (pares/impares) + full-90 con rejilla extendida:
  - **Ruta rerank (hibrido_rerank_chunks_top50) alpha=1.5**: F1@5 0.340→**0.425**,
    MRR 0.865→**0.950**, nDCG@10 0.513→**0.620**, F1@10 0.316→**0.379** (mejora
    consistente en A, B y full-90; óptimo interior en 1.5, cae en 2.0).
  - Ruta default (bm25) alpha=0.75: F1@5 0.233→0.396, MRR 0.901, nDCG@10 0.601.
  - Integrado en el producto v2 (`VinculoCP` `usar_hubs=True`, α base 0.75 /
    rerank 1.5) con equivalencia exacta producto↔benchmark verificada en 3 casos.
- **(b) EuroVoc-es — DECISIÓN: NO ADOPTAR.** Viabilidad técnica SÍ (41 conceptos
  raíz ES verificados vía SPARQL oficial; 1.268 términos; mapeo en
  `config/eurovoc_bloques.json`), pero ninguna variante mejoró:
  - D (cascada con taxonomía EuroVoc): test 0.2625 vs C 0.2980 (Δ -0.0355);
  - E (features de léxico EuroVoc): test 0.2771 vs C 0.2980 (Δ -0.0209).
  Coherente con la evaluación documental E1 (cobertura parcial para la CP
  colombiana: p. ej. organismos_control con 2 términos).
- **Alternatives Considered:**
  - PageRank de co-citación como prior: comparado (ρ=0.9828 con frecuencia,
    mismo top-5) y descartado por redundante (documentado en el código).
  - Extender la rejilla alpha más allá de 2.0: innecesario (curva desciende
    tras 1.5).
- **Consequences:**
  - (+) El retrieval del producto v2 gana ~+25 % F1@5 sobre la ruta v6 y ~+82 %
    sobre el inicio del ciclo (0.233→0.425), con validación cruzada honesta.
  - (+) El prior es estático, barato (un dict) y compatible con cualquier
    estrategia de ranking futura.
  - (-) El prior favorece artículos populares: puede penalizar artículos raros
    relevantes; el benchmark muestra ganancia neta, pero casos atípicos podrían
    verse afectados (riesgo residual documentado).
  - (-) EuroVoc no se integra: se preservan los archivos de mapeo/léxico como
    evidencia y base para trabajo futuro.
