# ADR-9 — Retrieval CP v2: pasajes + max-pooling de chunks (ventana corta refutada)

- **Status:** Aceptado (2026-09-08)
- **Date:** 2026-09-08
- **Context:** La recalibración de M7 (Fase K-2, petición de Carlos) partió de un
  diagnóstico previo del orquestador con 6 hipótesis de mejora. Una de ellas
  (query = fragmentos iniciales de la sentencia) fue REFUTADA con evidencia del
  corpus: la primera cita a la CP aparece a mediana en el 41 % del texto (solo
  29,5 % en el primer cuarto) y el sweep de ventanas mostró degradación
  monótona al recortar (ventana 3000 chars: F1@5 0.127 vs texto completo 0.204).
- **Decision:** Para el retrieval sentencia→CP en el producto (M7/M3):
  1. Indexar la Constitución por PASAJES (incisos/párrafos; 931 pasajes de 497
     artículos) y agregar score por artículo con max-pooling.
  2. Query = TEXTO COMPLETO de la sentencia procesado en chunks (512 tokens,
     solape 64) con max-pooling de scores por chunk (BM25 y E5).
  3. Tokenización BM25 con stopwords jurídicas + bigramas de colocaciones LLR
     del EDA (ablation: 0.218 vs 0.206 sin ellas).
  4. Estrategia híbrida BM25+E5 (min-max por caso, 0.5/0.5 pre-registrado).
- **Alternatives Considered:**
  - Ventana corta de query (considerandos): REFUTADA con evidencia (arriba).
  - Artículo completo como documento: descartado (señal diluida, v1 F1 0.119).
  - Re-ranking cruzado top-50→E5 por pares: futuro (siguiente iteración SOTA).
- **Consequences:** BM25 F1@5 0.119→0.218 (+84 %), MRR 0.717; E5 0.128; híbrido
  0.209. La "deriva temporal" del v1 no se mantiene con la muestra ampliada
  (90 casos): BM25 estable por periodo, E5/híbrido mejoran en 2012-2021. El gap
  restante vs cita-exacta (0.218 vs 0.649) motiva usar la extracción explícita
  de citas como feature de ranking en el producto.
- **Evidencia:** `resultados/vinculo_cp_benchmark_v2.json`,
  `scripts/12_evaluar_vinculo_cp_v2.py` (sweep de ventanas y ablation
  documentados en el propio script).
