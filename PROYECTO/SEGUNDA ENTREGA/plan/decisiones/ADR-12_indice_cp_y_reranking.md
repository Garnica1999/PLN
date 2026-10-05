# ADR-12: Limpieza del índice CP (transitorios) + reranking dos etapas por chunks

> **Actualización 2026-09-11 (T-iter3 + W-a, v6/v8):** la estrategia adoptada en el
> producto v2 evolucionó de `bm25_rerank_chunks` (v5) a la ganadora
> **`hibrido_rerank_chunks_top50` + prior de hubs α=1.5**:
> - v6 (T-iter3): candidatos top-50 del híbrido BM25+E5 (0.5/0.5) → rerank por
>   chunks → top-10; F1@5 0.285→0.340, MRR 0.850→0.865, nDCG@10 0.438→0.513,
>   F1@10 0.249→0.316 (3/3 metas del plan: P@5, MRR, F1@10).
> - v8 (W-a iter2): + prior de hubs α=1.5 (ver ADR-15): F1@5 **0.425**, MRR
>   **0.950**, nDCG@10 **0.620**, F1@10 **0.379** (90 casos; validación A/B).
> - Producto v2: ruta `--rerank` = ganadora v8; ruta default = BM25 + hubs α=0.75
>   (F1@5 0.396); equivalencia producto↔benchmark verificada (3 casos, ids+scores).
> - La comparación de estrategias v2 completa está en `vinculo_cp_benchmark_v6/v8.json`.



- **Status:** Aceptado
- **Date:** 2026-09-10
- **Context:** El diagnóstico en vivo del ciclo v2 (caso T-6251/01: NER extrae
  arts 21 y 209, BM25 recupera 0/2 con top-5 contaminado de transitorios)
  motivó dos cambios medibles en el módulo de vínculo CP: (a) el índice BM25
  incluía los 112 transitorios (24,4 % del top-5 en el benchmark v3) y 45 IDs
  malformados "Transitorio ACL01016-N" (código del Acto Legislativo 1 de 2016
  en la fuente del Senado, no una numeración de transitorios); (b) el pipeline
  v1 no tenía segunda etapa de reranking, estándar en statute retrieval 2026
  (DoSSIER@COLIEE, LegalMALR). La evidencia se midió ANTES de arreglar
  (benchmark v3 sobre índice sucio) y DESPUÉS (v3_limpio), y el reranking se
  probó en dos iteraciones (v4 query completa, v5 por chunks).

## Evidencia (resultados/*.json, 90 casos 3/año seed 42, índice permanentes)

| Estrategia | F1@5 | P@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 v3 (índice sucio) | 0.218 | 0.349 | 0.717 | 0.342 |
| bm25 v3_limpio | 0.233 | 0.371 | 0.765 | 0.380 |
| hibrido v3_limpio | 0.263 | 0.416 | 0.751 | 0.420 |
| bm25_rerank v4 (query completa) | 0.177 | 0.276 | 0.572 | 0.296 |
| **bm25_rerank_chunks v5** | **0.285** | **0.456** | **0.850** | **0.438** |

- Limpieza del índice: +0.015 F1@5, +0.048 MRR, +0.038 nDCG@10; contaminación
  24,4 % -> 0 en top-5/10; 45 IDs corregidos a serie canónica Transitorio
  68-112 (verificados contra el HTML del Senado, 0 fallos).
- Reranking v4 (query completa truncada a 512): EMPEORA −24 % F1@5 — el
  cross-encoder solo veía el inicio de la sentencia (~68k chars), perdiendo el
  contexto que BM25 explota con max-pooling de ~113 chunks (consistente con
  ADR-9). Resultado negativo conservado como evidencia reproducible.
- Reranking v5 (por chunks): top-20 BM25 -> 3 pares (chunk, pasaje) por
  candidato con el mejor pasaje BM25 del artículo -> cross-encoder
  bge-reranker-v2-m3 fp16 GPU -> max-pooling. +22 % F1@5 vs bm25 limpio,
  +8 % vs híbrido; max_tokens_par 405 <= 512 (el par cabe completo).
- Metas del plan: P@5 >= 0.45 CUMPLIDA (0.456), MRR >= 0.80 CUMPLIDA (0.850),
  F1@10 >= 0.30 NO (0.249) — techo estructural de recall@20 de la 1ª etapa
  (los artículos citados fuera del top-20 BM25 no pueden ser rescatados;
  caso 6251: arts 21/209 en posiciones 220/305).

- **Decision:**
  1. **Índice CP**: BM25 por defecto SOLO sobre artículos permanentes (384);
     transitorios opcionales vía flag (`incluir_transitorios=True` con cache
     `_full`). IDs canónicos Transitorio 68-112 + columna `acto_legislativo`.
     El silver del benchmark se valida sobre el parquet completo (497) para no
     perder transitorios citados legítimamente.
  2. **Reranking**: adoptar `bm25_rerank_chunks` (v5) como estrategia
     recomendada del vínculo CP. En el PRODUCTO queda como modo opcional
     (`--rerank` / `usar_reranker=True`), NO default, porque: (a) añade ~3 s
     y dependencia del modelo 2.2 GB por evaluación; (b) el caso de uso
     interactivo del producto valora la latencia; (c) el modo base (BM25
     top-10 limpio) ya mejora al v1. La decisión de default se revisa en el
     checkpoint del bloque 2.
  3. La estrategia v4 (query completa) se conserva en el benchmark como
     evidencia del fallo por truncamiento (lección: en docs largos, NUNCA
     truncar la query de un cross-encoder; usar chunks + max-pooling).
- **Alternatives Considered:**
  - Excluir transitorios del parquet: descartada — se citan legítimamente
    (desarme, JEP); el flag permite ambos modos con caches separados.
  - Reranking con query completa (v4): descartada con evidencia (−24 %).
  - Reranking sobre top-50 o candidatos híbridos: pendiente — ataca el techo
    de recall@20 (F1@10 0.249 vs meta 0.30); candidato a bloque 2.
  - RRF (fusión por rangos) en lugar de mezcla 0.5/0.5: pendiente bloque 2.
- **Consequences:**
  - (+) F1@5 0.218 -> 0.285 (+31 %) y MRR 0.717 -> 0.850 (+30 %) respecto al
    inicio del ciclo; 0 transitorios espurios en el top-k.
  - (+) El fallo v4 documentado evita repetir el error de truncamiento en
    futuros rerankers (incluida la Fase V del transformer).
  - (+) Producto: latencia 8.8 -> 5.5 s y RAM 8.0 -> 1.6 GB (Fase S2).
  - (-) El modo rerank del producto requiere GPU o CPU lenta (~batch 4) y
    2.2 GB de modelo en cache HF.
  - (-) F1@10 por debajo de la meta: la mejora requiere ampliar el recall
    de la 1ª etapa (top-50 / híbrido), no el reranker.
