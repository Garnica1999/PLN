# ADR-18: Adopción del producto v2.2 (retriever B2 + reranker B1)

- **Status:** Aceptado (2026-10-04)
- **Date:** 2026-10-04
- **Context:** El ciclo v3 (Sprint B) entrenó dos modelos de dominio con las citas propias del
  corpus: **B1** (reranker cross-encoder fine-tuneado con 299.999 pares; mejor val AUC 0,8513) y
  **B2** (retriever denso contrastivo ALIA-MrBERT con 150.974 tripletas + 101.366 pares). La
  evaluación v13 (90 casos) mostró que la ruta `híbrido BM25+B2 → rerank B1 → hubs α=1,5`
  supera a la ruta v2 (E5 + bge) con IC 95 % pareado, y el corpus C3 (2022-2026) confirmó que
  no hay deriva temporal.
- **Decision:** **Adoptar v2.2**: en la ruta `--rerank` del producto, sustituir E5 → B2
  (`modelos/b2_retriever`) y bge-reranker-v2-m3 → B1 (`modelos/b1_reranker`). El modo default
  rápido (BM25 + hubs α=0,75) no cambia. Fallbacks a E5/bge documentados con warning.
- **Evidence:**
  - v13 (silver, 90 casos): nDCG@10 **0,6926/0,6907** (sin/con hubs) vs 0,625 de v12 →
    Δ +0,0673 [+0,0418;+0,0892] y +0,0617 [+0,0381;+0,0852], prob=1,000; F1@5 0,478 (mejor no
    circular del proyecto); sin caídas por periodo.
  - Gold-LLM B3 (50 casos, grados 2/1/0): B2 nDCG@10 **0,783 [0,733;0,829]** vs B1 0,685/0,707
    (Δ +0,098 [0,069;0,128]); check inter-LLM kappa **0,651 (sustancial)**.
  - C3 (2022-2026, 100 casos): nDCG@10 **0,747 [0,719;0,777]** vs histórico 0,691 → sin deriva.
  - QA v2.2: **GO** — equivalencia producto↔benchmark exacta (ids+scores), latencias 4,6-5,5 s
    default / 13,9-14,5 s rerank en runs limpios (hasta ~23 s bajo contención de GPU; límite
    ≤30), `checks_qa_final_v3.py` 10/10, rollback viable.
- **Alternatives Considered:**
  - Mantener v2 (E5+bge): rechazado (mejora concluyente con IC en dos evaluaciones).
  - Reranker SINAI/ALIA y embeddings alternativos (A2/A3): ya rechazados con IC.
  - LLM ALIA zero-shot (C2: query understanding y rerank listwise): **negativo documentado**
    (Δ −0,198 y −0,217); opción futura: LoRA con datos propios.
- **Consequences:**
  - (+) Calidad del vínculo CP muy superior (nDCG@10 silver 0,693; gold-LLM 0,783) sin costo de
    latencia relevante; caches nuevas (`cache_b2_cp`) separadas de v1/v2.
  - (–) Dependencia de dos modelos locales (≈1,8 GB fp16); riesgo de fallback silencioso
    mitigado con warnings + checks v3 que inspeccionan los logs.
  - (–) Evidencia humana pendiente: el gate se cerró con **gold-LLM + check inter-LLM**
    (verificación automática); se recomienda spot-check humano posterior (10-15 casos).
  - (–) Pendiente formal: barrido md5 completo; `TOP10_ESPERADO` hard-coded en checks v3
    (documentado).
