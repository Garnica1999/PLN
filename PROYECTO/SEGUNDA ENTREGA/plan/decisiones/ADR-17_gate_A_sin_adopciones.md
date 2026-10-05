# ADR-17: Gate A del ciclo v3 — sin adopciones al producto (resultado nulo riguroso)

- **Status:** Aceptado
- **Date:** 2026-09-11
- **Context:** El Sprint A evaluó mejoras de recuperación (A1-A4) y clasificación (A5)
  contra el pipeline v2 congelado. El criterio pre-registrado (ADR-16, PLAN_v3) para
  adoptar era: Δ en la métrica norte (nDCG@10) superior al IC 95% bootstrap pareado,
  con consistencia entre mitades A/B. Se ejecutaron 5 experimentos con ~90 casos cada
  uno (benchmark) y test canónico (clasificador), todos con evidencia en JSON.
- **Decision:** **No adoptar ninguna variante.** El producto permanece en v2 sin cambios
  (retrieval: w05 0.5/0.5 + top-50 + bge-reranker-v2-m3 + hubs α=1.5; clasificador:
  LogReg v1 enmascarado). Los candidatos con señal parcial quedan documentados para
  Sprint B/C.
- **Evidencia (todos los JSON en `resultados/`, registro completo en `REGISTRO_EXPERIMENTOS.md`):**
  | Exp | Hipótesis | Resultado 90 casos | Gate |
  |---|---|---|---|
  | A1 | El diagnóstico con IC/recall revela techos | Recall candidatos@100 0.723; techo silver F1@5 ≤0.649; hubs aportan ~0.11 nDCG | Evidencia (sin adopción) |
  | A2 | Embeddings legales ES superan a E5 | ALIA: F1@10 +0.017 [+0.001,+0.031] ✓ / nDCG@10 +0.009 [−0.007,+0.024] ✗; modernbert solo sin hubs; BGE-M3 empata | **No** (norte no significativa) |
  | A3 | Reranker legal ES o RRF mejoran a bge | SINAI: −0.055 nDCG@10 (con hubs, p<0.01) ✗; RRF de rangos −0.020 ✗ | **No** (negativo) |
  | A4 | Otra fusión o más profundidad mejoran | w03/w07/rrf1a empatan con w05 (IC cruza 0); top-200 pierde; top-50 gana sin hubs pero empata con hubs (ya era default) | **No** (w05/top-50 se mantienen) |
  | A5 | Ensamble C+BETO o class-weights superan a C | Ensamble w=0.8: 0.2901 vs C 0.2980 ✗; class-weights 0.285-0.290 ✗; C vs v1: Δ +0.011 [−0.023,+0.030] ns | **No** (v1 se mantiene) |
- **Alternatives Considered:**
  - Adoptar ALIA por su mejora significativa en F1@10 (secundaria): descartado por
    criterio pre-registrado (norte = nDCG@10) y por costo (cache + QA); queda como
    candidato natural del Sprint B (el retriever de dominio la supera o la valida).
  - Adoptar C (calibración, +0.011 puntual pero IC cruza 0): descartado por el mismo
    gate; consistente con la decisión de Fase U (meta 0.32 no cumplida).
  - Barrido de profundidad sobre w07: descartado (w07 sin hubs roza el IC; con hubs empata).
- **Consequences:**
  - (+) El pipeline v2 queda **validado frente a 5 familias de alternativas** con
    evidencia estadística; no hay regresiones ni deuda de QA (producto intacto).
  - (+) Metodología anti-sobreajuste confirmada empíricamente: **los smokes n=10
    engañaron en 2 de 3 ocasiones** (A3 RRF +0.038→−0.007; A4 w07 0.514→empate).
    Lección documentada: los gates se deciden SOLO con n=90 + IC pareado.
  - (+) Herramientas permanentes para B: modos `--diagnostico/--a2/--a3/--a4` en el
    benchmark, rerankers parametrizables, caches de embeddings legales, bootstrap
    pareado reutilizable.
  - (-) Sprint A no mejoró métricas del producto: las ganancias quedan para Sprint B
    (fine-tuning de dominio) y C (gold + LLM).
  - (-) Coste: ~2 h GPU del sprint; documentado como inversión en diagnóstico y
    descarte fundamentado.
