# Objetivos — Juris-PLN

**Fecha:** 2026-10-03 (revisión tras la auditoría Fase 1 y las decisiones de Carlos).
**Alcance:** PRIMERA ENTREGA (E1) + SEGUNDA ENTREGA (E2), ciclos v1→v3.

## 1. Objetivo general (producto)

Sistema que, dada una sentencia de la Corte Constitucional de Colombia: (1) valida que sea
providencia (M0), (2) determina su área temática constitucional en 16 clases (M4),
(3) identifica los artículos de la CP que la sustentan con robustez 1992-2021 (M7) y (4) la
sitúa en su contexto jurisprudencial (M5/M6). Fuente: `SEGUNDA ENTREGA/plan/00_intent.md`.

## 2. Objetivo investigable (doble, aprobado por Carlos el 2026-10-03)

- **B — inmediato (silver):** predecir el **bloque constitucional predominante entre las citas
  detectadas** de una providencia, sin afirmar que sea "temática jurídica" validada. Base:
  etiquetas silver (bloque mayoritario de las citas NER; 16 clases).
- **A — upgrade (humano):** **clasificación temática validada por personas expertas** sobre la
  muestra anotada del pack O1 (200 documentos, doble anotador opcional; precisión del silver y
  kappa) y, si procede, reentrenamiento contra gold humano.
- Módulos de apoyo: M0 (filtro), M1/M2 (infraestructura), M5/M6 (contexto), M7 (retrieval con
  gold humano pendiente) y el producto CLI.

## 3. Objetivos específicos medibles (O1–O4)

- **O1.** Cuantificar la calidad del silver: precisión y acuerdo (kappa) en la muestra de 200;
  la pre-anotación automática ya ejecutada (LLM local) da exactitud 0,299 / kappa 0,203 como
  proxy y **no sustituye** la validación humana.
- **O2.** Baseline reproducible en espacio fijo de 16 clases (mayoría + LogReg TF-IDF) con F1
  por clase, IC bootstrap y soportes; split temporal y holdout bloqueado como protocolo.
- **O3.** Mejora con gates (IC 95 % + consistencia A/B). Ejecutados sin adopción: H1 (robustez
  al ruido), H2 (objetivo suave top3) y H2b (post-proceso). Pendientes: H3/H4 con hipótesis
  nuevas y Sprint B (retrieval M7).
- **O4.** Producto: wording "bloque predominante", probabilidades calibradas o marcadas como no
  calibradas, límites de uso documentados.

## 4. Estado por entrega y experimentos (cifras verificadas)

- **E1 (2026):** EDA del corpus: n-gramas (top tri "accion de tutela" 344.893), colocaciones
  (LLR 75 %), lemas 87,3 % vs stems 54,2 %, POS, MTLD 61,03, granularidad (subsentencia 4,64 %).
  Corpus: 23.742 sentencias (T 17.099 / C 6.246 / SU 397), 1992-2021; 250.012.101 tokens limpios.
- **E2/v1:** M0 accuracy 1.0 (52 casos); M2 cobertura CP 98,71 % (516.529 aristas); M3 cobertura
  99,63 %; M4 LogReg 0,2870 (obs) / 0,2691 (espacio fijo); M5 456.321 aristas (84,48 %);
  M7 BM25 F1@5 0,218; producto ~9 s/eval; QA v1 GO.
- **Ciclo v2:** limpieza del índice CP (transitorios 24,4 % → 0); reranking por chunks + híbrido
  top-50 + hubs α=1,5: F1@5 0,425 / MRR 0,950 / nDCG@10 0,620 / F1@10 0,379 (silver); BETO
  chunk-mean 0,2433; QA v2 GO.
- **Ciclo v3 — Sprint A (cerrado sin adopciones, ADR-17):** A1 diagnóstico con IC; A2 embeddings
  legales (ALIA recall@100 0,723; sin adopción); A3 rerankers (SINAI pierde vs bge); A4
  fusión/profundidad (w05 y top-50 se mantienen); A5 clasificador (C 0,298 sin adopción).
- **Auditoría Fase 1 (2026-10-03):** ruido del silver M4 ~31,8 % OOF (cleanlab; test 31,09 %);
  val 0 % in-sample (el joblib v1 se entrenó en train+val); OOF macro-F1 0,2597; baseline
  mayoría 0,0374; IC95 test [0,2461; 0,2912]. Discrepancias documentales detectadas y en
  corrección (ADR-13, `estado_arte.md`, terminología "gold NER"→silver).
- **Experimentos M4:** H1 negativo (ninguna variante robusta supera al baseline); H2 objetivo
  suave W2 val 0,3002 vs 0,2658 (Δ +0,0344 IC [−0,0051;+0,0749], sin adopción); H2b post-proceso
  con ganancia val-específica (+0,17) no transferible (test +0,02; temporal −0,16) → sin
  adopción. Producto mantiene v1.
- **Sprint B1 (en curso):** pares reranker de dominio (citas propias, 90 casos excluidos);
  fine-tune de `bge-reranker-v2-m3` en preparación. O1: pack de 200 docs + guía + herramienta
  listos; pre-anotación LLM hecha (proxy).

## 5. Metas vigentes y pendientes

- **M4:** macro-F1 0,32 (hoy 0,287 producto; espacio fijo 0,269; C 0,298 no adoptada). Sin gold
  humano no hay techo medible; el ruido del silver (~32 %) es el primer límite.
- **M7:** nDCG@10 ≥ 0,70 con gold humano (hoy 0,620 silver) y ≥ 0,60 por periodo; F1@10 ≥ 0,42.
  Sprints: B (B1 reranker de dominio, B2 retriever denso, B3 mini-gold) y C (gold 150, LLM,
  corpus 2022-2026, producto v3).
- **Producto:** ≤10 s default / ≤30 s con rerank o LLM.

## 6. Criterios de éxito y evidencia

QA GO v1/v2 (10/10 E2E, drift 0), `REGISTRO_EXPERIMENTOS.md`, ADRs 1-17, informes y zips de
entrega. Regla transversal: adopción solo con evidencia (IC 95 % bootstrap + consistencia A/B);
los negativos se documentan con el mismo rigor; terminología silver/gold estricta.

Referencias cruzadas: `arquitectura.md` (cómo está construido) y `AGENTS.md` (operación).
