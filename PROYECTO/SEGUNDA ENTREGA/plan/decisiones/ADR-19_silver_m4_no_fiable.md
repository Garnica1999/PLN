# ADR-19: El silver de M4 no es un gold fiable (validación humana O1)

- **Status:** Aceptado (2026-10-05)
- **Date:** 2026-10-05
- **Context:** La etiqueta `area_primaria` de M4 es **silver**: el bloque constitucional mayoritario
  de las citas CP detectadas (o un término léxico de respaldo). La auditoría de Fase 1 estimó
  ~31,8 % de ruido OOF (cleanlab) y el macro-F1 de M4 quedó en 0,287. Para medir la fiabilidad
  real del silver se anotaron humanamente 200 documentos estratificados (pack O1, modo lote ciego;
  Carlos, 200/200, 2026-10-05) y se contrastaron con dos proxies LLM ciegos.
- **Decision:** **No usar el silver de M4 como gold** para afirmar calidad del área temática ni
  para decisiones de producto; M4 se mantiene como característica diagnóstica con caveats
  explícitos. Cualquier mejora futura de M4 exige una etiqueta mejor (gold humano tipo C1) o una
  redefinición de la tarea.
- **Evidence:**
  - Acuerdo humano–silver: exactitud 23,5 % (kappa 0,153, "leve"; n=200; 0 ambiguo/ninguno).
  - Proxies LLM ciegos: gemma4 41,5 % (κ 0,269); preanotación 29,9 % (κ 0,203).
  - Acuerdo inter-anotador: carlos|gemma4 κ 0,249; carlos|llm κ 0,268; gemma4|llm κ 0,230;
    Fleiss 0,208 (leve) → la tarea también es ambigua entre anotadores.
  - Divergencia sistemática: silver `derechos_fundamentales` → humano DESC en 29/57; silver
    `principios_fundamentales` → 0/39 exactas (empates 1-1-1 resueltos por artículo menor, p. ej.
    art. 1); silver `proteccion_aplicacion_derechos` → DESC 6/11. Las tutelas de salud/pensión
    (tema DESC) reciben silver de bloques genéricos citados.
  - Spot-check cualitativo confirmado en 3 casos (T-853/02, T-231/20, T-235/20).
- **Alternatives Considered:**
  - Confiar en el silver para M4 y seguir optimizando sobre él: rechazado (techo de etiqueta).
  - Re-etiquetar M4 con LLM a escala: no resuelve la divergencia conceptual (los proxies LLM
    tampoco superan 41,5 % vs silver; κ humano–LLM ~0,26).
  - Redefinir "área" como el bloque del TEMA (no de las citas): cambio de tarea; queda como
    opción futura con gold humano (C1-like).
- **Consequences:**
  - (+) Mensaje honesto y verificable: el área del producto es un indicio con ruido alto; el
    macro-F1 ~0,29 tiene un techo de etiqueta, no solo de modelo.
  - (–) No habrá mejoras de M4 comparables a M7 sin una nueva etiqueta; la inversión adicional
    en M4 sobre el silver queda desaconsejada.
  - Artefactos: `resultados/validacion_m4/` (respuestas_carlos.jsonl, acuerdo_v1.json, CSV;
    respaldo solo-LLM en `backup/`), `metricas.json o1_validacion_humana`, REGISTRO `O1-humano`.
