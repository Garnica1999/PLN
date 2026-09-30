# V32 - Plan de reapertura: optimizacion del puntaje (2026-09-30)

Estado global: **EN EJECUCION**.
Campeon congelado (FINAL_REFERENCE): **V30 + plug-in por clase**, LB 0.63575
(CV global 0.6159732; nested 0.6154643; particion canonica).
Objetivo del usuario: mejorar el puntaje de Kaggle; referencia declarada: acercarse a 0.70 de F1.

## Invariantes (no negociables)

- Prohibido: transformers, LLMs, embeddings (preentrenados o entrenados); datos o etiquetas
  externas; metadatos de usuario.
- El **test** solo se usa para `transform -> predict` en una fase final autorizada.
  El **holdout** no se toca. El **leaderboard** es solo evidencia final (sin tuning).
- Un experimento = un **script nuevo y completo**; prohibido editar scripts ya ejecutados.
- **Pre-registro congelado** antes de ejecutar (representacion, escala, control, gates, arbol de
  decision y hashes de todo lo que se lee).
- Validacion **canonica** unica: `StratifiedGroupKFold` 5 sobre `dup_clusters`
  (`artifacts/p33_scores/todos/`). Nada se reagrupa.
- Gates de la serie: delta >= **+0.0015**, >= **4/5** folds, control favorable, bootstrap
  pareado sin cero, sin leakage. Variabilidad de entrenamiento de referencia: 8.9e-04.
- El campeon **no se modifica**: todo candidato es un challenger separado.

## Candidatos (auditoria por operacion, codebase + graphiti, 2026-09-30)

| Prioridad | Tecnica | Estado en el historial |
|---|---|---|
| P1 | Perdida surrogate de macro-F1 (entrenar el objetivo directamente) | No implementada; familia "objetivo" (solo cross-entropy usada) |
| P2 | CatBoost 1.2.10 sobre `matriz_279` congelada | No usada; boosting con ordered boosting |
| P3 | FTRL-Proximal sobre TF-IDF disperso | No implementada; cercana a SGD (ya usado) |
| - | Confident learning | Solo diagnostico; **prohibido corregir etiquetas** (riesgo pseudo-labeling) |

## Fases

- [x] **F0 Preparacion**: rama `proyecto-kaggle-age-2026-v32`, este plan, estado verificado.
- [x] **F1 Arquitectura y legalidad**: `ml-architect` eligio P1 (perdida surrogate de macro-F1)
  con protocolo congelable; `kaggle-specialist` = APTO con salvaguardas (hashes/alineacion,
  asserts de OOF, ES solo interno, determinismo, deltas con argmax puro, tercer brazo barajado);
  `dl-specialist` aprobo con correcciones (AdamW, one-hot float32, pasos calculados, guardas
  NaN, determinismo CPU/GPU, sanity de 1 lote) y recomendo ES. Carlos decidio: **early stopping
  estricto + entrenamiento en GPU** (D-007, guardian). GPU verificada libre el 2026-09-30.
- [ ] **F2 Pre-registro + implementacion + revision**: script de pre-registro congelado;
  `ml-coder` implementa `scripts/v32_01_surrogate_macrof1.py`; `ml-reviewer` revisa
  (lint/leakage/reglas); `ml-qa-engineer` verifica (Prove-It).
- [ ] **F3 Entrenamiento y evaluacion**: `ml-trainer` ejecuta, registra metricas y aplica gates;
  segunda particion si PROMISING; `ml-performance-auditor` si hay cuello de botella.
- [ ] **F4 Cierre**: notebook consolidado (seccion nueva "PHASE - V32 REOPENING"), actualizacion
  de `experiments.csv`, `PROJECT_STATE.md`, `DECISIONS.md`, `RUNLOG.md`, `FINAL_REPORT.md` y
  los informes del experimento; commits por fase; ingesta en graphiti y codebase.

## Criterio de parada

- Si P1 no pasa el gate: cerrar P1 documentado; solo entonces evaluar P2 con justificacion.
- Sin grids ni variantes encadenadas; maximo razonable 3-5 iteraciones por hipotesis.
- Ninguna submission se genera automaticamente; requeriria autorizacion explicita.

## Registro de fases

| Fecha | Fase | Evento | Commit |
|---|---|---|---|
| 2026-09-30 | F0 | plan cargado, rama creada, estado verificado | (este commit) |
