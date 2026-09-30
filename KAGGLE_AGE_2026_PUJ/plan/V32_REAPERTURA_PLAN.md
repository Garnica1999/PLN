# V32 - Plan de reapertura: optimizacion del puntaje (2026-09-30)

Estado global: **P1 CERRADA (NO_ATRIBUIBLE)**; **Ciclo 2 (P2 CatBoost) EN EJECUCION**.
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
- [x] **F2 Pre-registro + implementacion + revision**: pre-registro congelado (`9c5cca2` HEAD;
  artefactos en `1cdf7a1`); `ml-coder` implemento `scripts/v32_01_surrogate_macrof1.py`;
  `ml-reviewer` = APROBADO con observaciones (fix de degradacion de determinismo aplicado);
  `ml-qa-engineer` = APROBADO GO (47/47 checks, `--verificar` EXIT 0). Pendiente de aclarar en
  cierre: S1 (el prereg cita el commit 9c5cca2 = HEAD previo al commit de sus artefactos).
- [x] **F3 Entrenamiento y evaluacion**: ejecutado en GPU (exit 0, 59 s; relanzado a peticion
  de Carlos tras liberar p75; sonda de determinismo 0.0 identico). Veredicto
  **NO_ATRIBUIBLE**: A1 -0.0034165 (0/5), A2 -0.0009284 (control no favorable),
  A3 -0.0009011 (control negativo por encima de A1). Sin segunda particion; P1 cerrada
  (D-081). `ml-trainer` valido artefactos (9/9) y gates contra el arbol congelado.
- [x] **F4 Cierre**: notebook consolidado con la seccion "PHASE - V32 REOPENING" (310 celdas;
  salida real + aserciones); `experiments.csv`/`RUNLOG.md`/`PROJECT_STATE.md`/`FINAL_REPORT.md`
  y `DECISIONS.md` (D-081) actualizados; commits `5c998e6` y `8baaad7`; graphiti (episodio +
  3 tripletas en `kaggle_age_2026_puj`) y codebase re-indexado (42.768 nodos / 102.033 aristas).
  S1 aclarado: el commit `9c5cca2` citado en el prereg es el HEAD al generarlo; sus artefactos
  entraron en `1cdf7a1`.

## Criterio de parada

- Si P1 no pasa el gate: cerrar P1 documentado; solo entonces evaluar P2 con justificacion.
- Sin grids ni variantes encadenadas; maximo razonable 3-5 iteraciones por hipotesis.
- Ninguna submission se genera automaticamente; requeriria autorizacion explicita.

## Registro de fases

| Fecha | Fase | Evento | Commit |
|---|---|---|---|
| 2026-09-30 | F0 | plan cargado, rama creada, estado verificado | 9c5cca2 |
| 2026-09-30 | F1 | arquitectura + legalidad + reglas NN (ES estricto y GPU por decision de Carlos) | 9c5cca2 |
| 2026-09-30 | F2 | pre-registro congelado + runner verificado (reviewer APROBADO + QA GO 47/47) | 1cdf7a1 / d1a152f |
| 2026-09-30 | F3 | V32-E1 ejecutado en GPU; NO_ATRIBUIBLE; P1 cerrada (D-081) | 5c998e6 |
| 2026-09-30 | F4 | notebook + docs + memorias (graphiti + codebase) | 8baaad7 + este commit |

## Ciclo 2 - P2 CatBoost sobre matriz_279 (abierto 2026-09-30)

Autorizado por Carlos tras cerrar P1 (opcion A). Referencia de la auditoria por operacion:
CatBoost 1.2.10 no aparece en el historial; familia boosting con ordered boosting. Se heredan
las salvaguardas y el mecanismo de evaluacion de V32-E1 (hashes, asserts, integracion P6/P7,
gates, determinismo, compliance). Scripts previstos: `v32_02_preregistro_catboost.py` y
`v32_03_catboost_challenger.py`. Un experimento = un script nuevo.

- [ ] **F1' Arquitectura y legalidad**: `ml-architect` congela el protocolo (config unica
  justificada por precedente del proyecto, sin grid; control de atribucion elegido ex ante);
  `kaggle-specialist` valida reglas/leakage heredando las salvaguardas de V32-E1.
  (Sin `dl-specialist`: no hay red neuronal.)
- [ ] **F2' Pre-registro + implementacion + revision**: pre-registro congelado; `ml-coder`;
  `ml-reviewer`; `ml-qa-engineer` (GO/NO-GO).
- [ ] **F3' Entrenamiento y evaluacion**: `ml-trainer`; gates; arbol (PROMISING -> segunda
  particion; MARGINAL/CLOSED -> cierre). `ml-performance-auditor` si hay cuello de botella.
- [ ] **F4' Cierre**: notebook (seccion del ciclo), docs, commits, memorias (graphiti/codebase).
