# V32 - Plan de reapertura: optimizacion del puntaje (2026-09-30)

Estado global: reapertura V32 **TERMINADA** (P1 NO_ATRIBUIBLE D-081, P2 CLOSED D-082,
P3 NO_ATRIBUIBLE D-083) y **V33 TERMINADA** (V33-E1 NO_ATRIBUIBLE_BY_GATE, D-084; Via B
descartada por ILEGAL). Cuatro candidatos medidos y ninguno pasa los gates. Campeon V30 +
plug-in (LB 0.63575) intacto; sin submission. Reabrir la linea online exigiria un diseno
y un pre-registro nuevos.
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
| 2026-09-30 | F4 | notebook + docs + memorias (graphiti + codebase) | 8baaad7 + 1a0b971 |
| 2026-09-30 | F1' | P2: arquitectura (architect) + compliance (kaggle-specialist) | e31623f |
| 2026-09-30 | F2' | pre-registro + enmienda pre-ejecucion 1 + runner verificado (QA 7/7) | e31623f / f0bf731 / c38b8c7 |
| 2026-09-30 | F3' | V32-E2 ejecutado (1149 s); CLOSED; sin segunda particion (D-082) | b6902ed |
| 2026-09-30 | F4' | notebook (313 celdas) + docs + memorias | db69ed0 |
| 2026-09-30 | F1'' | ciclo 3 autorizado: P3 FTRL-Proximal (arquitectura + novedad) | 15c2be8 |
| 2026-09-30 | F2'' | pre-registro congelado + runner verificado (QA GO; auditoria del data_hash) | 7fcf499 / ee90975 |
| 2026-09-30 | F3'' | V32-E3 ejecutado (755 s); NO_ATRIBUIBLE; P3 cerrada; reapertura TERMINA (D-083) | a008e45 |
| 2026-09-30 | F4'' | notebook + docs + memorias | (commit de cierre) |

## Ciclo 4 - V33: combinacion autorizada (diseno online corregido + linea de otra magnitud)

Carlos autoriza combinar la opcion 2 (P3b: diseno online corregido, con semantica online real
en lugar de 10 pasadas proximales) y la opcion 3 (una linea de ideas de otra magnitud). La
reapertura V32 queda CERRADA (D-081, D-082, D-083) y esto es un expediente nuevo con su propio
plan. Restricciones heredadas: sin transformers/LLMs/embeddings; sin datos externos; sin metadatos;
test solo transform->predict; holdout intocable; leaderboard sin tuning; un experimento = un
script nuevo; pre-registro congelado; campeon V30 + plug-in (LB 0.63575) intacto.

- [x] **F1''' Investigacion y arquitectura**: `ml-architect` diagnostico la divergencia de P3
  (`sigma` inflado 10x por `alpha=0.05` mas 10 pasadas con semantica proximal; solo `alpha`
  tiene palanca) y diseno la **Via A**: FTRL-Proximal **softmax multiclase**, **una sola pasada
  online**, `alpha = 3/(2*sqrt(T))` DERIVADO (T = n_train_fold; cero valores evaluados),
  lambda1 0.0 / lambda2 1.0 / beta 1.0, intercepto x0=1.0, sin power_t; control SGD log_loss
  de 1 pasada con el mismo barajado; **puerta de convergencia G1-G6 fail-fast** en el fold 0
  antes de integrar. La Via B quedo DESCARTADA: B1 (grafo transductivo) **ILEGAL** segun
  `kaggle-specialist` (usa features de test para modificar predicciones de train; su forma legal
  es kNN = M09 cerrado), B2 descartada por analogia con C04, B3 refutada por algebra.
  `kaggle-specialist` = GO condicionado (condiciones incorporadas al pre-registro); sin
  `nlp-specialist` porque no entra ninguna representacion nueva. **Aprobado por Carlos.**
- [x] **F2''' Pre-registro + implementacion + revision**: pre-registro congelado (`950e17a`);
  `ml-coder` implemento `scripts/v33_01_ftrl_softmax_online.py`; **enmienda pre-ejecucion 1**
  (banda G1 y `shuffle=False`) + `scripts/v33_00c_correccion_puerta_md.py` para el documento
  normativo; `ml-reviewer` = **APROBADO** tras dos rondas (nucleo verificado con una 3a
  implementacion independiente, max|dw| = 2.4e-15); `ml-qa-engineer` = **NO-GO** y luego
  **GO** tras corregir 2 defectos criticos del nodo de cierre por puerta y la fila obsoleta
  del documento normativo. En total `--verificar` y las revisiones cazaron **6 bugs reales**:
  matmul con dimensiones incompatibles, `sum(axis=1)` que reducia las clases, `KeyError` de
  `control_atribucion`, typo `verdicto`, sonda de determinismo `None` y `KeyError` de `caps`.
  Lint limpio, `py_compile` OK, `--verificar` EXIT 0, 0 asserts de integridad (todo `Detencion`).
- [x] **F3''' Ejecucion + gates**: ejecutado en CPU (exit 0; **21.1 s**; fail-fast en el
  fold 0; RSS 0.71 GB; sondas (a)-(f) pass; AST 0 violaciones). Veredicto
  **NO_ATRIBUIBLE_BY_GATE**: A1 (FTRL softmax, 1 pasada, alpha = 3/(2*sqrt(135468)) =
  0.0040754249463367) dejo `ll_train` en 1.8047797, a 0.013 del anclaje uniforme ln 6 =
  1.7917595 -> fallo G1 (>1.6126), G3 (macro-F1 standalone 0.0409742 < 0.30 y < el trivial
  0.0485421) y G4 (frecuencias [0.2629, 0.1125, 0.0019, 0.00024, 0.0270, 0.5955], colapso a
  la clase 5); G2 y G6 pasaron. A2 (SGD 1 pasada) sano en perdida y F1 (ll 1.3899611,
  F1 0.3956246, G1-G3 y G6 OK) pero fallo G4 (clase 5 al 42.6 % > 0.30). Sonda de
  atribucion PUBLICADA: razon media |sigma*w|/|g| = 0.0419516 (mediana 0.0380133, p90
  0.0632781, max 0.2153788, sesgo 0.0038350, frac > 1 = 0.0) -> la correccion proximal es
  ~24x MAS DEBIL que el gradiente, no inflada. Sin integracion, sin bootstrap, sin segunda
  particion, sin submission; `delta` intacto. V33 cerrada (D-084).
- [x] **F4''' Cierre**: acta **D-084** (Observation / Result / Decision / Consequences /
  Evidence) con el hallazgo clave de que el punto de operacion viable de `alpha` esta
  entre 0.004 (correccion ~24x mas debil que el gradiente) y 0.05 (correccion inflada que
  hacia divergir en D-083), y que localizarlo exigiria un grid, prohibido por el
  protocolo; `PROJECT_STATE.md` y `FINAL_REPORT.md` actualizados; seccion
  **PHASE - V33 (P3b - FTRL SOFTMAX ONLINE)** en el notebook consolidado (319 celdas,
  salida real + 20 aserciones); artefactos de novedad NUEVOS
  (`reports/v33_novelty_registry.csv` y fila anexada en
  `reports/age_complete_experiment_history.csv`, sin editar ningun script ejecutado);
  RUNLOG y `experiments.csv` al dia; memorias (episodio + 3 tripletas en Graphiti del
  grupo `kaggle_age_2026_puj`; codebase re-indexado con 44.843 nodos y 107.796 aristas).
  Con este cierre **V33 queda cerrado** y la reapertura V32 sigue terminada.

## Ciclo 2 - P2 CatBoost sobre matriz_279 (abierto 2026-09-30)

Autorizado por Carlos tras cerrar P1 (opcion A). Referencia de la auditoria por operacion:
CatBoost 1.2.10 no aparece en el historial; familia boosting con ordered boosting. Se heredan
las salvaguardas y el mecanismo de evaluacion de V32-E1 (hashes, asserts, integracion P6/P7,
gates, determinismo, compliance). Scripts previstos: `v32_02_preregistro_catboost.py` y
`v32_03_catboost_challenger.py`. Un experimento = un script nuevo.

- [x] **F1' Arquitectura y legalidad**: `ml-architect` congelo el protocolo (A1 CatBoost
  MultiClass 3000/lr 0.03/depth 6/ES TotalF1 od_wait 150; A2 gemelo LightGBM con el ancla
  afinada V29/V30 como control decisional; A3 barajado; guarda de truncamiento; calibracion y
  timeout); `kaggle-specialist` = APTO con condiciones (bootstrap pareado de la diferencia,
  tolerancias de determinismo, AST-audit, truncamiento registrado). Sin `dl-specialist` (no
  hay red neuronal). Pre-registro congelado en `reports/v32_p2_preregistro.md` +
  `experiments/v32_02_preregistro_catboost.json` (script `v32_02`).
- [x] **F2' Pre-registro + implementacion + revision**: pre-registro congelado (`e31623f`) +
  **enmienda pre-ejecucion 1** (`f0bf731`: eval_metric -> `TotalF1:average=Macro`; el default
  Weighted de catboost 1.2.10 difiere de macro-F1); `ml-coder` implemento
  `scripts/v32_03_catboost_challenger.py`; `ml-reviewer` = APROBADO con observaciones (2
  correcciones: iteraciones A2 y calibracion fail-closed al reanudar; `metric:"None"` en A2
  aceptado por el lead como detalle de ejecucion); `ml-qa-engineer` = GO condicionado, cerrado
  con probe E2E 7/7 (PROMISING/NO_ATRIBUIBLE/caps/CLOSED/MARGINAL) y `--verificar` EXIT 0.
- [x] **F3' Entrenamiento y evaluacion**: ejecutado en CPU (exit 0; 1149 s; 15 fits; sin
  truncamiento; sondas de determinismo A1/A2 = 0.0 exactas; sanity TotalF1:average=Macro ==
  macro-F1 con delta 0.0). Veredicto **CLOSED**: A1 -0.0011258 (1/5 folds), A2 -0.0012871,
  A3 -0.0012138; control A1 vs A2 no favorable (+0.00017; IC95 incluye 0). Sin segunda
  particion; P2 cerrada (D-082). `ml-trainer` valido artefactos, gates y arbol del ciclo 2.
- [x] **F4' Cierre**: notebook actualizado con la seccion "PHASE - V32 REOPENING (P2 - CATBOOST)"
  (313 celdas; salida real + aserciones del veredicto CLOSED); `DECISIONS.md` (D-082),
  `PROJECT_STATE.md`, `FINAL_REPORT.md`, `RUNLOG.md` y `experiments.csv` al dia; graphiti
  (episodio + tripletas) y codebase re-indexado.

## Ciclo 3 - P3 FTRL-Proximal sobre TF-IDF (autorizado 2026-09-30)

Autorizado por Carlos tras cerrar P2. Es el ultimo candidato de la lista priorizada
(secundario: FTRL-Proximal es cercano al SGD ya usado; no aparece en el historial). Se heredan
las salvaguardas, el mecanismo de integracion P6/P7, los gates, la sonda de determinismo y el
compliance de los ciclos previos. Scripts previstos: `v32_04_preregistro_ftrl.py` (lead) y
`v32_05_ftrl_challenger.py` (runner). Un experimento = un script nuevo.

- [x] **F1'' Arquitectura y novedad**: `ml-architect` congelo el protocolo (FTRL-Proximal propio
  alpha 0.05 / lambda1 0.0 / lambda2 1.0 / beta 1.0 / 10 pasadas; TF-IDF IDENTICA a
  `v31_02_banco_representaciones.py` ajustada POR FOLD; A2 gemelo SGD con 10 pasadas; A3
  barajado; sondas (a)-(d) obligatorias con exit 4; regla de fallback escrita antes de ejecutar;
  cap por IC95 de la diferencia). `kaggle-specialist` = APROBADO CON EXIGENCIAS. Nota:
  `vowpalwabbit 9.11.8` esta instalado y se descarta (implementacion propia verificable).
  **Auditoria registrada**: el `data_hash` del manifiesto canonico es un hash de PUNTEROS
  (`tobytes()` sobre array de objetos) y no es reproducible; se congelan sha256 del archivo
  (`39c6c827...`) y del array de textos con receta determinista (`53c1a348...`).
  Pre-registro: `reports/v32_p3_preregistro.md` + `experiments/v32_04_preregistro_ftrl.json`
  (script `v32_04`).
- [x] **F2'' Pre-registro + implementacion + revision**: pre-registro congelado (`7fcf499`);
  `ml-coder` implemento `scripts/v32_05_ftrl_challenger.py` (sondas: vectorizado vs lazo
  6.66e-15, punto fijo analitico 0.0, TotalF1:average=Macro 0.0, p y log_loss finitos);
  `ml-reviewer` = APROBADO con observaciones (3 correcciones importantes: log_loss multiclase
  propio porque sklearn 1.8.0 no renormaliza, recarga validada del OOF crudo de A2 al reanudar,
  AST audit ampliado a modulos importados; mas 3 cosmeticas); `ml-qa-engineer` = GO con
  observaciones (54 checks; `--verificar` sin efectos secundarios; caps, precedencias,
  reanudacion y AST audit probados con harness sintetico en /tmp).
- [x] **F3'' Entrenamiento y evaluacion**: ejecutado en CPU (exit 0; 755 s; RSS 0.87 GB;
  sondas (a)-(d) pass; determinismo A1/A2 = 0.0 exacto). Veredicto **NO_ATRIBUIBLE**: la regla
  de fallback se activo (log_loss medio de train en la epoca 1 = 11.72 -> alpha 0.05 a 0.005
  con reinicio del fold 0) y aun asi el brazo FTRL **degenero** (standalone 0.040169, por debajo
  de la linea base de la clase mayoritaria 0.0485; p plana; log_loss OvR 1.7921) mientras el
  gemelo SGD si aprendo (0.493479; -0.0017002 integrado, 0/5). A1 y A3 colapsan al mismo argmax
  -> deltas identicos (-0.0011973); control A1 vs A2 no favorable (+0.00050; IC95 contiene 0)
  y cap por IC95 activo. Sin segunda particion; **P3 cerrada y la reapertura V32 termina**
  (D-083). `ml-trainer` recomputo 45/45 magnitudes desde los OOF.
- [x] **F4'' Cierre**: notebook (seccion del ciclo), `DECISIONS.md` (D-083), `PROJECT_STATE.md`,
  `FINAL_REPORT.md`, RUNLOG, experiments.csv, commits, memorias (graphiti/codebase). Con este
  cierre la reapertura V32 queda **terminada**: tres candidatos medidos (P1 NO_ATRIBUIBLE,
  P2 CLOSED, P3 NO_ATRIBUIBLE) y el campeon V30 + plug-in (LB 0.63575) intacto.
