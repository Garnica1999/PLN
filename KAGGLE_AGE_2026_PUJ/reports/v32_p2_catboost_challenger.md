# V32-E2 (P2) - Ejecucion: challenger CatBoost sobre la matriz 279

- Fecha de ejecucion: 2026-09-30
- Pre-registro: reports/v32_p2_preregistro.md (commit 9a2fbfd) y `experiments/v32_02_preregistro_catboost.json`
- Commit de ejecucion: `c38b8c73f9428391324bde7db4fa50c44821e835`
- Baseline primario recomputado: 0.6159732 (referencia nested 0.6154643)
- CPU con 32 hilos fijos; ES interno 90/10; od_wait/early_stopping = 150; iteraciones cap 3000
- Compliance AST: 52 literales de ruta revisados, 0 violaciones

## Entrenamiento (el fold-val solo se uso para el OOF; el ES miro el split interno)

| Brazo | fold | iters | best_iteration | metrica interna mejor | truncado | tiempo (s) |
|---|---|---|---|---|---|---|
| A1_catboost | 0 | 1539 | 1388 | 0.613265 | False | 105.1 |
| A1_catboost | 1 | 904 | 753 | 0.603842 | False | 61.8 |
| A1_catboost | 2 | 1036 | 885 | 0.609486 | False | 70.7 |
| A1_catboost | 3 | 1285 | 1134 | 0.606751 | False | 87.2 |
| A1_catboost | 4 | 1463 | 1312 | 0.614291 | False | 99.1 |
| A2_lgbm_twin | 0 | 522 | 372 | 0.615122 | False | 35.9 |
| A2_lgbm_twin | 1 | 440 | 290 | 0.611685 | False | 32.3 |
| A2_lgbm_twin | 2 | 724 | 574 | 0.617255 | False | 54.0 |
| A2_lgbm_twin | 3 | 447 | 297 | 0.608796 | False | 32.8 |
| A2_lgbm_twin | 4 | 550 | 400 | 0.617334 | False | 40.9 |
| A3_barajado | 0 | 989 | 838 | 0.164244 | False | 67.5 |
| A3_barajado | 1 | 676 | 525 | 0.151542 | False | 46.0 |
| A3_barajado | 2 | 1558 | 1407 | 0.158957 | False | 105.3 |
| A3_barajado | 3 | 1171 | 1020 | 0.169352 | False | 79.6 |
| A3_barajado | 4 | 754 | 603 | 0.159099 | False | 51.5 |

## Sanity de metricas (A1 fold 0)

- Criterio de la enmienda pre-ejecucion 1: TotalF1:average=Macro = 0.6132649 vs macro-F1 de sklearn = 0.6132649; delta = 0.000e+00 (tolerancia 1e-06)
- Diagnostico conservado: TotalF1 default (Weighted) = 0.6124681; delta contra macro-F1 de sklearn = 7.968e-04

## Sonda de determinismo

- A1_catboost: max|delta p| = 0.000e+00 (exacto, no_concluyente=False)
- A2_lgbm_twin: max|delta p| = 0.000e+00 (exacto, no_concluyente=False)
- Calibracion tras el fold 0 de A1: t = 105.1 s; proyeccion 9x = 0.26 h (limite 20.0 h)

## Integracion P6/P7 (combinador congelado, regla argmax(p_comb * w))

| Brazo | macro-F1 comb | delta vs baseline | folds+ | IC95 | delta sin regla | delta vs nested | sha256 OOF (16) |
|---|---|---|---|---|---|---|---|
| A1_catboost | 0.6148474 | -0.0011258 | 1/5 | [-0.0019378, -0.0003467] | -0.0030559 | -0.0006169 | `39fc48689ee99566` |
| A2_lgbm_twin | 0.6146861 | -0.0012871 | 1/5 | [-0.0020580, -0.0005027] | -0.0029133 | -0.0007782 | `8b738562366f5254` |
| A3_barajado | 0.6147594 | -0.0012138 | 1/5 | [-0.0019830, -0.0004530] | -0.0031274 | -0.0007049 | `b00fa2ccb83cf682` |

## Complementariedad standalone (argmax del OOF, sin pesos)

| Brazo | macro-F1 solo | diagonal 6x6 | rescate | dano | neto | rescate alta conf. | errores alta conf. campeon |
|---|---|---|---|---|---|---|---|
| A1_catboost | 0.6082005 | True | 4940 | 5589 | -649 | 9 | 9780 |
| A2_lgbm_twin | 0.6115437 | True | 4548 | 4609 | -61 | 4 | 9780 |
| A3_barajado | 0.1558278 | False | 11605 | 87272 | -75667 | 1718 | 9780 |

## Gates

| Gate | Resultado |
|---|---|
| baseline_ok | True |
| delta_A1_suficiente | False |
| folds_positivos_A1_suficiente | False |
| control_favorable_A1_sobre_A2 | False |
| ic95_A1_sin_cero | True |
| sin_leakage | True |
| sin_truncamiento_A1 | True |
| determinismo_concluyente | True |

## Control de atribucion y control negativo

- Bootstrap pareado de la diferencia (F1_combA1 - F1_combA2): delta medio +0.0001714, IC95 [-0.0003162, +0.0006557], favorable=False
- delta_int(A3) = -0.0012138 vs delta_int(A1) = -0.0011258: atribuible a la senal de CatBoost
- Truncamiento: A1 presupuesto agotado = False; A2 = False (control debilitado si True); A3 = False

## Veredicto

**CLOSED**: delta <= 0 -> se cierra P2; seria el ultimo candidato priorizado (P3 FTRL solo con autorizacion)

## Compliance

- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin leaderboard, sin submission.
- Auditoria AST del propio runner: 0 palabras vigiladas en literales de ruta; sin lecturas de particiones reservadas.
- El gemelo LightGBM se reentrena desde cero (control interno, no reutiliza pesos del pool); campeon V30 + plug-in intacto; sin tuning posterior.
