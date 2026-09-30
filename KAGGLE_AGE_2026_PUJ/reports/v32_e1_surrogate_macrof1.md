# V32-E1 - Ejecucion: challenger con perdida surrogate de macro-F1

- Fecha de ejecucion: 2026-09-30
- Pre-registro: reports/v32_e1_preregistro.md (commit 9c5cca2) y `experiments/v32_00_preregistro.json`
- Commit de ejecucion: `d1a152f1b8bdf4119d7e1d80bdd93b623659dd4d`
- Baseline primario recomputado: 0.6159732 (referencia nested 0.6154643)
- Dispositivo: AMD Radeon RX 9070 XT, torch 2.14.0+rocm7.2, matmul precision highest, 8 hilos CPU, guardian D-007 (require_gpu wait=10, poll=30 s; sin fallback a CPU)
- Determinismo: modo estricto; sonda A1 fold 0 max|delta p| = 0.000e+00 (referencia de variabilidad 0.0008876)

## Integracion por brazo (combinador P6/P7 congelado, regla argmax(p_comb * w))

| Brazo | macro-F1 comb | delta vs baseline | folds+ | IC95 | delta sin regla | delta vs nested | sha256 OOF (16) |
|---|---|---|---|---|---|---|---|
| A1_surrogate | 0.6125567 | -0.0034165 | 0/5 | [-0.0044246, -0.0023612] | -0.0058332 | -0.0029076 | `a9411269bf016f65` |
| A2_ce | 0.6150448 | -0.0009284 | 1/5 | [-0.0017714, -0.0000984] | -0.0026166 | -0.0004195 | `64cec549090ce7ea` |
| A3_barajado | 0.6150721 | -0.0009011 | 0/5 | [-0.0016839, -0.0001106] | -0.0030541 | -0.0003922 | `5e2d474e677b6549` |

## Entrenamiento (ES estricto; el fold-val solo se uso para el OOF/reporte)

| Brazo | fold | epocas | mejor epoca | F1 interna mejor | F1 fold-val (reporte) | NaN | pasos/epoca |
|---|---|---|---|---|---|---|---|
| A1_surrogate | 0 | 60 | 59 | 0.604752 | 0.601762 | 0 | 15 |
| A1_surrogate | 1 | 60 | 54 | 0.597995 | 0.601337 | 0 | 15 |
| A1_surrogate | 2 | 60 | 58 | 0.601676 | 0.600833 | 0 | 15 |
| A1_surrogate | 3 | 60 | 54 | 0.594066 | 0.598483 | 0 | 15 |
| A1_surrogate | 4 | 60 | 59 | 0.602978 | 0.592112 | 0 | 15 |
| A2_ce | 0 | 53 | 44 | 0.606472 | 0.606284 | 0 | 15 |
| A2_ce | 1 | 56 | 47 | 0.604458 | 0.604648 | 0 | 15 |
| A2_ce | 2 | 60 | 52 | 0.609802 | 0.605974 | 0 | 15 |
| A2_ce | 3 | 42 | 33 | 0.598766 | 0.604655 | 0 | 15 |
| A2_ce | 4 | 53 | 44 | 0.611643 | 0.593436 | 0 | 15 |
| A3_barajado | 0 | 14 | 5 | 0.167796 | 0.158339 | 0 | 15 |
| A3_barajado | 1 | 22 | 13 | 0.168636 | 0.180882 | 0 | 15 |
| A3_barajado | 2 | 9 | 0 | 0.166565 | 0.111696 | 0 | 15 |
| A3_barajado | 3 | 16 | 7 | 0.168201 | 0.179138 | 0 | 15 |
| A3_barajado | 4 | 14 | 5 | 0.166661 | 0.212174 | 0 | 15 |

## Complementariedad standalone (argmax del OOF, sin pesos)

| Brazo | macro-F1 solo | diagonal 6x6 | rescate | dano | neto | rescate alta conf. | errores alta conf. campeon |
|---|---|---|---|---|---|---|---|
| A1_surrogate | 0.5989288 | True | 6041 | 8972 | -2931 | 170 | 9780 |
| A2_ce | 0.6028281 | True | 6718 | 8422 | -1704 | 144 | 9780 |
| A3_barajado | 0.1741619 | False | 12034 | 85993 | -73959 | 1744 | 9780 |

## Gates

| Gate | Resultado |
|---|---|
| baseline_ok | True |
| delta_A1_suficiente | False |
| folds_positivos_A1_suficiente | False |
| control_favorable_A1_sobre_A2 | False |
| ic95_A1_sin_cero | True |
| sin_leakage | True |

## Control negativo

- delta_int(A3) = -0.0009011 vs delta_int(A1) = -0.0034165: NO_ATRIBUIBLE a la perdida

## Veredicto

**NO_ATRIBUIBLE**: no puede pasar a segunda particion (control negativo barajado)

## Compliance

- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin leaderboard, sin submission.
- Early stopping estricto registrado: metrica interna 90/10; el fold-val jamas se uso para parar.
- Entrenamiento GPU bajo guardian D-007; sin fallback a CPU; un script nuevo; campeon V30 + plug-in intacto.
