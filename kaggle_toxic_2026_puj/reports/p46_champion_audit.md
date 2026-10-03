# P46 - Auditoria del campeon real

Generado por `experiments/v46_00_champion_audit.py` (V46-00 (P0 auditoria del campeon real y reconciliacion del leaderboard)).

Fecha 2026-09-28T00:17:17+00:00. **0 modelos, 0 submissions, 0 consultas a Kaggle.**

## 1. Veredicto

**CASE = B**

- `leaderboard_max_verified` = **0.78306** (`v03_lr_union_thr`)
- `baseline_operativo` = **V03**, sha256 `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`
- `CHAMPION_STATUS` = **INCONSISTENTE_AUTOCORREGIDO**
- Afirmaciones sin evidencia primaria: [0.78535, 0.7865]

UNSUPPORTED_CLAIM_PREDICTION: los valores de la tabla de candidatas son el resultado de la formula causal declarada en el propio fichero, no puntuaciones del host. Las mediciones reales de esas mismas submissions son inferiores y estan marcadas REJECTED en el mismo CHAMPION.md.

## 2. Las 22 submissions, por score de leaderboard

| # | submission | experimento | local | LB | dLB | artefacto | sha256 | estado |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | `v03_lr_union_thr` | MOD-04 / TH-01 | 0.79890 | **0.78306** | -0.01584 | si | `7d7478f271e6` | **BEST_VERIFIED_LB** |
| 2 | `v08_meta_v2` | E2-02-META | 0.78740 | **0.78163** | -0.00577 | si | `b59852f6e715` | REJECTED |
| 3 | `v11_candidate_c` | E2-11 | 0.77970 | **0.78150** | +0.00184 | si | `066d46ca12e9` | REJECTED |
| 4 | `v02_lr_union` | MOD-03b_BEST | 0.79270 | **0.77993** | -0.01277 | si | `228068c6ce04` | REJECTED |
| 5 | `v10_candidate_b` | E2-10 | 0.77930 | **0.77978** | +0.00047 | si | `92caa41e379f` | REJECTED |
| 6 | `56599802` | V23-01 | 0.79041 | **0.77938** | -0.00368 | si | `8f750e910767` | REJECTED |
| 7 | `v09_candidate_a` | E2-09 | 0.78090 | **0.77895** | -0.00199 | si | `8e5a95d3bd6a` | REJECTED |
| 8 | `v16_ch1_tasa_optima_transportada` | V16-01 |  | **0.77828** | -0.00478 | si | `a1b91806f720` | REJECTED |
| 9 | `v09_p5_c_por_etiqueta` | V9-05 | 0.77752 | **0.77781** | -0.00525 | si | `610abaa16d03` | REJECTED |
| 10 | `v16_ch8_C2p5` | V16-08 | 0.77675 | **0.77707** | -0.00599 | si | `2a533a442ab0` | REJECTED |
| 11 | `56597267` | V20-01 | 0.77800 | **0.77663** | -0.00643 | si | `cbb39d44be34` | REJECTED |
| 12 | `v05_optuna` | OPT-01 | 0.80390 | **0.77602** | -0.02788 | si | `a7f68a8679da` | REJECTED |
| 13 | `v04_ens04_mediana` | ENS-03 | 0.79850 | **0.77510** | -0.02340 | si | `c37cf1f5f915` | REJECTED |
| 14 | `v16_ch5_tasa_optima_absoluta` | V16-05 |  | **0.77495** | -0.00811 | si | `dac0d815465a` | REJECTED |
| 15 | `v08_p1_fold_ensemble` | V8-04 | 0.77721 | **0.77470** | -0.00836 | si | `3bbcc6bb1c7e` | REJECTED |
| 16 | `v16_ch7_C1p5` | V16-07 | 0.77665 | **0.77296** | -0.01010 | si | `197feee3a4b6` | REJECTED |
| 17 | `v06_robustez_c1` | S04 | 0.79290 | **0.77227** | -0.02063 | si | `fc943e688259` | REJECTED |
| 18 | `v12_p7_promedio_repeticiones` | V12-P7 | 0.78228 | **0.77122** | -0.01184 | si | `6157ef15e2da` | REJECTED |
| 19 | `v07_robustez_mindf5` | S06 | 0.79930 | **0.77034** | -0.02896 | si | `4ba99f4daa62` | REJECTED |
| 20 | `v16_ch9_mindf1` | V16-09 | 0.76553 | **0.76978** | -0.01328 | si | `827da9e7f695` | REJECTED |
| 21 | `v01_baseline_tfidf_lr` | B1 | 0.75778 | **0.75677** | -0.00101 | si | `0c02c8aad81a` | REJECTED |
| 22 | `v00_plantilla` | E00 |  | **0.51494** |  | si | `0dfb4d07dffc` | REJECTED |

Filas: 22 | duplicados: 0 | scores nulos: 0 | scores fuera de [0,1]: 0 | sin artefacto: 0.

Ninguna fila supera 0.78306.

## 3. Las afirmaciones de CHAMPION.md

La columna se titula **`Tasa | Celdas | LB combinado | sha256`**, y el propio fichero declara la formula:

> `LB combinado` = 0.78306 + 0.12011 x delta_tasa + aciertos_netos_oof x 2.804e-05, y para

| clave | linea | afirmado | medido | discrepancia |
|---|---:|---:|---:|---:|
| ch1 | 115 | **0.78535** | 0.77828 | +0.00707 |
| ch5 | 116 | **0.78650** | 0.77495 | +0.01155 |
| ch2 | 117 | **0.78418** | no enviada | n/d |
| ch4 | 118 | **0.78242** | no enviada | n/d |
| ch3 | 119 | **0.77431** | no enviada | n/d |

## 4. La columna es una prediccion, no una medicion

Recalculo la formula declarada con los valores que el propio texto da:

| clave | delta_tasa | netos OOF | recalculado | afirmado | desviacion | reproduce |
|---|---:|---:|---:|---:|---:|---|
| ch1 | 0.019 | 0 | 0.785342 | 0.78535 | 7.91e-06 | True |
| ch5 | 0.0335 | -21 | 0.786495 | 0.78650 | 5.15e-06 | True |

Todas reproducen: **True**. Los valores de la tabla de candidatas de CHAMPION.md son predicciones del modelo causal declarado en ese mismo fichero, no puntuaciones medidas por el host.

## 5. La contradiccion es interna a CHAMPION.md

El mismo fichero contiene las dos cifras. En la seccion *Estado al cierre de V16*:

| clave | linea | valor medido | estado |
|---|---:|---:|---|
| ch1 | 226 | **0.77828** | REJECTED |
| ch5 | 227 | **0.77495** | REJECTED |

Las mediciones reales de esas mismas submissions son inferiores a las affirmaciones, y el propio CHAMPION.md las marca REJECTED. No hay dos fuentes en conflicto: hay una prediction y una medicion, en el mismo fichero, y la medicion es la que vale.

## 6. Las cuatro magnitudes que no deben confundirse

| magnitud | valor | quien |
|---|---:|---|
| mejor corrected_cv utilizable | 0.7772114525165721 | OOF canonico de 1483 filas |
| mejor local del log | 0.80390 | `v05_optuna`, LB 0.77602 |
| mejor submission por LB | **0.78306** | `v03_lr_union_thr` |
| mejor modelo congelado | 0.7772114525 | V03, LB 0.78306 |

El mejor local no es el mejor LB, y el mejor corrected_cv del registro tampoco. Las tres son magnitudes distintas y la unica que decide el campeon es la tercera.

