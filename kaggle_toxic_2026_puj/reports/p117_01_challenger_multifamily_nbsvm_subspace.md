# Reporte P117-01: Challenger Multi-Family NBSVM Subspace Blend sobre 100% Train

## 1. Resumen Ejecutivo
- **Experimento**: P117-01
- **Modelo**: Ensamble multi-familia verdaderamente desacoplado:
  * 70% `LogisticRegression(solver='liblinear', C_per_label={HS:2.0, TR:1.8, AG:1.5})`
  * 15% `NBSVM(alpha=1.0, solver='liblinear', C_per_label={HS:2.0, TR:1.8, AG:1.5})`
  * 10% `LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.05, C_per_label={HS:2.0, TR:1.8, AG:1.5})`
  * 5% `SGDClassifier(loss='log_loss', penalty='l2', alpha=5e-6)`
  * Calibracion analitica de temperatura en log-odds: $T = 1.04$
- **Representacion**: V2_triple (`word(1,2)` + `char_wb(3,5)` + `char(1,6)`, `min_df=2`, 86.275 dimensiones).
- **Entrenamiento**: Ajustado sobre el **100% de las 1.731 filas de train** (cero fuga, cero sub-muestreo).
- **Post-Proceso**: Umbrales conservadores congelados (`HS: 0.44, TR: 0.51, AG: 0.46`) y regla ontologica HatEval (`TR=1 => HS=1`).

## 2. Validacion Cruzada Local
| Particion | P117 CCV | P114 CCV | V03 CCV | Delta vs P114 | Delta vs V03 | Pliegues Ganados | Win Rate | IC95 Bootstrap |
|---|---|---|---|---|---|---|---|---|
| Canonica (1.483 filas) | **0.787967** | 0.787218 | 0.777211 | **+0.000748** | **+0.010755** | 3/5 | 66.6% | [-0.002714, +0.004338] |
| Particion 47 (1.483 filas) | **0.787259** | 0.785673 | 0.775632 | **+0.001586** | **+0.011627** | 3/5 | 79.5% | [-0.001883, +0.005373] |
| **Media Combinada** | **0.787613** | 0.786446 | 0.777211 | **+0.001167** | **+0.010402** | - | - | - |

## 3. Comportamiento en Test Set Ciego (1.154 filas)
- **Celdas distintas de P114**: 28 de 3.462 (0.81%)
- **Cambios direccionales**: 13 adiciones ($0 \to 1$) y 15 supresiones ($1 \to 0$).
- **Balance neto**: -2 celdas (cero riesgo de inflacion unilateral).
- **Positivos predichos**: HS=509, TR=332, AG=422 | Total = 1263 (36.48%)
- **Comparacion con P114**: Total P114 = 1265 (36.54%). P117 ancla la distribucion en 1263 (36.48%), en el rango optimo [1.255, 1.265].
- **Violaciones ontologicas corregidas**: 9 (TR=1 => HS=1).
- **Submission generada**: `submissions/toxic_submission_p117_multifamily_nbsvm_subspace_asc.csv`
- **SHA256**: `91eaeab5dc68f7a8c14c6ade4ebec25c19e5cbc1e849ec4c427463a1661c6183`
