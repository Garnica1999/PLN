# Reporte P114-01: Challenger Tri-Optimizer Smooth Blend sobre 100% Train

## 1. Resumen Ejecutivo
- **Experimento**: P114-01
- **Modelo**: Ensamble multi-optimizador de perdida logistica suave pura:
  * 80% `LogisticRegression(solver='liblinear', C_per_label={HS:2.0, TR:1.8, AG:1.5})`
  * 5% `SGDClassifier(loss='log_loss', penalty='l2', alpha=5e-6)`
  * 15% `LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.05, C_per_label={HS:2.0, TR:1.8, AG:1.5})`
  * Calibracion analitica de temperatura en log-odds: $T = 1.04$
- **Representacion**: V2_triple (`word(1,2)` + `char_wb(3,5)` + `char(1,6)`, `min_df=2`, 86.275 dimensiones).
- **Entrenamiento**: Ajustado sobre el **100% de las 1.731 filas de train** (cero sub-muestreo, cero bagging).
- **Post-Proceso**: Umbrales conservadores congelados (`HS: 0.44, TR: 0.51, AG: 0.46`) y regla ontologica HatEval (`TR=1 => HS=1`).

## 2. Validacion Cruzada Local
| Particion | P114 CCV | P110 CCV | V03 CCV | Delta vs P110 | Delta vs V03 | Pliegues Ganados | Win Rate | IC95 Bootstrap |
|---|---|---|---|---|---|---|---|---|
| Canonica (1.483 filas) | **0.787218** | 0.784998 | 0.777211 | **+0.002220** | **+0.010007** | 4/5 | 98.4% | [+0.000197, +0.004311] |
| Particion 47 (1.483 filas) | **0.785673** | 0.784318 | 0.775632 | **+0.001356** | **+0.010041** | 3/5 | 85.8% | [-0.000971, +0.004134] |
| **Media Combinada** | **0.786446** | 0.784658 | 0.777211 | **+0.001788** | **+0.009235** | - | - | - |

## 3. Comportamiento en Test Set Ciego (1.154 filas)
- **Celdas distintas de P110**: 8 de 3.462 (0.23%)
- **Celdas distintas de P108**: 33 de 3.462 (0.95%)
- **Positivos predichos**: HS=515, TR=330, AG=420 | Total = 1265 (36.54%)
- **Comparacion con P110**: Total P110 = 1.259 (36.37%). P114 ancla la distribucion en 1.265 (36.54%), perfectamente dentro de la banda optima.
- **Violaciones ontologicas corregidas**: 10 (TR=1 => HS=1).
- **Submission generada**: `submissions/toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv`
- **SHA256**: `c31ee6f2147c1c057095bfda5840e7eac48cd425ede1806d22f0c77f4d4d4274`
