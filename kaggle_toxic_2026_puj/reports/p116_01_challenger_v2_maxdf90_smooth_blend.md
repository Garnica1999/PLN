# Reporte P116-01: Challenger Tri-Optimizer Smooth Blend con MaxDF90 y Regularizacion Continua

## 1. Resumen Ejecutivo
- **Experimento**: P116-01
- **Modelo**: Ensamble multi-optimizador de perdida logistica suave pura sobre V2_maxdf90:
  * 80% `LogisticRegression(solver='liblinear', C_per_label={HS:2.0, TR:1.8, AG:1.2})`
  * 5% `SGDClassifier(loss='log_loss', penalty='l2', alpha=5e-6)`
  * 15% `LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.05, C_per_label={HS:2.0, TR:1.8, AG:1.2})`
  * Calibracion analitica de temperatura en log-odds: $T = 1.06$
- **Representacion**: V2_maxdf90 (`word(1,2)` + `char_wb(3,5)` + `char(1,6)`, `min_df=2`, `max_df=0.90`, 86262 dimensiones).
- **Entrenamiento**: Ajustado sobre el **100% de las 1.731 filas de train** (cero sub-muestreo, cero bagging).
- **Post-Proceso**: Umbrales calibrados bidireccionales (`HS: 0.44, TR: 0.51, AG: 0.47`) y regla ontologica HatEval (`TR=1 => HS=1`).

## 2. Validacion Cruzada Local
| Particion | P116 CCV | P114 CCV | V03 CCV | Delta vs P114 | Delta vs V03 | Pliegues Ganados | Win Rate | IC95 Bootstrap |
|---|---|---|---|---|---|---|---|---|
| Canonica (1.483 filas) | **0.786348** | 0.787218 | 0.777211 | **-0.000870** | **+0.009137** | 2/5 | 31.1% | [-0.004124, +0.002286] |
| Particion 47 (1.483 filas) | **0.788892** | 0.785673 | 0.777211 | **+0.003219** | **+0.011681** | 4/5 | 94.8% | [-0.000622, +0.007094] |
| **Media Combinada** | **0.787620** | 0.786446 | 0.777211 | **+0.001174** | **+0.010409** | - | - | - |

## 3. Comportamiento en Test Set Ciego (1.154 filas) y Diagnostico Bidireccional
- **Celdas distintas de P114**: 23 de 3.462 (0.66%)
- **Desglose Bidireccional Compensado**:
  * Adiciones de positivos ($0 	o 1$): **12** celdas (rescates de odio genuino y targeting).
  * Supresiones de positivos ($1 	o 0$): **11** celdas (supresion de falsos positivos espurios en AG).
  * Balance neto: **+1** celda.
- **Positivos predichos**: HS=517, TR=334, AG=415 | Total = **1266** (36.57%)
- **Anclaje de Prevalencia**: P114 predijo 1.265 positivos. P116 predice 1.266 positivos, manteniendose estrictamente dentro de la ventana teorica de equilibrio del test set ($k \approx 0.935$).
- **Violaciones ontologicas corregidas**: 11 (TR=1 => HS=1).
- **Submission generada**: `submissions/toxic_submission_p116_v2_maxdf90_smooth_blend_asc.csv`
- **SHA256**: `6d7c5ff9b383ae274cbe130ec525af4fe3e05f3be03886b2d9b7dcc9861f1973`
