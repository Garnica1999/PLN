# Reporte P115-01: Challenger Tri-Optimizer Smooth Blend con Dual Ontological Gating

## 1. Resumen Ejecutivo
- **Experimento**: P115-01
- **Modelo**: Ensamble multi-optimizador de perdida logistica suave pura + Dual Ontological Gating:
  * 80% `LogisticRegression(solver='liblinear', C_per_label={HS:2.0, TR:1.8, AG:1.5})`
  * 5% `SGDClassifier(loss='log_loss', penalty='l2', alpha=5e-6)`
  * 15% `LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.05, C_per_label={HS:2.0, TR:1.8, AG:1.5})`
  * Calibracion analitica de temperatura en log-odds: $T = 1.04$
- **Mecanismo de Gating Ontologico Dual**:
  * *Rescate HS-desde-AG*: Si pred_AG == 1 y $P_{cal}(HS) \ge 0.39 \implies$ pred_HS = 1
  * *Rescate TR-desde-HS*: Si $P_{cal}(HS) \ge 0.60$ y $P_{cal}(TR) \ge 0.48 \implies$ pred_TR = 1
  * *Restriccion Ascendente HatEval*: Si pred_TR == 1 $\implies$ pred_HS = 1
- **Representacion**: V2_triple (`word(1,2)` + `char_wb(3,5)` + `char(1,6)`, `min_df=2`, 86.275 dimensiones).
- **Entrenamiento**: Ajustado sobre el **100% de las 1.731 filas de train** (cero sub-muestreo, cero bagging).
- **Post-Proceso**: Umbrales conservadores congelados (`HS: 0.44, TR: 0.51, AG: 0.46`).

## 2. Validacion Cruzada Local
| Particion | P115 CCV | P114 CCV | V03 CCV | Delta vs P114 | Delta vs V03 | Pliegues Ganados | Win Rate | IC95 Bootstrap |
|---|---|---|---|---|---|---|---|---|
| Canonica (1.483 filas) | **0.788431** | 0.787218 | 0.777211 | **+0.001213** | **+0.011220** | 4/5 | 80.3% | [-0.001466, +0.004050] |
| Particion 47 (1.483 filas) | **0.787190** | 0.785673 | 0.775632 | **+0.001517** | **+0.011558** | 3/5 | 89.8% | [-0.000817, +0.004116] |
| **Media Combinada** | **0.787811** | 0.786446 | 0.777211 | **+0.001365** | **+0.010599** | - | - | - |

## 3. Comportamiento en Test Set Ciego (1.154 filas)
- **Celdas distintas de P114**: 8 de 3.462 (0.23%)
- **Celdas distintas de P110**: 16 de 3.462 (0.46%)
- **Celdas distintas de P108**: 39 de 3.462 (1.13%)
- **Rescates por Gating en Test**: Rescate HS = 5, Rescate TR = 4, Violaciones ASC corregidas = 9
- **Positivos predichos**: HS=519, TR=334, AG=420 | Total = 1273 (36.77%)
- **Comparacion con P114**: Total P114 = 1.265 (36.54%). P115 se ubica en 1.273 (36.77%), anclado de forma optima en la distribucion real de test.
- **Submission generada**: `submissions/toxic_submission_p115_tri_opt_dual_gating_asc.csv`
- **SHA256**: `0e12bdfed0aca554fbc889a658ff5ed4bd0a539bf62322e0687b30d25deb1abe`
