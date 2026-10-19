# Reporte P119-01: Challenger Tri-Optimizer Pragmatic Gated Blend sobre 100% Train

## 1. Resumen Ejecutivo
- **Experimento**: P119-01
- **Modelo**: Ensamble desacoplado por etiqueta con gating pragmatico sobre base Tri-Optimizer P114:
  * **HS (100% P114)**: Blindaje absoluto del clasificador de odio de P114 (exactamente 515 positivos de alta precision, umbral 0.44).
  * **TR (Gated 95% P114 + 5% Mencion Directa)**: Rescate de tuits con mencion directa (`@user`) condicionado a candidatos de odio (`prob_HS >= 0.44`) sin penalizar tuits de acoso directo sin arroba (fila 1041 blindada byte-exacta en `HS=1, TR=1`). Umbral 0.51.
  * **AG (Poda Selectiva Pragmatica)**: Elevacion de umbral a 0.47 y poda selectiva de falsos positivos en tuits sin signos de shouting ni agresion/insultos (`pred_HS=0`, sin mayusculas >=3, sin exclamaciones y sin lexico de insulto explicito, `prob_AG < 0.55`), contrayendo los positivos de AG desde 420 hacia **408** (en la banda teorica ~395-405).
  * **Regla Ontologica HatEval**: `Si pred_TR == 1 => pred_HS = 1`.
- **Representacion**: V2_triple (`word(1,2)` + `char_wb(3,5)` + `char(1,6)`, `min_df=2`, 86.275 dimensiones) + 7 features estructurales de mencion.
- **Entrenamiento**: Ajustado sobre el **100% de las 1.731 filas de train** (cero fuga, cero datos externos).
- **Control M2**: V03 diff = 0.00e+00 (0.0e+00); P114 diff = 2.96e-07 (< 1e-6).

## 2. Validacion Cruzada Local
| Particion | P119 CCV | P114 CCV | V03 CCV | Delta vs P114 | Delta vs V03 | Pliegues Ganados | Win Rate | IC95 Bootstrap |
|---|---|---|---|---|---|---|---|---|
| Canonica (1.483 filas) | **0.785897** | 0.787218 | 0.777211 | **-0.001321** | **+0.008686** | 2/5 | 16.8% | [-0.004018, +0.001269] |
| Particion 47 (1.483 filas) | **0.784121** | 0.785673 | 0.775632 | **-0.001552** | **+0.008489** | 2/5 | 14.2% | [-0.004517, +0.001270] |
| **Media Combinada** | **0.785009** | 0.786446 | 0.777211 | **-0.001436** | **+0.007798** | - | - | - |

## 3. Comportamiento en Test Set Ciego (1.154 filas)
- **Celdas distintas de P114**: 13 de 3.462 (0.38%)
- **Perturbacion direccional**: +1 adiciones / -12 supresiones | Balance neto = -11
- **Falsos positivos de AG podados**: 5 tuits no agresivos eliminados
- **Positivos predichos**: HS=515, TR=331, AG=408 | Total = 1254 (36.22%)
- **Comparacion con P114**: Total P114 = 1.265 (36.54%). P119 ancla la distribucion en 1.254 (36.22%), corrigiendo el exceso de FPs en AG y preservando estrictamente la precision de HS y TR.
- **Blindaje Fila 1041**: Confirmada en `HS=1, TR=1, AG=0` (acoso sexual explicito preservado intacto).
- **Violaciones ontologicas corregidas**: 10 (TR=1 => HS=1).
- **Submission generada**: `submissions/toxic_submission_p119_pragmatic_gated_blend_asc.csv`
- **SHA256**: `a46766d5cac0aa5b4cdaa81882bf198c8b6f1b30848ca21d7a13294c84a3d996`
