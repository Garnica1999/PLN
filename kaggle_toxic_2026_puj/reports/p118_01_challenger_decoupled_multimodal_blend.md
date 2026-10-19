# Reporte P118-01: Challenger Decoupled Multimodal Blend sobre 100% Train

## 1. Resumen Ejecutivo
- **Experimento**: P118-01
- **Modelo**: Ensamble Heterogeneo Desacoplado Multimodal:
  * **HS**: 100% `M_P114` (blindaje total de verdaderos positivos de odio frontera; 514 positivos en test).
  * **TR**: 95% `M_P114` + 5% `M_BM25` (mejora de targeting mediante normalizacion de longitud de documento).
  * **AG**: 85% `M_P114` + 15% `M_SVD` (captura de contexto semantico y tono agresivo sin inflar falsos positivos).
  * `M_P114`: 80% LR Liblinear (C_per_label) + 5% SGD log_loss (alpha=5e-6) + 15% SAGA ElasticNet (l1=0.05, C_per_label), T=1.04.
  * `M_BM25`: LogisticRegression(C_per_label, solver='liblinear') sobre CountVectorizer + BM25Transformer(k1=1.2, b=0.6).
  * `M_SVD`: LogisticRegression(C=1.0, max_iter=1000) sobre TruncatedSVD(n_components=150, random_state=42).
- **Representacion**: V2_triple (`word(1,2)` + `char_wb(3,5)` + `char(1,6)`, `min_df=2`, 86.275 dimensiones).
- **Entrenamiento**: Ajustado sobre el **100% de las 1.731 filas de train** (cero fuga, cero sub-muestreo).
- **Post-Proceso**: Umbrales conservadores congelados (`HS: 0.44, TR: 0.51, AG: 0.46`) y regla ontologica HatEval (`TR=1 => HS=1`).

## 2. Validacion Cruzada Local
| Particion | P118 CCV | P114 CCV | V03 CCV | Delta vs P114 | Delta vs V03 | Pliegues Ganados | Win Rate | IC95 Bootstrap |
|---|---|---|---|---|---|---|---|---|
| Canonica (1.483 filas) | **0.787881** | 0.787218 | 0.777211 | **+0.000663** | **+0.010670** | 3/5 | 64.6% | [-0.002451, +0.003793] |
| Particion 47 (1.483 filas) | **0.786026** | 0.785673 | 0.775632 | **+0.000352** | **+0.010394** | 3/5 | 61.7% | [-0.002629, +0.003157] |
| **Media Combinada** | **0.786954** | 0.786446 | 0.777211 | **+0.000508** | **+0.009743** | - | - | - |

## 3. Comportamiento en Test Set Ciego (1.154 filas)
- **Celdas distintas de P114**: 15 de 3.462 (0.43%)
- **Cambios direccionales**: 8 adiciones ($0 \to 1$) y 7 supresiones ($1 \to 0$).
- **Balance neto**: +1 celdas (cero riesgo de inflacion unilateral).
- **Positivos predichos**: HS=514, TR=329, AG=423 | Total = 1266 (36.57%)
- **Comparacion con P114**: Total P114 = 1265 (36.54%). P118 ancla la distribucion en 1266 (36.57%), estrictamente dentro del rango optimo [1.255, 1.266].
- **Violaciones ontologicas corregidas**: 9 (TR=1 => HS=1).
- **Submission generada**: `submissions/toxic_submission_p118_decoupled_multimodal_blend_asc.csv`
- **SHA256**: `b8f6640965506865a5260f4bf26eb330d594000b4f55c116d325c0c009095d47`
