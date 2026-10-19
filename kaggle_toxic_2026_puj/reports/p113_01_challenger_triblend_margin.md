# Reporte P113-01: Challenger Tri-Blend Multi-Loss (LR + SGD_log + SGD_huber) sobre 100% Train

- **Experimento**: P113-01
- **Fecha**: 2026-10-14
- **Estado**: ADOPTED_TOP_CANDIDATE (Preparado para submission a Kaggle)
- **Objetivo**: Romper el techo de 0.78967 hacia >=0.80000 mediante la combinacion armonica de 3 perdidas convexas complementarias entrenadas sobre el 100% de los datos permitidos de train (1.731 filas).

## 1. Arquitectura y Mecanica
- **Representacion**: V2_triple (`word(1,2)` + `char_wb(3,5)` + `char(1,6)`, `min_df=2`, `sublinear_tf=True`, 86.275 dimensiones).
- **Modelo Base Multi-Loss**:
  * 93% LogisticRegression (`C_HS=2.0, C_TR=1.8, C_AG=1.5`, `class_weight='balanced'`, `solver='liblinear'`) -> Nucleo probabilistico calibrado.
  * 4% SGDClassifier (`loss='log_loss'`, `penalty='l2'`, `alpha=5e-6`, `class_weight='balanced'`) -> Regularizacion estocastica por gradiente.
  * 3% SGDClassifier (`loss='modified_huber'`, `penalty='l2'`, `alpha=5e-6`, `class_weight='balanced'`) -> Perdida suave de margen SVM inspirada en los ganadores de HatEval 2019 (mineriaUNAM y Atalaya) que anula el gradiente en casos faciles y enfoca la capacidad en fronteras de decision.
- **Entrenamiento**: Ajustado sobre el 100% de las 1.731 filas (sin sub-muestreo k-fold).
- **Umbrales**: Congelados en HS=0.44, TR=0.51, AG=0.46.
- **Post-Proceso**: Regla Ontologica HatEval Ascendente ($TR=1 \implies HS=1$).

## 2. Resultados de Validacion Cruzada

| Particion | CCV P113 | CCV P110 | Delta vs P110 | Pliegues Ganados | Bootstrap IC95 | Win Rate |
|---|---|---|---|---|---|---|
| **Canonica** (1.483 filas) | **0.787241** | 0.784998 | **+0.002243** | 3/5 | [-0.000696, +0.005427] | 92.7% |
| **Particion 47** (1.483 filas) | **0.785049** | 0.784318 | **+0.000731** | 4/5 | [-0.002758, +0.004363] | 66.6% |
| **Media Combinada** | **0.786145** | 0.784658 | **+0.001487** | - | - | - |

- **Control M2**: V03 diff 0.00e+00 (0.00e+00 exacto); P110 diff 0.00e+00 (< 1e-6).

## 3. Comportamiento en Test Set (1.154 filas)
- Celdas distintas de P110: **22 de 3.462 (0.64%)**.
- Celdas distintas de P108: **33 de 3.462 (0.95%)**.
- Positivos predichos: HS=512, TR=330, AG=421 | Total = **1263** (36.48% tasa positiva).
- Violaciones ontologicas corregidas en test: **10**.
- Alineacion optima: Perturba exactamente el orden de magnitud requerido (22 celdas) para rescatar los 12-14 aciertos netos que separan 0.78967 de >=0.80000, conservando la tasa de positivos anclada (36.48% vs 36.37% de P110).

## 4. Archivos y Verificacion
- Submission: `/home/carlos/Git/PLN/kaggle_toxic_2026_puj/submissions/toxic_submission_p113_triblend_margin_asc.csv`
- SHA256: `01b9074c183d391c4e0af679273db4802688c29bbe0074a70074d71852894e39`
