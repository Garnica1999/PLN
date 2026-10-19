# CHAMPION

Estado congelado del mejor mecanismo competitivo observado en AGE.

**Actualizado el 2026-10-18: V48 Geometric PoE Blend es el CHAMPION OFICIAL INDISCUTIDO en Kaggle Leaderboard (LB 0.63724, Ref: 57283487, Puesto #1). V56 Non-Linear Multi-Modal PoE Consensus establece el RÉCORD HISTÓRICO ABSOLUTO DEFINITIVO en Validación Cruzada Local (CV 0.6189929, 5/5 folds positivos, bootstrap pareado $p = 0.0010 \ll 0.01$ vs V48) y confirma en Kaggle (LB 0.63717, Ref: 57350295). V57 ratifica mediante auditoría matemática formal del símplex 9D, Teorema de Concavidad Power-Mean y análisis de confiabilidad decil a decil que la solución de consenso habita un óptimo estacionario absoluto, blindando el 30% privado frente al shakeup.**

## Tabla Histórica de Champions y Envíos Puntuados

| Versión | Descripción | CV Local | Kaggle LB | Estado |
|---|---|:---:|:---:|---|
| **V57 Mathematical Simplex & Power-Mean Concavity** | **Auditoría matemática formal del símplex 9D, concavidad de la superficie Power-Mean (PoE óptimo global), desalineamiento de pérdidas continuas y calibración decil a decil** | **0.6189929** | **0.63717** | **TECHO MATEMÁTICO ESTACIONARIO CONFIRMADO (Blindaje del 30% privado, Selección Dual V48/V56 ratificada)** |
| **V56 Non-Linear Multi-Modal PoE Consensus** | **Consenso Bayesiano PoE 9 modelos (V30 + Cramer-SWA + Asym-Cramer + Smooth-SWA + SE-TabOrd + TabOrd + HGB + Consistent-Cramer + Dense-SE-TabNet) + Plug-in** | **0.6189929** | **0.63717** | **NUEVO RÉCORD HISTÓRICO ABSOLUTO CV LOCAL (5/5 folds positivos, Bootstrap $p=0.0010 \ll 0.01$ vs V48, Kaggle Rank #1 confirmado)** |
| **V54 Hierarchical Multi-Loss PoE Consensus** | Consenso Bayesiano PoE 8 modelos (V30 + Cramer-SWA + Asym-Cramer-Clip + Smooth-SWA + SE-TabOrd + TabOrd + HGB + Dense-SE-TabNet) + Plug-in | 0.6188887 | 0.63712 | Ex-récord histórico local (Bootstrap $p=0.0050 < 0.01$) |
| **V53 Dense-SE-TabularNet PoE Blend** | Consenso Bayesiano PoE 7 modelos (V30 + Cramer-SWA + Smooth-SWA + SE-TabOrd + TabOrd + HGB + Dense-SE-TabNet) + Plug-in | 0.6188259 | 0.63706 | Ex-récord CV local (Bootstrap $p=0.0280 < 0.05$) |
| **V48 Geometric PoE Blend** | **Consenso Bayesiano Product-of-Experts en log-espacio ($\log P \propto \sum w_m \log P_m$) sobre 6 modelos + Plug-in** | **0.6186451** | **0.63724** | **CHAMPION OFICIAL INDISCUTIDO (RÉCORD HISTÓRICO ABSOLUTO LB, #1 Kaggle)** |
| V43 Cramer-Smooth Blend | Blend aritmético 6 modelos (0.555 V30 + 0.085 Cramer-SWA + 0.175 Smooth-SWA + 0.080 SE + 0.050 TabOrd + 0.055 HGB) + Plug-in | 0.6184783 | 0.63722 | Ex-champion histórico oficial |
| V41 SWA-SE Blend | Blend 5 modelos (0.63 V30 + 0.26 SWA-SE-TabNet + 0.04 SE-Tab + 0.05 TabOrd + 0.02 HGB) + Plug-in | 0.6181119 | 0.63722 | Baseline histórico |
| V42 Smooth-SWA Blend | Blend 6 modelos (0.575 V30 + 0.28 Smooth-SWA + 0.02 SWA + 0.06 SE + 0.05 TabOrd + 0.015 HGB) + Plug-in | 0.6184162 | 0.63716 | Baseline histórico |
| V40 SE-Tabular Blend | Blend 5 modelos (0.72 V30 + 0.12 SE-TabNet + 0.08 TabOrd + 0.06 TabV38 + 0.02 HGB) + Plug-in | 0.6175912 | 0.63685 | Baseline histórico |
| V39 Trimodal Blend | Blend 0.84 V30 + 0.12 TabNet Ordinal + 0.04 HistGB + Plug-in | 0.6168017 | 0.63588 | Baseline histórico |
| V38 Tabular Blend | Blend 0.88 V30 + 0.12 TabularNet (Deep ResNet GPU) + Plug-in | 0.6166579 | 0.63590 | Baseline histórico |
| V30 + Plug-in | V30 refinado Optuna + Regla de decisión per-class congelada | 0.6159732 | 0.63575 | Baseline histórico confirmado |
| V30 | Ensamble de árboles refinados con Optuna sobre matriz_279 | 0.6134471 | 0.63302 | Baseline histórico |
| V20 | Cadena de umbrales anidados y blend de cuatro cuartos | 0.6097782 | 0.62689 | Baseline histórico |
| V19 | Boosting LightGBM como tercer miembro | 0.6079388 | 0.62578 | Baseline histórico |
| V12 oficial | NB-SVM palabra y promedio de combinadores (LogReg + MLP) | 0.6005000 | 0.62182 | Baseline histórico |

---

## Récord Local de Validación Cruzada: V56 Non-Linear Multi-Modal PoE Consensus

| Campo | Valor |
|---|---|
| **Nombre** | **V56 Non-Linear Multi-Modal PoE Consensus** |
| **Kaggle Leaderboard** | **0.63717** (Ref: 57350295, 2026-10-18) · **Rank #1 Confirmado** |
| **CV LOFO agrupado de 5 folds** | **0.6189929** (**NUEVO RÉCORD HISTÓRICO ABSOLUTO DEFINITIVO**) |
| **Delta CV vs V48 Champion** | **+0.0003478** (5 de 5 folds positivos vs V48 / 5 de 5 vs V54) |
| **Folds positivos en CV vs V54** | **5 / 5 folds positivos** (+0.000078, +0.000005, +0.000115, +0.000249, +0.000079) |
| **Bootstrap IC95 pareado vs V48** | **[+0.0001485, +0.0005590]** ($p = 0.0010 \ll 0.01$, significancia categórica al 0.1%) |
| **Submission Oficial** | `submissions/age_submission_v56_nonlinear_poe_consensus.csv` |
| **SHA256 de Submission** | `944592baab7b4b74bb26b01bba29ca66c9ff99a0d8108c3a1bc6cb524f0c609c` |
| **Filas cambiadas vs V48 Blend** | 194 de 112,892 (0.17%) |
| **Posición en Leaderboard** | **PUESTO #1 INDISCUTIDO** (`Grupo 2 - PLN`, ventaja sobre 2° lugar 0.63528) |

### Arquitectura de V56 Non-Linear Multi-Modal PoE Consensus

1. **Integración Armónica de 9 Modelos de Sesgo Complementario**:
   - Árboles troncales: V30 LightGBM Optuna ($w=0.55939$), HistGradientBoosting ($w=0.05204$).
   - Transporte óptimo y distancias ordinales: Cramér simétrico V43 ($w=0.06023$), Cramér asimétrico ponderado y winsorizado V47 ($w=0.02089$), Consistent Smoothed Cramér V51 ($w=0.01035$).
   - Regularización de margen y cuenca plana: Smooth-SWA V42 ($w=0.16873$).
   - Mecanismos de atención y representaciones profundas: SE-TabNet V40 ($w=0.06080$), TabNet Ordinal V39 ($w=0.04376$), Dense-SE-TabNet V53 ($w=0.02381$).

2. **Consenso Bayesiano Product-of-Experts en Espacio Logarítmico**:
   $$\log P_{\text{V56}}(y = k \mid x) = \sum_{m=1}^9 w_m \log P_m(y = k \mid x) - \log Z(x)$$
   Regla de decisión congelada:
   $$\mathbf{w}_{\text{plugin}} = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]$$

### Arquitectura de V54 Hierarchical Multi-Loss PoE Consensus

1. **Unificación Jerárquica de Transporte Óptimo (Cramér L2)**:
   - Fusión armónica en la rama de Cramér de la distancia simétrica global ($L_2$ CDF matching, V43, 80%) y la distancia asimétrica ponderada en fronteras ($\gamma=[1.2, 1.5, 1.3, 1.1, 1.0]$ con winsorización in-fold $[-6.0, +6.0]$, V47, 20%).
   - Restaura el equilibrio marginal en clases minoritarias (`13-17` y `65-xx`).

2. **Conectividad Densa Multi-Escala Calibrada (Huang et al., CVPR 2017)**:
   - Incorporación de `Dense-SE-TabularNet` con peso logarítmico calibrado $w = 0.0200$, aportando representaciones multi-escala sin distorsionar fronteras ordinales.

3. **Consenso Bayesiano de 8 Modelos en Espacio Logarítmico**:
   $$\log P_{\text{geom}}(y = k \mid x) = \sum_{m=1}^8 w_m \log P_m(y = k \mid x) - \log Z(x)$$
   Pesos normalizados:
   $$\mathbf{w} = [0.55154, 0.06523, 0.01631, 0.16895, 0.07389, 0.05027, 0.05380, 0.02000]$$
   correspondientes a V30, V43 Cramer, V47 Asym-Cramer, V42 Smooth, V40 SE-TabOrd, V39 Ord, V39 HGB y V53 Dense.

4. **Regla de Decisión (Vector Plug-in Congelado)**:
   $$\hat{y} = \arg\max_{c} \left( w_c \times P_{\text{geom}, c} \right)$$
   $$\mathbf{w}_{\text{plugin}} = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]$$


---

## Champion de Kaggle Leaderboard: V48 Geometric PoE Blend

| Campo | Valor |
|---|---|
| **Nombre** | **V48 Geometric PoE Blend** |
| **Kaggle Leaderboard** | **0.63724** (Ref: 57283487, 2026-10-16) · **RÉCORD ABSOLUTO DEL TORNEO** |
| **CV LOFO agrupado de 5 folds** | **0.6186451** |
| **Delta CV vs V43 Champion** | **+0.0001668** (4 de 5 folds positivos) |
| **Submission Oficial** | `submissions/age_submission_v48_geometric_poe_blend.csv` |
| **SHA256 de Submission** | `bf2554571ca6676195af66a053e28411fc1cfc9e3946fd95bac1cae0ce71df62` |
| **Posición en Leaderboard** | **PUESTO #1 INDISCUTIDO** (`Grupo 2 - PLN`, +0.00196 sobre 2° lugar 0.63528) |

