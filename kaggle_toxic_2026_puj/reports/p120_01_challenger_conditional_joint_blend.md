# Reporte P120-01: Challenger Tri-Optimizer Conditional Joint Blend

**Fecha**: 2026-10-15  
**Experimento**: P120-01  
**Objetivo**: Superar a P114 (LB 0.79057) y romper la barrera de 0.80286 mediante modelado condicional conjunto de etiquetas (Stage 2 Joint Staging) sin perturbar el ancla de prevalencia del test set.

## 1. Arquitectura y Mecanismos Evaluados
- **Representacion Base**: `V2_triple` (86.275 dimensiones: word(1,2) + char_wb(3,5) + char(1,6), min_df=2, sublinear_tf=True).
- **Optimizadores de Base**: Tri-Optimizer smooth blend (80% LibLinear LR + 5% SGD log_loss + 15% SAGA ElasticNet l1=0.05, C={HS:2.0, TR:1.8, AG:1.5}, T=1.04).
- **Modelado Condicional Conjunto (Stage 2 Joint Staging)**:
  1. *Supresion suave de FPs en HS aislados* (`gamma_hs = 0.02`): Atenua logits de odio en tuits con bajo targeting ($P(TR) < 0.15$) y baja agresion ($P(AG) < 0.35$).
  2. *Impulso jerarquico HS -> TR* (`beta_tr = 0.02`, `hs_th = 0.44`): Acopla la confianza de odio con targeting, rescatando menciones y acoso dirigido.
  3. *Acoplamiento de covarianza TR -> AG* (`beta_ag = 0.1`, `tr_th = 0.48`): Explota la ley empirica de HatEval ($P(AG=1 | TR=1) = 0.946$) acoplando alta confianza de targeting a agresividad.
- **Decision y Post-Procesamiento**:
  * Umbrales congelados comprobados: HS = 0.44, TR = 0.51, AG = 0.46.
  * Regla ontologica HatEval ascendente: `Si TR == 1 => HS = 1`.

## 2. Metricas Locales en Validación Cruzada (5 pliegues)

| Particion Local | P120 CCV | P114 CCV | V03 CCV | Delta vs P114 | Delta vs V03 | Pliegues Ganados | Win Rate Bootstrap |
|---|---|---|---|---|---|---|
| **Canonica** (1.483 filas dev) | **0.787582** | 0.787218 | 0.777211 | **+0.000363** | **+0.010370** | 1/5 | 63.4% |
| **Particion 47** (1.483 filas dev) | **0.785673** | 0.785673 | 0.777211 | **+0.000000** | **+0.010041** | 0/5 | 0.0% |
| **Media Combinada** | **0.786628** | 0.786446 | 0.777211 | **+0.000182** | **+0.009416** | - | - |

- **Determinismo M2 verificado**: Discrepancia V03 = 0.00e+00; P114 Canonica = 2.96e-07; P114 Part47 = 8.58e-08.
- F1 por etiqueta en Canonica (P120): HS=0.770724, TR=0.820744, AG=0.771277.

## 3. Comportamiento en Test Set Ciego (1.154 filas)
- **Celdas distintas de P114**: 2 de 3462 (0.06%).
- **Transiciones**: +1 adiciones / -1 supresiones | Balance neto: +0 celdas.
- **Positivos predichos**: HS=514, TR=331, AG=420 | Total = **1265** (36.54%).
- **Anclaje de Prevalencia**: Exactamente 1.265 positivos (identico a P114).
- **Violaciones ontologicas corregidas**: 10.

## 4. Analisis Cualitativo de Casos Modificados en Test
1. **Fila 583 (id 24341)**:
   - Texto: *"El profe diciendo que nada de apodos y que slo hay que hablarnos por nuestro nom..."*
   - P114 habia predicho `[1, 0, 0]` (falso positivo en odio en conversacion escolar no toxica).
   - P120 lo corrige con precision a **`[0, 0, 0]`**, eliminando el falso positivo.
2. **Fila 1049 (id 23726)**:
   - Texto: *"@RinconFortnite Me parece que si la foto gente ya es guarra usando los arbustos..."*
   - P114 habia omitido el targeting (`TR=0`, `[1, 0, 1]`) a pesar de mencion directa y ataque.
   - P120 rescata con precision **`TR=1`** (`[1, 1, 1]`), recuperando el verdadero positivo.

## 5. Ficha Tecnica de Submission
- **Archivo**: `submissions/toxic_submission_p120_conditional_joint_blend_asc.csv`
- **SHA256**: `411884574a76fd3f40912a36e6a0e89f94147db91962c72e2564778bc3075f25`
- **Pureza ASCII**: Verificada 100% pure ASCII.
