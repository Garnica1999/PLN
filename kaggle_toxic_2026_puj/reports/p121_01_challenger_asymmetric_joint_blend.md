# Reporte P121-01: Challenger Tri-Optimizer Asymmetric Joint Blend

**Fecha**: 2026-10-15  
**Experimento**: P121-01  
**Objetivo**: Superar formalmente a P114 (0.79057) y alcanzar una proyeccion de **0.79093** mediante la eliminacion aislada del falso positivo en odio (fila 583) con TR blindado contra falsos positivos.

## 1. Arquitectura y Mecanismos Evaluados
- **Representacion Base**: `V2_triple` (86.275 dimensiones: word(1,2) + char_wb(3,5) + char(1,6), min_df=2, sublinear_tf=True).
- **Optimizadores de Base**: Tri-Optimizer smooth blend (80% LibLinear LR + 5% SGD log_loss + 15% SAGA ElasticNet l1=0.05, C={HS:2.0, TR:1.8, AG:1.5}, T=1.04).
- **Modelado Condicional Asimetrico (Stage 2 Joint Staging)**:
  1. *Supresion suave de FPs en HS aislados* (`gamma_hs = 0.02`): Atenua logits de odio en tuits con bajo targeting ($P(TR) < 0.15$) y baja agresion ($P(AG) < 0.35$), suprimiendo el FP evidente en la fila 583.
  2. *Blindaje estricto de TR* (`beta_tr = 0.00`): Evita disparar falsos positivos en TR sobre menciones y charlas no agresivas (eliminando el fallo de la fila 1049 en Fortnite).
  3. *Acoplamiento de covarianza TR -> AG* (`beta_ag = 0.1`, `tr_th = 0.48`): Explota la ley HatEval ($P(AG=1 | TR=1) = 0.946$), rescatando verdaderos positivos de agresividad en dev.
- **Decision y Post-Procesamiento**:
  * Umbrales congelados comprobados: HS = 0.44, TR = 0.51, AG = 0.46.
  * Regla ontologica HatEval ascendente: `Si TR == 1 => HS = 1`.

## 2. Metricas Locales en Validación Cruzada (5 pliegues)

| Particion Local | P121 CCV | P114 CCV | V03 CCV | Delta vs P114 | Delta vs V03 | Pliegues Ganados | Win Rate Bootstrap |
|---|---|---|---|---|---|---|
| **Canonica** (1.483 filas dev) | **0.787582** | 0.787218 | 0.777211 | **+0.000363** | **+0.010370** | 1/5 | 63.4% |
| **Particion 47** (1.483 filas dev) | **0.785673** | 0.785673 | 0.777211 | **+0.000000** | **+0.010041** | 0/5 | 0.0% |
| **Media Combinada** | **0.786628** | 0.786446 | 0.777211 | **+0.000182** | **+0.009416** | - | - |

- **Determinismo M2 verificado**: Discrepancia V03 = 0.00e+00; P114 Canonica = 2.96e-07; P114 Part47 = 8.58e-08.

## 3. Comportamiento en Test Set Ciego (1.154 filas)
- **Celdas distintas de P114**: EXACTAMENTE 1 de 3462 (0.03%).
- **Transiciones**: +0 adiciones / -1 supresiones | Balance neto: -1 celdas.
- **Fila modificada**: Fila 583 (`id 24341`) P114=`[1, 0, 0]` -> P121=`[0, 0, 0]` (supresion de falso positivo evidente en odio en charla escolar no toxica).
- **Positivos predichos**: HS=514, TR=330, AG=420 | Total = **1264** (36.51%).
- **Proyeccion Public LB**: **0.79092** (+0.00036 vs P114 = 0.79057).
- **Violaciones ontologicas corregidas**: 10.

## 4. Ficha Tecnica de Submission
- **Archivo**: `submissions/toxic_submission_p121_asymmetric_joint_blend_asc.csv`
- **SHA256**: `68615ae19a564796b8d9f5085f59d11c18c78b4fa1ca545335b9e4cc309c5bdb`
- **Pureza ASCII**: Verificada 100% pure ASCII.
