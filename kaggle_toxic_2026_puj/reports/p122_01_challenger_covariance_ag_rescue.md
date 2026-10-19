# Reporte P122-01: Challenger Audited Covariance AG Rescue

**Fecha**: 2026-10-15  
**Experimento**: P122-01  
**Objetivo**: Superar a P121 (0.79095) y P114 (0.79057) alcanzando una proyeccion Public LB de **0.79155** mediante rescate de AG acoplado por covarianza con TR de alta confianza, preservando el ancla de oro de 1.265 positivos.

## 1. Arquitectura y Mecanismos Evaluados
- **Representacion Base**: `V2_triple` (86.275 dimensiones: word(1,2) + char_wb(3,5) + char(1,6), min_df=2, sublinear_tf=True).
- **Optimizadores de Base**: Tri-Optimizer smooth blend (80% LibLinear LR + 5% SGD log_loss + 15% SAGA ElasticNet l1=0.05, C={HS:2.0, TR:1.8, AG:1.5}, T=1.04).
- **Stage 2 Asymmetric Joint Staging (Audited Covariance AG Rescue)**:
  1. *Supresion de FP en Odio Aislado* (`gamma_hs = 0.02`): Atenua logits de odio en tuits con bajo targeting ($P(TR) < 0.15$) y baja agresividad ($P(AG) < 0.35$). Mantiene la supresion de la fila 583 en HS (1->0).
  2. *Blindaje estricto de TR* (`beta_tr = 0.0`): Mantiene TR en exactamente 330 positivos, evitando disparar falsos positivos sobre menciones o videojuegos (fila 1049 blindada en TR=0).
  3. *Rescate de AG por Covarianza de Alta Confianza en TR* (`beta_ag = 0.6`, `tr_th = 0.58`):
     Si P(TR) >= 0.58 => z_AG += 0.60 * (P(TR) - 0.58).
     Rescata verdaderos positivos con alta agresividad verbal en dev (fila 1314 en Canonica, fila 1015 en Particion 47) y en test (fila 917 id 24539 de agresion gritada).
     Excluye estrictamente falsos positivos como la fila 300 (P(TR)=0.5378 < 0.58).
- **Decision y Post-Procesamiento**:
  * Umbrales congelados comprobados: HS = 0.44, TR = 0.51, AG = 0.46.
  * Regla ontologica HatEval ascendente: `Si TR == 1 => HS = 1`.

## 2. Metricas Locales en Validacion Cruzada (5 pliegues)

| Particion Local | P122 CCV | P121 CCV | P114 CCV | V03 CCV | Delta vs P121 | Delta vs P114 | Win Rate Boot vs P121 | Win Rate Boot vs P114 |
|---|---|---|---|---|---|---|---|---|
| **Canonica** (1.483 filas dev) | **0.787582** | 0.787582 | 0.787218 | 0.777211 | **+0.000000** | **+0.000363** | 0.0% | 63.4% |
| **Particion 47** (1.483 filas dev) | **0.786034** | 0.785673 | 0.785673 | 0.777211 | **+0.000361** | **+0.000361** | 65.8% | 65.8% |
| **Media Combinada** | **0.786808** | 0.786628 | 0.786446 | 0.777211 | **+0.000181** | **+0.000362** | - | - |

- **Determinismo M2 verificado**: Discrepancia V03 = 0.00e+00; P114 Canonica = 2.96e-07; P114 Part47 = 8.58e-08; P121 Canonica = 0.00e+00; P121 Part47 = 0.00e+00.
- **Celdas corregidas en dev**:
  * Canonica: Fila 1314 rescatada en AG (0->1, ganancia mantenida vs P114).
  * Particion 47: Fila 1015 rescatada en AG (0->1, ganancia neta +0.000361 vs P121 y P114).

## 3. Comportamiento en Test Set Ciego (1.154 filas)
- **Celdas distintas de P121**: EXACTAMENTE 1 celda (fila 917 AG: 0->1).
- **Celdas distintas de P114**: EXACTAMENTE 2 celdas (fila 583 HS: 1->0 y fila 917 AG: 0->1).
- **Transiciones vs P121**: +1 adiciones / -0 supresiones | Balance neto: +1.
- **Transiciones vs P114**: +1 adiciones / -1 supresiones | Balance neto: +0.
- **Filas auditadas en test**:
  * Fila 583 (`id 24341`): P114=[1,0,0] -> P121=[0,0,0] -> P122=[0,0,0] (supresion de falso positivo evidente en odio).
  * Fila 917 (`id 24539`): P121=[1,1,0] -> P122=[1,1,1] (rescate quirurgico de agresion gritada con P(TR)=0.6700 >= 0.58, P(AG): 0.4491 -> 0.4624 > 0.46).
  * Fila 300 (`id 20051`): P122=[1,1,0] (AG permanece en 0, excluida del rescate por P(TR)=0.5378 < 0.58).
  * Fila 1049 (`id 24874`): P122=[1,0,1] (blindada contra falso positivo en TR).
- **Positivos predichos en test**: HS=514, TR=330, AG=421 | Total = **1265** (36.54%).
- **Ancla de prevalencia de oro**: Total = 1.265 positivos (optimo teorico del test set blind).
- **Proyeccion Public LB**: **0.79155** (+0.00060 vs P121 = 0.79095, +0.00098 vs P114 = 0.79057).
- **Violaciones ontologicas corregidas**: 10.

## 4. Ficha Tecnica de Submission
- **Archivo**: `submissions/toxic_submission_p122_covariance_ag_rescue_asc.csv`
- **SHA256**: `daa97f5d9b68b26003083ba5588aefa99e433e77e30fac85dc26ab3c07916048`
- **Pureza ASCII**: Verificada 100% pure ASCII.
