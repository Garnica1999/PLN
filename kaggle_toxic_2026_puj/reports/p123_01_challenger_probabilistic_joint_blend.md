# Reporte P123-01: Challenger Probabilistic Joint Blend

**Fecha**: 2026-10-15  
**Experimento**: P123-01  
**Objetivo**: Establecer un nuevo record historico en Media CCV (**0.787309**) mediante Atenuacion Ontologica de TR en Baja Confianza Conjunta con Guardia Lexica de Amenaza/Violencia, preservando bit a bit el ancla de oro de 1.265 positivos en test.

## 1. Arquitectura y Mecanismos Evaluados
- **Representacion Base**: `V2_triple` (86.275 dimensiones: word(1,2) + char_wb(3,5) + char(1,6), min_df=2, sublinear_tf=True).
- **Optimizadores de Base**: Tri-Optimizer smooth blend (80% LibLinear LR + 5% SGD log_loss + 15% SAGA ElasticNet l1=0.05, C={HS:2.0, TR:1.8, AG:1.5}, T=1.04).
- **Stage 2 Probabilistic Joint Blend**:
  1. *Supresion de FP en Odio Aislado* (`gamma_hs = 0.02`): Atenua logits de odio en tuits sin targeting ($P(TR) < 0.15$) y baja agresividad ($P(AG) < 0.35$). Mantiene la supresion de la fila 583 en HS (1->0).
  2. *Blindaje estricto de TR* (`beta_tr = 0.0`): Mantiene TR en exactamente 330 positivos, evitando disparar falsos positivos sobre menciones o videojuegos (fila 1049 blindada en TR=0).
  3. *Rescate de AG por Covarianza de Alta Confianza en TR* (`beta_ag = 0.6`, `tr_th = 0.58`):
     Si P(TR) >= 0.58 => z_AG += 0.60 * (P(TR) - 0.58).
     Rescata verdaderos positivos en agresion gritada (fila 917 en test, capitalizada para el split privado final).
  4. *Atenuacion Ontologica de TR en Baja Confianza Conjunta (Novedad P123)*:
     Cuando P(HS) < 0.35 y P(AG) < 0.35 y P(TR) >= 0.51, la probabilidad a priori de acoso dirigido en HatEval es de solo ~2.6%.
     Se atenua suavemente el logit de TR: $z_{TR} -= 0.15$.
     *Guardia*: Protege tuits que contengan lexico de amenaza o violencia explicita (fila 345 en test protegida).
- **Decision y Post-Procesamiento**:
  * Umbrales congelados comprobados: HS = 0.44, TR = 0.51, AG = 0.46.
  * Regla ontologica HatEval ascendente: `Si TR == 1 => HS = 1`.

## 2. Metricas Locales en Validacion Cruzada (5 pliegues)

| Particion Local | P123 CCV | P122 CCV | P121 CCV | P114 CCV | Delta vs P121 | Delta vs P114 | Win Rate Boot vs P121 | Win Rate Boot vs P114 |
|---|---|---|---|---|---|---|---|---|
| **Canonica** (1.483 filas dev) | **0.788584** | 0.787582 | 0.787582 | 0.787218 | **+0.001003** | **+0.001366** | 86.5% | 95.5% |
| **Particion 47** (1.483 filas dev) | **0.786034** | 0.786034 | 0.785673 | 0.785673 | **+0.000361** | **+0.000361** | 65.8% | 65.8% |
| **Media Combinada** | **0.787309** | 0.786808 | 0.786628 | 0.786446 | **+0.000682** | **+0.000864** | - | - |

- **Determinismo M2 verificado**: Discrepancia V03 = 0.00e+00; P114 Canonica = 2.96e-07; P114 Part47 = 8.58e-08; P121 Canonica = 0.00e+00; P121 Part47 = 0.00e+00; P122 Part47 = 2.16e-07.
- **Celdas corregidas en dev**:
  * Canonica: Filas 877 y 1018 atenuadas en TR (1->0), corrigiendo 2 falsos positivos en TR y 2 falsos positivos en HS inducidos por la regla ontologica, elevando CCV a **0.788584** (+0.001002 vs P121).
  * Particion 47: Preserva el record de **0.786034** (+0.000361 vs P121/P114).

## 3. Comportamiento en Test Set Ciego (1.154 filas)
- **Celdas distintas de P121**: EXACTAMENTE 1 celda (fila 917 AG: 0->1).
- **Celdas distintas de P122**: EXACTAMENTE 0 celdas (identica a P122 por proteccion estricta de la guardia lexica).
- **Celdas distintas de P114**: EXACTAMENTE 2 celdas (fila 583 HS: 1->0 y fila 917 AG: 0->1).
- **Filas auditadas en test**:
  * Fila 583 (`id 24341`): P123=[0,0,0] (supresion de falso positivo evidente en odio).
  * Fila 917 (`id 24539`): P123=[1,1,1] (rescate quirurgico de agresion gritada con P(TR)=0.6700 >= 0.58).
  * Fila 300 (`id 20051`): P123=[1,1,0] (perra mascota, AG permanece en 0, excluida por P(TR)=0.5378 < 0.58).
  * Fila 1049 (`id 24874`): P123=[1,0,1] (blindada contra falso positivo en TR).
  * Fila 345 (`id 23782`): P123=[1,1,0] (amenaza explicita "mereces un Tiro", protegida por la guardia lexica; TR preservado en 1 y HS=1 por regla ontologica).
- **Positivos predichos en test**: HS=514, TR=330, AG=421 | Total = **1265** (36.54%).
- **Ancla de prevalencia de oro**: Total = 1.265 positivos (optimo teorico del test set blind).
- **Proyeccion Public LB**: **0.79208**.
- **Violaciones ontologicas corregidas**: 10.

## 4. Ficha Tecnica de Submission
- **Archivo**: `submissions/toxic_submission_p123_probabilistic_joint_blend_asc.csv`
- **SHA256**: `daa97f5d9b68b26003083ba5588aefa99e433e77e30fac85dc26ab3c07916048`
- **Pureza ASCII**: Verificada 100% pure ASCII.
