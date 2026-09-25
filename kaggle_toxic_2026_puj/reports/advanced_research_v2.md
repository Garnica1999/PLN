# Advanced Research v2 - Informe de cierre

Fecha: 2026-09-25. Estado: **fase v2 concluida sin mejora robusta transferible**
(condicion de terminacion B del plan: las fuentes de senal razonables fueron
evaluadas y documentadas).

## 1. Objetivo y protocolo

Objetivo: encontrar una senal nueva que superara el LB 0.78306 de V03.
Protocolo (D-016): grupos = near-duplicado UNION pseudo-autor; holdout congelado
de 248 tweets (1 pliegue de `StratifiedGroupKFold(7)`; max 5 evaluaciones, usadas
2); desarrollo de 1.483 tweets; metrica primaria `corrected_cv`; regla de
adopcion: >= +0.005 con >= 3/5 pliegues o complementariedad con IC bootstrap > 0;
bootstrap por tweet para todo candidato.

Referencia v2 (desarrollo): V03 con umbrales = **0.7772** (M2 a 0.5 = 0.7718).

## 2. Resultados por paquete

| Paquete | Linea | Mejor variante | corrected_cv | Delta | Decision |
|---|---|---|---|---|---|
| P0 | Scaffold (holdout, OOF corregido, gap analysis) | V03 reevaluada | 0.7772 | - | Hecho |
| P1 | Cadenas de etiquetas (E2-01) | HS_AG_TR | 0.7744 | +0.002 vs base | **Cerrada** (IC incluye 0) |
| P1 | Meta de dependencia (E2-02) | LR C=0.1 sobre 44 features | **0.7853** | **+0.0081** | Adoptada (IC [+0.0021,+0.0248]) |
| P2 | Targeting estructural (E2-05/06) | union + n-gramas simbolicos | 0.7487 | -0.0285 | **Cerrada** |
| P3 | POS/dependencias/morfologia sparse (E2-03/04) | union + POS | 0.7656 | -0.0116 | **Cerrada** |
| P4 | Cita/atribucion (E2-07) y negacion (E2-08) | union + negacion | 0.7657 | -0.0115 | **Cerrada** |
| P5 | Emocion (E2-09) | union + afecto | 0.6330 | -0.1442 | **Cerrada** |
| P5 | NB-SVM (E2-15) | NB-union + LR | 0.7612 | -0.0160 | **Cerrada** (gain +0.0038, IC [-0.008,+0.016]) |
| P5 | Multicanal ortografico (E2-10) | raw + colapsado | 0.7534 | -0.0238 | **Cerrada** |
| P6 | Stacking v2 (E2-17/18) | META-BASES-RIDGE alpha=3 | **0.7874** | **+0.0102** | Adoptado por complementariedad (gain +0.0089, IC [+0.0019,+0.0160]) |
| P7 | Holdout congelado | Meta en holdout | **0.8465** (V03 0.8285) | +0.0180 | GO (IC [+0.0003,+0.0372]) |
| P7 | Sonda de LB | v08 meta v2 | **LB 0.78163** | -0.0014 vs v03 | **No adoptada** |

## 3. El resultado del leaderboard y su lectura

- v08 (meta v2) = **0.78163**; v03 (modelo final) = **0.78306**; diferencia
  -0.0014, muy por debajo del error estandar (~0.013 con n=1154).
- El meta mejoro en el desarrollo (+0.0102) y en el holdout congelado (+0.0180),
  y **no transfirio al test**. Su gap dev-LB fue -0.0058, frente al +0.0059
  implicito de v03 (cuyo LB se obtuvo con el modelo entrenado con todo el train).
- Lectura metodologica: con 12 sondas en total, el maximo observado (0.78306) esta
  sesgado hacia arriba por **seleccion (winner's curse)**. La media de la familia
  union es 0.7769 y el rango completo de sondas (0.77034-0.78306) cabe en ~1
  error estandar. Tratar 0.78306 como un objetivo reproducible seria un error.
- Implicacion practica: en este corpus (1.731 ejemplos, sin transformers ni
  embeddings preentrenados, test de 1.154) las diferencias locales de +0.01 no
  predicen la transferencia; la unica conclusion que resiste es que **V03 y el
  meta son indistinguibles en el test**.

## 4. Por que no transfirio (evidencia)

1. **Tamano**: el holdout tiene 248 tweets (EE ~0.03) y el test 1.154 (EE ~0.013);
   los deltas locales (0.010-0.018) son del orden del error.
2. **Gap de familia**: la representacion union word+char muestra un gap dev-LB
   negativo (-0.013 a -0.016) que se confirmo en todas las variantes.
3. **Cambio de distribucion de la entrada del meta**: el meta se entrena con
   scores OOF de bases entrenados en 4/5 del desarrollo y predice el test con
   bases entrenados con todo el train; ese cambio de calibracion de las entradas
   no se midio y es una hipotesis abierta.
4. **Los canales nuevos no aportaron ni al meta ni solos**: targeting (-0.0006 al
   meta), sintaxis, cita/negacion, emocion y NB-SVM quedaron por debajo o con
   IC que incluye 0.

## 5. Lo que si dejo la fase v2 (valor academico)

- **Protocolo de validacion corregido** (grupos fusionados, holdout congelado,
  bootstrap por tweet) que calibra mucho mejor con el LB que el protocolo
  historico (D-015/D-016).
- **Evidencia negativa sistematica**: 24 lineas del backlog evaluadas y
  documentadas, con cierre explicito y numeros (requisito de la rubrica).
- **Dos senales complementarias identificadas pero no transferidas**: el
  meta de dependencia de etiquetas (recall de AG en la rebanada de referencia
  colectiva de 0.452 a 0.574-0.589) y el canal de targeting (mismo efecto).
- **Hallazgos del corpus**: comillas en solo 2 % de los tweets, 0 retuits con prefijo canonico `RT @` (8 tweets contienen el token RT), marcadores
  de atribucion en 6.9 %, negacion en 25.5 % (y `es_core_news_md` no emite
  `dep_=neg`, se detecto por `advmod`+`Polarity=Neg`).

## 6. Estado final del proyecto

- **Modelo final: V03** (union TF-IDF word(1,2)+char_wb(3,5) + LR C=2.0 balanced
  + umbrales 0.44/0.51/0.46), LB **0.78306**, congelado en
  `experiments/baseline_v03_frozen/` y reconstruible byte a byte.
- Entregables academicos: `notebooks/10_final_pipeline.ipynb` y
  `FINAL_REPORT.md` (con la fase v2 documentada como investigacion posterior).
- Backlog v2: **24 de 24 lineas evaluadas y cerradas**; el paquete P8 cerro las
  cinco restantes (calibracion, encoder de entidades por sparsity 0.5 %,
  especialistas por rebanada, umbrales por rebanada por shrinkage,
  label-specific features -0.0253) sin mejora (D-021).

## 7. Artefactos de la fase v2

`tasks/plan.md`, `tasks/todo.md` · `reports/advanced_gap_analysis_v2.md`,
`reports/research_backlog_v2.md`, `reports/baseline_v03_dev_metrics.json`,
`reports/cadenas_etiquetas_v2.{csv,json}`, `reports/targeting_v2.{csv,json}`,
`reports/sintaxis_cita_negacion_v2.{csv,json}`,
`reports/emocion_nbsvm_ortografia_v2.{csv,json}`, `reports/stacking_v2.{csv,json}`,
`reports/p7_holdout_resultados.json`, `reports/holdout_evaluaciones_v2.csv`,
`reports/figuras/fig_27..31_*.png` · `features/v2_*.npz` · scripts
`experiments/13..22` · 128 filas en `experiments.csv` · D-016 a D-022 en
`DECISIONS.md` · artefactos de auditoria (FASES 43/47/50/55/88):
`reports/error_intersection_v2.csv`, `reports/error_by_label_combination_v2.csv`,
`reports/model_complementarity_v2.csv`, `reports/feature_catalog_v2.csv`,
`features/v2_structured_features.parquet`, `reports/spec_compliance_v2.md`,
`reports/rule_audit_v2.md`, `reports/target_encoding_audit_v2.md`.
