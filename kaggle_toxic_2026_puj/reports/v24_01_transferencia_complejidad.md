# V24-01 (P1): modelo de transferencia de complejidad

Generado por `experiments/v24_01_modelo_transferencia_complejidad.py` (V24-01 (P1: modelo de transferencia de complejidad sobre el historial real)).

**Auditoria diagnostica. Ningun modelo de prediccion para la competicion. Ninguna
submission.**

## 1. Por que esto existe

V23 envio la candidata con mejor evidencia interna de la historia del proyecto
(`corrected_cv` 0.790410, 5 de 5 pliegues, McNemar p = 0.0452) y el leaderboard bajo
0.00368. Si esa divergencia se puede detectar antes de enviar, las proximas fases
tienen un filtro que no tenian.

## 2. El historial real de leaderboard

22 submissions registradas, de las cuales 19 tienen `local_f1` numerico.

| submission | local | LB | dOOF | dLB |
|---|---:|---:|---:|---:|
| toxic_submission_v01_baseline_tfidf_lr.csv | 0.757780 | 0.75677 | -0.019431 | -0.02629 |
| toxic_submission_v02_lr_union.csv | 0.792700 | 0.77993 | +0.015489 | -0.00313 |
| toxic_submission_v03_lr_union_thr.csv | 0.798900 | 0.78306 | +0.021689 | +0.00000 |
| toxic_submission_v04_ens04_mediana.csv | 0.798500 | 0.77510 | +0.021289 | -0.00796 |
| toxic_submission_v05_optuna.csv | 0.803900 | 0.77602 | +0.026689 | -0.00704 |
| toxic_submission_v06_robustez_c1.csv | 0.792900 | 0.77227 | +0.015689 | -0.01079 |
| toxic_submission_v07_robustez_mindf5.csv | 0.799300 | 0.77034 | +0.022089 | -0.01272 |
| toxic_submission_v08_meta_v2.csv | 0.787400 | 0.78163 | +0.010189 | -0.00143 |
| toxic_submission_v09_candidate_a.csv | 0.780900 | 0.77895 | +0.003689 | -0.00411 |
| toxic_submission_v10_candidate_b.csv | 0.779300 | 0.77978 | +0.002089 | -0.00328 |
| toxic_submission_v11_candidate_c.csv | 0.779700 | 0.78150 | +0.002489 | -0.00156 |
| toxic_submission_v08_p1_fold_ensemble.csv | 0.777211 | 0.77470 | -0.000000 | -0.00836 |
| toxic_submission_v09_p5_c_por_etiqueta.csv | 0.777518 | 0.77781 | +0.000307 | -0.00525 |
| toxic_submission_v12_p7_promedio_repeticione | 0.782277 | 0.77122 | +0.005066 | -0.01184 |
| toxic_submission_v16_ch7_C1p5.csv | 0.776653 | 0.77296 | -0.000558 | -0.01010 |
| toxic_submission_v16_ch8_C2p5.csv | 0.776752 | 0.77707 | -0.000459 | -0.00599 |
| toxic_submission_v16_ch9_mindf1.csv | 0.765532 | 0.76978 | -0.011679 | -0.01328 |
| toxic_submission_v20_01_regla_hs_interaccion | 0.777998 | 0.77663 | +0.000787 | -0.00643 |
| toxic_submission_v23_01_svd_rbf_sigmoid.csv | 0.790410 | 0.77938 | +0.013199 | -0.00368 |

## 3. Los indicadores de complejidad, recalculados con un unico protocolo

Todos con los pliegues canonicos, las 1.483 filas de desarrollo y los umbrales
congelados de V03. Por eso son comparables entre si, cosa que los numeros de los
informes historicos no son.

| modelo | corrected_cv | f1 train | gap train-OOF | frac. no nula | sparsity | norma coef. | celdas | rescates | danos | neto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V03 LR L2 C=2.0 | 0.777211 | 0.9614 | +0.1842 | 1.0000 | 0.0000 | 22.8007 | 0 | 0 | 0 | +0 |
| LR L1 C=4.0 | 0.783642 | 0.9292 | +0.1455 | 0.0131 | 0.9869 | 65.8876 | 413 | 208 | 205 | +3 |
| LR L2 C=20.0 | 0.771989 | 1.0000 | +0.2280 | 1.0000 | 0.0000 | 54.6480 | 142 | 69 | 73 | -4 |
| LinearSVC C=1.0 | 0.771508 | 0.9968 | +0.2253 | 0.9838 | 0.0162 | 15.7537 | 276 | 105 | 171 | -66 |
| RidgeClassifier a=3.0 | 0.773301 | 0.7651 | -0.0082 | 1.0000 | 0.0000 | 8.2778 | 306 | 113 | 193 | -80 |
| V23 svd_rbf_sigmoid | 0.790409 | n/d | +0.0000 | n/d | 0.0000 | n/d | 240 | 136 | 104 | n/d |

La fila de V23 tiene `fraccion_no_nula` y `norma_coeficientes` en nan porque no
son recalculables sin reentrenar el kernel, y su indicador propio es la fraccion de
vectores de soporte: **70.9%** de las muestras.

## 4. La relacion con el delta LB

- Pearson(delta OOF, delta LB) = **+0.4876**, p = 0.0342, n = 19.
- Ajuste: `delta LB = -0.00920 + 0.2448 * delta OOF`.
- Para empatar el LB haria falta un delta OOF de **+0.03756**.

**Aviso que invalida parte de la tabla**: `local_f1` no significa lo mismo en todas las fases: V16 optimizo la tasa predicha, luego su dOOF no es comparable con el corrected_cv de V03. Los puntos de V16 estan contaminados y sesgan la correlacion hacia el eje.

## 5. Lo que este analisis NO puede concluir

1. **No produce un modelo predictivo del `delta LB`.** Con 22 puntos, y 4 de ellos
   contaminados por el esquema de V16, ajustar una relacion con varias variables
   seria sobreajustar el propio historial. La regresion simple se reporta como
   *descripcion*, no como predictor.
2. **Los modelos recalculados aqui no se enviaron nunca**, asi que no tienen
   `delta LB` propio. Emparejar el indicador de un modelo con el `delta LB` de otra
   submission seria un error de emparejamiento, y no se ha hecho.
3. **La correlacion con 22 observaciones no distingue nada.** Su intervalo de
   confianza es enorme y cualquier afirmacion causal seria falsa.
4. Los indicadores de V23 son los unicos emparejados con un `delta LB` medido, de
   modo que la direccion complejidad alta -> peor transferencia se apoya en un
   solo caso, que es exactamente el caso que motiva el analisis.

Completado en 5.7 s.

