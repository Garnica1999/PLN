# P54 - Atlas semantico de los errores no rescatables

Generado por `experiments/v54_00_atlas_no_rescatable.py` (V54-00 (atlas semantico de los errores no rescatables)).

Fecha 2026-09-28T02:35:28+00:00. **0 modelos, 0 submissions, 0 consultas a Kaggle, 0 uso de holdout o test.**

## 1. Poblaciones, recalculadas desde cero

| poblacion | celdas |
|---|---:|
| A = V03 correcto | 3692 |
| B = V03 mal y algun canal acierta | 715 |
| **C = V03 mal y ningun canal acierta** | **42** |
| errores totales | 757 |

Verificacion contra P49 (757 / 715 / 42): **True**.

> **Advertencia de soporte.** Las 42 celdas proceden de **33 tweets distintos**, luego el soporte efectivo es 33 observaciones, no 42. las 42 celdas NO son 42 observaciones independientes: proceden de 33 tweets distintos.

### Direccion del error en C

| direccion | n |
|---|---:|
| FP (V03=1, y=0) | 22 |
| FN (V03=0, y=1) | 20 |

## 2. Rescate frente a no rescate, que es la comparacion central

| atributo | n en C | pct C | pct B | lift C/B | p | lift C/A | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mi_mas_insulto` | 4 | 9.52 pct | 3.64 pct | 2.62 | 0.0656 | 7.33 | 0.0022 |
| `marcador_contraste` | 5 | 11.90 pct | 5.87 pct | 2.03 | 0.0983 | 2.11 | 0.0858 |
| `grupo_marginal` | 9 | 21.43 pct | 14.13 pct | 1.52 | 0.1295 | 2.68 | 0.0052 |
| `negacion` | 17 | 40.48 pct | 29.09 pct | 1.39 | 0.0758 | 1.81 | 0.0064 |
| `tiene_mencion_inicial` | 20 | 47.62 pct | 34.41 pct | 1.38 | 0.0528 | 1.14 | 0.2769 |
| `tiene_insulto` | 28 | 66.67 pct | 50.35 pct | 1.32 | 0.0242 | 1.25 | 0.0566 |
| `tiene_mayusculas_sostenidas` | 39 | 92.86 pct | 94.55 pct | 0.98 | 0.8053 | 0.98 | 0.7921 |
| `tiene_leet` | 9 | 21.43 pct | 22.38 pct | 0.96 | 0.6182 | 0.97 | 0.5953 |
| `tiene_elongacion` | 7 | 16.67 pct | 22.80 pct | 0.73 | 0.8736 | 0.94 | 0.6350 |
| `negacion_y_contraste` | 1 | 2.38 pct | 3.50 pct | 0.68 | 0.7757 | 0.76 | 0.7353 |
| `rt_presente` | 0 | 0.00 pct | 0.00 pct | nan | nan | nan | nan |

**El resultado que corrige una impresion mia.** Una lectura cualitativa de las 42 celdas sugiere que el residuo es sobre todo `insulto lexico sin escalada`. Al medirlo, el lexico de insulto esta en el **66,67 por ciento** de C pero tambien en el **53,36 por ciento** de A, luego el enriquecimiento es de apenas **1,25x** y no es discriminante. La impresion cualitativa era correcta sobre las celdas de AG y equivocada sobre el conjunto.

## 3. Margen y alta confianza

| bin de margen | A | B | C | pct de C |
|---|---:|---:|---:|---:|
| [0,0.10) | 368 | 280 | 2 | 4.8 pct |
| [0.10,0.20) | 541 | 203 | 4 | 9.5 pct |
| [0.20,1] | 2783 | 232 | 36 | 85.7 pct |

Errores de alta confianza (probabilidad a distancia >= 0.7 del corte): **36 de 42**, y P53 ya habia mostrado que ahi la decision esta bloqueada por la propia confianza de V03.

| label | errores | de alta confianza | pct | no rescatables | rescatables |
|---|---:|---:|---:|---:|---:|
| HS | 316 | 97 | 30.7 pct | 7 | 90 |
| TR | 164 | 72 | 43.9 pct | 11 | 61 |
| AG | 277 | 99 | 35.7 pct | 18 | 81 |

## 4. Reconciliacion con los 15 hallazgos del EDA original

Se reportan **dos** enrichments y no solo el de error, porque `error enrichment` solo dice donde V03 falla, mientras que `rescue enrichment` dice donde hay alternativa y `non-rescue enrichment` dice donde no la hay.

| hallazgo | lift C/B (rescate) | p | lift C/A (no rescate) | p | n en C | antecedente |
|---|---:|---:|---:|---:|---:|---|
| negacion | 1.55 | 0.0129 | 2.04 | 0.0003 | 21 | V19, V32 y P49 |
| mencion_inicial | 1.38 | 0.0528 | 1.14 | 0.2769 | 20 | V14 y P50 |
| insulto | 1.32 | 0.0242 | 1.25 | 0.0566 | 28 | V06, V17 y V18 |
| segunda_persona | 1.29 | 0.0718 | 1.20 | 0.1553 | 23 | V01 y P50 |
| propn | 1.05 | 0.3034 | 1.08 | 0.1751 | 37 | V01 y P50 |
| colectivo | 0.98 | 0.5874 | 1.21 | 0.2717 | 12 | V05, V20 y P50: analizado y cerrado |
| risas | 0.97 | 0.6159 | 0.82 | 0.7106 | 2 | P49: enriquecido en errores |
| leet | 0.96 | 0.6182 | 0.97 | 0.5953 | 9 | V01 y P50 |
| atribucion | 0.87 | 0.6846 | 1.08 | 0.5311 | 3 | V01 y P50 |
| imperativos | 0.75 | 0.7993 | 0.63 | 0.8976 | 4 | V01 |
| elongacion | 0.73 | 0.8736 | 0.94 | 0.6350 | 7 | P47 y V06-08 |
| urls | 0.63 | 0.9751 | 0.71 | 0.9274 | 9 | V31 lo analizo como estructura |
| cita | 0.39 | 0.9306 | 0.64 | 0.7981 | 1 | V19, V32 y P50: analizado y cerrado |
| hashtags | 0.22 | 0.9922 | 0.25 | 0.9853 | 1 | P49 lo encontro enriquecido en errores |
| rt | nan | nan | nan | nan | 0 | V29 |

## 5. Las 42 celdas, completas y auditables

Las 42 celdas estan en `reports/p54_nonrescuable_atlas.csv` con `tweet_id`, `label`, `y_true`, `v03`, `proba`, `margen`, `canales_que_acierten`, los atributos y el texto. La columna `error_type_retrospectivo` esta marcada como **retrospectiva**: el modelo ha visto texto y etiqueta a la vez, luego se usa para describir y **nunca** para inferencia.

| n | id | label | y | v03 | margen | canales OK | tipo retrospectivo |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | 20678 | AG | 1 | 0 | 0.120 | 0 | FN:narrativa_sobre_grupo_marginal |
| 2 | 21320 | AG | 0 | 1 | 0.370 | 0 | FP:insulto_lexico_sin_escalada |
| 3 | 21360 | AG | 1 | 0 | 0.262 | 0 | FN:narrativa_sobre_grupo_marginal |
| 4 | 21410 | AG | 1 | 0 | 0.398 | 0 | FN:insulto_con_agresividad_no_activada |
| 5 | 21497 | AG | 0 | 1 | 0.335 | 0 | FP:insulto_lexico_sin_escalada |
| 6 | 21498 | AG | 1 | 0 | 0.251 | 0 | FN:insulto_con_agresividad_no_activada |
| 7 | 21892 | AG | 0 | 1 | 0.436 | 0 | FP:insulto_lexico_sin_escalada |
| 8 | 22273 | AG | 0 | 1 | 0.372 | 0 | FP:insulto_lexico_sin_escalada |
| 9 | 22524 | AG | 0 | 1 | 0.463 | 0 | FP:insulto_lexico_sin_escalada |
| 10 | 22628 | AG | 0 | 1 | 0.465 | 0 | FP:insulto_lexico_sin_escalada |
| 11 | 23076 | AG | 0 | 1 | 0.384 | 0 | FP:insulto_lexico_sin_escalada |
| 12 | 23268 | AG | 1 | 0 | 0.360 | 0 | FN:agresividad_o_target_no_lexicalizada |
| 13 | 23414 | AG | 0 | 1 | 0.229 | 0 | FP:insulto_lexico_sin_escalada |
| 14 | 23587 | AG | 0 | 1 | 0.083 | 0 | FP:error_no_lexicalizado |
| 15 | 23628 | AG | 1 | 0 | 0.394 | 0 | FN:agresividad_o_target_no_lexicalizada |
| 16 | 23832 | AG | 1 | 0 | 0.396 | 0 | FN:narrativa_sobre_grupo_marginal |
| 17 | 24006 | AG | 0 | 1 | 0.345 | 0 | FP:insulto_lexico_sin_escalada |
| 18 | 24299 | AG | 1 | 0 | 0.283 | 0 | FN:insulto_con_agresividad_no_activada |
| 19 | 24317 | AG | 1 | 0 | 0.233 | 0 | FN:narrativa_sobre_grupo_marginal |
| 20 | 24946 | AG | 1 | 0 | 0.270 | 0 | FN:narrativa_sobre_grupo_marginal |
| 21 | 24962 | AG | 1 | 0 | 0.086 | 0 | FN:agresividad_o_target_no_lexicalizada |
| 22 | 20504 | HS | 0 | 1 | 0.277 | 0 | FP:insulto_lexico_sin_escalada |
| 23 | 20693 | HS | 1 | 0 | 0.151 | 0 | FN:insulto_con_agresividad_no_activada |
| 24 | 21497 | HS | 0 | 1 | 0.340 | 0 | FP:insulto_lexico_sin_escalada |
| 25 | 21835 | HS | 0 | 1 | 0.217 | 0 | FP:insulto_lexico_sin_escalada |
| 26 | 23985 | HS | 0 | 1 | 0.231 | 0 | FP:insulto_lexico_sin_escalada |
| 27 | 24006 | HS | 0 | 1 | 0.369 | 0 | FP:insulto_lexico_sin_escalada |
| 28 | 24078 | HS | 1 | 0 | 0.200 | 0 | FN:narrativa_sobre_grupo_marginal |
| 29 | 24655 | HS | 1 | 0 | 0.230 | 0 | FN:narrativa_sobre_grupo_marginal |
| 30 | 24829 | HS | 0 | 1 | 0.458 | 0 | FP:insulto_lexico_sin_escalada |
| 31 | 20295 | TR | 1 | 0 | 0.384 | 0 | FN:narrativa_sobre_grupo_marginal |
| 32 | 20504 | TR | 0 | 1 | 0.381 | 0 | FP:insulto_lexico_sin_escalada |
| 33 | 21497 | TR | 0 | 1 | 0.419 | 0 | FP:insulto_lexico_sin_escalada |
| 34 | 21498 | TR | 1 | 0 | 0.316 | 0 | FN:insulto_con_agresividad_no_activada |
| 35 | 21755 | TR | 1 | 0 | 0.291 | 0 | FN:insulto_con_agresividad_no_activada |
| 36 | 21822 | TR | 1 | 0 | 0.352 | 0 | FN:discurso_citado_o_atribuido |
| 37 | 23076 | TR | 0 | 1 | 0.428 | 0 | FP:insulto_lexico_sin_escalada |
| 38 | 23628 | TR | 1 | 0 | 0.399 | 0 | FN:agresividad_o_target_no_lexicalizada |
| 39 | 23832 | TR | 1 | 0 | 0.436 | 0 | FN:narrativa_sobre_grupo_marginal |
| 40 | 24006 | TR | 0 | 1 | 0.435 | 0 | FP:insulto_lexico_sin_escalada |
| 41 | 24232 | TR | 0 | 1 | 0.366 | 0 | FP:insulto_lexico_sin_escalada |
| 42 | 24865 | TR | 0 | 1 | 0.162 | 0 | FP:insulto_lexico_sin_escalada |

## 6. Taxonomia retrospectiva de C

| categoria | n | pct de C | lift C/B | fuerza de evidencia |
|---|---:|---:|---:|---|
| FP:insulto_lexico_sin_escalada | 21 | 50.0 pct | 0.99 | d razonable |
| FN:narrativa_sobre_grupo_marginal | 9 | 21.4 pct | 0.43 | d moderada |
| FN:insulto_con_agresividad_no_activada | 6 | 14.3 pct | 0.28 | d moderada |
| FN:agresividad_o_target_no_lexicalizada | 4 | 9.5 pct | 0.19 | d debil |
| FN:discurso_citado_o_atribuido | 1 | 2.4 pct | 0.05 | d debil |
| FP:error_no_lexicalizado | 1 | 2.4 pct | 0.05 | d debil |

## 7. Candidatos a hipotesis y por que no se abren

| atributo | n en C | lift C/B | p | celdas recuperables si se arreglara |
|---|---:|---:|---:|---:|
| `tiene_insulto` | 28 | 1.32 | 0.0242 | 28 |

Un enriquecimiento fuerte con soporte de un digito **no es una hipotesis operable**: 26,75 celdas valen aproximadamente 0,003 de macro-F1, luego 4 celdas valen unos 0,0005. Ademas, por el resultado ya establecido en P51 a P53, un cambio que acierte esas celdas tambien dana otras, y el neto tiende a ser negativo. Ningun candidato cumple los cinco criterios de la directiva a la vez.

## 8. Decision

**`P54_RESEARCH_GAP = NONE`.**

Lo que la evidencia sostiene, y es mas especifico que un cierre generico:

1. **El residuo no es falta de informacion en el texto.** El 66,67 por ciento de las celdas de C ya contienen el lexico de insulto, y solo el 2,85 por ciento de los errores de HS son no rescatables frente al 7,58 por ciento de AG. El informacion esta presente y el modelo la ve.
2. **El residuo es en su mayor parte un desacuerdo de frontera de etiqueta.** Los tipos retrospectivos dominantes son insulto lexico con etiqueta 0 y narrativa sobre grupo con etiqueta 1, y en ambos casos el texto es inequivoco y la interpretacion del anotador es una convencion que el modelo no puede derivar del texto.
3. **No queda un fenomeno semantico nuevo con soporte.** Los dos unicos atributos con enrichment significado tienen 4 y un numero pequeno de celdas, y su valor absoluto en macro-F1 es despreciable.

**Lo que no se afirma.** No se afirma que sea imposible mejorar. Se afirma que en este residuo no hay un mecanismo semantico con soporte suficiente y valor material, y que la via de datos observables esta cerrada por P52 y P53 con una cota, no por falta de intento.

