# P61 - EDA de las probabilidades OOF canonicas de V03

Generado por `experiments/p61_01_eda_probabilidades.py`. Fecha 2026-10-20T02:50:27+00:00.

> **Esta EDA es DIAGNOSTICA.** No se usa para ajustar thresholds, seleccionar modelos ni
> optimizar hiperparametros sobre este mismo OOF. Si de ella saliera una hipotesis
> nueva, habria que crear esa hipotesis con su propio protocolo anidado.

## La pregunta que esta EDA responde

No `que tan buenas son las probabilidades`, sino:

> que probabilidad produjo V03 finalmente para cada observacion, donde estaba respecto
> al threshold, y que decision final produjo esa probabilidad?

La tabla de esa pregunta es `reports/p61_final_probability_table.csv`, con una fila por
celda etiqueta-tweet, y las columnas `probabilidad_oof_final`, `umbral`,
`decision_final`, `correcto`, `distancia_al_umbral` y `margen_desde_el_umbral`.

## 11. Que probabilidad usa la decision, verificado y no supuesto

En este proyecto se llaman probabilidad a cuatro cosas distintas:

```text
score crudo del modelo     la funcion de decision del modelo lineal
probabilidad sigmoid       sigma del score, es decir predict_proba
probabilidad OOF           la anterior, en filas que el modelo no entreno
probabilidad calibrada     una reparametrizacion monotona, que esta CERRADA
```

La decision de V03 consume la **tercera**. No es una afirmacion del informe sino un
calculo: recomputar la decision desde `proba` y los umbrales reproduce exactamente la
decision almacenada, con **True**. Si no coincidiera, el script pararia, porque
describir estas probabilidades seria describir el objeto equivocado.

## 4. Distribucion de p(y=1|x)

| etiqueta | media | mediana | std | min | max | skew | kurt | f(p=0) | f(p=1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HS | 0.458552 | 0.402135 | 0.274322 | 0.015696 | 0.991997 | +0.3505 | -1.1179 | 0.000 por ciento | 0.000 por ciento |
| TR | 0.357063 | 0.198427 | 0.323000 | 0.014371 | 0.998639 | +0.7648 | -0.9448 | 0.000 por ciento | 0.000 por ciento |
| AG | 0.425376 | 0.350309 | 0.276773 | 0.016358 | 0.994876 | +0.5683 | -0.9477 | 0.000 por ciento | 0.000 por ciento |

| etiqueta | q01 | q05 | q10 | q25 | q50 | q75 | q90 | q95 | q99 | umbral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HS | 0.046614 | 0.086486 | 0.125660 | 0.225273 | 0.402135 | 0.692360 | 0.878214 | 0.933432 | 0.974614 | 0.44 |
| TR | 0.026141 | 0.042486 | 0.052985 | 0.084876 | 0.198427 | 0.626860 | 0.910675 | 0.963282 | 0.988125 | 0.51 |
| AG | 0.052358 | 0.086678 | 0.111298 | 0.192110 | 0.350309 | 0.641428 | 0.874110 | 0.938189 | 0.977683 | 0.46 |

| etiqueta | media TP | media TN | solapamiento de las dos densidades |
|---|---:|---:|---:|
| HS | 0.656677 | 0.304798 | 0.4190 |
| TR | 0.737356 | 0.194028 | 0.2399 |
| AG | 0.654371 | 0.283629 | 0.3965 |

## 7. Cerca del umbral

| etiqueta | ventana | n | errores | aciertos | tasa de error |
|---|---|---:|---:|---:|---:|
| HS | +/-0.01 | 32 | 17 | 15 | 0.5312 |
| HS | +/-0.02 | 63 | 31 | 32 | 0.4921 |
| HS | +/-0.05 | 145 | 67 | 78 | 0.4621 |
| TR | +/-0.01 | 9 | 4 | 5 | 0.4444 |
| TR | +/-0.02 | 22 | 10 | 12 | 0.4545 |
| TR | +/-0.05 | 58 | 25 | 33 | 0.4310 |
| AG | +/-0.01 | 24 | 14 | 10 | 0.5833 |
| AG | +/-0.02 | 55 | 25 | 30 | 0.4545 |
| AG | +/-0.05 | 151 | 66 | 85 | 0.4371 |

## 8. Distancia al umbral

| etiqueta | clase | n | media | mediana | std | q10 | q90 |
|---|---|---:|---:|---:|---:|---:|---:|
| HS | aciertos | 1167 | 0.258174 | 0.257536 | 0.139046 | 0.067186 | 0.461719 |
| HS | errores | 316 | 0.151865 | 0.124021 | 0.118434 | 0.021231 | 0.333184 |
| TR | aciertos | 1319 | 0.349261 | 0.391183 | 0.121169 | 0.144857 | 0.465000 |
| TR | errores | 164 | 0.192213 | 0.172776 | 0.135601 | 0.035125 | 0.409212 |
| AG | aciertos | 1206 | 0.262248 | 0.272950 | 0.131797 | 0.069618 | 0.438163 |
| AG | errores | 277 | 0.160672 | 0.130197 | 0.124533 | 0.021848 | 0.353386 |

## 9. Errores de alta confianza

Se definen solo para diagnostico, con `error` y `distancia al umbral >= delta`, y el
delta no se optimiza.

| etiqueta | delta | n errores de alta confianza | n errores totales | porcentaje |
|---|---:|---:|---:|---:|
| HS | 0.10 | 189 | 316 | 59.81 por ciento |
| HS | 0.20 | 90 | 316 | 28.48 por ciento |
| HS | 0.30 | 42 | 316 | 13.29 por ciento |
| TR | 0.10 | 111 | 164 | 67.68 por ciento |
| TR | 0.20 | 71 | 164 | 43.29 por ciento |
| TR | 0.30 | 40 | 164 | 24.39 por ciento |
| AG | 0.10 | 166 | 277 | 59.93 por ciento |
| AG | 0.20 | 96 | 277 | 34.66 por ciento |
| AG | 0.30 | 48 | 277 | 17.33 por ciento |

## 12. Rankings por probabilidad

| etiqueta | TP confiados | FN confiados | FP confiados | TN confiados |
|---|---:|---:|---:|---:|
| HS | 505 | 143 | 173 | 662 |
| TR | 363 | 82 | 82 | 956 |
| AG | 425 | 142 | 135 | 781 |

## 13. Combinacion multi-etiqueta por tweet

```text
p_max_media                                  0.493114
p_min_media                                  0.326659
p_max_std                                    0.278736
p_min_std                                    0.30227
p_media_de_las_medias                        0.413664
p_varianza_media                             0.009604
desacuerdo_medio                             1.039194
desacuerdo_cuartiles                         [0.999994, 1.09672]
violaciones_p_TR_alta_con_p_HS_baja          39
violaciones_p_AG_alta_con_p_HS_baja          32
violaciones_p_AG_alta_con_p_TR_baja          94
correlacion_HS_TR                            0.809056
correlacion_HS_AG                            0.931459
correlacion_TR_AG                            0.878867
```

Las violaciones estructurales se miden a nivel de probabilidad con umbral 0,5, y son
**diagnostico**: no se aplica ninguna correccion estructural.

## 16. Brier, log loss y ECE

| etiqueta | Brier | log loss | ECE |
|---|---:|---:|---:|
| HS | 0.14855146 | 0.46078122 | 0.04875444 |
| TR | 0.08930740 | 0.30819653 | 0.06822897 |
| AG | 0.13945370 | 0.44177054 | 0.05959063 |

**La calibracion sigue cerrada como hipotesis de mejora.** Estos numeros son
diagnosticos, y el ECE alto no es un hallazgo: V03 se ajusto con class_weight
balanced, que por construccion desplaza las probabilidades para compensar el
desequilibrio, luego sus probabilidades no son frequencies y un ECE grande es el
comportamiento esperado de un modelo correctamente ponderado para decision.

## Graficas

* `reports/figuras/p61/01_probability_distributions_by_label.png`
* `reports/figuras/p61/02_probability_bins_error.png`
* `reports/figuras/p61/03_threshold_distance_windows.png`
* `reports/figuras/p61/04_distance_to_threshold.png`
* `reports/figuras/p61/05_high_confidence_errors.png`
* `reports/figuras/p61/06_joint_probability_geometry_3d.png`
* `reports/figuras/p61/07_pairwise_probability_geometry.png`
* `reports/figuras/p61/08_reliability_curves.png`
* `reports/figuras/p61/09_error_correlation_heatmap.png`
* `reports/figuras/p61/10_unique_rescue_vs_damage.png`
* `reports/figuras/p61/11_ensemble_cv_vs_v03.png`
* `reports/figuras/p61/12_ensemble_weights.png`
* `reports/figuras/p61/13_v03_vs_candidate_probability_shift.png`
* `reports/figuras/p61/14_error_breakdown_v03_vs_candidate.png`

*Generado por `experiments/p61_01_eda_probabilidades.py`. Sin push.*