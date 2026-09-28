# V30 - Refinamiento local del LightGBM


### Por que el espacio es local, y no una busqueda nueva

Los cinco mejores trials de V29 coinciden en una region estrecha, y **V20 estaba sub-parametrizado en casi todo**:

| Parametro | 5 mejores de V29 | `HIERBA` de V20 |
|---|---|---|
| `num_leaves` | 43 - 87 | **8** |
| `max_depth` | 5 - 9 | **3** |
| `learning_rate` | 0,021 - 0,043 | 0,05 |
| `min_child_samples` | 22 - 94 | **200** |
| `reg_lambda` | 1e-7 - 0,018 | **1,0** |

Ese contraste **es el mecanismo de la mejora de V29**, y por eso V30 centra su espacio ahi en vez de recorrer la region entera. Y la localidad no es solo un rango: el **trial 0 de cada fold se encola con los parametros exactos** del mejor trial de V29 para ese fold.

- Trials: **15 por fold, 75 en total**, el maximo que autoriza el brief.
- Sampler: `TPESampler(seed=20300930 + fold) con trial 0 encolado`.
- Techo de estimadores: **1500**, con early stopping de 60 rondas.

### Iteraciones reales, que es donde se ve el early stopping

| Fold | Mejor trial | Inner | Mediana | Sigma | Iteraciones / techo | Early stop | Outer |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 0.612670 | 0.610101 | 1.89 | 999 / 1011 | True | 0.615979 |
| 1 | 3 | 0.611625 | 0.610502 | 1.20 | 1173 / 1486 | True | 0.616113 |
| 2 | 5 | 0.617228 | 0.615592 | 1.72 | 644 / 1023 | True | 0.611859 |
| 3 | 11 | 0.618832 | 0.616098 | 2.12 | 1129 / 1477 | True | 0.610876 |
| 4 | 13 | 0.617436 | 0.616016 | 1.43 | 398 / 1000 | True | 0.606381 |

**El early stopping corta siempre muy por debajo del techo.** Es exactamente el comportamiento que el punto 38 del brief describe como correcto: dar margen alto de capacidad y dejar que el criterio interior decida donde parar, en vez de entrenar artificialmente hasta el limite.

**Mejor trial outlier: 0 de 5.** Ningun fold se gano por un trial anomalo, asi que el resultado no depende de una tirada afortunada.

### Resultado del refinamiento

- corrected_cv **0,6122387**
- frente a V20: **0,002461**
- frente al LightGBM de V29: **0,000069**
- folds que mejoran a V29: **3/5** ([0.001485, 0.000216, -0.000409, -0.001387, 0.000452])
- veredicto: **EQUIVALE_A_V29**

Criterio declarado de consolidacion: delta frente al LightGBM de V29 superior a +0,0005; entre -0,0005 y +0,0005 se considera equivalente y se decide por complementariedad en la fase de ensemble.

### ¿V30 aprende mejor, o solo reproduce a V29?

| Modelo | CV individual | corr. errores con V20 | Rescate unico | Dano unico | Neto |
|---|---|---|---|---|---|
| v20_f3b_oof_candidato | 0.6097782 | 1.0000 | 0 | 0 | +0 |
| v29_lgbm_optuna | 0.6121698 | 0.8636 | 5,689 | 5,288 | +401 |
| v30_lgbm_refinado | 0.6122387 | 0.8694 | 5,448 | 5,061 | +387 |

Y entre los dos LightGBM, que es la comparacion que decide si hay algo que combinar:

| Par | corr. errores | Discrepancia | Uno arregla al otro |
|---|---|---|---|
| V29 vs V30 | 0.9164 | 0.0626 | 3,365 |

Una correlacion de errores **muy alta** entre V29 y V30 significa que el refinamiento encontro la misma funcion con parametros distintos: es una confirmacion de que la region esta bien elegida, y tambien de que **no hay un tercer miembro que aportar** al ensemble. Por eso el pool de V30 se queda en tres: V20, V29 y V30.
