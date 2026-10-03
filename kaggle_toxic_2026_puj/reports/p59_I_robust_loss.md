# P59-I - Perdida robusta redescendiente sobre V03

Generado por `experiments/p59_06_robust_loss.py`. Fecha 2026-09-28T06:33:15+00:00.

## Motivacion e hipotesis

La hipotesis NO es que los ejemplos difficiles deban pesar menos, ni que se detecten outliers para reponderarlos. Es sobre la **funcion objetivo**: la perdida logistica puede dejar que unas pocas observaciones de alto leverage con residuos extremos muevan la frontera global, y una perdida acotada y redescendiente podria limitar esa influencia DURANTE el ajuste, de forma endogena, sin nombrar ni excluir ningun ejemplo.

## Auditoria de novedad funcional

| precedente | veredicto |
|---|---|
| filas que mencionan tukey, biweight, geman, mcclure, cauchy, fair, redescend, m-estimador, rlm o irls | **0 de 323** |
| V34-01 | lista 'margin/robust loss' con profundidad **(0) UNEXPLORED** |
| V35-01 | `modified_huber` sobre `SGDClassifier`, otra resolucion. Huber esta acotada pero NO es redescendiente, y el registro dice 'NO es una nueva clase de funcion' |

**Funcion elegida: biweight de Tukey**, por criterio metodologico y no por enumerar. Es la redescendencia canonica: su influencia es exactamente nula mas alla de un corte fijo, que es justo la propiedad que la hipotesis necesita. `tau = 4,685` es la constante normal-consistente estandar, fijada ANTES de ejecutar y no ajustada. Una sola funcion, sin rejilla.

## Distincion frente a V30, que el encargo exige

| | V30 / V6-07 | aqui |
|---|---|---|
| peso | `1 - lambda * sospecha` | biweight del residuo actual del objetivo |
| de donde sale la senal | NearestNeighbors sobre TF-IDF, **externa** | **endogena** |
| cuando se fija | antes de ajustar, nunca se recalcula | en cada iteracion |
| puede llegar a cero | no, hay suelo 0,1 | si, por redescendencia |
| que se resuelve | la logistica con pesos reasignados | una ecuacion M-estimadora |

`CLOSED_BY_FUNCTIONAL_DUPLICATION = FALSE`.

## Control: tres brazos

| brazo | corrected_cv | comparacion |
|---|---:|---|
| A, V03 por sklearn | 0.7772114525165721 | max |proba - canonica| = 0.000e+00 |
| B, mi L-BFGS-B sobre la logistica pura | 0.7772647318762402 | max |proba - A| = 6.908e-03 |
| C, el mismo ajuste sobre la perdida de Tukey | 0.7785171129078273 | delta +0.001306 |

El brazo B es el que hace legible al C. Un optimizador propio puede diferir de liblinear en capacidad, en escala de regularizacion y en regla de parada, y cualquiera de esas diferencias se disfrazaria de efecto de la perdida. Si B no clona a A, entonces C - A seria una afirmacion sobre mi optimizador y el script se detiene. **B clona a A**, con una diferencia de 3 celdas de decision de 4449, que es la tolerancia del optimizador y no un efecto de la perdida.

## Comparabilidad de capacidad

```text
penalizacion      0,5 w'w + C suma_i omega_i perdida(s_i), identica a liblinear
intercepto        parametro explicito, no penalizado, igual que en liblinear
resolucion        L-BFGS-B con gradiente analitico, frente a liblinear
criterio          norma del gradiente < 1e-05 (1 + ||w||)
verificacion      gradiente analitico contra diferencias centrales, error 1e-09
perdida robusta   sigma^2 rho(sqrt(dev)/sigma), con sigma = 1
normalizacion     L coincide con el log-loss hasta segundo orden, luego la
                  regularizacion efectiva NO cambia
pliegues          los 5 canonicos, grupos near-duplicate UNION pseudo-autor intactos
umbrales          0,44 / 0,51 / 0,46 congelados, NO optimizados
semilla           42
```

Un detalle de implementacion que cambio el resultado y conviene registrar: la primera version uso un Newton propio con el intercepto absorbido por recentrado ponderado. Ese esquema resta dos terminos grandes y con d = 27.431 la resta pierde toda la precision: el mismo codigo daba coseno 0,99999 con d = 25 y **corrected_cv 0,4674** con d = 27.431. El control del parrafo 5, el brazo B, es precisamente lo que lo detecto y detuvo la ejecucion antes de interpretar nada. Con L-BFGS-B no hay recentrado, luego no hay cancelacion, y el brazo B clona a A.

## Resultados

```text
corrected_cv  V03 0.7772114525
corrected_cv  L-BFGS-B logistica 0.7772647319   (control del optimizador)
corrected_cv  Tukey             0.7785171129
delta_vs_V03  +0.0013056604
pliegues que mejoran 4/5
folds candidato 0.7970 0.7969 0.7756 0.7402 0.7838
folds V03       0.7960 0.7940 0.7789 0.7386 0.7791
rescate 8 | damage 4 | neto +4 | cambian 12
acierto en las celdas que cambian 0.6667
ambos correctos 3688 | ambos fallan 749
```

| etiqueta | F1 Tukey | F1 V03 | rescate | damage |
|---|---:|---:|---:|---:|
| HS | 0.7632 | 0.7617 | 2 | 0 |
| TR | 0.8175 | 0.8157 | 2 | 1 |
| AG | 0.7549 | 0.7542 | 4 | 3 |

## Gate

| criterio | valor | superado |
|---|---|---|
| delta >= +0,003 | +0.001306 | **False** |
| al menos 4 de 5 pliegues | 4/5 | **True** |
| gate completo | | **False** |

Bootstrap NO ejecutado: el gate no se supero y el parrafo 14 lo condiciona.

## Convergencia y optimizacion

| magnitud | valor |
|---|---:|
| todas las iteraciones del brazo B convergen | True |
| todas las iteraciones del brazo C convergen | True |
| iteraciones del brazo C | 27 a 45 de 100 |
| norma del gradiente, brazo B, maximo | 7.096e-06 |
| norma del gradiente, brazo C, maximo | 5.009e-06 |
| iteraciones del brazo B | 29 a 43 |
| celdas con influencia exactamente nula | 0.00 a 0.00 por ciento (media 0.00) |
| warnings | 0 |

La convergencia se mide con la norma del gradiente analitico del objetivo que se esta minimizando, verificada contra diferencias centrales con error relativo 1e-09.

## Analisis de mecanismo

| magnitud | valor |
|---|---:|
| coseno(V03, robusto) de coeficientes, medio | 0.999295 |
| coseno minimo | 0.998859 |
| cambio relativo de norma, medio | 0.045852 |
| cambio relativo de norma, maximo | 0.052762 |
| celdas que cambian | 12 de 4449 |
| cambios en error de V03 de alta confianza | 0 |
| cambios que rompen un acierto de V03 | 4 |
| |residual| mediano, antes -> despues | 0.5920 -> 0.5944 |
| |residual| percentil 99, antes -> despues | 1.2760 -> 1.3477 |
| |residual| mediano en la cola del 5 por ciento de V03 | 1.1195 -> 1.1577 |

### Influencia bajo V03 y bajo el modelo robusto, analisis POSTERIOR

- suma de la influencia: V03 7.8168e+03 | robusto 7.9041e+03 | ratio 1.0112
- influencia maxima: V03 2.4776e+00 | robusto 2.7366e+00
- Spearman entre las dos distribuciones: +0.9992

No se uso esta informacion para modificar I1. Es la pregunta del parrafo 10: si una funcion robusta atenua de manera endogena el efecto de observaciones influyentes sin tener que identificarlas ni ponderarlas explicitamente.

## Decision

```text
decision  MARGINAL. La perdida robusta mejora la metrica pero no alcanza el gate delta >= +0,003, luego no se demuestra. NO se prueba una segunda funcion
```

De las 42 celdas no rescatables de P54, el modelo robusto rescata **0**.

## Formulacion

> V03 remains the best verified leaderboard result.

*Seccion generada por `experiments/p59_06_robust_loss.py`. Sin push.*