# CHAR_SPECTRUM_KERNEL: V27-01 (kernel de espectro de caracteres con SVC de kernel precomputado)

Generado por `experiments/v27_01_char_spectrum_kernel.py`.
**Estado: `CHALLENGER`.** **Ninguna submission.**

## 1. Correccion de la motivacion, que hay que leer antes que los numeros

- Lo que se creia: el kernel spectrum de caracteres tiene mediana de vecino 0.3900.
- Lo que se habia medido: el coseno sobre la matriz binaria de la union TF-IDF de V03, word (1,2) junto a char_wb (3,5), calculado de una vez sobre las 1483 filas de desarrollo y fuera de los pliegues.
- Implicacion: la geometria menos memoristica medida en V26-01 era la de V03, no la de un kernel de espectro. La hipotesis cambia de forma: la pregunta pasa de si el spectrum evita la memorizacion a si un clasificador con esa geometria aporta algo que el hiperplano no aprovecha.

## 2. La configuracion

| parametro | valor |
|---|---|
| kernel | espectro de multiconjuntos de n-grams, overlap por minimo, normalizado por la masa |
| normalizacion | `K(i,j) = <min(f_i,f_j), min(f_i,f_j)> / sqrt(|f_i| |f_j|)` |
| analyzer / rango / min_df | char / (3, 5) / 2 |
| clasificador | `SVC(kernel=precomputed, C=2.0, class_weight=balanced)` |
| vectorizado ajustado dentro de cada pliegue | True |
| test leido | False |
| features por pliegue | [27980, 28754, 28322, 28678, 28659] |

## 3. Los controles del kernel

| control | resultado | ok |
|---|---|:-:|
| coincide con el calculo explicito por bucles | error maximo 0.000e+00 | si |
| Gram simetrica | max |K - K^T| = 0.000e+00 | si |
| diagonal igual a 1 en las filas con contenido | max |K_ii - 1| = 0.000e+00 en 38 filas; diagonal 0.000e+00 en las 2 vacias | si |
| sin NaN ni infinito | todos finitos: True | si |
| valores dentro de [0, 1] | [0.000000, 1.000000], filas vacias con K = 0.0e+00 | si |
| semidefinida positiva | minimo 0.000e+00, rango [0.000e+00, 10.533] | si |
| reproducible bit a bit | identico | si |

## 4. Resultados OOF

| modelo | corrected_cv | dOOF vs V03 | train F1 | gap | sd pliegues |
|---|---:|---:|---:|---:|---:|
| V03 (congelado) | 0.7772114525 | +0.0000000000 | 0.9614 | +0.1842 | -- |
| CHAR_SPECTRUM_KERNEL | 0.762122 | -0.015089 | 0.9698 | +0.2077 | 0.0106 |

F1 por etiqueta: HS 0.7446 (V03 0.7617) | TR 0.7933 (V03 0.8157) | AG 0.7485 (V03 0.7542)

F1 por pliegue: 0.7604 | 0.7639 | 0.7802 | 0.7573 | 0.7478  (media 0.7619, sd 0.0106)

## 5. Complementariedad con V03

| magnitud | valor |
|---|---:|
| V03 wrong y kernel correct (rescates) | 193 |
| V03 correct y kernel wrong (danos) | 238 |
| **neto** | **-45** |
| celdas coincidentes | 4018 de 4449 (0.9031) |
| correlacion de predicciones | 0.7936 |
| ambos incorrectos | 564 |

| etiqueta | F1 kernel | F1 V03 | cambios | rescates | danos |
|---|---:|---:|---:|---:|---:|
| HS | 0.7446 | 0.7617 | 174 | 79 | 95 |
| TR | 0.7933 | 0.8157 | 87 | 33 | 54 |
| AG | 0.7485 | 0.7542 | 170 | 81 | 89 |

## 6. Auditoria de memorizacion, spectrum frente a RBF exacto en los MISMOS pliegues

| pliego | spectrum mediana | spectrum >0.60 | RBF mediana | RBF >0.60 |
|---:|---:|---:|---:|---:|
| 1 | 0.3479 | 5.1 % | 0.7248 | 100.0 % |
| 2 | 0.3432 | 5.0 % | 0.7228 | 100.0 % |
| 3 | 0.3479 | 4.8 % | 0.7246 | 100.0 % |
| 4 | 0.3489 | 6.0 % | 0.7242 | 100.0 % |
| 5 | 0.3466 | 4.8 % | 0.7238 | 100.0 % |

Agregado: spectrum mediana 0.3469 frente a RBF 0.7240, diferencia -0.3771.

Comportamiento train frente a OOF: train 0.9698 contra OOF 0.7621, gap +0.2077.

## 7. Criterio de continuacion

| criterio | medida | se cumple |
|---|---|:-:|
| 1. mejora OOF clara respecto a V03 | dOOF -0.015089 | no |
| 2. OOF algo inferior pero fuerte complementariedad con V03 | rescates 193, danos 238, neto -45 | no |
| 3. estabilidad entre folds y geometria menos memoristica | sd 0.0106, mediana spectrum 0.3469 frente a RBF 0.7240 | si |
| 4. mecanismo que explique una clase donde V03 es debil | se evalua en el desglose por etiqueta del informe | no |

**Veredicto: `CHALLENGER`.**

Coste: 343.5 s, memoria 366 MB, submissions 0.

