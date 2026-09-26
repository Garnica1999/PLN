# V15 · A — Correlación entre entradas y asociación con la salida

## 1. Qué se mide y por qué dentro de cada fold

Matriz completa: **310 variables** (279 de V12 y 32 lingüísticas de V14).

Elegir variables por su correlación con la etiqueta calculada sobre todo el train es una forma de fuga aunque no se toque el test: el criterio de selección ya habría visto la validación. Por eso la estructura de clusters y la asociación se calculan **dentro de cada fold de entrenamiento** y se agregan. Además se mide la **estabilidad** de los clusters entre folds: una partición que cambia al cambiar de fold no sirve para podar.

## 2. Correlación entre variables de entrada

- Pares con |rho| >= 0,80: **402**
- Pares con |rho| >= 0.9: **245**
- Pares con duplicado exacto (|rho| = 1): **15**
- Clusters con |rho| >= 0.9: **58**, que cubren **171** variables de 310

### Los diez pares más correlacionados

| rho | Nivel | Variable A | Variable B |
|---|---|---|---|
| +1.0000 | redundante | `v12_estruct::yule_k` | `v12_estruct::simpson_d` |
| +1.0000 | redundante | `v12_densa::n_emoticon` | `v12_densa::has_emoticon` |
| +1.0000 | duplicado_exacto | `v12_densa::starts_mention` | `v12_estruct::first_token_is_mention` |
| +1.0000 | duplicado_exacto | `v12_densa::unique_char_ratio` | `v12_estruct::char_unique_ratio` |
| +1.0000 | duplicado_exacto | `v12_estruct::wlen_median` | `v12_estruct::wlen_p50` |
| -1.0000 | duplicado_exacto | `v12_miembro07_0` | `v12_miembro07_5` |
| -1.0000 | redundante | `v12_miembro06_0` | `v12_miembro07_0` |
| +1.0000 | redundante | `v12_miembro06_0` | `v12_miembro07_5` |
| +1.0000 | duplicado_exacto | `v12_supervisada_34` | `v12_supervisada_40` |
| -1.0000 | duplicado_exacto | `v12_estruct::compression_ratio` | `v12_estruct::compression_gain` |

## 3. Asociación con la salida (eta cuadrado)

La salida es categórica con seis clases, así que se usa **eta cuadrado** (variación explicada por la clase) en lugar de correlación de Pearson.

- Mediana de eta²: **0.0306**
- Máximo: **0.5599** (`v12_miembro07_5`)
- Variables con eta² < 0,01: **112** de 310
- Discrepancia entre la asociación global y la in-fold: mediana 0.000009

### Las diez variables más asociadas a la salida

| Variable | eta² in-fold | eta² global | Desviación entre folds |
|---|---|---|---|
| `v12_miembro07_5` | 0.55994 | 0.55994 | 0.00114 |
| `v12_miembro06_0` | 0.55561 | 0.5556 | 0.00107 |
| `v12_miembro07_0` | 0.55311 | 0.5531 | 0.00101 |
| `v12_miembro07_1` | 0.53807 | 0.53806 | 0.00096 |
| `v12_miembro07_4` | 0.51814 | 0.51814 | 0.00159 |
| `v12_miembro01_0` | 0.47175 | 0.47175 | 0.0018 |
| `v12_miembro01_5` | 0.46943 | 0.46943 | 0.00038 |
| `v12_miembro05_0` | 0.46902 | 0.46902 | 0.00175 |
| `v12_miembro00_0` | 0.46889 | 0.46889 | 0.00173 |
| `v12_miembro05_5` | 0.46878 | 0.46878 | 0.00049 |

### Las diez menos asociadas

| Variable | eta² in-fold | eta² global |
|---|---|---|
| `v14_conjuncion_ratio` | 0.0 | 0.0 |
| `v12_supervisada_50` | 0.0 | 0.0 |
| `v14_subordinacion_ratio` | 0.00013 | 0.00012 |
| `v12_estruct::maas` | 0.00014 | 0.00013 |
| `v14_verbo_gerundio_ratio` | 0.00021 | 0.0002 |
| `v12_estruct::yule_k` | 0.00034 | 0.00034 |
| `v12_densa::n_open_punct` | 0.00038 | 0.00038 |
| `v12_estruct::simpson_d` | 0.00041 | 0.0004 |
| `v14_verbo_infinitivo_ratio` | 0.00042 | 0.00041 |
| `v12_estruct::punct_comma_density` | 0.00044 | 0.00043 |

## 4. Estabilidad de los clusters

- Clusters que aparecen **completos en los cinco folds**: **0** de 58
- Jaccard medio de los **miembros** de cluster entre folds: **0.0157** (criterio muy estricto: exige que el cluster tenga exactamente los mismos miembros, y un solo par que cambie reordena el cluster entero por la transitividad de union-find)
- Jaccard medio de la **estructura de pares** entre folds: **1.0000** (mínimo 1.0000). Esta es la medida que responde a si la redundancia se reproduce al cambiar de fold.

### Los clusters más grandes y su representante

| Cluster | Variables | Correlación interna máxima | Representante | eta² del representante |
|---|---|---|---|---|
| 1 | 7 | 0.999 | `v12_miembro01_0` | 0.47175 |
| 5 | 7 | 0.999 | `v12_miembro01_4` | 0.38518 |
| 2 | 7 | 0.998 | `v12_miembro01_1` | 0.30163 |
| 6 | 7 | 0.999 | `v12_miembro01_5` | 0.46943 |
| 30 | 7 | 1.000 | `v12_estruct::repeated_token_ratio` | 0.00467 |
| 4 | 7 | 0.998 | `v12_miembro01_3` | 0.281 |
| 3 | 7 | 0.998 | `v12_miembro01_2` | 0.22848 |
| 12 | 6 | 0.998 | `v12_estruct::root_ttr` | 0.07858 |

## 5. Lo que hace V15b con esto

El criterio de poda será: **una variable por cluster**, y se queda la de mayor eta² **dentro del fold**, no la primera del cluster. Las variables que no están en ningún cluster se conservan todas. Con eso el número de variables pasa de 310 a 81 y la decisión se evalúa contra V12 con el gate D-042, incluida la segunda partición.

