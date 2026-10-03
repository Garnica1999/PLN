# V13 · P0b — Prueba de colinealidad de la matriz completa de V12

## 1. Que se midio y como

Matriz **completa** del combinador de produccion: **279 variables** sobre 169,336 filas, tal cual la usa el meta-modelo (sin quitar, sustituir ni reordenar ninguna columna): 78 scores de los 14 miembros, 35 densas, 87 estructuradas, 73 supervisadas y 6 de NB-SVM.

Correlacion de **Spearman** sobre una muestra de 20,000 filas, calculada con sumas explicitas (`grama_exacta`) porque `rangos @ rangos.T` devuelve valores erroneos en este entorno (D-040): con el camino erroneo el bloque entero pareceria no tener ninguna correlacion alta. Umbrales: |rho| >= 0.8 alta, >= 0.9 redundante, |rho| >= 1 - 1e-9 duplicado exacto.

**No se reentrena nada.** La referencia de V12 se lee de `experiments/phase39_fusion_v12.json` (el mismo combinador, los mismos folds y la misma semilla). Una comprobacion independiente de esta sesion reprodujo sus cinco folds uno a uno: [0.6007, 0.605, 0.6032, 0.6004, 0.5928].

## 2. Estructura de la colinealidad

- **Pares con |rho| >= 0.8**: **401** de 38,781 pares posibles
- Pares redundantes (>= 0.9): **229**
- **Pares con duplicado exacto (|rho| = 1)**: **15**
- **Clusters de variables redundantes**: **54**

### Duplicados exactos: la misma variable dos veces

| rho | Variable A | Variable B |
|---|---|---|
| -1.0000 | `miembro07_0` | `miembro07_5` |
| +1.0000 | `densa::punct_ratio` | `estruct::char_symbol_ratio` |
| +1.0000 | `densa::starts_mention` | `estruct::first_token_is_mention` |
| +1.0000 | `densa::unique_char_ratio` | `estruct::char_unique_ratio` |
| +1.0000 | `estruct::wlen_median` | `estruct::wlen_p50` |
| -1.0000 | `estruct::compression_ratio` | `estruct::compression_gain` |
| +1.0000 | `supervisada_34` | `supervisada_40` |
| +1.0000 | `supervisada_35` | `supervisada_41` |
| +1.0000 | `supervisada_36` | `supervisada_42` |
| +1.0000 | `supervisada_37` | `supervisada_43` |
| +1.0000 | `supervisada_38` | `supervisada_44` |
| +1.0000 | `supervisada_39` | `supervisada_45` |
| +1.0000 | `supervisada_48` | `supervisada_50` |
| +1.0000 | `supervisada_51` | `supervisada_62` |
| +1.0000 | `supervisada_57` | `supervisada_61` |

### Clusters mas grandes

| Cluster | Variables | Correlación interna máxima | Contenido |
|---|---|---|---|
| 1 | 15 | 1.000 | `miembro00_0, miembro01_0, miembro02_0, miembro03_0, miembro04_0, miembro05_0, miembro06_0, miembro07_0, miembro07_1, miembro07_4, miembro07_` |
| 43 | 12 | 0.963 | `densa::avg_word_len, densa::long_word_ratio, estruct::char_space_ratio, estruct::prop_len_ge10, estruct::prop_len_ge8, estruct::wlen_iqr, es` |
| 5 | 11 | 1.000 | `densa::cttr, densa::lexical_entropy, densa::n_chars, densa::n_unique_words, densa::n_words, densa::unique_char_ratio, estruct::char_unique_r` |
| 19 | 10 | 0.999 | `densa::n_mention, estruct::entity_count, estruct::entity_entropy, estruct::pos_mention_first, estruct::pos_mention_last, estruct::pos_mentio` |
| 20 | 10 | 0.999 | `miembro00_4, miembro01_4, miembro02_4, miembro03_4, miembro04_4, miembro05_4, miembro08_4, miembro10_4, miembro13_4, nbsvm_50-64` |
| 21 | 10 | 0.999 | `miembro00_5, miembro01_5, miembro02_5, miembro03_5, miembro04_5, miembro05_5, miembro08_5, miembro10_5, miembro13_5, nbsvm_65-xx` |
| 7 | 9 | 1.000 | `densa::hapax_ratio, densa::ttr, estruct::dis_legomena_ratio, estruct::log_ttr, estruct::maas, estruct::mattr_25, estruct::repeated_token_rat` |
| 18 | 8 | 0.999 | `densa::n_hashtag, estruct::pos_hashtag_first, estruct::pos_hashtag_last, estruct::pos_hashtag_mean, supervisada_29, supervisada_30, supervis` |

## 3. Gravedad de la colinealidad (sin entrenar)

El factor de inflacion de la varianza de la variable j es el elemento jj de la inversa de la matriz de correlacion. Como hay pares con |rho| = 1, la matriz es singular: se usa la pseudoinversa y esas variables se marcan aparte, porque su VIF real es infinito y no un numero grande. Un VIF por encima de 1e6 no es una medida utilizable sino la senal de que esa direccion es practicamente no invertible.

### Espectro de la matriz de correlacion

Contar autovalores por encima de 1e-8 solo detecta las direcciones exactamente redundantes, criterio demasiado severo para un bloque de variables de texto. El criterio util es cuantos autovalores hacen falta para explicar la varianza:

| Medida | Valor |
|---|---|
| Autovalores por encima de 1e-2 | 217 de 278 |
| Autovalores por encima de 1e-4 | 259 de 278 |
| Autovalores por encima de 1e-6 | 262 de 278 |
| Autovalores por encima de 1e-8 (rango efectivo estricto) | 263 de 278 |
| **Direcciones para el 95 % de la varianza** | **87** de 278 |
| Direcciones para el 99 % de la varianza | 146 de 278 |

- **Rango efectivo estricto**: 263 de 278. Las 15 direcciones que faltan son exactamente los 15 pares de duplicados exactos (30 variables): no hay degeneracion estructural mas alla de esas.
- Variables con VIF >= 10.0: **170** de 278
- Variables con VIF indefinido por duplicado exacto: **30**
- VIF mediano de la matriz completa: **18.869** (el umbral clasico de referencia es 10)

La razon de que el VIF de los scores de miembros sea enorme sin que haya duplicados exactos es que **catorce modelos sobre las mismas seis clases coinciden mucho en que clase dan como mas probable**: entre scores de la misma clase la correlacion tiene mediana 0.7014 y llega a 0.9987. El valor minimo, -1.0000, corresponde al miembro ordinal, cuya quinta columna es la primera por construccion.

Columnas constantes en la matriz de produccion: **1** (`supervisada_49`). No tienen correlacion definida y no entran en el analisis de VIF ni de rango efectivo; se declaran aparte porque son en si mismas un hallazgo: una variable que nunca varia no aporta nada al modelo.

### Por familia

| Familia | Variables | En pares correlacionados | VIF mediano | VIF máximo | VIF alto |
|---|---|---|---|---|---|
| estructuradas | 87 | 128 | 12.579 | 2.976e+01 | 61 |
| scores de miembros | 78 | 203 | 528.737 | 7.531e+10 | 74 |
| supervisadas | 72 | 47 | 3.113 | 7.327e+00 | 0 |
| densas | 35 | 89 | 42.755 | 7.160e+01 | 35 |
| nb_svm | 6 | 16 | 0.770 | 7.710e-01 | 0 |

**Reparto util para V13**: la familia menos saturada es **nb_svm** (VIF mediano 0.77) y la mas saturada es **scores de miembros** (VIF mediano 528.74). Anadir mas variables del tipo que ya esta saturado (textometria densa y estructurada) tiene poco margen; anadir señal de otro tipo, como evidencia supervisada calculada dentro del fold, es lo que puede aportar.


## 4. ¿Esta danando al modelo?

El combinador de produccion (0,5 LogReg + 0,5 MLP sobre estas 279 variables) alcanza **0.6004 de macro-F1** con 15 pares de variables identicas y un VIF mediano de 18.9. La respuesta es **no, en prediccion**:

- La colinealidad no cambia lo que el modelo acierta; cambia **cuanto se puede interpretar un coeficiente** y **cuanto esfuerzo de aprendizaje gasta el MLP** en direcciones casi duplicadas. Un coeficiente grande en una variable colineal no significa que esa variable sea la que explica la decision.
- **El bloque tiene 87 direcciones para el 95 % de la varianza**, asi que anadir variables nuevas solo aporta si caen fuera de esas direcciones. Por eso la criba de redundancia va **antes** de fusionar cualquier bloque nuevo.
- Donde si hay que mirar es el reparto por familia: las familias con VIF alto son las que ya estan saturadas, y las de VIF bajo son las unicas donde anadir signal puede aportar algo.

## 5. Regla operativa para V13

1. Ningun bloque nuevo se fusiona sin pasar por `criba_redundancia` contra las 279 variables de la matriz completa.
2. Una variable nueva con |rho| >= 0.8 contra V12 se documenta y no se incorpora, salvo que aporte algo que la correlacion lineal no capture (por ejemplo una interaccion no lineal), lo cual se argue en un experimento propio.
3. Ninguna conclusion de V13 puede apoyarse en el **valor** de un coeficiente: con este grado de colinealidad, solo en su rendimiento fuera de muestra.

## 6. Validacion cruzada de la criba

La criba se ejecuto sobre las 72 variables lexicas de la fase cerrada y su resultado coincide con el que ya estaba calculado con **otra ruta de codigo** (`reports/age_lexicon_redundancy.csv`, obtenido con producto matricial matriz-vector, que D-040 declara sana): **3 redundantes en ambos casos** y diferencia maxima 0.0005 en la correlacion maxima de cada variable.

Es la segunda via independiente que confirma el diagnostico de D-040: si el entorno rompiese tambien la ruta matriz-vector, los dos calculos no podrian coincidir.

