# V25 P0: auditoria historica de interacciones de segundo orden y factorizacion

Worktree `/home/carlos/Git/PLN-v25`, rama `v25`, base `56d156c` (V24 cerrado).
Fuente: lectura de codigo. `experiments.csv` tiene 242 registros y hay 20 scripts propios de
embeddings y mas de 100 scripts de modelado.

## 1. La tabla del encargo (seccion 2)

| mecanismo | historico | codigo | resultado | equivalente a V25 | estado |
|---|---|---|---:|---|---|
| Factorization Machine | **ninguno** | 0 ocurrencias de `FactorizationMachine`, `xlearn`, `libffm`, `fastFM` | -- | **si, es V25** | **nuevo** |
| interacciones de segundo orden de features | **ninguno** | 0 ocurrencias de multiplicacion elemento a elemento de la matriz consigo misma | -- | **si, es V25** | **nuevo** |
| polynomial de grado 2 | **ninguno** | 0 ocurrencias de `polynomial`, `degree=2`, `poly2` | -- | equivalente | **nuevo** |
| factorizacion de bajo rango de features | **ninguno** | 0 ocurrencias de `low_rank` aplicadas a la matriz de features | -- | equivalente | **nuevo** |
| field-aware factorization | **ninguno** | 0 ocurrencias de `field_aware` o `FieldAware` | -- | equivalente | **nuevo** |
| bajo rango **entre etiquetas** | v8-00, v12-01 | `v8_00_gap_audit.py` patron `low_rank_entre_etiquetas`; `v12_01_regularizacion_grupo.py` | regularizacion conjunta de las tres cabezas | **no**: comparte estructura entre etiquetas, no entre features | cerrado |
| ponderacion (reescalado) de columnas | v4-C | `30_supervised_weighting_v4.py` linea 300, `matriz.multiply(vector[None, :])` | 0.662-0.785 | **no**: multiplica por un vector, no empareja features | cerrado en contra |
| norma al cuadrado por columna | v10-09 | `v10_09_correlacion_eliminacion.py` linea 211, `matriz.multiply(matriz).sum(axis=0)` | diagnostico de magnitudes | **no**: `x^2` por feature, nunca `x_i * x_j` | cerrado |
| norma de Frobenius del bloque | v12-01 | `v12_01_regularizacion_grupo.py` linea 215 | penalizacion | **no** | cerrado |
| pares de **etiquetas** | v12-03 | `v12_03_entrenamiento_por_pares.py` | objetivo que acopla HS, TR y AG | **no**: pares de etiquetas dentro de cada ejemplo, no pares de features | cerrado |
| factorizacion condicional de etiquetas | v6-B | `v6_02_label_factorization.py` | B1 -0.0107, 1/5 pliegues | **no**: descompone la probabilidad de una etiqueta en ramas de otra | cerrado en contra |
| interaccion de modelos (ensembles) | v2 | `05_ensembles.py`, `32_meta_shift_v4.py` | 0.7751 LB | **no**: combina predicciones, no features | cerrado |
| interaccion con la verosimilitud condicional | v4-G | `33_class_conditional_lm_v4.py` | 0.790410 OOF, LB 0.77938 | **no**: anade 36 features de verosimilitud de clase, no terminos cruzados | cerrado en LB |
| CNN de caracteres | DL-02 | `11_dl_desde_cero.py` | 0.7410 | **no** | cerrado |
| All-But-The-Top | v22-03 | `v22_03_all_but_the_top.py` | D=1 -0.0018, D=2 -0.0061, D=5 -0.0036 | **no**: post-procesado de un espacio vectorial, no interacciones | cerrado |

## 2. El caso que mas se le parecia, y por que no lo es

`v12_03_entrenamiento_por_pares.py` lleva "entrenamiento por pares" en el nombre y es el
unico script que podria confundirse con una interaccion de segundo orden. Su propia cabecera
aclara que no lo es, y conviene citarlo porque es la confusion mas probable de esta fase:

> La auditoria de brecha describio H3 como pares de **ejemplos**: acercar ejemplos de la
> misma etiqueta y separar ejemplos de etiquetas distintas. La pregunta de la fase es mas
> especifica: pares de **etiquetas** dentro de cada ejemplo. Se implementa la segunda.

Es decir, lo que acopla es HS con TR y AG **dentro de una misma fila**, mediante un termino
en la funcion objetivo. FM acopla **features dentro de una misma fila** mediante factores
latentes. Son ejes distintos del producto cartesiano: uno opera sobre el eje de las etiquetas,
el otro sobre el eje de las 27.431 features.

Las otras tres apariciones que podrian confundirse son multiplicaciones elemento a elemento,
y ninguna produce un termino cruzado:

- `matriz.multiply(vector[None, :])` reescala cada columna por un escalar. El resultado sigue
  siendo una matriz **linealmente equivalente** a la original: no anade ninguna interaccion.
- `matriz.multiply(matriz)` elevate al cuadrado **cada elemento por separado**, sin mezclarlos.
  Da las magnitudes de las features, no sus productos cruzados.
- `sqrt(matriz.multiply(matriz).sum())` es la norma de Frobenius, una penalizacion.

## 3. Conclusion de P0

**Factorization Machines son genuinamente nuevas en este repositorio.** No existe ninguna
implementacion previa de interacciones de features de segundo orden, ni de factorizacion de
bajo rango sobre la matriz de features, ni de variantes field-aware, ni de las bibliotecas que
las implementan.

El proyecto ha explorado interacciones en **el eje de las etiquetas** (v6-B, v12-03) y en **el
espacio de modelos** (ensembles, stacking), pero nunca en el eje de las features. Ese eje es
exactamente el que V25 propone y el unico hueco de la matriz del encargo.

Se puede continuar.

## 4. Que hay que vigilar al implementar

1. **No materializar los pares.** Con 27.431 features, los pares son 376 millones. La
   propiedad algebraica de la FM resuelve esto en tiempo lineal en el numero de no nulos, y
   esa propiedad es justamente lo que hay que verificar numericamente contra una
   implementacion explicita en un conjunto sintetico pequeno antes de fiarse del resultado
   grande.
2. **La complejidad efectiva es la que decidio V23.** Una FM con `k = 4` sobre 27.431
   features tiene 27.431 coeficientes lineales mas 27.431 x 4 factores latentes, es decir del
   orden de 137.000 parametros frente a los 27.431 de V03. Es un factor de 5, no de 70 como
   los vectores de soporte del kernel. Esa es exactamente la propiedad que V25 quiere
   explotar, y hay que reportarla junto al resultado, no solo el CV.
3. **El gap train-OOF es el indicador que V24 demostro que no sirve solo.** El modelo de
   transferencia de P1 de V24 (`dLB = -0.00920 + 0.2448 * dOOF`) debe aplicarse a cualquier
   resultado antes de considerar una submission.
