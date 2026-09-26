# Clasificadores alternativos: discriminante lineal, arbol y bosque (V8-13)

Generado por `experiments/v8_13_clasificadores_alternativos.py` (V8-13 (clasificadores alternativos)).

## 1. Por que tiene sentido aparte de la cifra

El analisis de V8-12 mostro que promediar cinco repeticiones del **mismo** modelo da
+0,0043 sin cambiar nada de la tecnica. Aqui no hay promediado: cada clasificador se
mide por separado, y una mejora que aparezca en todas las repeticiones seria algo que
ninguna linea anterior de la fase ha conseguido.

El discriminante lineal se ajusta sobre el subespacio mas discriminante porque sus
supuestos de normalidad y covarianza independiente no se cumplen sobre veinte mil
caracteristicas dispersas. Los arboles, por el mismo motivo de coste. La seleccion del
subespacio usa informacion mutua con la etiqueta y se calcula solo con el tren.

## 2. Resultados

| rama | metrica | diferencia | repeticiones que mejoran | celdas | tasa |
|---|---|---|---|---|---|
| v03_control (control) | 0.780036 | +0.002824 | 1/3 | 192 | 0.5208 frente a 0.5500 |
| lda | 0.747157 | -0.030054 | 0/3 | 486 | 0.4177 frente a 0.5198 |
| arbol_2000_d6 | 0.704043 | -0.073168 | 0/3 | 785 | 0.3529 frente a 0.5122 |
| bosque_2000_d6 | 0.713274 | -0.063937 | 0/3 | 613 | 0.3393 frente a 0.5157 |

| rama | repeticion 1 | repeticion 2 | repeticion 3 |
|---|---|---|---|
| v03_control | 0.770550 | 0.782809 | 0.773841 |
| lda | 0.727280 | 0.742162 | 0.734275 |
| arbol_2000_d6 | 0.675179 | 0.693070 | 0.679820 |
| bosque_2000_d6 | 0.711781 | 0.717911 | 0.710470 |

## 3. El control que hace legible la tabla

El brazo `v03_control` es **V03 sin cambiar nada**: la misma
representacion, los mismos hiperparametros y los mismos umbrales, medido con el
protocolo de 3 repeticiones. Da **+0.002824**
respecto del OOF canonico y mejora en **1 de
3** repeticiones.

Eso significa que **la diferencia del control es el efecto del promediado, no el de la
tecnica**. Cualquier rama que se mida con este protocolo y dé mas que el control no está
midiendo su tecnica, está midiendo el promediado por segunda vez. Por eso el criterio
que decide es **cuantas repeticiones mejoran**, y no la cifra final.

## 4. Integridad del fichero de envio

Resultado global: **todas pasan**.

| comprobacion | resultado | detalle |
|---|---|---|
| numero de filas | correcto | 1154 frente a 1154 en la prueba |
| nombres y orden de columnas | correcto | ['id', 'HS', 'TR', 'AG'] |
| valores de etiqueta | correcto | [0, 1] |
| sin valores ausentes | correcto | 0 celdas ausentes |
| sin identificadores duplicados | correcto | 0 repetidos |
| identificadores en el orden de la prueba | correcto | coinciden con el orden original |
| filas distintas del entrenamiento | correcto | 1731 frente a 1154 |

## 5. Decision

La mejor alternativa es **lda**, con 0.747157, es decir
-0.030054 frente a V03.

**No se recomienda enviar.** Ninguna alternativa supera al control en numero de
repeticiones, y esa es la razon de la comparacion: un clasificador distinto que no
mejora en mas particiones que el propio V03 medido con el mismo protocolo no aporta
evidencia. **El envio es manual por D-063.**

Fecha: 2026-09-26T05:14:19+00:00. Duracion: 1416.6 s.
