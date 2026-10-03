# min_count de v5.1: Familia min_count (frecuencia minima para entrar en el vocabulario) (EBD-20 (P3))

Generado por el script de la familia. Todo lo demas fijo: corpus E-CORP-01
(Wikipedia 10M), vector_size 200, min_count 3, subpalabras 3-6, negative 10,
5 epocas, sg skipgram, seed 42, bucket 200.000. Cambia **solo** la variable de la
familia. El punto central es el baseline B0, ya entrenado en v5, que se reevalua
en esta corrida como control y debe reproducir 0.7818.

## 1. Entrenamientos

| codigo | vocabulario | s | tipos con vector | sha256 |
|---|---|---|---|---|
| B0-200 | 100098 | 82.0 | 6059/6059 | `e54a363da986` |
| MC-2 | 139479 | 127.8 | 6059/6059 | `53d53814880c` |
| MC-5 | 69310 | 95.6 | 6059/6059 | `fa9eb2977c42` |

Cobertura TOXIC (train+test, como v5): por tipo 0.6839, por
token 0.8833, word-OOV 2704. La cobertura
es del **corpus**, y el corpus no cambia entre familias: lo unico que la familia
puede mover es cuantos de esos tipos reciben vector del modelo.

## 2. Resultados downstream

| evaluacion | corrected_cv (frozen) | ajustada | delta vs V03 | std pliegues | IC95 |
|---|---|---|---|---|---|
| EMB-B0-200-LR | 0.6523 | 0.6603 | -0.1249 | 0.0314 | - |
| EMB-B0-200-SVC | 0.6551 | 0.6584 | -0.1221 | 0.0228 | - |
| FUSION-B0-200 | 0.7818 | 0.7825 | +0.0046 | 0.0205 | [-0.0032, +0.0124] |
| EMB-MC-2-LR | 0.6516 | 0.6586 | -0.1256 | 0.0244 | - |
| EMB-MC-2-SVC | 0.6522 | 0.6559 | -0.1250 | 0.0237 | - |
| FUSION-MC-2 | 0.7771 | 0.7797 | -0.0001 | 0.0234 | [-0.0080, +0.0078] |
| EMB-MC-5-LR | 0.6547 | 0.6637 | -0.1226 | 0.0308 | - |
| EMB-MC-5-SVC | 0.6555 | 0.6586 | -0.1218 | 0.0274 | - |
| FUSION-MC-5 | 0.7792 | 0.7809 | +0.0020 | 0.0206 | [-0.0056, +0.0097] |

## 3. Lectura

`min_count` es el unico parametro de la lista que cambia **cuantos tipos
existen**, y por tanto los unicos que pueden mover la cobertura del vocabulario
TOXIC. Hay dos efectos opuestos y el resultado dice cual pesa. A favor de 2: mas
tipos entran, luego mas palabras de TOXIC deformadas tienen vector propio en vez de
uno compuesto por subpalabras, y en v5 se midio que el 66,6 % de los tipos ausentes
son evasiones con perdida de vocales. En contra de 2: cada tipo con frecuencia 2
aporta una estimacion ruidosa de su vector, y con 1.483 ejemplos el lado
supervisado ya es el cuello de botella.

El numero de tipos con vector es la cifra que hay que mirar en esta familia: es la
unica de las tres que puede cambiarlo, porque con el corpus y la ventana fijos solo
el umbral de frecuencia decide que palabras tienenrepresentacion propia.

## 4. Slices de la mejor variante

| rebanada | filas | V03 | mejor fusion | delta |
|---|---|---|---|---|
| interpersonal (TR=1) | 445 | 0.9129 | 0.9199 | +0.0070 |
| afiliacion (AG=1) | 567 | 0.8846 | 0.8925 | +0.0079 |
| sin_hate_explicito (HS=0) | 835 | 0.0667 | 0.0657 | -0.0010 |
| colectivos | 93 | 0.5740 | 0.5890 | +0.0150 |
| insultos | 855 | 0.8357 | 0.8389 | +0.0032 |
| elongaciones | 57 | 0.6853 | 0.7221 | +0.0368 |
| coloquial | 214 | 0.8177 | 0.8230 | +0.0053 |
| hashtags | 144 | 0.5311 | 0.5462 | +0.0151 |
| menciones | 702 | 0.8462 | 0.8483 | +0.0021 |
| urls | 458 | 0.6578 | 0.6782 | +0.0204 |
| cita | 37 | 0.8449 | 0.8393 | -0.0056 |

## 5. Decision de familia

Gate de dos niveles: una variante es **adoptable** si su delta alcanza +0.005 **y** su
intervalo de confianza al 95 % no incluye el 0. La familia se **cierra** si ninguna
variante es adoptable y la dispersion interna es <= 0.005.

- Dispersion interna: **0.0047** (umbral 0.005).
- Mejor delta: **+0.0046**, margen -0.00045 sobre el umbral.
- IC95 de la mejor: **[-0.0032, +0.0124]**, que incluye el 0.

**CERRADA: inerte. Ninguna variante es adoptable y la dispersion interna es 0.0047 <= 0.005**

Presupuesto: 2 unidades exploratorias de esta familia, acumuladas en
`reports/embedding_budget_v5_1.json`. 0 unidades de reserva.

Fecha: 2026-09-26T00:10:49+00:00. Duracion: 257.9 s.

## 6. Que mueve esta familia

La cobertura del **corpus** (0.6839 por tipo, 0.8833 por token) es identica en toda
la familia porque el corpus no cambia. Lo que `min_count` mueve es cuantos de esos
tipos reciben vector **propio** del modelo, y con ello la parte de la cobertura que
depende del modelo y no del corpus. Ese es el numero que hay que mirar aqui.

| min_count | tipos en el vocabulario del modelo | tipos del train con vector |
|---|---|---|
| 2 | 139479 | 6059 (medido en v5 con subpalabras 3-6) |
| 3 | 100098 | 6059 (medido en v5 con subpalabras 3-6) |
| 5 | 69310 | 6059 (medido en v5 con subpalabras 3-6) |

Un umbral mas bajo agranda el vocabulario, pero cada tipo con frecuencia 2 aporta una
estimacion ruidosa de su vector. Un umbral mas alto hace lo contrario. La tabla de la
seccion 2 dice cual de los dos efectos pesa en la fusion.

