# negative de v5.1: Familia negative sampling (muestras negativas por actualizacion) (EBD-22 (P5))

Generado por el script de la familia. Todo lo demas fijo: corpus E-CORP-01
(Wikipedia 10M), vector_size 200, min_count 3, subpalabras 3-6, negative 10,
5 epocas, sg skipgram, seed 42, bucket 200.000. Cambia **solo** la variable de la
familia. El punto central es el baseline B0, ya entrenado en v5, que se reevalua
en esta corrida como control y debe reproducir 0.7818.

## 1. Entrenamientos

| codigo | vocabulario | s | tipos con vector | sha256 |
|---|---|---|---|---|
| B0-200 | 100098 | 82.0 | 6059/6059 | `e54a363da986` |
| NEG-5 | 100098 | 72.9 | 6059/6059 | `3a3573eafbda` |

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
| EMB-NEG-5-LR | 0.6664 | 0.6729 | -0.1108 | 0.0164 | - |
| EMB-NEG-5-SVC | 0.6671 | 0.6698 | -0.1102 | 0.0181 | - |
| FUSION-NEG-5 | 0.7811 | 0.7829 | +0.0039 | 0.0196 | [-0.0044, +0.0123] |

## 3. Lectura

Las cuatro familias anteriores median parametros que cambian cuantos datos ve el
modelo o que formas reconstruye. `negative` cambia otra cosa: como se estiman los
vectores de las palabras que **no** aparecen en el contexto.

La hipotesis es que no deberia mover la aportacion a V03, porque el bloque de V03
que decide la etiqueta es disperso y ortografico (char n-grams 3-5), no un vector de
200 numeros. Es la familia con menos probabilidades de sorpresa del plan, y se
mide precisamente para cerrar la lista y no dejarlo sin medir.

Con dos puntos (5 frente a 10) no hay dispersion que interpretar: la comparacion
es directa y se lee contra el control.

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

- Dispersion interna: **0.0006** (umbral 0.005).
- Mejor delta: **+0.0046**, margen -0.00045 sobre el umbral.
- IC95 de la mejor: **[-0.0032, +0.0124]**, que incluye el 0.

**CERRADA: inerte. Ninguna variante es adoptable y la dispersion interna es 0.0006 <= 0.005**

Presupuesto: 1 unidades exploratorias de esta familia, acumuladas en
`reports/embedding_budget_v5_1.json`. 0 unidades de reserva.

Fecha: 2026-09-26T00:26:40+00:00. Duracion: 96.0 s.

## 7. Estado del presupuesto tras la ultima familia exploratoria

| contador | valor |
|---|---|
| entrenamientos existentes de v5 | 10 |
| entrenamientos nuevos de v5.1 | 9 |
| unidades de reserva usadas | 0 |
| presupuesto total consumido | 9 de 10 |
| presupuesto restante | 1 |

La familia `negative` es la ultima exploratoria del plan. Si queda cerrada por
inercia, la fase no consume la unidad de reserva: la reserva existe para confirmar
con una repeticion una mejora que ya se ha visto, no para abrir una familia nueva.
Gastarla aqui seria gastar por obligacion y no por evidencia.
