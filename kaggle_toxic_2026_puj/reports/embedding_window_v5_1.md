# window de v5.1: Familia window (tamano de ventana de contexto) (EBD-19 (P2))

Generado por el script de la familia. Todo lo demas fijo: corpus E-CORP-01
(Wikipedia 10M), vector_size 200, min_count 3, subpalabras 3-6, negative 10,
5 epocas, sg skipgram, seed 42, bucket 200.000. Cambia **solo** la variable de la
familia. El punto central es el baseline B0, ya entrenado en v5, que se reevalua
en esta corrida como control y debe reproducir 0.7818.

## 1. Entrenamientos

| codigo | vocabulario | s | tipos con vector | sha256 |
|---|---|---|---|---|
| B0-200 | 100098 | 82.0 | 6059/6059 | `e54a363da986` |
| WS-3 | 100098 | 82.6 | 6059/6059 | `bd90ad0fd006` |
| WS-7 | 100098 | 147.6 | 6059/6059 | `406ae952394e` |

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
| EMB-WS-3-LR | 0.6694 | 0.6765 | -0.1078 | 0.0119 | - |
| EMB-WS-3-SVC | 0.6678 | 0.6717 | -0.1094 | 0.0117 | - |
| FUSION-WS-3 | 0.7830 | 0.7835 | +0.0057 | 0.022 | [-0.0021, +0.0136] |
| EMB-WS-7-LR | 0.6498 | 0.6670 | -0.1274 | 0.0183 | - |
| EMB-WS-7-SVC | 0.6513 | 0.6619 | -0.1259 | 0.02 | - |
| FUSION-WS-7 | 0.7806 | 0.7814 | +0.0034 | 0.0222 | [-0.0040, +0.0110] |

## 3. Lectura

La ventana controla cuanta distribucion se reune alrededor de cada palabra. Con
10M tokens, cada palabra del vocabulario ya aparece decenas de veces, asi que una
ventana mayor **no deberia** aportar senal nueva: el limite esta en cuantos datos
hay, no en cuanta distribucion se ve. Si la ventana tampoco cambia nada, se acumula
evidencia de que el embedding esta limitado por los 1.483 ejemplos etiquetados.

## 4. Slices de la mejor variante

| rebanada | filas | V03 | mejor fusion | delta |
|---|---|---|---|---|
| interpersonal (TR=1) | 445 | 0.9129 | 0.9208 | +0.0079 |
| afiliacion (AG=1) | 567 | 0.8846 | 0.8932 | +0.0086 |
| sin_hate_explicito (HS=0) | 835 | 0.0667 | 0.0639 | -0.0028 |
| colectivos | 93 | 0.5740 | 0.5850 | +0.0110 |
| insultos | 855 | 0.8357 | 0.8405 | +0.0048 |
| elongaciones | 57 | 0.6853 | 0.7213 | +0.0360 |
| coloquial | 214 | 0.8177 | 0.8234 | +0.0057 |
| hashtags | 144 | 0.5311 | 0.5310 | -0.0001 |
| menciones | 702 | 0.8462 | 0.8504 | +0.0042 |
| urls | 458 | 0.6578 | 0.6787 | +0.0209 |
| cita | 37 | 0.8449 | 0.8501 | +0.0052 |

## 5. Decision de familia

Gate de dos niveles: una variante es **adoptable** si su delta alcanza +0.005 **y** su
intervalo de confianza al 95 % no incluye el 0. La familia se **cierra** si ninguna
variante es adoptable y la dispersion interna es <= 0.005.

- Dispersion interna: **0.0024** (umbral 0.005).
- Mejor delta: **+0.0057**, margen +0.00075 sobre el umbral.
- IC95 de la mejor: **[-0.0021, +0.0136]**, que incluye el 0.

**CERRADA: inerte. Ninguna variante es adoptable y la dispersion interna es 0.0024 <= 0.005**

Presupuesto: 2 unidades exploratorias de esta familia, acumuladas en
`reports/embedding_budget_v5_1.json`. 0 unidades de reserva.

Fecha: 2026-09-26T00:05:20+00:00. Duracion: 263.3 s.
