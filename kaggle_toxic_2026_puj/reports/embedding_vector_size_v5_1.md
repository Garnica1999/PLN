# P1 de v5.1: familia vector_size (EBD-18)

Generado por `experiments/embedding_12_vector_size_v5_1.py` (EBD-18 (P1)).
Todo lo demas fijo: corpus E-CORP-01 (Wikipedia 10M), window 5, min_count 3,
subpalabras 3-6, negative 10, 5 epocas, sg skipgram, seed 42, bucket 200.000.
Cambia **solo** vector_size. El 200d ya existia y **no se reentreno**: se reevalua
en esta misma corrida para que la comparacion sea celda a celda.

## 1. Entrenamientos

| codigo | dim | vocabulario | s | tipos con vector | sha256 |
|---|---|---|---|---|---|
| B0-200 | 200 | 100098 | 82.0 | 6059/6059 | `e54a363da986` |
| VS-100 | 100 | 100098 | 0.0 | 6059/6059 | `f524b2db80b6` |
| VS-300 | 300 | 100098 | 0.0 | 6059/6059 | `c5df34c5f0c6` |

Cobertura del vocabulario TOXIC (identica en los tres puntos: el corpus y el
min_count no cambian): por tipo 0.6839, por token
0.8833, word-OOV 2704. La dimension **no**
afecta a la cobertura: afecta al numero de coordinadas con las que se resume cada
documento, que es otra cosa.

## 2. Resultados downstream

| evaluacion | corrected_cv (frozen) | ajustada | delta vs V03 | std pliegues | IC95 |
|---|---|---|---|---|---|
| EMB-B0-200-LR | 0.6523 | 0.6603 | -0.1249 | 0.0314 | - |
| EMB-B0-200-SVC | 0.6551 | 0.6584 | -0.1221 | 0.0228 | - |
| FUSION-B0-200 | 0.7818 | 0.7825 | +0.0046 | 0.0205 | [-0.0032, +0.0124] |
| EMB-VS-100-LR | 0.6420 | 0.6541 | -0.1353 | 0.0277 | - |
| EMB-VS-100-SVC | 0.6473 | 0.6536 | -0.1299 | 0.0299 | - |
| FUSION-VS-100 | 0.7823 | 0.7825 | +0.0050 | 0.0213 | [-0.0028, +0.0130] |
| EMB-VS-300-LR | 0.6516 | 0.6632 | -0.1256 | 0.0126 | - |
| EMB-VS-300-SVC | 0.6410 | 0.6510 | -0.1362 | 0.0165 | - |
| FUSION-VS-300 | 0.7805 | 0.7829 | +0.0033 | 0.0195 | [-0.0043, +0.0109] |

## 3. Lectura

- El embedding aislado **no cambia de banda** con la dimension: las tres
  variantes quedan muy por debajo de V03, igual que en v5.
- La fusion es lo que se mueve, y dentro de la familia la dispersion interna es de
  **0.0017**, muy por debajo de 0.005.
- La mejor fusion de la familia es FUSION-VS-100 con
  0.7823 (delta +0.0050).

## 4. Slices

| rebanada | filas | V03 | mejor fusion | delta |
|---|---|---|---|---|
| interpersonal (TR=1) | 445 | 0.9129 | 0.9195 | +0.0066 |
| afiliacion (AG=1) | 567 | 0.8846 | 0.8925 | +0.0079 |
| sin_hate_explicito (HS=0) | 835 | 0.0667 | 0.0690 | +0.0023 |
| colectivos | 93 | 0.5740 | 0.5873 | +0.0133 |
| insultos | 855 | 0.8357 | 0.8399 | +0.0042 |
| elongaciones | 57 | 0.6853 | 0.7197 | +0.0344 |
| coloquial | 214 | 0.8177 | 0.8280 | +0.0103 |
| hashtags | 144 | 0.5311 | 0.5462 | +0.0151 |
| menciones | 702 | 0.8462 | 0.8491 | +0.0029 |
| urls | 458 | 0.6578 | 0.6759 | +0.0181 |
| cita | 37 | 0.8449 | 0.8557 | +0.0108 |

## 5. Decision

Criterio pre-registado: una variante es **adoptable** si su delta de fusion alcanza
+0.005 **y** su intervalo de confianza al 95 % no incluye el 0. La familia se
**cierra** si ninguna variante es adoptable y la dispersion interna es <= 0.005.

- Dispersion interna de la familia: **0.0017** (umbral 0.005).
- Mejor delta: **+0.0050**, un margen de +0.00005 sobre el umbral.
- Intervalo de la mejor variante: **[-0.0028, +0.0130]**, que incluye el 0.
- Regla aplicada: gate alcanzado en numero y intervalo de confianza que no lo respalda.

**CERRADA: inerte. Ninguna variante es adoptable y la dispersion interna es 0.0017 <= 0.005**

El punto delicate es que el mejor delta cae practicamente **sobre** el umbral (margen +0.00005) con un intervalo que incluye el 0. Tocar el umbral por cinco diezmilésimas no es
una mejora: es una variable que se ha movido dentro del ruido del protocolo, que es
lo que v5 demostro cinco veces con cinco configuraciones distintas.

El dato que si queda es que la capacidad dimensional **no es la palanca**: con
1.483 ejemplos etiquetados, 100, 200 y 300 distancias dan practicamente lo mismo, y
ninguna llega al umbral que de verdad importa, que es el intervalo de confianza.

Presupuesto: esta familia consumio 2 unidades exploratorias de 9,
mas 0 de 1 de reserva. Un +0.003, aunque sea positivo, no se adopta: es el orden de
magnitud que v5 ya demostro que no es distinguible del ruido del protocolo.

Fecha: 2026-09-25T23:59:06+00:00.
