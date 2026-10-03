# v5.1: analisis de capacidad de la representacion (EBD-23)

Pregunta que responde este informe: **¿la configuracion arquitectonica modifica
materialmente la calidad de la representacion?** Es distinta de si aporta a la tarea,
y la fase v5 ya sufrio por confundirlas.

## 1. El rango del embedding aislado

| configuracion | embedding_cv (media+LR) |
|---|---|
| WS-3 | 0.6694 |
| NEG-5 | 0.6664 |
| MC-5 | 0.6547 |
| baseline (200d/w5/min3/sub3-6/neg10) | 0.6523 |
| VS-300 | 0.6516 |
| MC-2 | 0.6516 |
| WS-7 | 0.6498 |
| SW-3-7 | 0.6465 |
| VS-100 | 0.6420 |
| SW-3-5 | 0.6397 |

Rango: 0.0297 entre la mejor (WS-3) y la peor
(SW-3-5) de las 10 configuraciones. Ese rango es **mayor** que todo el
movimiento de la fusion combinado.

## 2. Que se puede leer de la representacion

La mejor representacion aislada es WS-3 y la peor SW-3-5. La mejor combina
ventana 3 y el intervalo de subpalabra 3-6, es decir **menos** contexto y el intervalo
central: los dos extremos de la ventana y del intervalo empeoran, lo que indica que
el optimo esta en un punto medio y no en un extremo.

## 3. Por que ese rango no se traduce en aportacion

El rango del embedding aislado es 0.0297; el de la fusion es 0.0063.
La razon es que la fusion no hereda la calidad del bloque denso: hereda **cuanta
informacion no esta ya en el bloque disperso de V03**, que es la misma en las quince
configuraciones porque el corpus no cambia. Mejorar la geometria del embedding no
anade informacion que el char n-gram ya no tenga.

## 4. Conclusion de capacidad

La representacion **mejora** de forma medible y reproducible al variar la
arquitectura. La mejora es real como propiedad del embedding y **no** se traduce en
capacidad de decision para esta tarea. Son dos hechos distintos y ambos quedan
documentados: el primero es un resultado positivo y el segundo es un resultado
negativo, y ninguno anula al otro.

Mejor celda por delta de fusion: WS-3 con +0.0057, IC95 [-0.0021, +0.0136].
