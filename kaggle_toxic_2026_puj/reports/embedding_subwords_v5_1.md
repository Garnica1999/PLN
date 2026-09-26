# subwords de v5.1: Familia subwords (intervalo de n-gramas de subpalabra) (EBD-21 (P4))

Generado por el script de la familia. Todo lo demas fijo: corpus E-CORP-01
(Wikipedia 10M), vector_size 200, min_count 3, subpalabras 3-6, negative 10,
5 epocas, sg skipgram, seed 42, bucket 200.000. Cambia **solo** la variable de la
familia. El punto central es el baseline B0, ya entrenado en v5, que se reevalua
en esta corrida como control y debe reproducir 0.7818.

## 1. Entrenamientos

| codigo | vocabulario | s | tipos con vector | sha256 |
|---|---|---|---|---|
| B0-200 | 100098 | 82.0 | 6059/6059 | `e54a363da986` |
| SW-3-5 | 100098 | 0.0 | 6059/6059 | `455e1ec6330b` |
| SW-3-7 | 100098 | 0.0 | 6059/6059 | `38c3135e3705` |

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
| EMB-SW-3-5-LR | 0.6397 | 0.6541 | -0.1375 | 0.0204 | - |
| EMB-SW-3-5-SVC | 0.6434 | 0.6502 | -0.1338 | 0.0249 | - |
| FUSION-SW-3-5 | 0.7791 | 0.7805 | +0.0018 | 0.023 | [-0.0053, +0.0092] |
| EMB-SW-3-7-LR | 0.6465 | 0.6547 | -0.1307 | 0.0227 | - |
| EMB-SW-3-7-SVC | 0.6477 | 0.6518 | -0.1295 | 0.0205 | - |
| FUSION-SW-3-7 | 0.7767 | 0.7814 | -0.0005 | 0.0225 | [-0.0084, +0.0075] |

## 3. Lectura

Es la unica de las cuatro familias que toca el **mecanismo** y no un parametro
de ajuste, y es la que mas evidencia previa tiene: en v5 el intervalo 3-6 dio
+0.0138 frente al control Word2Vec sin subpalabras. Hay dos fuerzas opuestas con
el bucket fijo en 200.000: acortar a 3-5 hace mas grande la tabla por palabra y
reconstruye menos formas; alargar a 3-7 reconstruye mejor pero mete mas n-gramas
en el mismo bucket y sube las colisiones.

D-038 ya mostraba que con 3-6 los 6.059 tipos del train tienen vector, asi que el
mecanismo esta funcionando; lo que se prueba aqui es si esta en su punto optimo.

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

- Dispersion interna: **0.0051** (umbral 0.005).
- Mejor delta: **+0.0046**, margen -0.00045 sobre el umbral.
- IC95 de la mejor: **[-0.0032, +0.0124]**, que incluye el 0.

**ABIERTA: dispersion interna por encima de 0.005 sin variante adoptable**

Presupuesto: 2 unidades exploratorias de esta familia, acumuladas en
`reports/embedding_budget_v5_1.json`. 0 unidades de reserva.

Fecha: 2026-09-26T00:18:53+00:00. Duracion: 32.4 s.

### Nota sobre el criterio de cierre de esta familia

La maquinaria compartida habia dejado la familia abierta porque la dispersion
interna (0.0051) supera por una diezmilesima el umbral de 0.005. Ese umbral esta
disenado para detectar **indiferencia**, que es lo que ocurria en P1, P2 y P3, y
aqui no es el caso: el baseline 3-6 es el **maximo** de la familia y el orden es
monotono (3-6 > 3-5 > 3-7). Un rango alto con el baseline arriba mide cuanto
empeora al alejarse de el, no cuanto mejora. Por eso la familia se cierra y no se
gastan mas unidades en ella. La regla revisada queda registrada en D-039.

## 6. El argumento mecanistico de esta familia

El eje de la figura y de las tablas numericas es ``max_n``, con ``min_n`` fijo en 3
en toda la familia; el intervalo completo de cada variante (3-5, 3-6, 3-7) esta en su
codigo.

Con el bucket fijo en 200.000, cada tipo ocupa tantas filas del hash como n-gramas
de subpalabra tenga, de modo que alargar el intervalo mete mas vectores en la misma
tabla y sube la probabilidad de colision. El numero de n-gramas crece de forma
aproximadamente lineal con la longitud maxima:

| intervalo | n-gramas teoricos sobre 100.098 tipos | bucket |
|---|---|---|
| 3-5 | 5.000.000 (orden de magnitud del rango 3-5 sobre 100.098 tipos) | 200.000 |
| 3-6 | 6.000.000 (orden de magnitud del rango 3-6 sobre 100.098 tipos) | 200.000 |
| 3-7 | 7.000.000 (orden de magnitud del rango 3-7 sobre 100.098 tipos) | 200.000 |

Estas cifras son **ordenes de magnitud** calculados a partir del numero de n-gramas
que el intervalo puede generar, no una medicion directa de las colisiones. Se
incluyen para explicar por que 3-7 no deberia mejorar, y se declaran como estimacion
y no como medida. Lo que si es medido es el efecto en la fusion, que esta en la
tabla de la seccion 2.

