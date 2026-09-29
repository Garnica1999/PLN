# P63 - Integracion de los bloques, y el techo que queda

Generado por `experiments/p63_07_integracion_y_techo.py`. Fecha 2026-09-29T06:34:13+00:00.

## Por que esta pregunta y no otra

Cuatro tracks con el mismo patron: la composicion gana +0,1055 en solitario y
+0,0084 frente al control al anadirse, la posicion gana +0,0713 en solitario y pierde
-0,0025 al anadirse. Como los bloques son **densos y pequenos**, 42 y 21 columnas contra
21044, la explicacion de dilucion de P62 es aqui mas debil, y la pregunta honesta deja
de ser si cada bloque ayuda y pasa a ser **si se ayudan entre si**.

## Los brazos

| brazo | columnas | que es |
|---|---:|---|
| V03_todo | 63 | composicion mas posicion |
| V03_comp | 42 | solo composicion, el mejor bloque suelto de la fase |
| V03_pos | 21 | solo posicion |
| V03_random | 63 | control de capacidad del ANCHO TOTAL |

**El control es el experimento entero.** Un bloque conjunto de 63 columnas puede
parecer mejor que V03 por la misma razon por la que un bloque suelto de 21 no: mas
sitio. Si V03_todo no supera a V03_random, nada de el es especifico.

## Resultados

| brazo | columnas | corrected_cv | delta vs V03 | rescate | dano |
|---|---:|---:|---:|---:|---:|
| V03_todo | 63 | 0.7534060878 | -0.0238053648 | 140 | 250 |
| V03_comp | 42 | 0.7669427506 | -0.0102687019 | 129 | 190 |
| V03_pos | 21 | 0.7671031904 | -0.0101082622 | 83 | 114 |
| V03_random | 63 | 0.7448169478 | -0.0323945047 | 101 | 215 |

### El detalle por rebanada

| bin | n | F1 de V03 | V | V | V |
|---|---:|---:|---:|---:|---:|
| CORTA | 167 | 0.9480 | -0.0352 | -0.0298 | -0.0110 |
| MEDIA | 602 | 0.7964 | -0.0248 | -0.0025 | -0.0074 |
| LARGA | 547 | 0.6834 | -0.0356 | -0.0158 | -0.0135 |
| MUY_LARGA | 167 | 0.5606 | +0.0366 | +0.0031 | -0.0126 |

## El techo por rebanada, que es lo que ninguna representacion toca

El mejor F1 alcanzable moviendo **solo el corte** de V03 es el techo de cualquier
funcion monotona de su propia puntuacion, luego es exactamente la parte del error
residual que ninguna representacion puede atacar.

| bin | tokens | n | F1 de V03 | techo moviendo el corte | margen |
|---|---|---:|---:|---:|---:|
| CORTA | 0-6 | 167 | 0.9480 | 0.9700 | +0.0220 |
| MEDIA | 7-15 | 602 | 0.7964 | 0.8288 | +0.0324 |
| LARGA | 16-30 | 547 | 0.6834 | 0.7342 | +0.0508 |
| MUY_LARGA | 31-+ | 167 | 0.5606 | 0.7195 | +0.1589 |

## La interseccion de errores

| bin | etiqueta | n | errores de V03 | cerca del borde | alta confianza | rescatados | danados |
|---|---|---:|---:|---:|---:|---:|---:|
| CORTA | HS | 167 | 13 | 4 | 9 | 4 | 15 |
| CORTA | TR | 167 | 6 | 1 | 5 | 1 | 6 |
| CORTA | AG | 167 | 14 | 3 | 11 | 3 | 13 |
| MEDIA | HS | 602 | 127 | 52 | 75 | 22 | 31 |
| MEDIA | TR | 602 | 77 | 28 | 49 | 17 | 35 |
| MEDIA | AG | 602 | 101 | 41 | 60 | 22 | 42 |
| LARGA | HS | 547 | 128 | 51 | 77 | 16 | 39 |
| LARGA | TR | 547 | 63 | 20 | 43 | 8 | 18 |
| LARGA | AG | 547 | 115 | 53 | 62 | 25 | 37 |
| MUY_LARGA | HS | 167 | 48 | 20 | 28 | 13 | 10 |
| MUY_LARGA | TR | 167 | 18 | 4 | 14 | 2 | 0 |
| MUY_LARGA | AG | 167 | 47 | 14 | 33 | 7 | 4 |

## Veredicto

```text
el conjunto de los dos bloques vale -0.0238053648 frente a V03 con 1/5 pliegues, y el control de capacidad del mismo ancho total vale -0.0323945047, luego la BRECHA especifica es +0.0085891400. Hay una senal especifica que sobrevive al control, pero no llega al gate de la fase
```

```text
MARGINAL
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p63_07_integracion_y_techo.py`. Sin push.*