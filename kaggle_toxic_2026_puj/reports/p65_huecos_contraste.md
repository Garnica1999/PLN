# P65 - Pares lexicos de hueco grande y alcance de contraste

Generado por `experiments/p65_02_huecos_contraste.py`. Fecha 2026-09-29T14:40:22+00:00.

## Por que el hueco tiene que ser grande para que valga la pena medirlo

**V6-05 ya pregunto** si una palabra y otra separadas por una palabra intermedia
llevan algo que los bigramas adyacentes no llevan. Cambio el analizador del bloque de
palabras para que los pares con hueco compitan por el presupuesto L2, probo hueco 1 y
huecos 1 y 2, y dio **-0,0001 con IC95 que incluye el cero**. El vocabulario crecio un
83 por ciento y no paso nada.

Ese resultado es la razon para mirar mas lejos, no la razon para parar. A distancia uno
o dos, en un corpus con mediana de 15 tokens, el par casi siempre coaparece en otra
parte del corpus y es redundante con un bigrama ya visto. A distancia de seis a diez el
par se vuelve especifico de la composicion de ESE documento.

**El control de implementacion va incluido.** El rango de hueco 1 a 2 es exactamente el
de V6-05 y esta ahi para comprobar que la implementacion reproduce su resultado neutral.

## La EDA de alcance

| rango | filas alcanzables | porcentaje | pares | pares por fila |
|---|---:|---:|---:|---:|
| hueco_1_2 | 1483 | 100.0% | 45193 | 30.5 |
| hueco_3_5 | 1472 | 99.3% | 56744 | 38.5 |
| hueco_6_10 | 1314 | 88.6% | 68018 | 51.8 |

| contraste | filas | porcentaje | marcas | error con | error sin |
|---|---:|---:|---:|---:|---:|
| marcadores | 85 | 5.7% | 485 | 18.43% | 16.93% |

## Resultados

| brazo | corrected_cv | delta vs V03 | pliegues | neto | ancho |
|---|---:|---:|---:|---:|---:|
| hueco_1_2 | 0.7761598617 | -0.0010515908 | 2/5 | +1 | 13053 |
| hueco_3_5 | 0.7706593428 | -0.0065521098 | 2/5 | -18 | 13299 |
| hueco_6_10 | 0.7697723394 | -0.0074391131 | 1/5 | -19 | 13914 |
| CONTRASTE | 0.7748262592 | -0.0023851933 | 1/5 | -8 | 53 |
| hueco_3_10 | 0.7717492829 | -0.0054621696 | 1/5 | -10 | 22664 |
| V03_margen | 0.7661312520 | -0.0110802005 | 1/5 | -90 | 0 |
| V03_aleatorio | 0.7774237692 | +0.0002123167 | 3/5 | +10 | 0 |

### Las brechas

| brecha | valor |
|---|---:|
| hueco_1_2_contra_aleatorio | -0.0012639075 |
| hueco_1_2_contra_margen | +0.0100286097 |
| hueco_3_5_contra_aleatorio | -0.0067644265 |
| hueco_3_5_contra_margen | +0.0045280907 |
| hueco_6_10_contra_aleatorio | -0.0076514298 |
| hueco_6_10_contra_margen | +0.0036410874 |
| CONTRASTE_contra_aleatorio | -0.0025975101 |
| CONTRASTE_contra_margen | +0.0086950072 |
| hueco_3_10_contra_aleatorio | -0.0056744863 |
| hueco_3_10_contra_margen | +0.0056180309 |

## Veredicto

```text
Ninguno de los rangos ni el contraste supera al control de anchura. Y hay un dato de la propia ejecucion que conviene: el rango de hueco 1 a 2, que es exactamente el de V6-05, da -0.0010515908. El control de este script REPRODUCE el resultado neutral de V6-05, luego la implementacion de los pares con hueco es la misma, y los rangos largos que no se habian medido tampoco aportan

LECTURA: SIN_APORTE
DECISION: CLOSED
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p65_02_huecos_contraste.py`. Sin push.*