# V16-03: tres sondas de capacidad y regularizacion alrededor de V03

Generado por `experiments/v16_03_vecinos_de_capacidad.py` (V16-03 (tres sondas de capacidad y regularizacion alrededor de V03)).

## 1. Correccion al encargo, y por que era necesaria

El encargo tomaba V03 con min_df=1 y proponia min_df=2 como tercera sonda. V03 tiene min_df=2, verificado en las constantes PARAMS_WORD y PARAMS_CHAR del arnes validado, en el manifiesto de la submission campeona y en el pipeline congelado. Ademas el proyecto ya envio V03 hoy, ref 56571480, y obtuvo 0.78306. La tercera sonda habria sido V03 misma. Se corrige a min_df=1, que es el vecino inmediato por debajo del valor real y el mismo salto de una muesca que el encargo pretendia

## 2. Las tres sondas

| sonda | unico parametro que cambia | corrected_cv | delta CV | celdas que cambian | neto |
| --- | --- | --- | --- | --- | --- |
| v16_ch7_C1p5 | C, de 2.0 a 1.5 | 0.776653 | -0.000558 | 101 | +5 |
| v16_ch8_C2p5 | C, de 2.0 a 2.5 | 0.776752 | -0.000459 | 115 | -17 |
| v16_ch9_mindf1 | min_df, de 2 a 1 | 0.765532 | -0.011680 | 140 | -22 |

## 3. La pregunta que responde cada sonda

**v16_ch7_C1p5**: V03 esta en el pico de la curva LB contra capacidad, con v01 en 4k features y C=1 dando 0.75677 y v05 en 32k y C=3.26 dando 0.77602. El pico esta en C=2.0. Esta sonda pregunta si el pico es ancho o agudo, reduciendo la regularizacion un paso

**v16_ch8_C2p5**: La direccion contraria de la misma sonda. Si el pico fuese agudo, subir C empeoraria de forma medida y la curva quedaria cerrada por los dos lados

**v16_ch9_mindf1**: El corte de frecuencias controla el tamaño del vocabulario y por tanto la capacidad de forma distinta de C. Con min_df=2 el espacio tiene 24646 dimensiones; con min_df=1 crecera. Esta sonda separa 'mas capacidad' de 'mas regularizacion', que hasta ahora solo se han explorado juntas

## 4. Invariantes comprobados

- Los thresholds son los de V03 en las tres sondas, sin cambio
- Un solo hiperparametro cambia por sonda
- Ninguna feature nueva, ningun modelo nuevo
- El test se usa solo para transformar; ningun modelo se ajusta con el
- Mismo protocolo de generacion: ajuste sobre todo el desarrollo y prediccion del test

## 5. Como se leen los resultados

El criterio de adopcion es el leaderboard, no el CV local. El historial de esta fase ha
demostrado que el CV local **anti-correlaciona** con el LB: las cuatro sondas con CV
superior a V03 que se han enviado transferieron a la baja. Por eso las tres sondas de
aqui se envian aunque su CV local sea peor, y el veredicto se toma del score externo.


Fecha: 2026-09-27T01:53:25+00:00. Duracion: 27.5 s.
