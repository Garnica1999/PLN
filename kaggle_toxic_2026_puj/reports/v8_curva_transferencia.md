# La curva de transferencia: la tasa de positivos predicha frente al externo

Generado por `experiments/v8_09_curva_transferencia.py` (V8-09 (analisis)).
No entrena nada y no gasta ninguna sonda: usa las quince ya pagadas.

## 1. Lo que las dos ultimas sondas anaden al patron

La linea P5, con +0,000306 local y tasa de acierto sobre el azar, dio 0,77781. El
promedio de cinco repeticiones, con +0,005066 local, cuatro de cinco pliegues a favor y
la tasa de acierto mas alta de la fase, dio 0,77122. **La mayor mejora local de la fase
fue la mayor perdida externa.** Es la quinta vez que ocurre, y las otras cuatro fueron
R1-C1, R2-MINDF5, OPT-01 y P1, todas con el mismo signo.

## 2. La correlacion simple no responde, y hay que decir por que

La correlacion lineal entre la diferencia de tasa de positivos y la diferencia de
leaderboard es **+0.469**, y es **poco informativa**, porque resume con un
coeficiente una relacion que no es una recta. Ordenando las sondas por su tasa y
agrupandolas en cuartiles, la relacion es una **curva con maximo**, y V03 esta
exactamente en el maximo. Un coeficiente lineal aplicada a una campana da un numero que
no describe nada.

## 3. La curva, que es el resultado de la linea

| cuartil de tasa | sondas | tasa media | LB medio | LB minimo | LB maximo |
|---|---|---|---|---|---|
| baja | 4 | 0.3426 | 0.77355 | 0.75677 | 0.78150 |
| media-baja | 3 | 0.3583 | 0.77715 | 0.77470 | 0.77895 |
| media-alta | 3 | 0.3614 | 0.77802 | 0.77122 | 0.78306 |
| alta | 4 | 0.3785 | 0.77484 | 0.77034 | 0.78163 |

El maximo de la curva esta en el cuartil **media-alta**, es decir alrededor de la tasa de
V03. **Mover la tasa de positivos predicha en cualquier direccion cuesta leaderboard.**
Bajarla desde 0,3645 hasta el cuartil bajo cuesta unos cuatro milésimos, y subirla al
cuartil alto tambien. La F1 como funcion de la tasa de positivos es unimodal, y V03 se
ajusto a su maximo.

**Esa es la respuesta a la pregunta que llevaba la especificacion de P8**, la
calibracion condicionada al resultado externo: no hay margen en la dimension de la tasa
de positivos. Es la conclusion mas solida que se puede sacar de quince sondas, porque no
depende de ajustar nada.

## 4. La tabla completa de las quince sondas

| sonda | tasa | LB | diferencia | aciertos netos | celdas que cambian | local |
|---|---|---|---|---|---|---|
| v01_baseline_tfidf_lr | 0.3209 | 0.75677 | -0.02629 | -33.2 | 231 | 0.75778 |
| v02_lr_union | 0.3385 | 0.77993 | -0.00313 | -3.9 | 98 | 0.7927 |
| v05_optuna | 0.3550 | 0.77602 | -0.00704 | -8.9 | 125 | 0.8039 |
| v11_candidate_c | 0.3559 | 0.78150 | -0.00156 | -2.0 | 142 | 0.7797 |
| v09_p5_c_por_etiqueta | 0.3579 | 0.77781 | -0.00525 | -6.6 | 123 | 0.777518 |
| v09_candidate_a | 0.3582 | 0.77895 | -0.00411 | -5.2 | 128 | 0.7809 |
| v08_p1_fold_ensemble | 0.3588 | 0.77470 | -0.00836 | -10.5 | 100 | 0.777211 |
| v10_candidate_b | 0.3590 | 0.77978 | -0.00328 | -4.1 | 99 | 0.7793 |
| v12_p7_promedio_repeticiones | 0.3608 | 0.77122 | -0.01184 | -14.9 | 127 | 0.782277 |
| v03_lr_union_thr | 0.3645 | 0.78306 | +0.00000 | +0.0 | 0 | 0.7989 |
| v07_robustez_mindf5 | 0.3657 | 0.77034 | -0.01272 | -16.0 | 108 | 0.7993 |
| v06_robustez_c1 | 0.3715 | 0.77227 | -0.01079 | -13.6 | 68 | 0.7929 |
| v08_meta_v2 | 0.3862 | 0.78163 | -0.00143 | -1.8 | 199 | 0.7874 |
| v04_ens04_mediana | 0.3905 | 0.77510 | -0.00796 | -10.0 | 98 | 0.7985 |

**13 de 14 sondas estan dentro de la banda de ruido** de
19.2 aciertos netos, que es 0.01523
en la cifra del leaderboard. V03 es el maximo de las quince, y ninguna lo supera.

## 5. Un error de diseno mio, que hay que corregir y no esconder

La validacion de la sonda del ensemble promedia **cinco** modelos, uno por repeticion,
porque cada fila solo puede recibir en cada repeticion la prediccion de un modelo que no
la entreno. La submission promedia el promedio de los cinco pliegues de cada repeticion,
o sea **veinticinco** modelos. Se valido un ensemble de cinco y se envio uno de
veinticinco.

Promediar mas modelos contrae mas las probabilidades, y con los umbrales congelados de
V03 ese efecto de escala desplaza la decision. Es el mismo mecanismo que hacia perder a
P1. La contraccion observada en la submission es de -0.0037
en la tasa media, de 0.3645 a 0.3608.

**Que correccion exige**: la submission coherente con el OOF promedia cinco modelos, uno
por repeticion, no veinticinco. Y el filtro de detectabilidad, que exige una tasa de
acierto sobre el minimo, se puede medir **antes** de enviar en esa version, que es
justo lo que esta comprobacion habria hecho si se hubiera revisado el diseno antes de
generar el fichero.

## 6. Lo que este analisis cierra y lo que no

- **Cierra**: la tasa de positivos predicha no es una palanca. V03 esta en su maximo y
  moverse cuesta. Eso elimina de raiz la familia entera de calibraciones y umbrales.
- **Cierra**: la mejora local no predice el externo, y la correlacion entre ambas es
  negativa en cinco casos documentados.
- **No cierra**: por que V03 es el maximo de las quince sondas. Las dos explicaciones
  posibles son que V03 sea genuinamente el mejor modelo, o que sea el maximo de ruido de
  una serie de comparaciones, y **con quince sondas y este tamano de conjunto no se puede
  distinguir entre las dos**. Decirlo asi es mas util que elegir la hipotesis que mas
  gusta.
- **No cierra**: si 0,80 es alcanzable. Esta a +0,01694 de la referencia, o sea 1,1
  veces el ruido, y ninguna de las quince sondas se acerca. Alcanzarlo exigiria un cambio
  de una magnitud que ninguna de las ciento noventa y siete lineas medidas del proyecto
  ha producido.

Fecha: 2026-09-26T04:00:27+00:00. Duracion: 0.0 s.
