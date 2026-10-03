# P67 - Reconciliacion de las cifras de P66

Generado por `experiments/p67_00_reconciliacion.py`. Fecha 2026-09-29T16:41:33+00:00.

## Por que este script va antes que cualquier fusion

P66 dejo dos cifras que no coinciden, y el encargo prohibe continuar hasta que cada una se trace a un script, una version de codigo y un estado de anidamiento. Y la conciliacion no es un tramite: una revision independiente encontro que una de las dos cifras es un **mecanismo distinto** y no una version corregida del mismo, y que las cifras de atribucion que se construyeron encima estaban calculadas contra un suelo y no contra un cero.

## Lo que la reconciliacion concluye

```text
+0,0033068923   MI_arbol con MEZCLA LINEAL de probabilidades, 3 de 5 pliegues
+0,0026743167   MI_arbol con CORRECCION RESIDUAL anidada,     4 de 5 pliegues
```

No son dos versiones de un experimento. La primera es una mezcla convexa de dos scores y la segunda es una regresion del candidato sobre V03 seguida de una correccion de lo que la regresion no explica. Difieren en +0,0006, que a esta resolucion es **menos de una fila** de las 1483, luego la eleccion entre ellas se tomo sobre una diferencia menor que el ruido.

## CUANTAS CELDAS ES CADA CIFRA, que es la unidad real del efecto

| magnitud | delta | celdas sobre 1483 |
|---|---:|---:|
| blend de probabilidad | +0.0033068923 |   +4.96 |
| correccion residual | +0.0026743167 |   +4.01 |
| IC95 inferior | -0.0044311634 |   -6.65 |
| IC95 superior | +0.0102108745 |  +15.32 |
| control barajado del blend | -0.0060147443 |   -9.02 |
| control barajado del residual | -0.0014390421 |   -2.16 |

El efecto que se discute son **unas 4 celdas** sobre 1483, y su intervalo de confianza abarca de **-7 a 15 celdas**.

Eso no es una mejora pequena con imprecision. Es un efecto **por debajo de la resolucion** con la que la metrica puede distinguirlo, y ninguna cantidad de experimentos con el mismo diseno lo va a convertir en otra cosa.

## La tabla de reconciliacion

| cifra | mecanismo | clasificacion | estado final |
|---:|---|---|---|
| +0.0033068923 | MI_arbol con MEZCLA LINEAL de probabilidades | NON_COMPARABLE | NO es la version corregida de la otra cifra. Es otro mecanismo, con menos pl |
| +0.0026743167 | MI_arbol con CORRECCION RESIDUAL anidada | FINAL_VALID_P66_RESULT | es el unico valor que cumple delta >= +0,0015 y >= 4/5 pliegues. Es el resul |
| +0.0041133588 | delta del residual menos el delta del control baraja | CORRECTED | reinterpretada: no es 'informacion atribuida'. Es 'cuanto le costaria a la m |
| +0.0093216366 | delta del blend menos el delta del control barajado | INVALIDATED | casi dos tercios de esta cifra son el dano evitado de usar ruido, no senal d |
| -0.0060147443 | la MISMA operacion con el score del candidato baraja | ORIGINAL | el suelo. Su profundidad es en si misma informacion: la operacion con ruido  |
| -0.0044311634 | extremo inferior del IC95 del residual, por grupos | ORIGINAL | equivale a perder unas 7 celdas. Es el otro extremo del hecho de que la ince |
| +0.0102108745 | extremo superior del IC95 del residual, por grupos | ORIGINAL | equivale a ganar unas 15 celdas |

### FINAL_VALID_P66_RESULT = +0.0026743167

Mecanismo: **MI_arbol con CORRECCION RESIDUAL anidada**. Es el unico valor que cumple `delta >= +0,0015` y `>= 4/5 pliegues`, y es el unico que P67 debe intentar confirmar.

## Lo que cambia: el significado de las cifras de atribucion

P66 reporto `brecha de atribucion = delta REAL - delta del control BARRAJADO`. El control barajado es un **SUELO**, no un cero: es lo que cuesta la MISMA operacion cuando el candidato no lleva senal. Restar un suelo resta el **dano evitado**.

En el blend, el control vale -0,0060 y el real +0,0033. La lectura correcta es:

```text
la operacion con senal real gana 5 celdas
la operacion con RUIDO pierde 9 celdas
la diferencia de 14 celdas son 5 de senal y 9 de dano evitado
```

**El valor defendible del blend es +0,0033 y el del residual +0,0027.** Las brechas de +0,0093 y +0,0041 dejan de ser 'informacion atribuida' y pasan a ser 'cuanto cuesta la misma operacion con ruido'.

## Lo que NO cambia

**Ninguno de los dos mecanismos se promueve.** El IC95 del residual incluye el cero y eso no depende de la aritmetica de la atribucion. Lo que cambia es el vocabulario, para que ninguna fase posterior herede una 'atribucion' que no significa nada.

## Auditoria del protocolo nested

| comprobacion | presente en el codigo | nota |
|---|---|---|
| _inner_split YA NO devuelve (ir, ir) | False | el bug era devolver el mismo conjunto dos veces. Se verifica que el codigo ACTUAL ya no lo hace, que es la forma de comprobar que la correccion llego antes del numero reportado |
| ahora usa StratifiedKFold | True | el split interno es real y estratificado por la etiqueta |
| beta se selecciona en i_va | True | beta se elige sobre inner_validation |
| a y b0 se ajustan en i_tr | True | la regresion residual se ajusta solo en inner_train |
| la normalizacion de longitud usa el entrenamiento | True | la media y la desviacion de longitud salen del entrenamiento del pliegue |

El numero reportado se produjo **despues** de las correcciones: False.

## Lo que una revision independiente apunto y P66 no midio

| falta | por que importa |
|---|---|
| tasa base de los errores en la banda del umbral | sin el total de errores de V03 y de errores en la banda, el '47 de 53' no se puede interpretar en ninguna direccion |
| AUC del residuo contra la etiqueta, estratificado por margen | es el discriminante directo entre representacion y calibracion, y no se midio |
| el perfil de margen del control barajado | sin el, el '47 de 53' no se puede ni apoyar ni refutar |
| reentreno anidado del umbral | un escalar por etiqueta. Si reproduce el efecto, todo es calibracion |
| cuantos grupos hay para el bootstrap | el IC95 depende de los grados de libertad del remuestreo |
| significacion del conteo de 4 de 5 pliegues | bajo la nula de signos, P(al menos 4 de 5) = 0,1875, luego no es evidencia |

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p67_00_reconciliacion.py`. Sin push.*