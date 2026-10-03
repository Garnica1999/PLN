# P3: estabilidad de features entre submuestras (V8-03)

Generado por `experiments/v8_03_estabilidad_features.py` (V8-03 (P3)).

## 1. De donde viene la idea

La estabilidad de rasgos es el capitulo 8 de *Inference and Disputed Authorship*, de
Mosteller y Wallace. El libro la mide traduciendo cada documento a cinco idiomas y
viendo que palabras sobreviven a la traduccion de ida y vuelta: sobreviven las que son
solidas, no las que dependen del contexto. La traduccion automatica no esta disponible
en este proyecto, asi que la pregunta se contesta con el instrumento que si hay: en
lugar de pedir que palabras sobreviven a una traduccion, se pide que pesos sobrevivan
a un **resubconjunto de las filas de entrenamiento**. Es la misma pregunta.

## 2. Como se construye, y por que no puede haber fuga

- El vectorizador se ajusta **una vez por pliegue sobre su tren completo**, de modo
  que el espacio de caracteristicas es fijo. Solo varian las filas.
- Se toman 7 submuestras sin reemplazo del
  80 % de las filas de ese tren.
- Un rasgo cuenta como estable si su peso supera el percentil
  90 de la distribucion de magnitudes en la submuestra.
- Se conservan los rasgos cuya estabilidad alcanza un umbral y se reentrena con ellos.

**La seleccion nunca ve las filas de validacion**, porque la funcion que mide la
estabilidad no recibe ningun indice exterior: solo recibe la matriz del tren del
pliegue. No hay forma de que lo haga: la funcion no recibe indices de validacion.

Control de codigo: el OOF canonico de V03 da 0.777211 sobre las mismas
filas, y los identificadores estan en el mismo orden.

## 3. Los tres umbrales de estabilidad

El espacio completo tiene **24646 caracteristicas**.

| umbral | caracteristicas conservadas | metrica | diferencia | IC95 | pliegues | gate |
|---|---|---|---|---|---|---|
| 0.5 | 1636 (1609 a 1674) | 0.773904 | -0.003308 | [-0.01218, +0.00558] | 1/5 | no supera |
| 0.7 | 927 (900 a 954) | 0.772583 | -0.004628 | [-0.01533, +0.00631] | 1/5 | no supera |
| 0.9 | 455 (432 a 479) | 0.758321 | -0.018890 | [-0.03154, -0.00610] | 0/5 | no supera |

## 4. El filtro de detectabilidad, que es el que manda

El analisis de la resolucion del leaderboard (V8-02) mostro que el externo solo
resuelve diferencias de unos 19.2 aciertos netos. Una candidata que
cambie C celdas solo es resoluble si la tasa de acierto entre esas celdas supera
`0,5 + ruido / (2C)`. **Gana en local no basta: hay que superar tambien esto.**

| umbral | celdas que cambian | filas | tasa de acierto | minimo exigido | filtro |
|---|---|---|---|---|---|
| 0.5 | 184 | 160 | 0.4402 | 0.5522 | **no pasa** |
| 0.7 | 266 | 224 | 0.4436 | 0.5361 | **no pasa** |
| 0.9 | 374 | 294 | 0.3930 | 0.5257 | **no pasa** |

## 5. Decision

**No se genera submission**: ninguna variante supera el gate de mejora.

Esta es la respuesta correcta y no un fracaso. El filtro de V8-02 existe para esto:
una linea que gana en local por debajo del minimo de acierto no es una linea que
vaya a mejorar el externo, es una linea cuyo resultado no se podria distinguir del
azar. Enviarla gastaria la sonda para obtener un numero que no informa.

## 6. El hallazgo que va mas alla del resultado

Lo importante de esta linea no es que pierde, es **como pierde**. Las tres
variantes mueven entre 184 y 374 celdas, y en esas celdas aciertan un **39 % a un
44 %**, por debajo del 50 % que daria una moneda al aire. Es decir, la seleccion por
estabilidad **mueve la decision de forma sistematica hacia peor** en las filas que
toca.

La explicacion es coherente con lo que hace el metodo. Un rasgo que mantiene peso
alto bajo siete submuestras es un rasgo de magnitud grande y muy repetida en el
corpus, no necesariamente un rasgo que separa las clases: es mas probable que sea un
rasgo **frecuente y neutro** que uno discriminante y especifico. Al conservar solo
esos rasgos se pierde justamente la senal minoritaria que separa el insulto de la
toxicidad.

El capitulo 8 de Mosteller y Wallace mide estabilidad para **atribuir autoria**,
donde un rasgo frecuente y neutro si es informativo, porque el objetivo es la
**forma** del texto y no su pertenencia a una clase. **Traducir ese criterio a un
problema de clasificacion no conserva su sentido, y las cifras lo confirman.**

Esa conclusion sobrevive a la fase: la estabilidad de features es el mecanismo
correcto para el problema del que viene y el equivocado para este. No se vuelve a
intentar con mas agresividad ni con otra constante de submuestreo, porque la
tendencia es monotona y va en la direccion contraria.

### Una distincion que hay que hacer explicita

El intervalo de confianza del umbral 0.9 **excluye cero**: la perdida es
detectable en el desarrollo. Y a la vez el filtro de detectabilidad dice que no se
veria en el externo. **No es una contradiccion**, y conviene no mezclarlas: la
primera afirmacion es sobre la metrica de validacion agrupada, que tiene mil veces
mas filas que el conjunto de prueba; la segunda es sobre si el leaderboard de 1154
filas puede distinguir un movimiento de esa magnitud. Una cosa y la otra pueden ser
ciertas a la vez, y por eso hacen falta las dos.

Fecha: 2026-09-26T05:58:16+00:00. Duracion: 12.1 s. Conjunto de prueba de 1154 filas, sin usar para nada de este script.
