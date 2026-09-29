# P63 - Auditoria predictiva del indice de coleccion

Generado por `experiments/p63_03_id_auditoria_predictiva.py`. Fecha 2026-09-29T06:17:01+00:00.

## La pregunta, y por que merece un experimento

P63-02 clasifico el id como `D. STRUCTURED_IDENTIFIER` con confianza alta: un entero
dentro del bloque contiguo [20001, 24999], del que train y test son muestras disjuntas
que en conjunto ocupan 2885 de 4999 posiciones. Eso es estructura, y la estructura es
una precondicion de la senal, no una garantia.

El gap analysis encontro ademas que el id **nunca fue entrada de un modelo**: V15-00 lo
excluyo declarandolo contador, y P63-02 demonstro que no lo es. La pregunta viva es
estrecha y concreta: el indice porta informacion de etiqueta que V03 no tenga ya?

## Tres controles, porque un positivo aqui seria el artefacto mas facil del proyecto

Un modelo que leyera el indice seria el positivo mas facil de falsificar del proyecto,
y por una razon estructural, no moral. Si el corpus se ensamblo en bloques y las
etiquetas derivan entre bloques por motivos que no tienen que ver con el texto, el
modelo lee el **orden de ensamblado**, brilla en validacion cruzada y muere en el
leaderboard. Tres controles separan eso de un efecto real:

- **capacidad**: el mismo modelo sobre rasgos ALEATORIOS de dimension identica. Si el
  bloque real puntua como el aleatorio, no hay senal especifica, solo la capacidad de
  memorizar una funcion suave de 1483 filas.
- **permutacion**: el mismo pipeline con las etiquetas mezcladas. Destruye cualquier
  relacion id-etiqueta y conserva la distribucion de ids y las tasas marginales.
- **segunda particion**: **NO se ejecuta.** El encargo prohibe usar el holdout durante
  la exploracion, y esto sigue siendo exploracion. Se reserva para un finalista.

## El bloque

Percentil de rango del id contra el pliegue de entrenamiento, 10 bins gruesos y
10 modulos pequenos: **21 columnas**. Deliberadamente NO se usa el id
crudo como categoria, ni millones de cubos, ni target encoding, ni label lookup, que
serian cuatro maneras de memorizar.

## Resultados

| brazo | corrected_cv | delta vs V03 | descripcion |
|---|---:|---:|---|
| V03 | 0.7772114525 | +0.0000000000 | control congelado, leido del OOF canonico |
| id_solo | 0.4191359898 | -0.3580754627 | percentil de rango, 10 bins y 10 modulos |
| random_solo | 0.4240383000 | -0.3531731525 | control de capacidad, 21 rasgos aleatorios |
| id_solo_permutado_max | 0.4379545936 | -0.3392568589 | maximo de 20 permutaciones, el suelo a superar |
| random_solo_permutado_media | 0.4192740806 | -0.3579373719 | media del control de capacidad con etiquetas mezcladas |

### El suelo de la permutacion

| brazo | media | minimo | maximo |
|---|---:|---:|---:|
| id_solo con etiquetas mezcladas | 0.4187704600 | 0.3919046168 | 0.4379545936 |
| random_solo con etiquetas mezcladas | 0.4192740806 | 0.4021571286 | 0.4328686943 |

El brazo real queda **-0.0188186038** por encima
del maximo de las 20 permutaciones del mismo brazo, y la brecha contra
el control de capacidad es **-0.0049023102**.

### Una nota sobre el gate

El delta contra V03 es -0.3580754627. El id va **solo**, luego no PUEDE alcanzar el
gate de la fase de +0.0015 ni aunque fuera perfecto, porque compite contra
un modelo que tiene todo el texto. Su valor es diagnostico: dice si la estructura del
indice contiene informacion de etiqueta, no si mejora el modelo.

## Veredicto

```text
El control de capacidad iguala o supera al bloque id, con brecha -0.0049023102, luego no hay senal especifica y solo capacidad. el brazo real no supera el maximo de las permutaciones, luego parte de lo que parecio senal era ruido. El id queda CLOSED como predictor: lo que parecia estructura se disuelve en capacidad y permutacion, que es justo lo que se esperaba de un indice de coleccion cuyo unico contenido es la posicion en un ensamblado
```

```text
CLOSED. El indice de coleccion no aporta senal especifica
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p63_03_id_auditoria_predictiva.py`. Sin push.*