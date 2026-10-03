# P67 - Interpretacion correcta del resultado

Generado por `experiments/p67_03_cierre_documental.py`. Fecha 2026-09-29T17:10:47+00:00.

## La cadena de afirmaciones que este documento separa

```text
existe senal
  != la senal es atribuible a MI_arbol
  != la senal es explotable
  != la senal es suficientemente robusta para adopcion
```

Son cuatro afirmaciones distintas. La evidencia de P67 sostiene **solo la primera**.
Mezclarlas es el error que este documento existe para evitar.

## Lo que la evidencia SI permite afirmar

### MI_arbol contiene cierta informacion predictiva residual

Dos piezas independientes lo sostienen:

- El AUC de su residuo contra la etiqueta **fuera** de la banda de margen es `0.579597`, sobre 0,5.
  Fuera de la banda no hay umbral que mover, luego un desplazamiento de frontera no
  puede producir ese numero para la etiqueta.
- Su residuo tiene desviacion `0.8338` frente a `0.2193` de `huecos_1_2`, el mas redundante de los seis: un factor `3.80`.
  Los seis restantes tienen correlacion con `z_V03` entre `+0,9294` y `+0,9988`; MI_arbol
  es el menos colineal de todos, y el unico no lineal.

## Lo que la evidencia NO permite afirmar

### NO: MI_arbol es la fuente exclusiva del beneficio

La sustitucion cruzada da delta positivo con **4 de los 7 candidatos**
bajo exactamente la misma operacion:

| candidato | delta real | delta barajado |
|---|---:|---:|
| `MI_arbol` | +0.0026743167 | -0.0014390421 |
| `negacion` | +0.0011549795 | -0.0054335321 |
| `word_1_3` | -0.0001400584 | +0.0038763165 |
| `cohesion` | +0.0000325761 | -0.0070646241 |
| `huecos_1_2` | +0.0003505310 | -0.0064731429 |
| `contraste` | +0.0000000000 | -0.0017878950 |
| `huecos_6_10` | -0.0028447979 | +0.0004009382 |

MI_arbol es el mas alto de los siete, pero cuatro positivos no son una firma. Parte del
efecto pertenece a la correccion residual **como operacion**, no a la senal de MI_arbol.
Por eso la categoria lleva `ATRIBUCION_PARCIAL` en el nombre.

### NO: la calibracion explica el efecto

Al reves. El control mas barato posible de calibracion, un escalar por etiqueta
reentrenado de forma anidada, da `-0.0029731513`: **peor** que el umbral
congelado. Si algo queda demostrado con claridad en P67 es que el efecto **no** es
calibracion. El umbral congelado de V03 ya estaba bien puesto en esta particion.

### NO: el residual blending es suficientemente robusto para adopcion

El IC95 por grupos de near-duplicados del mecanismo residual es
`[-0.004431163444677563, 0.010210874472249717]`,
que **incluye el cero**. Y el efecto medido son unas `4` celdas de 1483 filas.

## Dos formulaciones, y por que solo una se usa

| formulacion | veredicto |
|---|---|
| `MI_arbol es inutil` | **falso**. Tiene AUC de residuo sobre 0,5 fuera de la banda, y es el mas alto de los siete en la sustitucion cruzada |
| `V03 es optimo` | **no afirmable**. Solo se sabe que sigue siendo el mejor resultado verificado en leaderboard |

La formulacion precisa, y la que se adopta, es:

> Se detecto senal residual, pero los mecanismos de integracion evaluados no han
> conseguido aislar una mejora robusta y atribuible sobre V03.

## Por que P67 no se convierte en `CLOSED` ni en `ADOPTED`

`CLOSED` diria que la linea esta agotada. No lo esta: MI_arbol **si** tiene senal, y
queda sin explotar.

`ADOPTED` diria que hay una mejora que llevar. No la hay: nada paso el bootstrap.

`MIXED_CON_ATRIBUCION_PARCIAL` es la unica etiqueta que describe lo que hay: una senal
real, un efecto cuya atribucion no es unica, y un tamano por debajo de la resolucion de
la metrica. Convertirlo retrospectivamente en cualquiera de las otras dos seria
simplificar el resultado en una direccion u otra, y las dos serian falsas.

## El control que separa senal de operacion, y que se anadio en el cierre

El cierre necesitaba una referencia de **azar** para su figura, y al construirla
aparecio el control mas limpio de toda la fase. Vale la pena contarlo entero porque el
primer intento estaba mal.

### El error del primer intento, porque es instructivo

La primera version seleccionaba el `beta` sobre la validacion **externa** y lo aplicaba
a esa misma particion. Es leakage, y el resultado lo delataba: el azar daba
`+0,0054964453`, **el doble que la senal real**. Un control de ruido que gana a la senal
no es un hallazgo, es la firma de un error.

Corregido a la validacion **interna**, el azar da:

```text
azar crudo,    sin igualar la escala    -0.0017593058
azar igualado, escala 0.8338        -0.0030780979
senal real de MI_arbol                  +0,0026743167
```

### Y ahora el control sin grados de libertad

El `beta` seleccionado por pliegue anade una eleccion. Con el `beta` **fijo** y aplicado
a todo el mundo no hay ninguna eleccion, y la comparacion se vuelve directa:

| beta | MI_arbol, senal real | azar, escala igualada |
|---:|---:|---:|
| 0.05 | +0.0028927811 | -0.0012044456 |
| 0.10 | +0.0032694685 | -0.0025007984 |
| 0.15 | +0.0034274786 | -0.0036763898 |
| 0.20 | +0.0029507631 | -0.0055030518 |
| 0.30 | +0.0007597901 | -0.0098575594 |
| 0.50 | +0.0018953867 | -0.0239503053 |

**La senal real es positiva en los seis betas. El azar es negativo en los seis.**

Eso separa senal de operacion sin depender de ninguna seleccion, y es la evidencia mas
limpia que tiene el proyecto a favor de que MI_arbol lleva informacion que la operacion
consigue extraer. La desviacion del residuo es la misma en los dos casos, luego no se
esta midiendo magnitud de perturbacion sino contenido.

## La equivalencia algebraica que explica todo lo anterior

El mecanismo residual de P66 se presento como distinto de un blend. **No lo es.**

```text
R = z_C - (a + b0 * z_V03)
z + beta*R = (1 - beta*b0)*z_V03 + beta*z_C - beta*a
```

Comprobado numericamente: la diferencia entre las dos formas es `8.882e-16`, que es
ruido de coma flotante. El mecanismo `residual` es **exactamente** un blend lineal de
los dos logits mas un desplazamiento de umbral.

Consecuencia para el gap analysis: P66 creyo medir cinco mecanismos distintos y en
realidad midio menos. `residual` pertenece a la familia de `blend_logit`, y lo unico
que aporta de nuevo es el desplazamiento de umbral.

## El control de azar, que faltaba

La misma operacion con un score de **puro ruido** da `-0.0030780979` con la
escala igualada. Es un suelo mas profundo que el barajado, porque el barajado conserva
la distribucion marginal del candidato y el ruido no conserva nada. Los cuatro niveles
quedan asi:

| referencia | delta | que mide |
|---|---:|---|
| score real | +0,0026743167 | la senal del candidato mas la operacion |
| score barajado | -0,0014390421 | la operacion sin senal, con la marginal intacta |
| score de azar, crudo | -0.0017593058 | la operacion con nada, perturbando 1.68x mas |
| score de azar, escala igualada | -0.0030780979 | la operacion con nada, perturbando lo mismo |

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p67_03_cierre_documental.py`. Sin push.*