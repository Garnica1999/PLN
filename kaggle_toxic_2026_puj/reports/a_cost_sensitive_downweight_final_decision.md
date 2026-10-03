# A_cost_sensitive_downweight - Decision final

Generado por el cierre de P69. Fecha 2026-09-29.

## Los estados, que no se sustituyen entre si

| estado | valor | nota |
|---|---:|---|
| ORIGINAL | +0,0082045862 | 2026-09-25, nunca enviado, nunca auditado |
| REPRODUCED | +0,0082045862 | **R3**, diferencia 3,2e-11 |
| SECOND_PARTITION | +0.0105253130 | 5 de 5 pliegues |
| THIRD_PARTITION | +0.0048825807 | 4 de 5 pliegues |
| PERMUTATION | p = 0.000999 | 0 de 1000 superan al observado |
| FINAL_STATUS | **PROMISING_UNCONFIRMED** | 5 de 6 componentes |

## Las tres particiones

| particion | V03 | delta | folds+ | rescate | damage | neto |
|---|---:|---:|---:|---:|---:|---:|
| canonica | 0.7772114525 | +0.0082045862 | 5/5 | 99 | 74 | +25 |
| segunda | 0.7674405718 | +0.0105253130 | 5/5 | 86 | 74 | +12 |
| tercera | 0.7739813065 | +0.0048825807 | 4/5 | 80 | 70 | +10 |

```text
media     +0.0078708266
mediana   +0.0082045862
minimo    +0.0048825807
maximo    +0.0105253130
positivas 3 de 3
```

**Tres de tres positivas.** La media de las tres es `+0,0078708266`. El minimo, que es la
tercera, sigue siendo positivo y con 4 de 5 pliegues.

## El test de permutacion

La pregunta era distinta de la de las particiones: no si el efecto sobrevive a otro reparto
de pliegues, sino si depende de **emparejar los pesos con los ejemplos concretos**.

```text
delta real                  +0.0082045862
nulo: media                 -0.0013209910
nulo: desviacion            0.0020508746
nulo: minimo                -0.0078934583
nulo: maximo                +0.0045971670
nulo: percentil 99          +0.0030856337
permutaciones >= observado  0 de 1000
p empirico                  0.000999
```

**El maximo de las mil permutaciones es `+0,0045971670`**, apenas la mitad del delta real.
Ninguna asignacion equivalente de los mismos pesos se acerca. La regla de pesos funciona por
**que** ejemplos reciben menos peso, no solo por cuanto peso se retira.

## La decision, con los seis componentes

| componente | cumple |
|---|---|
| tres_particiones_positivas | **SI** |
| permutacion_favorable | **SI** |
| bootstrap_por_grupos_compatible | NO |
| rescue_damage_favorable | **SI** |
| compliance_completo | **SI** |
| provenance_completo | **SI** |

```text
PROMISING_UNCONFIRMED
```

El unico componente que no cumple es el **bootstrap por grupos**, cuyo intervalo sigue
incluyendo el cero. Por eso **no se clasifica ADOPTED**: la adopcion exigiria tambien ese
componente, y es una decision separada y explicita.

## El defecto que se encontro por el camino

La primera version del test de permutacion se **nego a continuar**: los pesos reales no
reproduc?an el delta observado. La causa era real y merece quedar escrita.

```text
donde   30_supervised_weighting_v4.py, funcion ajustar_predecir
que     `if not np.isclose(multiplicadores[posicion], 1.0)` deja
        class_weight='balanced' cuando el multiplicador es 1.0
por que sklearn recalcula 'balanced' CON los sample_weight pasados a fit,
        y un diccionario explicito construido desde y no hace eso
```

Los clasificadores son distintos: los coeficientes difieren hasta `1,4e-2`. El historico
fija **siempre** el diccionario explicito; el guard no.

**A donde afecto**: la reproduccion canonica no se vio afectada, porque **carga** el OOF
historico en vez de reentrenar. Las particiones segunda y tercera si, porque se reentrenan
aqui: 2 de 15 y 3 de 15 combinaciones pliegue-etiqueta caian en el guard.

**La correccion**: un `ajustar_exacto` que replica el bloque inline historico, sin guard.
Con el, la canonica da `0,7854160387`, identica a `3,2e-11`.

```text
segunda   +0,0105253130 -> +0,0105253130   sin cambio
tercera   +0,0047946703 -> +0,0048825807   cambio en la sexta cifra
```

El signo, el numero de pliegues positivos y la conclusion **no cambian**. Pero los numeros
que se reportan son los corregidos.

## Lo que este ciclo NO hace

```text
no adopta el candidato
no genera ninguna submission
no abre una cuarta particion
no abre una grid de pesos
no toca V03
no usa el test, ni el holdout, ni el leaderboard
```

## Lo que el test de permutacion no responde

```text
no dice si V03 esta definitivamente peor
no dice si la mejora en Kaggle sera positiva
```

Son preguntas distintas, con evidencias distintas, y confundirlas es la forma mas facil de
sobreafirmar un test de permutacion.

## La respuesta a la pregunta del ciclo

> El efecto de A_cost_sensitive_downweight es suficientemente consistente como para pasar de
> MARGINAL a una evidencia mas fuerte: **tres particiones positivas y un test de permutacion
> que ninguna de mil asignaciones equivalentes alcanza**. Pero el bootstrap por grupos sigue
> incluyendo el cero, luego el estado es `PROMISING_UNCONFIRMED` y no `ADOPTED`.

> V03 remains the best verified leaderboard result.

*Sin push.*