# P74: null aleatorio replicado

`experiment_id` P74 | pre-registro `c8856cd` | inicio 2026-09-30T08:07:32+00:00

## La amenaza que ataca

P73 midio `delta A - B = +0.0316259472` con un intervalo por clusters que excluia el cero, pero con **una sola tirada** del embedding aleatorio. La pregunta es si esa ventaja se explica por una realizacion particularmente debil del null.

## El diseno

Diez replicas. En cada una, las dos ramas se **reentrenan** con la misma semilla de modelo:

```text
A_i  Word2Vec 100d CONGELADO    + TextCNN semilla i
B_i  ALEATORIO 100d CONGELADO + TextCNN semilla i
```

La unica diferencia dentro de una replica es la matriz de embeddings. Congelar una sola TextCNN A habria roto el emparejamiento.

## Verificacion de pareamiento

```text
mismas filas por pliegue     True
mismo orden de lotes         True
mismo estado del RNG         True
misma mascara de dropout     True (16163 de 16163)
misma arquitectura           70339 parametros entrenables
mismos folds, preprocessing, cobertura, vocabulario, umbrales y evaluacion
```

## Control A contra A

```text
misma semilla     delta +0.000000000000   identicas True
distinta semilla  delta -0.0007645557   (varianza de inicializacion)
```

Control del instrumento, **no** evidencia de mejora.

## Resultado por replica

| # | seed | A_CV | B_CV | delta | folds A | folds B | A gana |
|---|---|---|---|---|---|---|---|
| 1 | 20261111 | 0.597812 | 0.585865 | +0.011947 | 3/5 | 2/5 | si |
| 2 | 20261112 | 0.601477 | 0.575766 | +0.025711 | 3/5 | 2/5 | si |
| 3 | 20261113 | 0.589549 | 0.566741 | +0.022808 | 3/5 | 2/5 | si |
| 4 | 20261114 | 0.589207 | 0.580535 | +0.008672 | 3/5 | 2/5 | si |
| 5 | 20261115 | 0.592732 | 0.591042 | +0.001690 | 3/5 | 2/5 | si |
| 6 | 20261116 | 0.603531 | 0.581845 | +0.021686 | 4/5 | 1/5 | si |
| 7 | 20261117 | 0.614455 | 0.586855 | +0.027600 | 4/5 | 1/5 | si |
| 8 | 20261118 | 0.619284 | 0.570930 | +0.048354 | 5/5 | 0/5 | si |
| 9 | 20261119 | 0.616064 | 0.575383 | +0.040681 | 4/5 | 1/5 | si |
| 10 | 20261120 | 0.610744 | 0.577189 | +0.033556 | 5/5 | 0/5 | si |

## Distribucion de los diez deltas

```text
media     +0.0242705458      mediana  +0.0242594813
sd        0.0143858559      min      +0.0016901167      max +0.0483543126
IC 95 t de Student sobre replicas  [+0.0139795245, +0.0345615672]
A gana en 10 de 10 replicas
```

## Variabilidad de cada rama

```text
A  media 0.6034857136  sd 0.0112090702  rango [0.589207, 0.619284]
B  media 0.5792151678  sd 0.0074986963  rango [0.566741, 0.591042]
```

## Banda de equivalencia

```text
epsilon 0.01 | |media| 0.024271 | dentro de la banda False
```

Se reporta por separado: diferencia estimada, IC95, relacion con epsilon y numero de replicas ganadas. **La banda no convierte un positivo en negativo.**

## Bootstrap secundario

```text
replicas cuyo intervalo por clusters excluye el cero: 7 de 10
```

Secundario. El primario es el t sobre replicas y no se desplaza.

## Veredicto

**CASO 1 EVIDENCIA_REPRODUCIBLE**
