# P63 - La estructura interna del tweet largo, donde V03 falla de verdad

Generado por `experiments/p63_06_estructura_tweet_largo.py`. Fecha 2026-09-29T06:31:21+00:00.

## El hallazgo que hace esta la linea abierta mas prioritaria de la fase

La correccion de cierre de P62 establecio que V03 es MEJOR en los tweets cortos, no
peor: alcanza entre 0,9615 y 0,9872 de AUC dentro de la region de hasta seis tokens y
 puntua F1 0,94 frente a 0,72 en los largos. El margen hasta el techo de umbral en la
region corta es de +0,0003, luego no hay nada ahi. El problema residual esta en los
tweets LARGOS y ambiguos.

Y un diagnostico de la fase anterior nombro el mecanismo sin llegar a probarlo. V30-01
midio, sobre el OOF canonico, que:

```text
el mayor deposito de error es la region de texto largo con densidad lexica baja y
caracteres repetidos, estable en 5 de 5 pliegues
```

y, en el mismo informe, que el efecto de la URL y de la MENCION no es de signo fijo
sino que cambia de signo con la longitud del texto, lo que describio, correctamente,
como exactamente el tipo de interaccion que un modelo lineal sobre n-gramas no puede
capturar. **Esa frase es una hipotesis que quedo escrita y nunca gasto un
experimento.** Es la hipotesis mas antigua sin gastar de todo el repositorio.

Por eso este track no abre una linea nueva: continua una que estaba abierta desde
V30-01. El mecanismo es la interaccion entre POSICION y LONGITUD, y por eso ninguna
familia cerrada lo alcanza: P62 probo la identidad del token condicionada a su
categoria, esto prueba donde se situa una estructura y como cambia con la longitud.

## Los cortes, congelados antes de calcular nada

| bin | tokens | n | F1 de V03 |
|---|---|---:|---:|
| CORTA | CORTA-0 | 167 | 0.9480 |
| MEDIA | MEDIA-7 | 602 | 0.7964 |
| LARGA | LARGA-16 | 547 | 0.6834 |
| MUY_LARGA | MUY_LARGA-31 | 167 | 0.5606 |

Estan declarados en el fuente, antes de mirar ningun resultado, y son los mismos para
todos los brazos y para todas las tablas. No se ajustan para obtener una diferencia
positiva.

## Resultados

| brazo | corrected_cv | delta vs V03 | delta en LARGA | delta en MUY_LARGA | rescate | dano |
|---|---:|---:|---:|---:|---:|---:|
| V03_pos | 0.7671031904 | -0.0101082622 | -0.0135 | -0.0126 | 83 | 114 |
| V03_random | 0.7696052430 | -0.0076062095 | -0.0139 | -0.0032 | 105 | 138 |
| pos_solo | 0.5475223449 | -0.2296891076 | -0.2701 | -0.1981 | 287 | 1427 |
| random_solo | 0.4762486729 | -0.3009627796 | -0.2593 | -0.0970 | 390 | 1955 |

### El detalle por bin, que es la lectura que la hipotesis pide

| bin | n | F1 de V03 | F1 de V03_pos | delta | delta del control aleatorio |
|---|---:|---:|---:|---:|---:|
| CORTA | 167 | 0.9480 | 0.9370 | -0.0110 | -0.0073 |
| MEDIA | 602 | 0.7964 | 0.7890 | -0.0074 | -0.0058 |
| LARGA | 547 | 0.6834 | 0.6698 | -0.0135 | -0.0139 |
| MUY_LARGA | 167 | 0.5606 | 0.5480 | -0.0126 | -0.0032 |

## La pregunta que decide el veredicto

```text
global:    bloque real -0.0101082622 (0/5) contra aleatorio -0.0076062095, brecha -0.0025020526
en LARGA:    brecha +0.000399
en MUY_LARGA: brecha -0.009437
```

La hipotesis de V30-01 situaba el mecanismo en la region larga, asi que las brechas por
bin son la lectura relevante y el global es solo el agregado.

## Veredicto

**La hipotesis de V30-01 queda refutada en su prediccion, no solo no confirmada.**
La tabla de rebanadas reproduce de forma independiente el diagnostico de P62: el F1 de
V03 cae de 0,9480 en la region CORTA a 0,5606 en MUY_LARGA, y el margen de error esta
ahi. Pero el bloque de posiciones es PEOR que un bloque aleatorio del mismo ancho
precisamente en MUY_LARGA, con una brecha de -0.009437. Si la interaccion
entre posicion y longitud fuera el mecanismo, ahi es donde tendria que aparecer.

Lo que si queda confirmado es el techo: el error crece de forma monotona con la
longitud, y eso no lo explica ninguna de las representaciones probadas en esta fase ni
en P62.

```text
el bloque de posiciones vale -0.0101082622 frente a V03 con 0/5 pliegues y la BRECHA especifica global es -0.0025020526. Y aqui esta la lectura que importa: la hipotesis de V30-01 situaba el mecanismo en la region larga, y en MUY_LARGA la brecha es -0.009437, es decir el bloque de posiciones es PEOR que un bloque aleatorio del mismo ancho precisamente donde se suponia que estaba la senal. La interaccion entre posicion y longitud NO es la que explica el deposito de error, y la hipotesis de V30-01 queda refutada en su prediccion, no solo no confirmada
```

```text
CLOSED. La estructura interna no aporta sobre V03
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p63_06_estructura_tweet_largo.py`. Sin push.*