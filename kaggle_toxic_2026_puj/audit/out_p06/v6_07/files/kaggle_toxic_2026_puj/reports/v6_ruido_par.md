# Track F: ruido a nivel de par instancia-etiqueta (V6-07)

Generado por `experiments/v6_07_ruido_par.py` (V6-07 (F0-F3)).

## 1. Que se prueba y por que no es una repeticion de V4-A

V4-A midio el ruido por **duplicado** y concluyo que es escaso: 2 clusters
contradictorios, 84 tuits, 4.85 % del corpus, y V03 acertaba bien dentro de ellos
(error 0.046 frente a 0.218 fuera). Ese error 0.046 se midio por *duplicado exacto*.
La pregunta abierta es si hay ruido que **no** es duplicado exacto, es decir, pares
similares pero no identicos con etiqueta opuesta. La senal `estructural` de este
track es exactamente esa generalizacion.

## 2. Deteccion: el techo se mide antes de ponderar

La sospecha se calcula en una **ronda separada** de la ponderacion, a partir del OOF
**congelado** de V03, cuya prediccion de cada fila se hizo sin esa fila. Si se
ponderara con una sospecha definida por el mismo modelo que luego se corrige, el
resultado no significaria nada.

| senal | pares marcados | que mide |
|---|---|---|
| `confianza` | 1802 de 4449 | baja confianza en la frontera: `1 - abs(2p - 1)` |
| `margen` | 1802 de 4449 | mismo eje con otro corte: `1 - 2abs(p - 0.5)` |
| `estructural` | 12 de 4449 | hay un vecino muy cercano (coseno < 0.25) con etiqueta opuesta |
| `densidad` | 855 de 4449 | fraccion de los vecinos cercanos con etiqueta opuesta |
| combinada | 739 de 4449 | media de las cuatro |

**Techo medido: 449 filas sospechosas de 1483 (30.28 %).**

Y lo que V03 hace con ellas: acierta **47 de 449** (10.5 %), frente a **293 de 1034** (28.3 %) en las no sospechosas.

### Este numero contradice la hipotesis, y por eso hay que decirlo bien

La version de este informe escrita antes de ejecutar daba por hecho que V03 ya
clasificaba bien las filas marcadas y que por eso bajar su peso no podia aportar
nada. **Los datos dicen lo contrario**: V03 acierta **mas del doble de porcentaje**
en las filas NO sospechosas que en las sospechosas
(28.3 % frente a 10.5 %). La deteccion **si** identifica
filas donde el modelo falla.

Eso abre dos lecturas posibles, y los resultados de la ponderacion las separan:

1. La deteccion encuentra **dificultad genuina**, no ruido. Las filas sonarias y
   estan en la frontera; bajar su peso **aleja al modelo de donde estan los errores**,
   que es exactamente lo que se ve al subir `lambda`.
2. La deteccion encuentra **ruido de etiqueta** y ponderar deberia ayudar. En ese
   caso un `lambda` mayor deberia mejorar, y no lo hace.

El comportamiento observado es el de la lectura 1, asi que **lo que la deteccion
marca es donde hay que estar, no donde hay que limpiar**. La senal tiene valor
predictivo de dificultad aunque no tenga valor de correccion, que son dos cosas
distintas y aqui conviene no confundirlas.

### Y la senal estructural casi no detecta nada

La senal `estructural` marca **12 pares de 4449**, frente a los 84 tuits del duplicado exacto de V4-A. Es decir, la generalizacion del duplicado a la similitud no encuentra mas
ruido: encuentra **mucho menos**. El precedente de V4-A era ya adverso, y al ampliar
la definicion de conflicto el ruido se hace mas escaso, no mas abundante.

## 3. Resultados de la ponderacion

| variante | componentes | lambda | corrected_cv | delta | IC95 | HS | TR | AG |
|---|---|---|---|---|---|---|---|---|
| F0-control | ninguno | 0.0 | 0.777211 | +0.0000 | [+0.0000, +0.0000] | 0.7617 | 0.8157 | 0.7542 |
| F1-completo | confianza, margen, estructural, densidad | 0.5 | 0.777353 | +0.0001 | [-0.0033, +0.0035] | 0.7619 | 0.8161 | 0.7540 |
| F2-estructural | estructural | 1.0 | 0.776179 | -0.0010 | [-0.0026, +0.0004] | 0.7615 | 0.8157 | 0.7513 |
| F3-agresivo | confianza, margen, estructural, densidad | 1.0 | 0.773089 | -0.0041 | [-0.0111, +0.0027] | 0.7650 | 0.8058 | 0.7484 |

## 4. Gate de dos niveles

| variante | delta >= +0.003 | IC95 excluye 0 | pliegues positivos | Gate 1 |
|---|---|---|---|---|
| F0-control | no (+0.0000) | no | 0/5 | no supera |
| F1-completo | no (+0.0001) | no | 2/5 | no supera |
| F2-estructural | no (-0.0010) | no | 1/5 | no supera |
| F3-agresivo | no (-0.0041) | no | 1/5 | no supera |

Fecha: 2026-09-26T05:53:14+00:00. Duracion: 30.1 s.
