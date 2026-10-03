# Track P9: perdida orientada a la F1 (V7-01)

Generado por `experiments/v7_01_perdida_f1.py` (V7-01 (P9)).

## 1. Que se prueba

Un **unico** cambio respecto a la referencia: la funcion de perdida. Las features
son las de V03, el modelo es lineal, el protocolo es el mismo y los umbrales son
los congelados. El balanceo de clases es identico en las tres perdidas, asi que la
forma de la perdida es lo unico que varia.

La perdida es la capa de decision: es la funcion que el modelo minimiza para llegar
al punto de operacion. Y es el mecanismo mas barato de los que quedan abiertos,
porque no exige ninguna representacion nueva.

## 2. Las tres perdidas

| perdida | que es | por que se prueba |
|---|---|---|
| `BCE-control` | entropia binaria con peso de clase balanceado, el objetivo de V03 | control: demuestra que el montaje es correcto |
| `FOCAL` | perdida focal con decaimiento 2, que baja el peso de los ejemplos faciles | la F1 no cuenta los negativos verdaderos, de modo que un ejemplo facil y negativo aporta poco al objetivo aunque su perdida logaritmica sea alta |
| `F1-SUAVE` | sustituto directo de la F1 binaria sobre el conjunto entero | sustituto directo del objetivo real, sin aproximaciones intermedias |

La F1 suave se calcula sobre el conjunto entero y no por lotes, para que no dependa
de la composicion del lote, que con 1.186 filas de entrenamiento seria una fuente
de ruido pequena pero real. Por eso el ajuste es de conjunto completo.

## 3. Control de calidad

- referencia de V03: **0.777211**
- control BCE con este codigo: **0.777165**
- hueco: **-0.000046**, con tolerancia declarada de 0.0060
- veredicto: **dentro de tolerancia**

El control es BCE **con este mismo codigo**, y no los 0.777211 de V03, porque un
modelo lineal ajustado de forma iterativa no es identico a la solucion dual exacta
de liblinear. Lo que exige la seccion 32 es que el control se acerque a la
referencia, y ademas se informa del hueco: **todo delta de este script se mide
contra el control y no contra V03**, porque es la unica manera de que el delta sea
atribuible a la perdida y no al cambio de optimizador.

## 4. Resultados

| variante | contra | corrected_cv | delta | IC95 | HS | TR | AG | rescate | dano |
|---|---|---|---|---|---|---|---|---|---|
| BCE-control | V03 | 0.777165 | -0.0000 | [-0.0046, +0.0043] | 0.7603 | 0.8167 | 0.7544 | 1 | 4 |
| FOCAL | control BCE | 0.774810 | -0.0024 | [-0.0086, +0.0039] | 0.7609 | 0.8105 | 0.7530 | 6 | 4 |
| F1-SUAVE | control BCE | 0.773821 | -0.0033 | [-0.0100, +0.0032] | 0.7627 | 0.8104 | 0.7484 | 4 | 13 |

## 5. Gate de adopcion

| variante | delta al menos +0.003 | IC95 excluye 0 | pliegues positivos | gate |
|---|---|---|---|---|
| FOCAL | no (-0.0024) | no | 2/5 | no supera |
| F1-SUAVE | no (-0.0033) | no | 3/5 | no supera |

## 6. Que significa este resultado

El control BCE da 0.777165 frente a los 0.777211 de V03, un
hueco de 0.000046. Con un hueco tan pequeno, **el montaje esta
verificado**: las features, el protocolo, el modelo lineal, el optimizador y el
equivalente de la regularizacion reproducen la referencia. Por eso el resultado
negativo es atribuible a la perdida y no a la implementacion. Esa es la razon de que
el control sea la parte mas importante del experimento.

Las dos perdidas orientadas a la F1 quedan **ligeramente por debajo** del control,
y la mas directa de las dos, la F1 suave, es la peor. Ese orden tiene lectura:

- La F1 suave es el sustituto **literal** del objetivo, sin suavizado intermedio. Con
  1.186 filas de entrenamiento, optimizar el objetivo directamente acopla el
  descenso a la composicion de esa muestra concreta y sobreajusta. Es exactamente
  lo que le paso a OPT-01, donde buscar la metrica fuera de pliegue de forma directa
  dio el mejor OOF del proyecto y el peor leaderboard de los tres intentos.
- La focal suaviza ese acoplamiento y se acerca mas al control, pero sigue sin
  ganarlo.
- La entropia binaria ponderada por clase es un objetivo **proXY**: no es la F1,
  pero es un campo de Kraft suave, bien condicionado, y en 1.186 filas esa
 Propiedad vale mas que la fidelidad al objetivo.

La conclusion que sobrevive al cierre es util para el resto de la fase: **en este
problema, aproximar mejor el objetivo no es mejor**. La F1 no premia al modelo que la
optimiza directamente, sino al que tiene una condicionamiento bien puesta. Eso
refuerza el freno de transferencia: el camino a 0.80 no pasa por buscar el maximo de
la metrica local.

## 7. Freno de transferencia

El freno de la fase se aplica a partir de +0.005 de mejora en la metrica fuera de
pliegue: por encima de ese valor, un candidato solo llega a la mesa de finales si
**no** aumenta la dimension ni la brecha entre ajuste y validacion. Es la regla que
incumplieron los tres fracasos del proyecto (R1-C1, R2-MINDF5 y OPT-01), que
ganaron entre +0.015 y +0.025 fuera de pliegue y perdieron entre 0.007 y 0.013 de
leaderboard. Aqui el cambio no aumenta la dimension, porque las features son las
mismas de V03, de modo que el freno no llegaria a activarse.

Fecha: 2026-09-26T05:56:13+00:00. Duracion: 148.0 s.
