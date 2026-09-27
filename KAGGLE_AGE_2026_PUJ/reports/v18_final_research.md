# V18 — Informe final: cambiar el objetivo, no la representacion

- **Rama** `kaggle-age-2026-puj-v18` · **commit** `8163215`
- **Experimentos**: 8 de 14 previstos, todos con protocolo anidado
- **Resultado global**: **ninguna adopcion**. V12 sigue en **0.6005017473** pooled y 0,62188 en el leaderboard.

## 1. La pregunta y la respuesta

> ¿Puede un modelo global de seis clases usar mejor la informacion de las fronteras
> cambiando su funcion de objetivo, sin aumentar capacidad y sin nuevas representaciones?

**Respuesta: no, y la razon esta medida con cinco mecanismos distintos.**

## 2. Lo que cambio la premisa de la fase: D-048

La auditoria de P0 concluyo que la ventaja del especialista de frontera venia de la
**representacion**, y uso usar para concluirlo el 0,4944 que V13 P3a publico. Ese numero **no es la
exactitud del especialista**: `filas_actuables: 57.019` es el recuento sobre todo el dataset,
no dentro del par, y dentro de las 57.023 filas del par solo 38.424 eran actuables mientras
que 18.599 quedaban congeladas con la prediccion de V12. La mezcla daba exactamente 0,4944.

Ejecutando el codigo y las variables exactos de P3a, el especialista sobre las **279 variables
de V12** saca **0,7136** y **0,7493**, mas que el 0,6864 y 0,7218 del TF-IDF disperso, y sin
fuga: las 72 columnas supervisadas se construyen con el mismo `make_folds` agrupado.

**La senal de frontera estaba en el propio espacio de variables del modelo, todo el tiempo.**
Y el combinador de V12 extrae 0,4963 de 0,7136 disponibles.

## 3. Las cinco vias, y por que ninguna cobro la senal

Las fronteras se dan en el **sistema combinado**, que es lo comparable con V12. Para P3 se anade
la medida en la **cabeza lineal sola**, que es donde actua la perdida de margen y donde el
acoplamiento de las dos fronteras se ve sin el amortiguamiento del MLP. Las dos medidas son
correctas y responder a preguntas distintas; confundirlas seria repetir justo el error que
D-048 corrige.

| Via | Mecanismo | Frontera 1 | Frontera 2 | Frontera 1 (lineal) | Frontera 2 (lineal) | Global | Delta |
|---|---|---|---|---|---|---|---|
| P1 re-peso | `sample_weight` en las tres clases centrales | 0,5441 | 0,5341 | - | - | 0.6001 | -0.0004 |
| P3 margen | penalizacion de adyacencia en el objetivo | 0.5172 | 0.4706 | 0.5257 | 0.4281 | 0.599 | -0.0015 |
| P4 auxiliar | termino aditivo, cero parametros | 0.4955 | 0.4962 | 0.4842 | 0.4868 | 0.5993 | -0.0012 |
| P5 destilacion | teacher cross-fitted, sin routing | 0.4969 | 0.4973 | - | - | 0.5991 | -0.0014 |
| P1 re-peso (nivel fuerte) | mismo mecanismo, peso mayor | 0,5786 | 0,5699 | - | - | 0,5841 | -0,0164 |
| **V12** | referencia | 0,4963 | 0,4974 | 0,4865 | 0,4833 | **0,6005** | - |

**Los mecanismos son distintos y los cinco fallan por razones distintas y coherentes:**

1. **P1 redistribuye.** La frontera sube +0,082, pero el intercambio es de suma cero y su
   localizacion es exacta: `18-24` gana +0,0170 y `13-17` paga -0,0166. El coste entero cae
   sobre la clase facil vecina.
2. **P3 pone a competir las fronteras.** En la **cabeza lineal**, donde actua la perdida de
   margen, la frontera `18-24|25-34` sube **+0,0392** (0,4865 a 0,5257) y `25-34|35-49` baja
   **-0,0552** (0,4833 a 0,4281) con lambda 0,1. En el sistema combinado el mismo efecto se
   amortigua a +0,0209 y -0,0268. Las dos fronteras comparten la clase `25-34`, luego estan
   acopladas, y con lambda 0,5 la segunda se desploma hasta 0,4013.
3. **P4 no mueve nada.** El termino aditivo, con **cero parametros extra**, mueve la frontera
   **+0,0038** como maximo, y esa cifra es de la **cabeza global** (0,4858 a 0,4896 en el brazo
   C2); en el sistema combinado el movimiento es de +0,0017. Compárese con las +0,08 que movia el
   re-peso de P1: **dos ordenes de magnitud menos con el mismo numero de parametros**. En un
   problema de seis clases la restriccion de frontera resulta redundante con el ajuste global.
4. **P5 transfiere poco.** Todos los brazos destilados superan a su propio control, asi que
   la senal del teacher **no es inerte**, pero la mejora es de +0,0008 como maximo, con 0 de
   5 folds positivos, y la frontera recupera el **0,8 %** de lo que el teacher tiene.

## 4. La explicacion comun

La macro-F1 de seis clases es una **media**, y la frontera es una **minoria de la masa**.
Todo mecanismo que la mejore pagando fuera de ella pierde, y todo mecanismo que la mejore sin
pagar fuera no consigue moverla. Eso no es una conjetura: es lo que muestran los cinco
experimentos, y ademas lo explica el caso de V17, donde un enrutamiento con la senal correcta
tampoco cobraba nada.

## 5. Respuesta a la pregunta cientifica de la fase

| Opcion | Veredicto |
|---|---|
| A. cuello de botella del objetivo | **Parcial**: el objetivo mueve la frontera (P1 +0,082),
  pero no la cobra globalmente |
| B. cuello de botella de informacion | **Falso**: hay 0,7136 disponibles (D-048) |
| C. cuello de botella de arquitectura | **No**: la destilacion no anade parametros y tampoco |
  rinde |
| D. el conocimiento del especialista se puede destilar | **Marginal**: se transfiere el 0,8 % |
| E. la senal de frontera sigue siendo local | **Cierto**: con la media de seis clases, la
  frontera no puede pesar mas |
| F. no queda senal explotable | **Cierto en la practica**: 0,0000 alcanzable frente a

## 6. Lo que queda abierto, y no es un experimento de esta fase

Una sola cosa, y es la unica accion que este informe recomienda:

> **Cambiar la funcion objetivo a la que se optimise, no la que se informa.** El problema
> esta en las fronteras de edad y el modelo se optimiza con una media de seis clases que las
> diluye. Los datos existen y estan en el espacio de variables del propio modelo. Cualquier
> esquema que entrene o seleccione **sobre las fronteras directamente** es la via que la
> evidencia deja abierta; el resto esta cerrado con medicion.

Y una consecuencia futura que este trabajo deja a mano: **V12 comprime la salida de cada
miembro a seis numeros**, y el especialista binario recupera el 0,7136 de esas mismas seis
columnas por miembro. Esa compresion es, por si sola, un objeto de estudio con nombre propio.
