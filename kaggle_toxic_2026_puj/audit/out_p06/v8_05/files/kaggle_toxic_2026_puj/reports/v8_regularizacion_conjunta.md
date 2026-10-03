# P4: regularizacion conjunta entre las tres etiquetas (V8-05)

Generado por `experiments/v8_05_regularizacion_conjunta.py` (V8-05 (P4)).

## 1. Por que esta linea y no las otras tres

Con P1, P3 y P2 cerradas, quedan P4, P5, P6 y P7. Las tres ultimas solo mueven la capa
de decision, y el filtro de detectabilidad de V8-02 las deja con prioridad baja.
**P4 es la unica que toca el modelo y no la decision**, y por eso es la unica que puede
cambiar muchas celdas de golpe. Ademas es el unico mecanismo de estructura compartida
entre etiquetas que no cae en cadenas, ni en powerset, ni en apilamiento, los tres
cerrados en fases anteriores.

## 2. Que hace el termino nuevo

V03 ajusta **tres regresiones logisticas independientes**, y cada una penaliza solo su
propio vector de pesos. Nada obliga a que las tres cabeceras se parezcan. El termino
nuevo penaliza la dispersion entre las tres:

```text
minimizar  suma de las tres perdidas logaritmicas, con la clase equilibrada dentro
         mas la penalizacion cuadrada de cada cabecera, equivalente a la constante 2.0
         mas un peso por la distancia de cada cabecera a su media
```

Con el peso en cero esta implementacion es **una reimplementacion de V03**, porque usa
un optimizador propio. Por eso la comparacion de la linea es **interna**, de peso
cero contra peso mayor. Medir contra el OOF canonico mezclaria el efecto de la
regularizacion conjunta con el de haber cambiado de optimizador, y no se podrian
separar.

Es regularizacion **blanda**: no dice que las cabeceras sean iguales, solo que alejarlas
de su media cuesta. Esa es la diferencia con las cadenas y con el apilamiento, que
imponen relaciones exactas y ya estan cerrados.

## 3. Validacion del codigo

- OOF canonico de V03: 0.777211
- control interno, con peso conjunto cero: 0.775652
- diferencia: -0.001559, con tolerancia declarada de
  0.004

La tolerancia es mas ancha que la de otras lineas porque aqui se compara una
reimplementacion contra el original, no el mismo codigo dos veces. La diferencia
esperable viene de que el optimizador propio converge a un punto ligeramente distinto
que el de la biblioteca, no de un error.

## 4. Los tres pesos, medidos contra el control interno

| peso | metrica | diferencia | IC95 | pliegues | celdas | tasa | filtro |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.775652 | +0.000000 | [+0.00000, +0.00000] | 0/5 | 0 | nan frente a nan | no pasa |
| 0.5 | 0.774181 | -0.001471 | [-0.00503, +0.00234] | 0/5 | 43 | 0.4186 frente a 0.7233 | no pasa |
| 2.0 | 0.768872 | -0.006781 | [-0.01242, -0.00096] | 0/5 | 107 | 0.3925 frente a 0.5897 | no pasa |

| peso | HS | TR | AG |
|---|---|---|---|
| control interno (peso 0) | 0.7646 | 0.8170 | 0.7454 |
| 0.0 | 0.7646 | 0.8170 | 0.7454 |
| 0.5 | 0.7586 | 0.8166 | 0.7474 |
| 2.0 | 0.7532 | 0.8121 | 0.7413 |

## 4 bis. Identidad de los pesos contra la biblioteca

Comparar solo la metrica con una tolerancia ancha deja pasar a un modelo equivocado
que por casualidad acierta. Comparar los vectores de pesos no. Si el coseno de cada
cabecera pasa de 0,99, la reimplementacion esta haciendo lo mismo que la biblioteca.

| etiqueta | coseno | norma propia | norma de la biblioteca | diferencia de probabilidad |
|---|---|---|---|---|
| HS | 0.999376 | 23.3275 | 23.3332 | 3.17e-02 |
| TR | 0.987005 | 21.8567 | 21.8042 | 1.40e-01 |
| AG | 0.994423 | 23.2293 | 23.2497 | 9.50e-02 |

## 5. El mecanismo actua de verdad

Un riesgo de esta linea es que el termino no haga nada y los resultados sean ruido del
optimizador. La comprobacion es directa: si el mecanismo funciona, **subir su peso debe
parejar las tres cabeceras**, y eso se mide con el coseno entre sus vectores de pesos.

| peso | coseno medio | coseno minimo | norma media |
|---|---|---|---|
| 0.0 | 0.7380 | 0.6977 | 22.8045 |
| 0.5 | 0.8265 | 0.8014 | 22.1164 |
| 2.0 | 0.9216 | 0.9064 | 21.4552 |

## 6. Ajustes y convergencia

| pliegue | peso | iteraciones | objetivo |
|---|---|---|---|
| 1 | 0.0 | 34 | 0.976042 |
| 1 | 0.5 | 35 | 0.998571 |
| 1 | 2.0 | 37 | 1.033067 |
| 2 | 0.0 | 35 | 0.978336 |
| 2 | 0.5 | 33 | 1.000721 |
| 2 | 2.0 | 36 | 1.034878 |
| 3 | 0.0 | 34 | 0.984924 |
| 3 | 0.5 | 34 | 1.007377 |
| 3 | 2.0 | 37 | 1.041742 |
| 4 | 0.0 | 34 | 0.963020 |
| 4 | 0.5 | 34 | 0.985314 |
| 4 | 2.0 | 36 | 1.019473 |
| 5 | 0.0 | 35 | 0.982507 |
| 5 | 0.5 | 34 | 1.004796 |
| 5 | 2.0 | 36 | 1.038952 |

## 7. Decision

**No se genera submission**: no superan el gate: 0.0, 0.5, 2.0.

Que el mecanismo actue no significa que mejore, y conviene separar las dos cosas.
Parejar las tres cabeceras es justamente lo que haria un modelo mas simple, y V03 ya
ocupa ese sitio con la constante 2.0 sobre la representacion completa. Compartir
estructura entre cabeceras solo tendria sentido si el techo no estuviera en la
representacion, y el analisis de V8-02 situa el techo ahi.

Fecha: 2026-09-26T05:58:48+00:00. Duracion: 31.2 s.
