# Track B: factorizacion condicional de etiquetas (V6-02)

Generado por `experiments/v6_02_label_factorization.py` (V6-02 (B1-B3)).

## 1. Que se prueba y por que no es una repeticion

Lo que ya se cerro: cadenas de clasificadores (E2-01, seis ordenes) y label powerset
(V4-E, doce variantes, mejor 0.7713). **Ninguna restringe el dominio de entrenamiento del
condicional**: la cadena entrena P(TR|X) con todas las filas y la anterior como entrada.

Aqui el modelo de TR **solo ve ejemplos con HS=1**, que es su dominio natural, y la
prediccion se compone como producto:

```text
p(TR|X) = p(HS|X) * p(TR | HS=1, X)
p(AG|X) = p(HS|X) * p(AG | HS=1, X) + (1 - p(HS|X)) * p(AG | HS=0, X)
```

No hay routing duro: la restriccion viene del producto, no de imponer TR=0 cuando
HS=0. Esa es exactamente la invariante que el diagnostico (V6-01) encontro en los datos.

## 2. Resultados

| variante | corrected_cv (frozen) | delta | IC95 | HS | TR | AG | salvage | damage |
|---|---|---|---|---|---|---|---|---|
| B0-control-marginales | 0.7772 | +0.0000 | [+0.0000, +0.0000] | 0.7617 | 0.8157 | 0.7542 | 0 | 0 |
| B1-TR-condicional | 0.7665 | -0.0107 | [-0.0184, -0.0033] | 0.7617 | 0.7836 | 0.7542 | 1 | 35 |
| B1-TR-AG-condicional | 0.7657 | -0.0115 | [-0.0231, -0.0005] | 0.7617 | 0.7836 | 0.7517 | 1 | 37 |
| B2-AG-dos-ramas | 0.7764 | -0.0008 | [-0.0075, +0.0057] | 0.7617 | 0.8157 | 0.7517 | 0 | 12 |
| B3-HS-propio | 0.7657 | -0.0115 | [-0.0231, -0.0005] | 0.7617 | 0.7836 | 0.7517 | 1 | 37 |

Referencia V03: 0.777211. El termino AG de las filas con HS=0 recibe una probabilidad
media de 0.0764, coherente con P(AG|HS=0) = 0.026
observado. La maxima p(TR) que la factorizacion puede dar a una fila con HS=0 es
0.9066, mientras que un clasificador TR
independiente puede llegar a 1.0.

## 3. Gate de dos niveles

| variante | delta >= +0.003 | IC95 excluye 0 | pliegues positivos | Gate 1 |
|---|---|---|---|---|
| B0-control-marginales | no (+0.0000) | no | 0/5 | no supera |
| B1-TR-condicional | no (-0.0107) | no | 1/5 | no supera |
| B1-TR-AG-condicional | no (-0.0115) | no | 1/5 | no supera |
| B2-AG-dos-ramas | no (-0.0008) | no | 2/5 | no supera |
| B3-HS-propio | no (-0.0115) | no | 1/5 | no supera |

El Gate 2 (segunda particion agrupada con otra semilla) solo se aplica si alguna
variante supera el Gate 1. El holdout no se usa: esta agotado.

## 4. Por que la restriccion no aporta: la prediccion de V03 ya la cumple

La invariante TR implica HS es un hecho de las **etiquetas**, y el diagnostico (V6-01)
la confirmo con 0 casos de TR=1 con HS=0. Lo que decide si factorizar sirve no es
que la invariante exista, sino si **V03 la viola en sus predicciones**:

- V03 predice **TR=1 con HS=0 en 20 filas** de 1483 (1.35 %).
- V03 predice TR=1 en **78 filas que en realidad son HS=0**: en
  esas filas el TR=1 predicho es un falso positivo **garantizado** por la invariante,
  porque ningun ejemplo del corpus tiene TR=1 sin HS=1.
- p(TR) de V03 en las filas HS=0: media 0.2115, maximo 0.9605, con umbral TR = 0.51.

**La restriccion se aplica y funciona**: la variante factorizada predice TR=1 con
HS=0 en **0 filas**, frente a 20 de V03. La invariante deja
de violarse en la prediccion, que era el objetivo declarado.

El problema es que **la restriccion no es gratis**, y ahi esta el resultado. El
producto p(HS)p(TR|HS=1) no solo prohibe TR=1 cuando HS=0: tambien **atenua** p(TR)
en toda fila con p(HS) intermedia, porque multiplica por un factor menor que 1. V03
acertaba justamente en esas filas. De ahi el saldo de B1: **1 fila rescatada frente
a 35 rotas**. El dano no viene de una restriccion mal puesta, viene del efecto de
escala sobre las filas que V03 acierta con p(HS) intermedia.

La lectura del resultado negativo es por tanto cuantitativa y no moral: **la
incoherencia que la invariante corrige (20 filas de prediccion,
1.35 % del desarrollo) es demasiado pequena para pagar
el coste de reescalar la probabilidad de TR en todas las filas.** Contra una
referencia que violara la invariante en muchas mas filas, o con un umbral de TR mas
alto, que haria subir la probabilidad de las filas coherentes, la misma linea
tendria otro resultado. Para V03 congelada, no.


## 5. Advertencia sobre los umbrales ajustados

La columna "ajustado" de la tabla reoptimiza un umbral por etiqueta con barrido
greedy **sobre el mismo OOF que se evalua**, asi que esta sesgada al alza y no es
comparable con los umbrales congelados de V03: por eso B0 sube a 0.7808 solo por
reajustar umbrales, sin haber cambiado el modelo. **El gate se evalua siempre con los
umbrales congelados**, y la columna de ajustados queda solo como dato de diagnostico.

Fecha: 2026-09-26T01:25:58+00:00. Duracion: 17.0 s.
