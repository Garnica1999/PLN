# Track C1: n-gramas de palabra con salto (V6-05)

Generado por `experiments/v6_05_skipgrams.py` (V6-05 (C1-0 a C1-2)).

## 1. Que se prueba y por que no es una repeticion de V4-D

V04-D cerro "interacciones como tokens" (`INSULT_NEAR_MENTION` y similares) y
"canales posicionales" (ventanas), y E2-10 cerro combinaciones forma x token. La
diferencia es que aquellos eran **canales anadidos** sobre una representacion ya
construida: una matriz extra apilada a la de V03, con sus propios pesos. Aqui se
modifica el **analizador del bloque de palabras**, de modo que las relaciones con
hueco compiten con las palabras por el mismo presupuesto de regularizacion. No se
puede anadir sin dao.

## 2. Resultados

| variante | huecos | carac. palabras | corrected_cv | delta | IC95 | HS | TR | AG |
|---|---|---|---|---|---|---|---|---|
| C1-0-control | ninguno | 2744 | 0.777211 | +0.0000 | [+0.0000, +0.0000] | 0.7617 | 0.8157 | 0.7542 |
| C1-1-hueco1 | 1 | 3874 | 0.775055 | -0.0022 | [-0.0084, +0.0039] | 0.7648 | 0.8103 | 0.7500 |
| C1-2-huecos12 | 1,2 | 5025 | 0.777107 | -0.0001 | [-0.0085, +0.0082] | 0.7644 | 0.8123 | 0.7547 |

El control C1-0 debe reproducir 0.777211 exactamente: sin skip-grams, el
tokenizador propio emite los mismos tokens que el bloque de palabras de V03 y el
vocabulario queda en 2744 caracteristicas. Si no coincide, la comparacion no vale.

## 3. Gate de dos niveles

| variante | delta >= +0.003 | IC95 excluye 0 | pliegues positivos | Gate 1 |
|---|---|---|---|---|
| C1-0-control | no (+0.0000) | no | 0/5 | no supera |
| C1-1-hueco1 | no (-0.0022) | no | 2/5 | no supera |
| C1-2-huecos12 | no (-0.0001) | no | 2/5 | no supera |

## 4. Como se lee un resultado negativo aqui

Los skip-grams anaden dimensionalidad, asi que un empeoramiento **no demuestra que
la relacion con hueco no exista**: puede ser que exista pero no compense el coste de
mas parametros libres. Por eso importa distinguir los dos casos:

- Si el vocabulario crece mucho y el resultado cae, la relacion es real pero la
  representacion ya tiene todo lo que cabe en 1.483 filas.
- Si el vocabulario crece mucho y el resultado **no se mueve**, la relacion con
  hueco no aporta senal pero tampoco molesta: es el caso neutro.
- Si el vocabulario crece poco y el resultado no se mueve, la relacion no existe.

**Este es el segundo caso, y conviene nombrarlo bien.** El bloque de palabras pasa de
2744 a 3874 caracteristicas con un hueco y a
5025 con dos huecos: un crecimiento del
83 %, y el resultado se
queda en -0.0001 con IC95 que incluye el 0. Los
skip-grams son **neutros**: ni anaden senal aprovechable ni restan capacidad.

Eso es un resultado distinto del de V4-D, donde las interacciones como tokens
costaron -0.0346. Aqui la relacion con hueco no es dannosa, lo que **confirma que no
era una repeticion de V4-D** y que la intuicion de partida era razonable; lo que
falla es la premisa de que quede senal por representar. El intercambio por etiqueta
lo muestra: con un hueco HS sube de 0.7617 a
0.7648 mientras TR baja de 0.8157 a
0.8103. No es una ganancia, es un reparto.

Fecha: 2026-09-26T05:51:31+00:00. Duracion: 50.0 s.
