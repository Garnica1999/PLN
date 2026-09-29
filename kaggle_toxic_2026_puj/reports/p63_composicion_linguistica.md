# P63 - Composicion agregada, en forma de ratios

Generado por `experiments/p63_05_composicion_linguistica.py`. Fecha 2026-09-29T06:28:07+00:00.

## La distincion sobre la que se construye este experimento

P62 cerro una familia, y leer QUEE cerro importa mas que el hecho. Lo que midio fue un
canal POS lexico de alta dimension, `palabra__POS`, con un coste de dilucion de
+0,011585 y un valor de contenido de +0,002014, que repago el 17,4 por ciento de su
coste. Eso es una afirmacion sobre una **representacion**, no sobre la gramatica.

La lectura correcta, y la que el encargo fija, es que la composicion gramatical puede
existir como senal mientras que su representacion explicita de alta dimensionalidad no
compensa su coste en el regimen de datos disponible.

Este experimento no repite la familia cerrada: cambia la **forma** del bloque, que es
justo el eje que la curva de presupuesto de P62 marco como relevante. Un canal de
varios cientos de columnas dispersas compite por el presupuesto L2 de un espacio de
21000 dimensiones y diluye. Un punado de proporciones acotadas no compite.

## El bloque

**42 columnas**, todas proporciones acotadas e interpretables:

```text
lexico        type-token, hapax, longitud media y mediana, unicos, largas, cortas
funcional     pronombres, determinantes, preposiciones, conjunciones, auxiliares,
              adverbios, interjecciones, particulas
persona       primera, segunda y tercera, singular y plural
categoria     sustantivos, verbos, adjetivos, y las razones verbo/sustantivo y
              adjetivo/sustantivo, que es donde un modelo lineal de n-gramas no llega
ortografia    mayusculas, digitos, interrogaciones, exclamaciones, elipsis,
              elongation, puntuacion repetida, comillas, URL, menciones, hashtags
```

Las proporciones de persona y de categoria salen de la anotacion de spaCy ya
autorizada por D-003 y ya construida en P62. Ningun lexicon cerrado, ningun corpus
externo, ningun tagger nuevo. La proporcion de una categoria es un mecanismo
**distinto** del token que lleva la categoria, y de ahi la diferencia con P62.

**Seguridad de pliegue de la frecuencia.** El hapax y el recuento de unicos dependen de
la frecuencia de documento, que seria una estadistica supervisada si se calculara sobre
todo. Se calcula SOLO con las filas de entrenamiento de cada pliegue.

## Resultados

| brazo | corrected_cv | delta vs V03 | pliegues | delta vs V03_random |
|---|---:|---:|---:|---:|
| V03_comp | 0.7669427506 | -0.0102687019 | 2/5 | 0.0083528268 |
| V03_random | 0.7585899239 | -0.0186215287 | 1/5 | - |
| comp_solo | 0.5753556705 | -0.2018557820 | 0/5 | - |
| random_solo | 0.4698541557 | -0.3073572969 | 0/5 | - |

## La pregunta que decide el veredicto

```text
V03 mas composicion:  -0.0102687019 con 2/5 pliegues
V03 mas ALEATORIO:    -0.0186215287 con 1/5 pliegues
BRECHA especifica:    +0.0083528268
en solitario:         0.5753556705 contra 0.4698541557 del aleatorio
```

**El veredicto no es si V03_comp le gano a V03.** Es si le gano a **V03_random**.
Una ganancia que un bloque aleatorio del mismo ancho tambien produce es capacidad, no
composicion, y este proyecto ya se dejo enganar exactamente por esa pregunta en P59-10,
donde un regularizador que parecia real era treinta veces menor que la dispersion entre
pliegues.

## Veredicto

```text
el bloque de composicion en forma de ratios vale -0.0102687019 frente a V03 con 2/5 pliegues, el control de capacidad del mismo ancho vale -0.0186215287 con 1/5, luego la BRECHA especifica es +0.0083528268. En solitario el bloque vale 0.5753556705 frente a 0.4698541557 del aleatorio, o sea +0.1055015148. Eso separa dos cosas que P62 no pudo separar: la composicion categorial SI porta informacion, y lo que no porta es una mejora que sobreviva al control de capacidad
```

```text
CLOSED. La composicion agregada no aporta sobre V03
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p63_05_composicion_linguistica.py`. Sin push.*