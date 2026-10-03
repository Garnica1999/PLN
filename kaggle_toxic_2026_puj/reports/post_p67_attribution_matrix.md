# Post-P67 - Matriz de atribucion entre las siete alternativas

Generado por `experiments/p67_04_gap_analysis.py`. Fecha 2026-09-29T17:15:50+00:00.

## Que mide

Para cada par de candidatos, cuantas celdas rescatan **los dos** y cuantas **solo uno**.
El coeficiente de Jaccard sobre los rescates dice si dos alternativas estan
descubriendo el mismo mecanismo latente o dos distintos.

- Poblacion B (V03 falla y alguien acierta): **215 celdas**.
- Rescates exclusivos de un solo candidato, sumados: **122 celdas**, el
  `56.7 por ciento` de B.

## Rescate unico por candidato

| candidato | rescata | exclusivo | compartido | porcentaje de B |
|---|---:|---:|---:|---:|
| `MI_arbol` | 139 | 77 | 62 | 35.8% |
| `huecos_1_2` | 70 | 16 | 54 | 7.4% |
| `negacion` | 53 | 14 | 39 | 6.5% |
| `cohesion` | 36 | 6 | 30 | 2.8% |
| `huecos_6_10` | 44 | 5 | 39 | 2.3% |
| `word_1_3` | 33 | 4 | 29 | 1.9% |
| `contraste` | 6 | 0 | 6 | 0.0% |

## Los pares

| par | rescate compartido | solo el primero | solo el segundo | Jaccard | dano compartido |
|---|---:|---:|---:|---:|---:|
| `word_1_3|huecos_1_2` | 26 | 7 | 44 | 0.3377 | 37 |
| `word_1_3|huecos_6_10` | 19 | 14 | 25 | 0.3276 | 28 |
| `huecos_1_2|huecos_6_10` | 27 | 43 | 17 | 0.3103 | 32 |
| `cohesion|huecos_6_10` | 13 | 23 | 31 | 0.1940 | 17 |
| `MI_arbol|huecos_1_2` | 32 | 107 | 38 | 0.1808 | 21 |
| `negacion|huecos_6_10` | 14 | 39 | 30 | 0.1687 | 14 |
| `cohesion|huecos_1_2` | 15 | 21 | 55 | 0.1648 | 19 |
| `negacion|cohesion` | 12 | 41 | 24 | 0.1558 | 14 |
| `word_1_3|cohesion` | 9 | 24 | 27 | 0.1500 | 13 |
| `negacion|word_1_3` | 11 | 42 | 22 | 0.1467 | 12 |
| `MI_arbol|huecos_6_10` | 23 | 116 | 21 | 0.1437 | 22 |
| `MI_arbol|negacion` | 24 | 115 | 29 | 0.1429 | 14 |
| `negacion|huecos_1_2` | 15 | 38 | 55 | 0.1389 | 16 |
| `MI_arbol|word_1_3` | 14 | 125 | 19 | 0.0886 | 17 |
| `MI_arbol|cohesion` | 14 | 125 | 22 | 0.0870 | 13 |
| `negacion|contraste` | 4 | 49 | 2 | 0.0727 | 2 |
| `contraste|huecos_6_10` | 2 | 4 | 42 | 0.0417 | 5 |
| `MI_arbol|contraste` | 4 | 135 | 2 | 0.0284 | 5 |
| `huecos_1_2|contraste` | 2 | 68 | 4 | 0.0270 | 5 |
| `word_1_3|contraste` | 1 | 32 | 5 | 0.0263 | 5 |
| `cohesion|contraste` | 1 | 35 | 5 | 0.0244 | 2 |

## La lectura

Un Jaccard **alto** entre dos candidatos dice que ven lo mismo. Un Jaccard **bajo** dice
que aportarian diversidad, pero tambien que cada uno ve muy poco.

**No se promedian predicciones aqui.** La matriz sirve para decidir si un ensemble
tendria sentido, y la regla del proyecto es que no se crea un ensemble solo porque
exista diversidad superficial: P61 ya lo cerro.

> V03 remains the best verified leaderboard result.
