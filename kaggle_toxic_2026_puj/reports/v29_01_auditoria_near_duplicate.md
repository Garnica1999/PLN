# V29-01: auditoria del metodo de near-duplicate actual

Generado por `experiments/v29_01_auditoria_near_duplicate.py` (V29-01 (auditoria del metodo de near-duplicate actual y de su cobertura)).

**Estado `AUDIT_ONLY`. Alcance: auditoria documental y numerica. No se entrena nada, no se produce ninguna senal para un modelo y no se genera submission**

## 1. El metodo, documentado

| elemento | valor |
|---|---|
| normalizacion | minusculas, URLs sustituidas por espacio, espacios colapsados |
| no normaliza | menciones, hashtags, puntuacion ni digitos |
| representacion | `TfidfVectorizer(analyzer='char_wb', ngram_range=(4,4), min_df=1)` |
| similitud | coseno (producto de filas con norma L2) |
| umbral de union | 0.85 |
| agrupacion | componentes conexas con union-find |
| datos | train + test concatenados (2885 filas); el test participa en las componentes |

Los otros dos umbrales del modulo (`0.9` y `0.8`) solo se usan para el resumen descriptivo, no para agrupar.

## 2. Controles

| control | resultado |
|---|---|
| identidad del particionado frente al CSV congelado | **True** |
| coseno de si mismo, desviacion maxima | 0.00e+00 |
| simetria, asimetria maxima | 0.00e+00 |
| coseno explicito vs producto, error | 2.22e-16 |
| pares sobre el umbral | 29 |
| grupos de tamano 2 o mas | 4 |

El control 1 compara el **particionado**, no las etiquetas enteras de grupo: lo que tiene
que reproducirse es quien esta junto a quien, no el numero que el union-find asigne.

## 3. De donde sale el 7,08 %

Esta es la tabla que reencuadra la fase.

| fuente | filas del desarrollo en grupo >= 2 | cobertura |
|---|---:|---:|
| near-duplicate puro | 6 | **0.0040** |
| pseudo-autor (primera mencion `@usuario`) | 99 | 0.0668 |
| grupos fusionados, los que usa V03 | 105 | 0.0708 |

**99 de las 105 filas entran solo por la
heuristica de pseudo-autor.** Hay {d['autores_distintos']} autores distintos y solo
822 filas tienen mencion, de las cuales se agrupan muy pocas.

La consecuencia practica es directa: **relajar la similitud textual no es relajar el
criterio que produjo el 7,08 %**, porque ese criterio es otro. El techo al que se puede
aspirar relajando similitud es el 0,40 %, no el 7,08 %. Y el 0,40 % es exactamente el
grupo que V28-03 encontro contradictorio.

## 4. Por que 0,40 %, separando umbral de normalizacion

| umbral | pares | grupos >= 2 | cobertura del desarrollo | grupo maximo |
|---:|---:|---:|---:|---:|
| 0.95 | 22 | 3 | 0.0020 | 6 |
| 0.90 | 24 | 4 | 0.0027 | 6 |
| 0.85 | 29 | 4 | 0.0040 | 6 |
| 0.80 | 37 | 7 | 0.0067 | 7 |
| 0.75 | 59 | 10 | 0.0115 | 13 |
| 0.70 | 96 | 17 | 0.0196 | 20 |
| 0.60 | 222 | 29 | 0.0303 | 27 |
| 0.50 | 519 | 58 | 0.0668 | 47 |
| 0.40 | 1163 | 116 | 0.1436 | 71 |
| 0.30 | 2709 | 126 | 0.2825 | 336 |

Y el efecto de la normalizacion, manteniendo el mismo metodo:

| umbral | normalizacion historica | normalizacion agresiva | ganancia |
|---:|---:|---:|---:|
| 0.85 | 0.0040 | 0.0303 | **+0.0263** |
| 0.80 | 0.0067 | 0.0337 | **+0.0270** |
| 0.75 | 0.0115 | 0.0465 | **+0.0351** |
| 0.70 | 0.0196 | 0.0607 | **+0.0411** |
| 0.60 | 0.0303 | 0.0924 | **+0.0620** |
| 0.50 | 0.0668 | 0.1639 | **+0.0971** |

La similitud maxima entre pares es 1.0000 y hay
29 pares por encima de 0,85. Con solo
1731 filas de train hay 1497315 pares train-train, asi que
con 1731 filas de train hay 1,5 millones de pares, de modo que MinHash o LSH no aportan nada: la matriz densa cabe en memoria y es mas simple de auditar.

## 5. Lo que este script no hace

- No entrena ningun modelo.
- No produce ninguna senal para un predictor.
- No genera submission.
- El test se lee unicamente para reproducir el calculo historico identico.

Coste: 1.5 s, memoria 426 MB, 0 modelos, 0 submissions.

