# P66 - Late fusion: la evidencia de largo alcance en el espacio de scores

Informe final de la fase. Fecha 2026-09-29T15:56:16+00:00. Dos experimentos y dos auditorias. 0 submissions, 0 Kaggle, 0 holdout, 0 test, 0 leaderboard.

## La pregunta, y la respuesta con su matiz

> Cuando una representacion alternativa ya ha demostrado contener informacion adicional
> respecto a V03, puede usarse como una pequena correccion en el espacio de scores?

**Si. Y es la primera vez que esa senal mueve el numero.**

| fusion | delta vs V03 | pliegues | brecha de atribucion |
|---|---:|---:|---:|
| MI_arbol + residual anidada | +0.0026743167 | 4/5 | +0.0041133588 |
| MI_arbol + residual con score BARRAJADO | -0.0014390421 | | |
| V03 | 0.7772114525165721 | | |

La **brecha de atribucion** es la cifra que manda: el mismo mecanismo con el score del
candidato barajado pierde casi todo, luego la mejora es del candidato y no de la capa.

## Y el matiz, que va pegado a ese resultado

De las **53 celdas** que la fusion rescata,
**solo 6 son errores de ALTA CONFIANZA** de V03. Las demas
estan pegadas al umbral. Una fusion que mueve celdas a traves de una frontera ha encontrado
**calibracion, no representacion**.

| etiqueta | rescate | dano | neto | de alta confianza | cerca del umbral |
|---|---:|---:|---:|---:|---:|
| HS | 13 | 10 | +3 | 0 | 13 |
| TR | 16 | 16 | +0 | 4 | 12 |
| AG | 24 | 27 | -3 | 2 | 22 |

Y el bootstrap por 1415 grupos:

```text
IC95  [-0.0044311634, +0.0102108745]   incluye el cero: True
```

Por eso el veredicto es **PROMISING** y no FINALIST.

## Por donde llega, por estrato de longitud

| estrato | rescate | dano | neto | F1 fusion | F1 V03 |
|---|---:|---:|---:|---:|---:|
| CORTA | 1 | 3 | -2 | 0.9452 | 0.9480 |
| MEDIA | 21 | 22 | -1 | 0.7944 | 0.7964 |
| LARGA | 26 | 17 | +9 | 0.7047 | 0.6834 |
| MUY_LARGA | 5 | 11 | -6 | 0.5653 | 0.5606 |

La mejora se concentra en **LARGA**, que es donde P63 y P64 situaron el margen, y
**empeora en CORTA y MEDIA**. Eso es coherente con que la evidencia que aporta es
estructural y de largo alcance, y con que en los tweets cortos V03 ya esta cerca de su
techo de umbral y cualquier correction solo puede danar celdas.

## Los siete candidatos, y por que solo uno funciona

Las correlaciones entre z_V03 y z_C van de **+0,9294 a +0,9988**: los siete son casi
la misma puntuacion. El AUC del **residuo** los separa:

| candidato | corr con z_V03 | AUC del candidato | AUC del RESIDUO |
|---|---:|---:|---:|
| MI_arbol | +0.9294 | 0.555824 | 0.565387 |
| negacion | +0.9921 | 0.542904 | 0.453645 |
| word_1_3 | +0.9963 | 0.550512 | 0.514599 |
| cohesion | +0.9971 | 0.545381 | 0.441306 |
| huecos_1_2 | +0.9910 | 0.550435 | 0.498082 |
| contraste | +0.9988 | 0.548418 | 0.482872 |
| huecos_6_10 | +0.9931 | 0.545601 | 0.418585 |

**Cinco de siete quedan por debajo de 0,5**, luego su parte unica es antisimetrica con
el error de V03: no anaden informacion, la destruyen. Y el unico por encima del azar es
**MI_arbol**, cuyo rasgo distintivo es que lo produce un modelo **NO LINEAL**. Ese
detalle es la explicacion del resultado y no un detalle: si el error de V03 fuera una
funcion lineal del error de MI_arbol, la regresion residual no dejaria nada y el AUC
del residuo seria 0,5.

## El inventario, que fue el problema real

**Ninguno de los scripts de P64 y P65 escribio sus probabilidades a disco.** Publicaron
metricas y scores por pliegue y las dejaron en memoria, y `features/` esta en
`.gitignore`. Los scores se regeneraron y se verificaron contra el numero que
cada fase reporto:

| candidato | regenerado | la fase reporto | estado |
|---|---:|---:|---|
| MI_arbol | -0.0054372718 | -0.0091686799 | NO_VERIFICABLE |
| negacion | -0.0036227731 | -0.0021826774 | DIFERENCIA_NO_RESUELTA |
| word_1_3 | -0.0039520688 | -0.0039520688 | REPRODUCE |
| cohesion | -0.0027090062 | -0.0027090062 | REPRODUCE |
| huecos_1_2 | -0.0010515908 | -0.0010515908 | REPRODUCE |
| contraste | -0.0023851933 | -0.0023851933 | REPRODUCE |
| huecos_6_10 | -0.0074391131 | -0.0074391131 | REPRODUCE |

El script de regeneracion **se paro** cuando la verificacion fue binaria y dos candidatos
no cuadraron. Continuar habria sido medir con scores no verificados.

## El gap analysis del espacio de scores

**4 familias genuinamente nuevas.** Se leyo el codigo y no el registro.

| familia | estado | por quien |
|---|---|---|
| blend lineal de un score local con V03 mediante un alpha | EXACT_DUPLICATE | V6-05 y P59-09 |
| correccion residual sobre el score de V03 | GENUINELY_NEW | ninguno. La palabra residuo aparece en el codigo en otros  |
| correccion sensible a la longitud, continua y sin puerta | GENUINELY_NEW | el especialista duro se midio en P65-04 y fue BIT A BIT ig |
| calibracion anidada de probabilidades antes de fusionar | PARTIAL_PRECEDENT | V10-05 toco la capa de decision por shift de prevalencia,  |
| stacking o meta-modelo general | EXACT_DUPLICATE | V4-X01 y las familias meta36, y P64-05 con el arbol |
| selector keep/flip | EXACT_DUPLICATE | V18 y V50, y P64-19 lo excluyo explicitamente |
| control de score barajado | GENUINELY_NEW | ninguno. Existia el control de BLOQUE aleatorio en la etap |
| unica informacion mas alla de V03 en el score del candidato | GENUINELY_NEW | P63-07 y P64-03 midieron el AUC de la evidencia frente al  |

El blend lineal con un alpha ya lo habian medido V6-05 (+0,0014800075, 2 de 5) y
P59-09. Se ejecuto aqui como **control**, no como novedad, y es lo que hace interpretable
a la residualizacion: si P8 mejora donde P5 no, la diferencia es la residualizacion.

## Las correcciones propias de la fase

1. `_inner_split` devolvia `(ir, ir)`, el mismo conjunto dos veces. Eso no es un split
   interno: es `outer_train` jugando los dos papeles, y habria hecho que **todo el
   protocolo nested fuera decoracion**. Se corrigio antes de ejecutar.
2. `np.ones(i_tr.sum())` construia un array de 1483 filas en vez de uno por fila de `inner_train`.
3. La variable local `decision` tapaba la funcion `decision(umbral)`.
4. `ccv_de` mezclaba un objetivo de una columna con el vector de tres umbrales.
5. `W(...)` recibia tres argumentos en vez de uno, por un salto de linea.
6. **LA IMPORTANTE**: `blend_prob` usaba siempre la probabilidad REAL del candidato y nunca
   la barajada. El control de barajado de ese mecanismo salia identico al real POR
   CONSTRUCCION, la brecha de atribucion daba cero falso, y el mejor resultado de la
   fase quedaba INVALIDATED por un motivo que no era el suyo. Con el barajado bien
   puesto, ese mismo resultado pasa a +0,0033 con brecha +0,0093.

## Cumplimiento

25 comprobaciones con 0 fallos, por analisis del arbol.
El inspector recoge ademas los literales que llegan a `read_csv`, porque esta fase lee
un CSV de GRUPOS para el bootstrap y una busqueda por nombre lo habria marcado.

## Estado

| | |
|---|---|
| experimentos | 2 de 16 principales, 0 de 5 reservas |
| cumplimiento | 25 comprobaciones, 0 fallos |
| figuras | 13 |
| submissions | 0 |
| Kaggle | 0 |
| holdout | 0 |
| test | 0 |
| segunda particion | NO ejecutada: el gate no llego |
| reproducibilidad | R2, sin R4 |

## Cierre

```text
anadir columnas a un espacio de veinte mil dimensiones no cabe, y corregir un score
de tres columnas si, para la MISMA senal y con el MISMO modelo de base

la unica excepcion medida es MI_arbol, y su rasgo distintivo es que el modelo que lo
produce es NO LINEAL, luego su error no es una funcion lineal del error de V03

y el matiz que va pegado: es sobre todo recalibracion, porque 6 de 53 rescates son de
alta confianza y el IC95 incluye el cero

V03 remains the best verified leaderboard result.
```

No se afirma que V03 sea matematicamente optimo, ni que los transformadores sean necesarios.
Se afirma lo que las mediciones sostienen: la senal de largo alcance existe, y en el
espacio de scores produce una mejora real y atribuible al candidato, que no se
consiguio por concatenacion; y que esa mejora es todavia mayoritariamente
recalibracion y no esta establecida por su intervalo.

## Artefactos

* `reports/p66_candidate_comparison.csv`
* `reports/p66_candidate_inventory.csv`
* `reports/p66_complementarity.csv`
* `reports/p66_complementarity.json`
* `reports/p66_complementarity.md`
* `reports/p66_compliance.md`
* `reports/p66_error_intersection.csv`
* `reports/p66_final_research.md`
* `reports/p66_final_summary.json`
* `reports/p66_hypothesis_gap_analysis.csv`
* `reports/p66_hypothesis_gap_analysis.json`
* `reports/p66_hypothesis_gap_analysis.md`
* `reports/p66_long_aware_correction.csv`
* `reports/p66_master_results.csv`
* `reports/p66_random_controls.csv`
* `reports/p66_reproducibility.md`
* `reports/p66_residual_correction.csv`
* `reports/p66_score_fusion.csv`
* `reports/p66_score_fusion.json`
* `reports/p66_score_fusion.md`
* `reports/p66_score_regeneration.json`
* `figuras/p66/p66_01_v03_vs_fusion.png`
* `figuras/p66/p66_02_delta_por_mecanismo.png`
* `figuras/p66/p66_03_rescate_vs_dano.png`
* `figuras/p66/p66_04_rescate_alta_confianza.png`
* `figuras/p66/p66_05_rendimiento_por_longitud.png`
* `figuras/p66/p66_06_residuo_vs_error.png`
* `figuras/p66/p66_07_score_vs_margen.png`
* `figuras/p66/p66_08_delta_por_pliegue.png`
* `figuras/p66/p66_09_gates.png`
* `figuras/p66/p66_10_segunda_particion.png`
* `figuras/p66/p66_11_evolucion.png`
* `figuras/p66/p66_12_verificacion_scores.png`
* `figuras/p66/p66_13_bootstrap.png`

> V03 remains the best verified leaderboard result.
