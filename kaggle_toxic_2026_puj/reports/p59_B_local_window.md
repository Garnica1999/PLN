# P59-B - Representacion por ventanas locales sobre V03

Generado por `experiments/p59_07_local_window.py`. Fecha 2026-09-28T14:06:18+00:00.

## Motivacion e hipotesis

No es detectar mejor las celdas difiles. La hipotesis es sobre la **unidad de representacion**: V03 comprime un tweet entero en un unico vector sparse, y ese vector puede diluir evidencia toxica concentrada en pocos tokens. Una representacion por ventanas locales la preserva, porque el clasificador nunca ve el documento entero.

## Auditoria de novedad funcional

| precedente | veredicto |
|---|---|
| V4-D01-D04, 'ventanas locales como tokens' (spec 24, ventana +-3) | **NO es duplicacion**: alli la ventana es un canal sparse EXTRA apilado al documento, luego la unidad de puntuacion sigue siendo el tweet. Aqui la ventana ES la unidad de puntuacion |
| DL-02 CharCNN kernels 3 y 5 (campos locales + max) | mismo principio de agregacion, pero en familia densa y desde cero, 0,741036, descartado. B lo prueba en la familia sparse lineal |
| EBD-18/19/21/22/23, ventana de entrenamiento de FastText | mecanismo sin relacion |
| filas con multiple-instance learning | **0** |
| informes con nombre de ventana o segmento | ninguno fuera de la familia FastText |

`B_NOVELTY = TRUE`, `FUNCTIONAL_EQUIVALENCE = FALSE`.

## Configuracion, fijada antes de mirar validation

```text
ventana 32 tokens | stride 16 (solapamiento 16) | agregacion max | umbrales 0,44/0,51/0,46
tokenizacion  el patron por defecto de TfidfVectorizer
tweets de <= 32 tokens  una sola ventana = el tweet entero, sin duplicar ni relleno
entrenamiento  union de todas las ventanas del pliegue; cada ventana hereda la etiqueta de su tweet
clasificador  el de V03 | features word(1,2)+char_wb(3,5) | rejilla ninguna
```

## Factibilidad: un techo registrado antes de las metricas

Los tweets son cortos. Mediana de tokens 15, maximo 51. Con ventana 32 solo el **8.23 por ciento** de los tweets tiene mas de una ventana (122 de 1483).

```text
ventanas por tweet: media 1.0856 | max 3
tweets multi-ventana: 122 (8.23 por ciento)
celdas alcanzables: 366 de 4449 (8.23 por ciento)
distribucion: {'1': 1361, '2': 117, '3': 5}
```

Fuera de esas celdas la representacion es identica a la de V03 por construccion, luego el cambio no puede excederlas. Esto acota el experimento desde el principio y es la razon por la que un resultado negativo aqui no refuta la hipotesis local en general.

## Control

```text
brazo A corrected_cv 0.7772114525165721 | max |proba - canonica| = 0.000e+00 | celdas distintas 0
```

## Resultados

```text
corrected_cv  B1 0.7765868373
corrected_cv  V03 0.7772114525
delta_vs_V03  -0.0006246152
pliegues que mejoran 3/5
folds B1  0.7805 0.8006 0.7745 0.7400 0.7877
folds V03 0.7960 0.7940 0.7789 0.7386 0.7791
rescate 39 | damage 40 | neto -1 | cambian 79
solo B1 correcto 39 | solo V03 correcto 40
```

| etiqueta | F1 B1 | F1 V03 | rescate | damage |
|---|---:|---:|---:|---:|
| HS | 0.7654 | 0.7617 | 18 | 14 |
| TR | 0.8139 | 0.8157 | 4 | 6 |
| AG | 0.7504 | 0.7542 | 17 | 20 |

## Donde se concentran los cambios

| estrato | n tweets | cambios | ccv B1 | ccv V03 |
|---|---:|---:|---:|---:|
| corto | 478 | 14 | 0.8765 | 0.8802 |
| medio | 514 | 27 | 0.7444 | 0.7398 |
| largo | 491 | 38 | 0.6228 | 0.6272 |

```text
cambios dentro de multi-ventana:  26
cambios fuera de multi-ventana:   53
cambios con una sola ventana:    53
cambios con varias ventanas:     26
delta ccv en multi-ventana:      +0.003367
delta ccv en una-sola-ventana:   +0.000041
```

## Complementariedad por estrato alcanzable

La pregunta central del parrafo 13 es si B1 corrige errores que V03 no puede corregir con su representacion global. Separada por estrato, la respuesta es distinta a la global.

| estrato | n tweets | cambios | rescate | damage | neto | acierto en las que cambian |
|---|---:|---:|---:|---:|---:|---:|
| multi_ventana | 122 | 26 | 11 | 15 | -4 | 0.4231 |
| una_sola_ventana | 1361 | 53 | 28 | 25 | +3 | 0.5283 |

## Slices del proyecto

| rebanada | n tweets | cambios |
|---|---:|---:|
| tiene_url | 458 | 32 |
| tiene_hashtag | 144 | 9 |
| tiene_mencion | 702 | 30 |
| tiene_mencion_inicial | 605 | 23 |
| tiene_retuit | 0 | 0 |
| tiene_pregunta | 132 | 14 |
| tiene_comillas | 72 | 7 |
| texto_corto_40 | 127 | 4 |
| region_regimen_noticia | 510 | 37 |
| region_sin_url_ni_hashtag | 973 | 42 |
| cuartil_longitud_Q1 | 390 | 11 |
| cuartil_longitud_Q2 | 361 | 16 |
| cuartil_longitud_Q3 | 361 | 16 |
| cuartil_longitud_Q4 | 371 | 36 |

`tiene_insulto_lexico` NO se incluye: depende de un lexico de insultos inducido fold-local que no esta en disco, y rehacerlo seria crear una rebanada nueva. Se declara el hueco en vez de sustituirlo.

## Gate

| criterio | valor | superado |
|---|---|---|
| delta >= +0,003 | -0.000625 | **False** |
| al menos 4 de 5 pliegues | 3/5 | **False** |
| gate completo | | **False** |

Bootstrap NO ejecutado: el parrafo 16 lo condiciona al gate.

## Decision

```text
decision  CLOSED. La representacion por ventanas locales con agregacion max NO mejora la metrica. NO se rescata con mean, top-k, softmax ni max+mean (parrafo 15). Y con el techo de viabilidad registrado, el resultado no puedeerse como refutacion de la hipotesis local en general, solo de esta configuracion
```

De las 42 celdas no rescatables de P54, B1 rescata **0**.

## Formulacion

> V03 remains the best verified leaderboard result.

*Seccion generada por `experiments/p59_07_local_window.py`. Sin push.*