# Analisis de error - MOD-03b_BEST (Fase 12, 2026-09-22)

Tarea: entender donde y por que falla el mejor modelo, cuantificar los tipos de error y derivar hipotesis dirigidas. Fuente de predicciones: `features/oof_mod03b.npz` (out-of-fold, umbrales de D-012). Sin reentrenamiento, sin test y sin tocar los scripts previos.

## Resumen ejecutivo

- Modelo analizado: `MOD-03b_BEST` con probabilidades OOF verificadas (F1 0.7927 con umbral 0.5 y 0.7989 con umbrales 0.44/0.51/0.46).
- Errores con umbrales por etiqueta: 425 falsos positivos y 375 falsos negativos sobre 3 x 1731 decisiones; 251 ejemplos fallan 1 etiqueta, 123 fallan 2 y 101 fallan las 3.
- Con umbral 0.5 el balance es 349 FP y 445 FN; los umbrales por etiqueta cambian el sesgo (mas FP, menos FN) y ganan +0.0063 de F1.
- La regla estructural `HS_pred = 0 => TR = 0` vale -0.0004 de F1 y su version completa (tambien AG) -0.0049: no es una palanca util a estos umbrales.
- Las 5 rebanadas mas debiles con n >= 40 son: `tiene_hashtag=1` (n=161, F1 0.586); `tiene_pregunta=1` (n=158, F1 0.633); `tiene_referencia_colectiva=1` (n=418, F1 0.639); `cuartil_longitud=Q4` (n=432, F1 0.645); `tiene_mencion=0` (n=909, F1 0.697).
- La region mas fragil es el regimen noticia (URL o hashtag): F1 0.702 frente a 0.829 sin URL ni hashtag, con n=599.
- Los falsos positivos de HS y AG se concentran en el registro de insulto interpersonal: de 526 textos con insulto, mencion inicial y sin referencia colectiva, 213 (40.49 %) estan etiquetados (0,0,0).
- El error de TR vive en los textos sin destinatario explicito: F1_TR 0.875 en respuestas frente a 0.728 sin mencion ni colectiva y 0.597 sin mencion con colectiva.
- Los FN de AG son agresion implicita contra grupos: 45.5 % de los FN de AG tiene referencia colectiva frente a 13.9 % de los TP.
- Se proponen 5 hipotesis nuevas (H11-H15) con evidencia, mecanismo, experimento y criterio de exito; H15 es una hipotesis de cierre de la via de calibracion fina.

## 1. Metodo y verificacion

OBSERVACION: el analisis parte de las probabilidades out-of-fold ya guardadas del mejor modelo; cualquier conclusion depende de que esas probabilidades sean las publicadas y de que el umbral este declarado.

HIPOTESIS: las probabilidades de `features/oof_mod03b.npz` reproducen el F1 documentado en `PROJECT_STATE.md` y en `reports/fase13a_refit_umbral_metricas.json`.

METODO: carga con `src.data.read_train()`, alineacion por `id`, recuento de las predicciones con los dos juegos de umbrales de D-012 (0.5 y 0.44/0.51/0.46) y calculo de las metricas con `src.evaluation.metricas_multietiqueta`. No se reentrena nada y no se re-optimiza ningun umbral.

RESULTADO: replicacion exacta; F1 0.792659 con umbral 0.5 (esperado 0.792659) y 0.798933 con umbrales por etiqueta (esperado 0.798933). Dimensiones por escenario:

| escenario     | etiqueta   |   umbral |   positivos_reales |   positivos_predichos |
|:--------------|:-----------|---------:|-------------------:|----------------------:|
| umbral_0_5    | HS         |     0.5  |                756 |                   698 |
| umbral_0_5    | TR         |     0.5  |                519 |                   526 |
| umbral_0_5    | AG         |     0.5  |                662 |                   617 |
| umbral_optimo | HS         |     0.44 |                756 |                   799 |
| umbral_optimo | TR         |     0.51 |                519 |                   520 |
| umbral_optimo | AG         |     0.46 |                662 |                   668 |

INTERPRETACION: el analisis es valido como diagnostico del modelo publicado con los umbrales ya decididos.

DECISION: usar el escenario de umbrales por etiqueta (v03, la apuesta vigente) como escenario principal del analisis y reportar el de 0.5 en paralelo.

SIGUIENTE PASO: matrices de confusion por etiqueta (seccion 2).

## 2. Matrices de confusion por etiqueta

OBSERVACION: se quieren los conteos y las tasas de acierto y error de cada etiqueta en los dos escenarios de umbral decididos en D-012.

HIPOTESIS: el modelo es mejor en TR que en HS y AG, y el ajuste de umbrales mueve el error de falsos negativos hacia falsos positivos.

METODO: matriz 2x2 por etiqueta (TP, FP, FN, TN) con precision, recall, especificidad, F1, tasa de falsos positivos (FPR), tasa de falsos negativos (FNR) y tasa de error por etiqueta.

RESULTADO:

| escenario     | etiqueta   |   umbral |   TP |   FP |   FN |   TN |   precision |   recall |     f1 |    fpr |    fnr |   tasa_error_etiqueta |
|:--------------|:-----------|---------:|-----:|-----:|-----:|-----:|------------:|---------:|-------:|-------:|-------:|----------------------:|
| umbral_0_5    | HS         |     0.5  |  567 |  131 |  189 |  844 |      0.8123 |   0.75   | 0.7799 | 0.1344 | 0.25   |                0.1849 |
| umbral_0_5    | TR         |     0.5  |  433 |   93 |   86 | 1119 |      0.8232 |   0.8343 | 0.8287 | 0.0767 | 0.1657 |                0.1034 |
| umbral_0_5    | AG         |     0.5  |  492 |  125 |  170 |  944 |      0.7974 |   0.7432 | 0.7694 | 0.1169 | 0.2568 |                0.1704 |
| umbral_optimo | HS         |     0.44 |  614 |  185 |  142 |  790 |      0.7685 |   0.8122 | 0.7897 | 0.1897 | 0.1878 |                0.1889 |
| umbral_optimo | TR         |     0.51 |  431 |   89 |   88 | 1123 |      0.8288 |   0.8304 | 0.8296 | 0.0734 | 0.1696 |                0.1023 |
| umbral_optimo | AG         |     0.46 |  517 |  151 |  145 |  918 |      0.774  |   0.781  | 0.7774 | 0.1413 | 0.219  |                0.171  |

INTERPRETACION: con umbral 0.5 el modelo subpredice (FN > FP en las tres etiquetas); con los umbrales por etiqueta el signo se invierte (FP > FN). La etiqueta peor sigue siendo AG (F1 0.7774) y la mejor TR (0.8296). HS es la etiqueta con mas errores absolutos (185 FP + 142 FN).

DECISION: considerar AG y HS como los objetivos primarios de mejora; TR tiene menos margen absoluto y ya esta en 0.83.

SIGUIENTE PASO: analizar como se combinan los errores entre etiquetas (seccion 3).

## 3. Combinaciones de errores y estructura de etiquetas

OBSERVACION: las tres etiquetas estan asociadas (phi HS-TR 0.743, HS-AG 0.831, TR-AG 0.759; seccion 6 del EDA) y TR=1 implica HS=1 en el 100 % de train, por lo que los errores no son independientes.

HIPOTESIS: parte de los errores de pares se debe a la dependencia estructural (predecir HS=0 arrastra TR) y la regla estructural deberia recuperar esos casos.

METODO: conteo de etiquetas erroneas por ejemplo, tabla por combinacion verdadera HS x TR x AG, tabla de errores conjuntos por par y evaluacion de la regla estructural en sus dos variantes sobre cada escenario de umbral.

RESULTADO (umbrales por etiqueta):

|   etiquetas_erroneas |    n |   proporcion |   falla_HS |   falla_TR |   falla_AG |
|---------------------:|-----:|-------------:|-----------:|-----------:|-----------:|
|                    0 | 1256 |       0.7256 |          0 |          0 |          0 |
|                    1 |  251 |       0.145  |        125 |         35 |         91 |
|                    2 |  123 |       0.0711 |        101 |         41 |        104 |
|                    3 |  101 |       0.0583 |        101 |        101 |        101 |

Errores por combinacion verdadera:

| combinacion_true   |   n |   ejemplos_con_alguna_mal |   ejemplos_con_3_mal |   errores_HS |   errores_TR |   errores_AG |
|:-------------------|----:|--------------------------:|---------------------:|-------------:|-------------:|-------------:|
| 0,0,0              | 949 |                       193 |                   68 |          178 |           81 |          110 |
| 1,1,1              | 491 |                        84 |                   33 |           42 |           76 |           55 |
| 1,0,1              | 145 |                        90 |                    0 |           50 |            6 |           81 |
| 1,0,0              |  92 |                        65 |                    0 |           42 |            0 |           23 |
| 1,1,0              |  28 |                        27 |                    0 |            8 |           12 |           18 |
| 0,0,1              |  26 |                        16 |                    0 |            7 |            2 |            9 |

Errores conjuntos por par de etiquetas:

| par   |   n_ambos_correctos |   n_solo_primera_mal |   n_solo_segunda_mal |   n_ambas_mal |   n_ambas_falso_negativo |   n_ambas_falso_positivo |   n_ambas_mixto |   errores_totales_primera |   errores_totales_segunda |
|:------|--------------------:|---------------------:|---------------------:|--------------:|-------------------------:|-------------------------:|----------------:|--------------------------:|--------------------------:|
| HS-TR |                1347 |                  207 |                   57 |           120 |                       44 |                       76 |               0 |                       327 |                       177 |
| HS-AG |                1291 |                  144 |                  113 |           183 |                       82 |                      101 |               0 |                       327 |                       296 |
| TR-AG |                1381 |                   54 |                  173 |           123 |                       49 |                       69 |               5 |                       177 |                       296 |

Efecto de la regla estructural:

| escenario     | variante            |   f1_media_columnas |   f1_HS |   f1_TR |   f1_AG |   delta_vs_sin_regla |   casos_TR1_con_HS0 |   casos_TR1_con_HS0_y_TR_true_1 |   casos_AG1_con_HS0 |   casos_AG1_con_HS0_y_AG_true_1 |
|:--------------|:--------------------|--------------------:|--------:|--------:|--------:|---------------------:|--------------------:|--------------------------------:|--------------------:|--------------------------------:|
| umbral_0_5    | sin regla           |              0.7927 |  0.7799 |  0.8287 |  0.7694 |               0      |                  32 |                              15 |                  38 |                              23 |
| umbral_0_5    | HS=0 => TR=0        |              0.7915 |  0.7799 |  0.8253 |  0.7694 |              -0.0011 |                  32 |                              15 |                  38 |                              23 |
| umbral_0_5    | HS=0 => TR=0 y AG=0 |              0.787  |  0.7799 |  0.8253 |  0.7558 |              -0.0056 |                  32 |                              15 |                  38 |                              23 |
| umbral_optimo | sin regla           |              0.7989 |  0.7897 |  0.8296 |  0.7774 |               0      |                  13 |                               6 |                  29 |                              20 |
| umbral_optimo | HS=0 => TR=0        |              0.7985 |  0.7897 |  0.8285 |  0.7774 |              -0.0004 |                  13 |                               6 |                  29 |                              20 |
| umbral_optimo | HS=0 => TR=0 y AG=0 |              0.7941 |  0.7897 |  0.8285 |  0.764  |              -0.0049 |                  13 |                               6 |                  29 |                              20 |

INTERPRETACION: 101 ejemplos (5.8 %) fallan las tres etiquetas y son el nucleo duro del error; 251 fallan solo una. El par con mas errores conjuntos es HS-AG (183), seguido de TR-AG (123) y HS-TR (120). Los errores conjuntos de HS-TR son 76 falsos positivos y 44 falsos negativos: el modelo pierde o inventa el par completo, no una de las dos etiquetas. TR-AG es el unico par con casos mixtos (5), coherente con que AG no esta condicionada a HS en 26 casos de train.

La regla estructural aporta -0.0004 de F1 en su version solo TR y -0.0049 en su version completa: con los umbrales por etiqueta hay 13 casos con TR=1 y HS=0, de los que 6 son TR verdaderos, y 29 con AG=1 y HS=0, de los que 20 son AG verdaderos. La regla intercambia casi tantos aciertos como errores.

DECISION: no adoptar la regla estructural; H8 queda cerrada como palanca con esta representacion (ya se habia medido -0.0057 en la Fase 4-5 y aqui -0.0048 con umbrales por etiqueta). El esfuerzo debe ir a la frontera real (secciones 4-5).

SIGUIENTE PASO: identificar por rebanadas donde se concentra el error (seccion 4).

## 4. Analisis por rebanadas

OBSERVACION: el F1 global esconde una fuerte heterogeneidad entre regimenes del corpus (respuestas cortas frente a textos noticiosos largos).

HIPOTESIS: el error se concentra en los textos largos con URL o hashtag y en los que tienen referencia colectiva, donde la anotacion es mas dificil.

METODO: 18 rebanadas definidas con regex y spaCy (POS y lemas, sin vectores), evaluadas con las predicciones OOF del escenario de umbrales por etiqueta; se exige n >= 40 para entrar en el ranking. Las rebanadas marcadas como `artefacto_lexico` usan el lexico inducido in-sample y se separan del ranking.

RESULTADO: tabla completa de rebanadas evaluables.

| rebanada                    | grupo   |    n |   f1_HS |   f1_TR |   f1_AG |   f1_media |   delta_vs_global |   tasa_error_etiquetas |   tasa_error_exacta |
|:----------------------------|:--------|-----:|--------:|--------:|--------:|-----------:|------------------:|-----------------------:|--------------------:|
| tiene_insulto_lexico        | 0       |  437 |  0.4237 |  0.2727 |  0.4062 |     0.3676 |           -0.4314 |                 0.0931 |              0.0183 |
| tiene_insulto_nucleo        | 0       |  682 |  0.5677 |  0.5067 |  0.4824 |     0.5189 |           -0.28   |                 0.1325 |              0.0264 |
| tiene_hashtag               | 1       |  161 |  0.6222 |  0.5455 |  0.5902 |     0.5859 |           -0.213  |                 0.1429 |              0.0248 |
| tiene_pregunta              | 1       |  158 |  0.6815 |  0.6531 |  0.5634 |     0.6326 |           -0.1663 |                 0.192  |              0.057  |
| tiene_referencia_colectiva  | 1       |  418 |  0.6971 |  0.6437 |  0.576  |     0.6389 |           -0.16   |                 0.1994 |              0.0311 |
| cuartil_longitud            | Q4      |  432 |  0.6959 |  0.6286 |  0.6113 |     0.6453 |           -0.1537 |                 0.1898 |              0.0486 |
| tiene_mencion               | 0       |  909 |  0.7032 |  0.707  |  0.681  |     0.6971 |           -0.1019 |                 0.1746 |              0.0528 |
| tiene_mencion_inicial       | 0       | 1030 |  0.7066 |  0.7118 |  0.6784 |     0.6989 |           -0.1    |                 0.1712 |              0.0495 |
| tiene_url                   | 1       |  540 |  0.703  |  0.7403 |  0.6903 |     0.7112 |           -0.0878 |                 0.1636 |              0.0352 |
| cuartil_longitud            | Q3      |  433 |  0.7012 |  0.7692 |  0.7341 |     0.7348 |           -0.0641 |                 0.1647 |              0.0531 |
| tiene_exclamacion           | 1       |  150 |  0.7391 |  0.775  |  0.7238 |     0.746  |           -0.053  |                 0.1844 |              0.0667 |
| tiene_mayusculas_sostenidas | 1       | 1013 |  0.7612 |  0.7853 |  0.745  |     0.7638 |           -0.0351 |                 0.1586 |              0.0582 |
| tiene_segunda_persona       | 0       |  950 |  0.762  |  0.8029 |  0.7331 |     0.766  |           -0.0329 |                 0.1509 |              0.0463 |
| texto_corto_40              | 0       | 1577 |  0.7676 |  0.8009 |  0.7544 |     0.7743 |           -0.0246 |                 0.1625 |              0.0615 |
| tiene_comillas              | 1       |   88 |  0.7356 |  0.8108 |  0.8    |     0.7821 |           -0.0168 |                 0.1932 |              0.0455 |
| cuartil_longitud            | Q2      |  431 |  0.797  |  0.7947 |  0.7768 |     0.7895 |           -0.0094 |                 0.1709 |              0.0858 |
| tiene_risa                  | 0       | 1611 |  0.7834 |  0.8187 |  0.7695 |     0.7905 |           -0.0084 |                 0.1558 |              0.0571 |
| tiene_imperativo            | 0       | 1355 |  0.7902 |  0.8244 |  0.7713 |     0.7953 |           -0.0036 |                 0.1557 |              0.0605 |
| tiene_elongacion            | 0       | 1641 |  0.7907 |  0.8259 |  0.7769 |     0.7978 |           -0.0011 |                 0.1525 |              0.0567 |
| tiene_retuit                | 0       | 1731 |  0.7897 |  0.8296 |  0.7774 |     0.7989 |            0      |                 0.1541 |              0.0583 |
| tiene_comillas              | 0       | 1643 |  0.7929 |  0.8303 |  0.7755 |     0.7996 |            0.0007 |                 0.152  |              0.059  |
| tiene_exclamacion           | 0       | 1581 |  0.7946 |  0.8342 |  0.782  |     0.8036 |            0.0047 |                 0.1512 |              0.0576 |
| tiene_hashtag               | 0       | 1570 |  0.8    |  0.8358 |  0.7864 |     0.8074 |            0.0085 |                 0.1552 |              0.0618 |
| tiene_pregunta              | 0       | 1573 |  0.8    |  0.8384 |  0.7895 |     0.8093 |            0.0104 |                 0.1502 |              0.0585 |
| tiene_imperativo            | 1       |  376 |  0.7881 |  0.8475 |  0.7987 |     0.8114 |            0.0125 |                 0.148  |              0.0505 |
| tiene_elongacion            | 1       |   90 |  0.7742 |  0.875  |  0.7857 |     0.8116 |            0.0127 |                 0.1815 |              0.0889 |
| tiene_insulto_lexico        | 1       | 1294 |  0.8198 |  0.8417 |  0.7962 |     0.8192 |            0.0203 |                 0.1747 |              0.0719 |
| tiene_url                   | 0       | 1191 |  0.8202 |  0.8452 |  0.8073 |     0.8242 |            0.0253 |                 0.1497 |              0.0688 |
| tiene_segunda_persona       | 1       |  781 |  0.8168 |  0.8471 |  0.8164 |     0.8268 |            0.0278 |                 0.1579 |              0.073  |
| tiene_referencia_colectiva  | 0       | 1313 |  0.819  |  0.8466 |  0.8241 |     0.8299 |            0.031  |                 0.1396 |              0.067  |
| tiene_mayusculas_sostenidas | 0       |  718 |  0.8219 |  0.8691 |  0.8141 |     0.835  |            0.0361 |                 0.1476 |              0.0585 |
| tiene_insulto_nucleo        | 1       | 1049 |  0.8435 |  0.8548 |  0.8294 |     0.8425 |            0.0436 |                 0.1681 |              0.0791 |
| tiene_mencion               | 1       |  822 |  0.8595 |  0.8828 |  0.8472 |     0.8631 |            0.0642 |                 0.1314 |              0.0645 |
| tiene_mencion_inicial       | 1       |  701 |  0.8742 |  0.887  |  0.8652 |     0.8755 |            0.0765 |                 0.1289 |              0.0713 |
| tiene_risa                  | 1       |  120 |  0.8571 |  0.9153 |  0.8548 |     0.8757 |            0.0768 |                 0.1306 |              0.075  |
| cuartil_longitud            | Q1      |  435 |  0.9106 |  0.9291 |  0.9007 |     0.9134 |            0.1145 |                 0.0912 |              0.046  |
| texto_corto_40              | 1       |  154 |  0.9436 |  0.9667 |  0.9231 |     0.9444 |            0.1455 |                 0.0671 |              0.026  |

Rebanadas no evaluables:

| rebanada   | grupo   | estado   | n   |
|------------|---------|----------|-----|

Las 5 mas debiles (excluyendo artefactos del lexico inducido):

|   posicion | rebanada                   | grupo   |   n |   f1_HS |   f1_TR |   f1_AG |   f1_media |   delta_vs_global |   tasa_error_etiquetas |   tasa_error_exacta |
|-----------:|:---------------------------|:--------|----:|--------:|--------:|--------:|-----------:|------------------:|-----------------------:|--------------------:|
|          1 | tiene_hashtag              | 1       | 161 |  0.6222 |  0.5455 |  0.5902 |     0.5859 |           -0.213  |                 0.1429 |              0.0248 |
|          2 | tiene_pregunta             | 1       | 158 |  0.6815 |  0.6531 |  0.5634 |     0.6326 |           -0.1663 |                 0.192  |              0.057  |
|          3 | tiene_referencia_colectiva | 1       | 418 |  0.6971 |  0.6437 |  0.576  |     0.6389 |           -0.16   |                 0.1994 |              0.0311 |
|          4 | cuartil_longitud           | Q4      | 432 |  0.6959 |  0.6286 |  0.6113 |     0.6453 |           -0.1537 |                 0.1898 |              0.0486 |
|          5 | tiene_mencion              | 0       | 909 |  0.7032 |  0.707  |  0.681  |     0.6971 |           -0.1019 |                 0.1746 |              0.0528 |

Las 5 mas debiles incluyendo artefactos del lexico inducido:

|   posicion | rebanada                   |   grupo |   n |   f1_HS |   f1_TR |   f1_AG |   f1_media |   delta_vs_global |   tasa_error_etiquetas |   tasa_error_exacta |
|-----------:|:---------------------------|--------:|----:|--------:|--------:|--------:|-----------:|------------------:|-----------------------:|--------------------:|
|          1 | tiene_insulto_lexico       |       0 | 437 |  0.4237 |  0.2727 |  0.4062 |     0.3676 |           -0.4314 |                 0.0931 |              0.0183 |
|          2 | tiene_insulto_nucleo       |       0 | 682 |  0.5677 |  0.5067 |  0.4824 |     0.5189 |           -0.28   |                 0.1325 |              0.0264 |
|          3 | tiene_hashtag              |       1 | 161 |  0.6222 |  0.5455 |  0.5902 |     0.5859 |           -0.213  |                 0.1429 |              0.0248 |
|          4 | tiene_pregunta             |       1 | 158 |  0.6815 |  0.6531 |  0.5634 |     0.6326 |           -0.1663 |                 0.192  |              0.057  |
|          5 | tiene_referencia_colectiva |       1 | 418 |  0.6971 |  0.6437 |  0.576  |     0.6389 |           -0.16   |                 0.1994 |              0.0311 |

INTERPRETACION: la debilidad se concentra en el registro noticia/comentario (hashtag F1 0.586, pregunta 0.633, referencia colectiva 0.639, URL 0.711, cuartil largo 0.645) frente a textos cortos (0.944) y respuestas directas. La rebanada `tiene_insulto_lexico=0` (F1 0.368, n=437) es un artefacto in-sample: el lexico se indujo con las etiquetas de train y estas rebanadas separan casi por construccion. Retuits: cero casos en train (`tiene_retuit`), por lo que la rebanada no existe en este corpus.

DECISION: priorizar el regimen noticia y la region sin destinatario explicito en los siguientes experimentos; no usar las rebanadas del lexico como evidencia.

SIGUIENTE PASO: cuantificar que rasgos acompanan a cada tipo de error (seccion 5) y revisar los casos (seccion 6).

## 5. Regiones del corpus y rasgos que acompanan al error

OBSERVACION: las rebanadas simples se solapan (URL y hashtag suelen ir juntas); conviene definir regiones mutuamente legibles y medir alli donde vive el error.

HIPOTESIS: el error se divide en dos regimenes: la respuesta insultante (persona a persona) y el comentario o noticia con referencia colectiva.

METODO: siete regiones definidas por combinacion de rasgos (mencion inicial, referencia colectiva, URL o hashtag) y perfil de rasgos en TP, FP, FN y TN por etiqueta.

RESULTADO: metricas por region.

| region                            | condicion                                           |    n |   f1_HS |   f1_TR |   f1_AG |   f1_media |   delta_vs_global |   prevalencia_real_TR |   prevalencia_predicha_TR |
|:----------------------------------|:----------------------------------------------------|-----:|--------:|--------:|--------:|-----------:|------------------:|----------------------:|--------------------------:|
| responde_mencion_inicial          | tiene_mencion_inicial == 1                          |  701 |  0.8742 |  0.887  |  0.8652 |     0.8755 |            0.0765 |                0.4807 |                    0.5164 |
| no_responde_sin_colectiva         | mencion_inicial == 0 y referencia_colectiva == 0    |  712 |  0.7181 |  0.7333 |  0.7327 |     0.728  |           -0.0709 |                0.2191 |                    0.2022 |
| no_responde_con_colectiva         | mencion_inicial == 0 y referencia_colectiva == 1    |  318 |  0.6873 |  0.55   |  0.555  |     0.5974 |           -0.2015 |                0.0818 |                    0.044  |
| insulto_responde_sin_colectiva    | insulto == 1, mencion_inicial == 1 y colectiva == 0 |  526 |  0.9066 |  0.9051 |  0.8959 |     0.9026 |            0.1036 |                0.5837 |                    0.6388 |
| insulto_no_responde_sin_colectiva | insulto == 1, mencion_inicial == 0 y colectiva == 0 |  451 |  0.7596 |  0.7543 |  0.7588 |     0.7576 |           -0.0414 |                0.3259 |                    0.3149 |
| regimen_noticia                   | url == 1 o hashtag == 1                             |  599 |  0.7014 |  0.7186 |  0.6849 |     0.7016 |           -0.0973 |                0.1536 |                    0.1252 |
| sin_url_ni_hashtag                | url == 0 y hashtag == 0                             | 1132 |  0.8248 |  0.8509 |  0.8124 |     0.8294 |            0.0305 |                0.3772 |                    0.3931 |

Perfil de rasgos en aciertos y errores (tasas y sobrerrepresentacion):

| etiqueta   | rasgo                       |   tasa_TP |   tasa_FP |   tasa_FN |   tasa_TN |   sobrerrepresentacion_FP_vs_TN |   sobrerrepresentacion_FN_vs_TP |
|:-----------|:----------------------------|----------:|----------:|----------:|----------:|--------------------------------:|--------------------------------:|
| HS         | tiene_insulto_lexico        |    0.9593 |    0.8108 |    0.7676 |    0.5646 |                          1.4362 |                          0.8002 |
| HS         | tiene_segunda_persona       |    0.5228 |    0.4703 |    0.4014 |    0.4    |                          1.1757 |                          0.7678 |
| HS         | tiene_referencia_colectiva  |    0.2117 |    0.3027 |    0.4014 |    0.2215 |                          1.3665 |                          1.8959 |
| HS         | tiene_mencion_inicial       |    0.5489 |    0.4    |    0.162  |    0.338  |                          1.1835 |                          0.2951 |
| HS         | tiene_risa                  |    0.0928 |    0.0649 |    0.0493 |    0.0557 |                          1.1646 |                          0.531  |
| HS         | tiene_elongacion            |    0.0586 |    0.0865 |    0.0352 |    0.0418 |                          2.0704 |                          0.6005 |
| HS         | tiene_mayusculas_sostenidas |    0.5114 |    0.6    |    0.6056 |    0.6354 |                          0.9442 |                          1.1843 |
| HS         | tiene_url                   |    0.2313 |    0.3135 |    0.4366 |    0.3519 |                          0.8909 |                          1.8879 |
| HS         | tiene_hashtag               |    0.0456 |    0.1027 |    0.1056 |    0.1253 |                          0.8195 |                          2.3164 |
| HS         | tiene_entidad_html          |    0      |    0      |    0.007  |    0.0013 |                          0      |                        nan      |
| TR         | tiene_insulto_lexico        |    0.993  |    0.9438 |    0.875  |    0.6278 |                          1.5034 |                          0.8811 |
| TR         | tiene_segunda_persona       |    0.6172 |    0.6292 |    0.4545 |    0.3731 |                          1.6864 |                          0.7365 |
| TR         | tiene_referencia_colectiva  |    0.065  |    0.0787 |    0.2727 |    0.3197 |                          0.246  |                          4.1981 |
| TR         | tiene_mencion_inicial       |    0.7193 |    0.5843 |    0.3068 |    0.2778 |                          2.103  |                          0.4266 |
| TR         | tiene_risa                  |    0.1253 |    0.0674 |    0.0455 |    0.0499 |                          1.3519 |                          0.3628 |
| TR         | tiene_elongacion            |    0.0812 |    0.0899 |    0.0227 |    0.0401 |                          2.2432 |                          0.2799 |
| TR         | tiene_mayusculas_sostenidas |    0.4455 |    0.5169 |    0.6705 |    0.6376 |                          0.8107 |                          1.505  |
| TR         | tiene_url                   |    0.1323 |    0.1236 |    0.3295 |    0.3945 |                          0.3133 |                          2.4918 |
| TR         | tiene_hashtag               |    0.0139 |    0.0674 |    0.0455 |    0.1291 |                          0.5221 |                          3.2652 |
| TR         | tiene_entidad_html          |    0      |    0      |    0.0114 |    0.0009 |                          0      |                        nan      |
| AG         | tiene_insulto_lexico        |    0.9749 |    0.9073 |    0.8345 |    0.5795 |                          1.5656 |                          0.856  |
| AG         | tiene_segunda_persona       |    0.559  |    0.4967 |    0.3793 |    0.3943 |                          1.2596 |                          0.6786 |
| AG         | tiene_referencia_colectiva  |    0.1393 |    0.2649 |    0.4552 |    0.2614 |                          1.0132 |                          3.2684 |
| AG         | tiene_mencion_inicial       |    0.5899 |    0.4437 |    0.1931 |    0.3279 |                          1.3532 |                          0.3273 |
| AG         | tiene_risa                  |    0.1025 |    0.0596 |    0.0621 |    0.0534 |                          1.1166 |                          0.6055 |
| AG         | tiene_elongacion            |    0.0638 |    0.0795 |    0.0414 |    0.0425 |                          1.8706 |                          0.6483 |
| AG         | tiene_mayusculas_sostenidas |    0.5087 |    0.5828 |    0.6345 |    0.6209 |                          0.9386 |                          1.2473 |
| AG         | tiene_url                   |    0.2263 |    0.2848 |    0.4276 |    0.3464 |                          0.8221 |                          1.8894 |
| AG         | tiene_hashtag               |    0.0348 |    0.0861 |    0.0828 |    0.1285 |                          0.6698 |                          2.377  |
| AG         | tiene_entidad_html          |    0      |    0      |    0.0138 |    0      |                        nan      |                        nan      |

INTERPRETACION: en la region de respuesta directa el F1 es alto en las tres etiquetas; el dano se concentra en `no_responde_con_colectiva` (TR y AG) y en el regimen noticia. Los FP de HS llevan mencion inicial en el 40.0 % de los casos y el insulto en el 81.1 %; los FN de HS llevan URL en el 43.7 % y referencia colectiva en el 40.1 %. En AG, el 45.5 % de los FN tiene referencia colectiva frente al 13.9 % de los TP: los positivos que se escapan son agresion contra grupos, no insulto interpersonal.

DECISION: atacar por separado los dos regimenes (features de registro para la respuesta y features de objetivo colectivo o individual para el comentario).

SIGUIENTE PASO: revision humana de las muestras (seccion 6).

## 6. Analisis cualitativo

OBSERVACION: las rebanadas dicen donde falla el modelo, no por que; hace falta leer casos.

HIPOTESIS: los FP y FN responden a un numero pequeno de patrones recurrentes (ironia, insulto sin odio, odio implicito, agresividad sin objetivo, textos muy cortos, ruido de anotacion).

METODO: exportar 15 FP (los mas confiados) y 15 FN (los mas olvidados) por etiqueta a `reports/eda/error_samples_para_revision.csv` (texto truncado a 200 caracteres) y revisar los casos a mano; el resumen de rasgos de esas muestras es automatico.

RESULTADO: perfil de las muestras exportadas.

| etiqueta   | tipo_error     |   n |   mediana_caracteres |   tasa_tiene_mencion_inicial |   tasa_tiene_insulto_lexico |   tasa_tiene_referencia_colectiva |   tasa_tiene_segunda_persona |   tasa_tiene_risa |   tasa_tiene_hashtag |   tasa_tiene_url |
|:-----------|:---------------|----:|---------------------:|-----------------------------:|----------------------------:|----------------------------------:|-----------------------------:|------------------:|---------------------:|-----------------:|
| AG         | falso_positivo |  15 |                   52 |                       0.7333 |                      1      |                            0.0667 |                       0.6667 |            0.1333 |               0.0667 |           0.1333 |
| AG         | falso_negativo |  15 |                   88 |                       0      |                      0.5333 |                            0.2667 |                       0.5333 |            0.0667 |               0.1333 |           0.5333 |
| HS         | falso_positivo |  15 |                   49 |                       0.5333 |                      0.9333 |                            0.0667 |                       0.4    |            0.0667 |               0      |           0.1333 |
| HS         | falso_negativo |  15 |                   75 |                       0.1333 |                      0.4667 |                            0.3333 |                       0.3333 |            0.0667 |               0.0667 |           0.4667 |
| TR         | falso_positivo |  15 |                   54 |                       0.8    |                      1      |                            0.1333 |                       0.5333 |            0.0667 |               0      |           0.0667 |
| TR         | falso_negativo |  15 |                  102 |                       0.2    |                      0.7333 |                            0.4667 |                       0.2667 |            0.0667 |               0.0667 |           0.5333 |

Conteo de las muestras que caen en cada patron operacionalizado (buckets no excluyentes; la regla de cada bucket esta en el codigo):

| patron               | etiqueta   | tipo_error     |   n |   n_grupo |   proporcion |
|:---------------------|:-----------|:---------------|----:|----------:|-------------:|
| broma_afiliacion     | AG         | falso_positivo |   6 |        15 |       0.4    |
| broma_afiliacion     | AG         | falso_negativo |   0 |        15 |       0      |
| broma_afiliacion     | HS         | falso_positivo |   6 |        15 |       0.4    |
| broma_afiliacion     | HS         | falso_negativo |   1 |        15 |       0.0667 |
| broma_afiliacion     | TR         | falso_positivo |   7 |        15 |       0.4667 |
| broma_afiliacion     | TR         | falso_negativo |   0 |        15 |       0      |
| cita_metalenguaje    | AG         | falso_positivo |   2 |        15 |       0.1333 |
| cita_metalenguaje    | AG         | falso_negativo |   1 |        15 |       0.0667 |
| cita_metalenguaje    | HS         | falso_positivo |   2 |        15 |       0.1333 |
| cita_metalenguaje    | HS         | falso_negativo |   1 |        15 |       0.0667 |
| cita_metalenguaje    | TR         | falso_positivo |   1 |        15 |       0.0667 |
| cita_metalenguaje    | TR         | falso_negativo |   0 |        15 |       0      |
| noticia_o_atribucion | AG         | falso_positivo |   2 |        15 |       0.1333 |
| noticia_o_atribucion | AG         | falso_negativo |   8 |        15 |       0.5333 |
| noticia_o_atribucion | HS         | falso_positivo |   2 |        15 |       0.1333 |
| noticia_o_atribucion | HS         | falso_negativo |   8 |        15 |       0.5333 |
| noticia_o_atribucion | TR         | falso_positivo |   1 |        15 |       0.0667 |
| noticia_o_atribucion | TR         | falso_negativo |   9 |        15 |       0.6    |
| sin_insulto_lexico   | AG         | falso_positivo |   0 |        15 |       0      |
| sin_insulto_lexico   | AG         | falso_negativo |   7 |        15 |       0.4667 |
| sin_insulto_lexico   | HS         | falso_positivo |   1 |        15 |       0.0667 |
| sin_insulto_lexico   | HS         | falso_negativo |   8 |        15 |       0.5333 |
| sin_insulto_lexico   | TR         | falso_positivo |   0 |        15 |       0      |
| sin_insulto_lexico   | TR         | falso_negativo |   4 |        15 |       0.2667 |

Muestras representativas (5 por etiqueta y tipo, texto truncado a 100 caracteres):

| etiqueta   | tipo_error     |   probabilidad | etiquetas_true   | etiquetas_pred   | texto_100                                                                                            |
|:-----------|:---------------|---------------:|:-----------------|:-----------------|:-----------------------------------------------------------------------------------------------------|
| AG         | falso_negativo |          0.056 | 1,0,1            | 0,0,0            | m da miedo cuando me sigue un rabe                                                                   |
| AG         | falso_negativo |          0.081 | 1,1,1            | 0,0,0            | Comemela mientras te meto los dedos https://t.co/R1gEttjD4S                                          |
| AG         | falso_negativo |          0.103 | 1,1,1            | 0,0,0            | racismo es que en la fila del banco me pidan guardar el celular a m y no al gero de adelante         |
| AG         | falso_negativo |          0.117 | 0,0,1            | 0,0,0            | Trump arremete contra las mujeres que lo acusan de acoso sexual; asegura que las denuncias de las 16 |
| AG         | falso_negativo |          0.122 | 1,0,1            | 0,0,0            | Indios hijos de puta. Soltad al gabacho                                                              |
| AG         | falso_positivo |          0.965 | 1,1,0            | 1,1,1            | @DeenisseMb Jeje callate perra                                                                       |
| AG         | falso_positivo |          0.954 | 0,0,0            | 1,1,1            | @saancisco Callate puta tu eres el traidor.                                                          |
| AG         | falso_positivo |          0.939 | 0,0,0            | 1,1,1            | @defays_mikael @rickcenturion cllate zorra ac los leprosos no opinan                                 |
| AG         | falso_positivo |          0.933 | 1,1,0            | 1,1,1            | @kleopatreus T eres bien perra                                                                       |
| AG         | falso_positivo |          0.93  | 0,0,0            | 1,1,1            | @agos_goroso CALLATE HIJA DE PUTA JAJAJAJAJAJAJAJAJAJA                                               |
| HS         | falso_negativo |          0.062 | 1,1,1            | 0,0,0            | Comemela XDDD y la cara tambin!!   No nos pasemos  https://t.co/t3o5yHY0zb                           |
| HS         | falso_negativo |          0.071 | 1,0,1            | 0,0,0            | Indios hijos de puta. Soltad al gabacho                                                              |
| HS         | falso_negativo |          0.075 | 1,1,1            | 0,0,0            | Comemela mientras te meto los dedos https://t.co/R1gEttjD4S                                          |
| HS         | falso_negativo |          0.094 | 1,1,1            | 0,0,0            | Nunca confes en el gordito gero orejon &gt;:(                                                        |
| HS         | falso_negativo |          0.114 | 1,0,0            | 0,0,0            | El Arabe es un flojo                                                                                 |
| HS         | falso_positivo |          0.951 | 0,0,0            | 1,1,1            | @CamilaArvayo Cllate todo te queda bien zorra te amo                                                 |
| HS         | falso_positivo |          0.949 | 0,0,0            | 1,1,1            | @defays_mikael @rickcenturion cllate zorra ac los leprosos no opinan                                 |
| HS         | falso_positivo |          0.947 | 0,0,0            | 1,1,1            | @saancisco Callate puta tu eres el traidor.                                                          |
| HS         | falso_positivo |          0.93  | 0,0,0            | 1,1,1            | @Troi_2_Tata que puta novia??? tu eres mi novia!!                                                    |
| HS         | falso_positivo |          0.911 | 0,0,0            | 1,1,1            | @GvyMC @Wismichu Callate puta rata ya estas llorando??                                               |
| TR         | falso_negativo |          0.024 | 1,1,0            | 1,0,0            | El agressor de una mujer de 25 ha sido detenido. La agresin se produjo en la estacin de Miln. Es un  |
| TR         | falso_negativo |          0.035 | 1,1,1            | 0,0,0            | Comemela XDDD y la cara tambin!!   No nos pasemos  https://t.co/t3o5yHY0zb                           |
| TR         | falso_negativo |          0.036 | 1,1,1            | 0,0,0            | Comemela mientras te meto los dedos https://t.co/R1gEttjD4S                                          |
| TR         | falso_negativo |          0.044 | 1,1,0            | 0,0,0            | @Victor_EstradaG seor presidente me robaron los tapones de mi auto. A las afueras de suburbano de le |
| TR         | falso_negativo |          0.045 | 1,1,0            | 0,0,0            | Estimados @FcpysUNCuyo quisiera saber por qu tienen a este antisemita dando clase en su universidad  |
| TR         | falso_positivo |          0.966 | 0,0,0            | 1,1,1            | @saancisco Callate puta tu eres el traidor.                                                          |
| TR         | falso_positivo |          0.966 | 0,0,0            | 1,1,1            | @Troi_2_Tata que puta novia??? tu eres mi novia!!                                                    |
| TR         | falso_positivo |          0.954 | 0,0,0            | 1,1,1            | @mariogigasanche Tu eres mi perra                                                                    |
| TR         | falso_positivo |          0.951 | 0,0,0            | 1,1,1            | @CamilaArvayo Cllate todo te queda bien zorra te amo                                                 |
| TR         | falso_positivo |          0.948 | 0,0,0            | 1,1,1            | @GvyMC @Wismichu Callate puta rata ya estas llorando??                                               |

Patrones recurrentes observados en la revision humana:

- **Insulto interpersonal en registro de broma, afiliacion o vocativo** (etiquetas: HS, TR, AG (falsos positivos); ids: 23076, 24006, 23172, 20504, 21150, 22524, 21497, 22924): Falso positivo dominante de HS, TR y AG: el texto insulta con lexico duro (zorra, perra, puta, hijueputa) pero el contexto marca broma o afiliacion entre usuarios que se conocen: vocativo con mencion, posesivos de primera persona ('mi novia', 'mi perra', 'mi favorita'), apelativos afectivos ('te amo', 'socia'), risa (JAJAJA, jeje) o marcas de juego ('beer pong'). La anotacion los considera no odiosos y el modelo los dispara con probabilidad 0.87-0.97 en las tres etiquetas. Es el error mas costoso y el mas sistematico del analisis.
- **Uso metalinguistico: cita, mencion o reproduccion del insulto** (etiquetas: HS, AG (falsos positivos); HS, AG (falsos negativos); ids: 21362, 23972, 24061): El texto no insulta: cita el insulto o habla de el entre comillas ('#Cllate Zorra el cllate perro de este semestre', 'Es como maldita sea cllate ese maldito hocico perra de mierda', 'los estupidos rabes de la universidad'). El modelo lineal reacciona al lexico citado tanto en falso positivo (cuando la cita es inocua) como en falso negativo (cuando la cita es el vehiculo del odio).
- **Regimen noticia o atribucion a terceros** (etiquetas: HS, TR, AG (falsos negativos); ids: 24269, 20144, 21733, 21041, 23631, 20295): Falso negativo de HS, TR y AG en textos largos (180-302 caracteres) con URL: titulares, cronicas o denuncias que describen violencia contra un grupo y que la anotacion marca como toxicas. Varios casos describen a un individuo concreto ('el agressor ... inmigrante ilegal de origen Nigeriano', 'este antisemita dando clase'), lo que refuerza la lectura de D-005 de que TR=1 puede marcar objetivo individual y no solo respuesta directa.
- **Odio implicito contra un grupo sin insulto del lexico** (etiquetas: HS, AG (falsos negativos); ids: 21799, 21408, 23832, 22174, 23922, 22678): Falso negativo de HS y AG sin ninguna palabra del lexico inducido: estereotipo ('El Arabe es un flojo'), miedo estigmatizante ('m da miedo cuando me sigue un rabe'), despersonalizacion ('Al final un iletrado sin papeles'), animalizacion ('mas digna ... la vida de una linda gallina que acabar en un KFC'), consigna ('fuera la escoria inmigrante') o deseo ('lastima que no soy un arabe de mierda'). El modelo necesita el insulto explicito para activar AG y HS.
- **Texto corto cuyo sentido depende del contenido enlazado** (etiquetas: HS, TR, AG (falsos negativos); ids: 20082, 20852, 23283): Falso negativo de HS, TR y AG en textos muy cortos con URL ('Comemela XDDD y la cara tambin!! ...', 'Comemela mientras te meto los dedos', '@TaigaDeLimon gritame y pisame la polla con una falda puesta'): el texto por si solo no explica la etiqueta positiva, que probablemente depende de la imagen o el video enlazado, informacion que el modelo no puede ver. Es una fraccion irreducible del error con las features de texto.
- **Frontera de anotacion no observable en el texto** (etiquetas: HS, TR, AG (falsos positivos y negativos); ids: 21892, 24454, 21865, 20463): Casos de superficie casi identica con etiquetas opuestas: 'Callate puta tu eres el traidor' y 'Cllate negro Ql! ...' estan etiquetados (0,0,0) mientras textos del mismo tipo llevan (1,1,1). La region completa lo confirma: de 526 textos con insulto, mencion inicial y sin referencia colectiva, 213 (40.5 %) son negativos. La frontera depende de criterio humano (quien habla, a quien, con que intencion) que no esta en el texto y que ningun modelo de bolsas de n-gramas puede reconstruir; acota el techo alcanzable.

INTERPRETACION: la revision confirma que el error no es homogeneo. El bloque dominante de falsos positivos es el insulto interpersonal en registro de broma o afiliacion, que se repite en las tres etiquetas y con probabilidades muy altas. El bloque dominante de falsos negativos es el odio implicito contra grupos sin insulto del lexico y el regimen noticia o atribucion a terceros, donde TR ademas se comporta como objetivo individual (D-005). Los casos de frontera (mismo patron de superficie con etiquetas opuestas, citas, textos cuyo sentido depende de un enlace) muestran ambiguedad de anotacion que ningun modelo de bolsas de n-gramas puede resolver: acotan el techo alcanzable con estas features.

DECISION: convertir los patrones dominantes en experimentos dirigidos (seccion 7) en lugar de seguir ampliando la representacion a ciegas.

SIGUIENTE PASO: registrar las hipotesis H11-H15 y ejecutarlas en la Fase 14 con la regla de decision vigente.

## 7. Hipotesis nuevas (H11 en adelante)

Las cinco hipotesis nacen del analisis de error y se ejecutan en la Fase 14. Todas heredan la regla de decision del proyecto: una mejora se adopta con >= +0.005 absoluto en OOF y >= 3/5 pliegues, y por D-012 se exige >= +0.010 local en la familia union antes de gastar una sonda de leaderboard.

Tabla resumen:

| id   | enunciado                                                                                                                                                                                                                                              | experimento   | criterio                                                                                                                                                                                                  |
|:-----|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| H11  | Los falsos positivos de HS y AG se concentran en el registro de insulto interpersonal entre usuarios que la anotacion considera no odioso (broma, afiliacion, pelea entre conocidos); un bloque explicito de interacciones de registro reduce esos FP. | EXP-14a       | F1 OOF >= +0.005 y >= 3/5 pliegues; por D-012 exigir >= +0.010 para considerarlo realista antes de enviar una sonda de LB; ademas reducir >= 25 % el FP de HS y AG en la region de insulto interpersonal. |
| H12  | El error de TR vive en los textos sin destinatario explicito: la frontera depende de si el objetivo es una persona identificable o un colectivo, y el modelo no tiene un bloque que represente esa distincion.                                         | EXP-14b       | F1_TR OOF >= +0.005 y >= 3/5 pliegues, con la region sin mencion con colectiva mejorando >= +0.02 y sin degradar HS ni AG mas de 0.002; exigir >= +0.010 global para sonda de LB (D-012).                 |
| H13  | El modelo es fuerte en el registro de respuesta y debil en el registro noticia/comentario (URL, hashtag, pregunta, textos largos), donde el texto cita o describe odio ajeno en lugar de producirlo.                                                   | EXP-14c       | F1 OOF >= +0.005 y >= 3/5 pliegues; mejora >= +0.02 en la rebanada de regimen noticia; exigir >= +0.010 global para sonda de LB (D-012).                                                                  |
| H14  | Los falsos negativos de AG son agresion implicita sin insulto del lexico (estigma, deshumanizacion, cosificacion de un grupo); los falsos positivos de AG vienen del registro bromista, no de la agresion implicita.                                   | EXP-14d       | F1_AG OOF >= +0.005 y >= 3/5 pliegues, con el FN de AG sin insulto reduciendose >= 20 %; exigir >= +0.010 global para sonda de LB (D-012).                                                                |
| H15  | Ajustar el umbral por etiqueta y por rebanada (corto frente a largo, noticia frente a respuesta) NO aporta una mejora explotable; es una via de cierre, no de mejora.                                                                                  | EXP-14e       | Se adopta solo si la ganancia anidada es >= +0.002 y >= 3/5 pliegues; en caso contrario se cierra la linea de calibracion fina y se documenta el cierre.                                                  |

### H11

- OBSERVACION (evidencia del analisis): De los 526 textos con insulto del lexico, mencion inicial y sin referencia colectiva, 213 (40.49 %) estan etiquetados (0,0,0); la tasa de FP de HS en esa region es 0.0951 frente a 0.1897 global. Los FP mas confiados de HS y AG (probabilidad 0.88-0.97) pertenecen a ese grupo.
- HIPOTESIS (enunciado): Los falsos positivos de HS y AG se concentran en el registro de insulto interpersonal entre usuarios que la anotacion considera no odioso (broma, afiliacion, pelea entre conocidos); un bloque explicito de interacciones de registro reduce esos FP.
- METODO (experimento propuesto): EXP-14a: agregar a MOD-03b un bloque denso de interacciones calculadas in-fold (insulto x risa, insulto x posesivo de primera persona, insulto x vocativo, insulto x ausencia de colectivo, insulto x segunda persona, mencion multiple) con LR C=2.0 balanced x3 y el mismo StratifiedGroupKFold(5, shuffle=True, random_state=42).
- RESULTADO: pendiente; se ejecuta en la Fase 14 y se registra en `experiments.csv` con su `experiment_id`.
- INTERPRETACION (mecanismo esperado): Un modelo lineal sobre TF-IDF suma evidencia de palabras y no puede representar conjunciones: la marca que desactiva la toxicidad es la coaparicion de insulto con marcadores de afecto/risa/vocativo y la ausencia de un objetivo colectivo.
- DECISION (criterio de adopcion): F1 OOF >= +0.005 y >= 3/5 pliegues; por D-012 exigir >= +0.010 para considerarlo realista antes de enviar una sonda de LB; ademas reducir >= 25 % el FP de HS y AG en la region de insulto interpersonal.
- SIGUIENTE PASO: implementar el bloque de features dentro del pliegue (sin fuga), medir OOF y por rebanada con este mismo script de analisis para comparar antes y despues.

### H12

- OBSERVACION (evidencia del analisis): Dentro de HS=1, F1_TR es 0.8754729978049847 en respuestas con mencion inicial, 0.7280349885597547 en textos sin mencion ni colectiva y 0.5974196817857421 en textos sin mencion con referencia colectiva. 50 de los 94 errores de TR dentro de HS=1 ocurren en el grupo sin mencion.
- HIPOTESIS (enunciado): El error de TR vive en los textos sin destinatario explicito: la frontera depende de si el objetivo es una persona identificable o un colectivo, y el modelo no tiene un bloque que represente esa distincion.
- METODO (experimento propuesto): EXP-14b: bloque denso de rasgos de objetivo (individual frente a colectivo) inducidos dentro de cada pliegue con spaCy, agregado a la union; comparar F1 de TR dentro de HS=1 y global contra MOD-03b.
- RESULTADO: pendiente; se ejecuta en la Fase 14 y se registra en `experiments.csv` con su `experiment_id`.
- INTERPRETACION (mecanismo esperado): La etiqueta TR exige un objetivo individual; los rasgos que lo indican (nombre propio, gentilicio en singular, demostrativo + sustantivo singular) y los que indican colectivo (determinante plural + sustantivo plural) quedan dispersos entre n-gramas y el modelo lineal no los combina.
- DECISION (criterio de adopcion): F1_TR OOF >= +0.005 y >= 3/5 pliegues, con la region sin mencion con colectiva mejorando >= +0.02 y sin degradar HS ni AG mas de 0.002; exigir >= +0.010 global para sonda de LB (D-012).
- SIGUIENTE PASO: implementar el bloque de features dentro del pliegue (sin fuga), medir OOF y por rebanada con este mismo script de analisis para comparar antes y despues.

### H13

- OBSERVACION (evidencia del analisis): F1 medio columnas 0.7016172823880535 en el regimen noticia (URL o hashtag, n=599) frente a 0.8293835026730169 sin URL ni hashtag; las rebanadas hashtag=0.5859469007009991, pregunta=0.6326409958871394, referencia colectiva=0.6389096997524474 y cuartil largo=0.6452662053456493 estan entre las mas debiles.
- HIPOTESIS (enunciado): El modelo es fuerte en el registro de respuesta y debil en el registro noticia/comentario (URL, hashtag, pregunta, textos largos), donde el texto cita o describe odio ajeno en lugar de producirlo.
- METODO (experimento propuesto): EXP-14c: bloque denso de cita y atribucion inducido in-fold (verbos de reporte, comillas, patron de titular, dominio de la URL, hashtag de campana) mas su interaccion con la longitud; medir en las rebanadas del regimen y en OOF.
- RESULTADO: pendiente; se ejecuta en la Fase 14 y se registra en `experiments.csv` con su `experiment_id`.
- INTERPRETACION (mecanismo esperado): El vocabulario noticioso comparte terminos con el odio (gentilicios, migracion) y el modelo no distingue uso de mencion; el modelo lineal no tiene forma de representar la atribucion a terceros ni el metalenguaje.
- DECISION (criterio de adopcion): F1 OOF >= +0.005 y >= 3/5 pliegues; mejora >= +0.02 en la rebanada de regimen noticia; exigir >= +0.010 global para sonda de LB (D-012).
- SIGUIENTE PASO: implementar el bloque de features dentro del pliegue (sin fuga), medir OOF y por rebanada con este mismo script de analisis para comparar antes y despues.

### H14

- OBSERVACION (evidencia del analisis): El 45.5 % de los FN de AG tiene referencia colectiva frente al 13.9 % de los TP (y 26.1 % de los TN); F1_AG en la region con referencia colectiva es 0.5974196817857421 frente a 0.7774436090225564 global.
- HIPOTESIS (enunciado): Los falsos negativos de AG son agresion implicita sin insulto del lexico (estigma, deshumanizacion, cosificacion de un grupo); los falsos positivos de AG vienen del registro bromista, no de la agresion implicita.
- METODO (experimento propuesto): EXP-14d: inducir in-fold un lexico de deshumanizacion por log-odds sobre los lemas de documentos AG=1 sin insulto del lexico (frecuencia >= 3, presente en >= 2 documentos) y agregar conteos, banderas y la interaccion con referencia colectiva.
- RESULTADO: pendiente; se ejecuta en la Fase 14 y se registra en `experiments.csv` con su `experiment_id`.
- INTERPRETACION (mecanismo esperado): El lexico inducido cubre la agresion explicita; la deshumanizacion usa metaforas y giros que no estan en el lexico y que los n-gramas de caracter no aislan como categoria.
- DECISION (criterio de adopcion): F1_AG OOF >= +0.005 y >= 3/5 pliegues, con el FN de AG sin insulto reduciendose >= 20 %; exigir >= +0.010 global para sonda de LB (D-012).
- SIGUIENTE PASO: implementar el bloque de features dentro del pliegue (sin fuga), medir OOF y por rebanada con este mismo script de analisis para comparar antes y despues.

### H15

- OBSERVACION (evidencia del analisis): Los umbrales globales por etiqueta dieron +0.0014 en verificacion anidada (2/5 pliegues) y +0.0031 real en el LB (D-012), muy por debajo del +0.0063 optimista. La heterogeneidad de F1 entre rebanadas (0.586 a 0.944) invita a umbrales por rebanada, pero las celdas pequenas favorecen el sobreajuste.
- HIPOTESIS (enunciado): Ajustar el umbral por etiqueta y por rebanada (corto frente a largo, noticia frente a respuesta) NO aporta una mejora explotable; es una via de cierre, no de mejora.
- METODO (experimento propuesto): EXP-14e: ajustar umbrales por etiqueta y rebanada (corto/largo y noticia/respuesta) en 4 pliegues y evaluar en el restante con el mismo protocolo anidado de D-012, comparando contra los umbrales globales anidados.
- RESULTADO: pendiente; se ejecuta en la Fase 14 y se registra en `experiments.csv` con su `experiment_id`.
- INTERPRETACION (mecanismo esperado): La heterogeneidad de F1 entre rebanadas se explica por prevalencia y dificultad, no necesariamente por un sesgo de calibracion explotable; el ajuste de umbrales consume grados de libertad con n por celda pequeno.
- DECISION (criterio de adopcion): Se adopta solo si la ganancia anidada es >= +0.002 y >= 3/5 pliegues; en caso contrario se cierra la linea de calibracion fina y se documenta el cierre.
- SIGUIENTE PASO: implementar el bloque de features dentro del pliegue (sin fuga), medir OOF y por rebanada con este mismo script de analisis para comparar antes y despues.

## 8. Limitaciones y honestidad

- Las rebanadas del lexico inducido (`tiene_insulto_lexico`, `tiene_insulto_nucleo`) son in-sample: el lexico se construyo con las etiquetas de train (150 lemas), por lo que su F1 no es evidencia de generalizacion. Se incluyen por pedido explicito y se excluyen del ranking de debilidad.
- La rebanada de risa usa un patron laxo (subcadena) distinto del patron con frontera de palabra del EDA: el corpus escribe series como `JAJAJAJAJJAJAJAJA` que el patron estricto no captura (19 frente a 120 documentos). El cambio esta documentado en el modulo.
- Los rasgos de segunda persona, imperativo y referencia colectiva dependen de spaCy `es_core_news_md` como anotador (D-003); en texto sin diacriticos ni puntuacion normalizada hay ruido de anotacion.
- Las predicciones son OOF: provienen de modelos que vieron los otros pliegues, por lo que las tasas por rebanada son estimaciones honestas del rendimiento esperado pero no una validacion independiente adicional.
- No se re-optimiza ningun umbral ni se usa el test; las conclusiones son sobre train y sus etiquetas, con la advertencia conocida de que el test tiene ~7 % menos positivos (E00) y que el gap local-LB de la familia union es ~0.013-0.016 (D-012).
- Este analisis no entrena modelos: las hipotesis quedan propuestas y sin validar. Si la Fase 14 no encuentra ganancias >= +0.005 con >= 3/5 pliegues, la conclusion honesta es que el techo de la familia TF-IDF + modelo lineal esta cerca y el esfuerzo debe ir a la calibracion final y al notebook, no a mas features.

Numeros de control para leer el reporte: F1 global 0.7927 (0.5) y 0.7989 (umbrales), con 425 FP y 375 FN; lexico inducido de 150 lemas (19 en el nucleo).

## 9. Artefactos

- `reports/eda/error_matrices_confusion.csv`
- `reports/eda/error_etiquetas_erroneas.csv`
- `reports/eda/error_combinaciones_etiquetas.csv`
- `reports/eda/error_pares_etiquetas.csv`
- `reports/eda/error_regla_estructural.csv`
- `reports/eda/error_rebanadas.csv`
- `reports/eda/error_regiones.csv`
- `reports/eda/error_perfil_rasgos.csv`
- `reports/eda/error_samples_para_revision.csv`
- `reports/eda/error_resumen_muestras.csv`
- `reports/eda/error_resumen_patrones.csv`
- `reports/figuras/fig_20_matrices_confusion_umbral_05.png`
- `reports/figuras/fig_21_matrices_confusion_umbrales_optimos.png`
- `reports/figuras/fig_22_errores_por_etiqueta_y_por_ejemplo.png`
- `reports/figuras/fig_23_rebanadas_mas_debiles.png`
- `reports/figuras/fig_24_regiones_f1.png`
- `reports/error_analysis.md` (este reporte)
- `reports/contexto.md` (seccion de handoff de la Fase 12)

Comando de reproduccion:

```bash
/home/carlos/venv/bin/python experiments/06_error_analysis.py
```
