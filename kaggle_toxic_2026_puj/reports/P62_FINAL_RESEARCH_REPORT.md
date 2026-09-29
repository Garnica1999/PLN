# P62 - Representacion, estructura linguistica y error de alta confianza

Informe final de la fase. Fecha 2026-09-29T05:50:51+00:00. Cinco experimentos, 0 submissions, 0 Kaggle, 0 holdout, 0 test.

## La respuesta a la pregunta de la fase

> V03 falla porque no tiene suficiente informacion textual, o porque la representacion word+char no expresa adecuadamente la estructura con la que esas palabras se combinan?

**Las dos cosas son verdad a la vez, y por eso hace falta separarlas.** La interaccion entre la palabra y su categoria es REAL y vale +0.0657, el 17.41 por ciento de la senal de la palabra. El problema no es que falte informacion: es que no cabe, y porque no cabe no compensa ni compensada por ampliar la cobertura.

## El recorrido, y por que cada experimento era necesario

Los tres primeros no fueron tres intentos: cada uno lo pidio una medicion del anterior.

```text
P62-01  L palabra__POS                 -0,009572  1/5   la interaccion vale
                                                    +0,002014 frente al azar
P62-02  M triple lexico               -0,011236  0/5   la estructura vale
                                                    -0,010793, PEOR que el azar
P62-03  curva de presupuesto           ningun K mejora a V03
P62-04  POS contra lemas sueltos      ambos pierden 8/8, el POS ayuda 6/8
P62-05  cobertura min_df=1             -0,003351  la cobertura sube al 99,5
```

## El coste de anadir un canal, separado del contenido

Un candidato que anade rasgos puede perder porque el contenido no sirve, o porque anadir rasgos en un espacio de 21.000 dimensiones con 1.186 filas diluye. Se separan con un control aleatorio independiente de tamano parecido.

| | L palabra__POS | M triple lexico |
|---|---:|---:|
| (1) coste de anadir el bloque | +0.0115854689 | +0.0004432824 |
| (2) valor de la estructura | +0.0020137043 | -0.0107929661 |
| (3) saldo | -0.0095717646 | -0.0112362485 |

En L la interaccion **existe** y repaga el 17.4 por ciento de su coste. En M el canal es mucho mas pequeno, el coste casi nulo, y la estructura es **peor que el azar**.

## La cobertura, que no era lo que creia

El canal de palabras de V03 tiene 2673 terminos en el pliegue 1, y `min_df=2` conserva solo el **28.3 por ciento** de las formas superficiales presentes. Con `min_df=1` sube al **99.5 por ciento** y **resta**: A -0.0040227339, B -0.0116799367, C -0.0033508936.

Las palabras que `min_df=2` descartaba no eran las que faltaban.

## La correccion mas importante de la fase

**Yo lei mal mi propia EDA y lo digo sin rodeos.** El perfil de longitud se llamaba `estructura_corta`, pero su medida era el numero de tokens, luego su cuartil superior es lo LARGO. La delta negativa, -0,2330, no decia que V03 fallara en los cortos: decia que V03 es **mejor** en los largos.

La comprobacion independiente lo desmiente de forma rotunda:

| region | n | F1 de V03 | AUC interno | error de alta confianza |
|---|---:|---:|---:|---:|
| corta, hasta 6 tokens | 136 | 0.9704 | 0.9872 | 0.0294 |
| larga | 1347 | 0.7795 | 0.9242 | 0.0794 |

V03 alcanza **0.9639** de AUC dentro de la region corta. El margen hasta el techo moviendo **solo el corte** es de +0.000323, luego en los cortos V03 ordena bien y la linea esta casi donde debe.

**El limite esta en el orden de los tweets largos**, los que tienen ambiguedad y sentimiento mixto. Ahi el AUC baja a 0,85-0,92 y el techo moviendo el corte deja apenas +0,006. Y V28-01 ya habia mostrado que los errores de las tres etiquetas se solapan con coeficientes de +0,42 a +0,51, luego fallan sobre los mismos textos. Las dos lineas convergen.

La correccion se aplico EN EL ORIGEN: el perfil se renombro `longitud_del_tweet` y su informe lleva un aviso de lectura.

## Un hallazgo de instrumentacion que conviene no perder

Un control construido reetiquetando el vocabulario con nombres nuevos da **EXACTAMENTE** el mismo resultado, y no es un fallo: **permutar columnas de la matriz de diseno es una simetria de un modelo lineal regularizado**, porque la penalizacion es la norma al cuadrado del vector de coeficientes. Ese control es un no-op demostrable. El reflejo natural ante un desajuste entre dos representaciones es reetiquetar una para igualarla a la otra, y eso no mide nada.

## Errores propios encontrados y corregidos en la fase

- El `corrected_cv` de la curva se calculaba **dentro** del bucle de pliegues, sobre un array con una sola quinta parte rellena, y cada pliegue sobrescribia al anterior. Daba 0,2518 en todos los brazos. Anadido un guardia permanente que verifica el control en el pliegue 1.
- La EDA de secuencias POS y dependencias no filtraba por clase, luego las tres etiquetas obtenian conteos identicos y la EDA era una tabla repetida.
- Los perfiles linguisticos se guardaban como booleanos y el cuartil era degenerado, luego la rebanada rica era la misma que la pobre.
- Un lema multipalabra, `quedar el`, rompia el tokenizador del canal de dependencias.
- La sintaxis de un f-string con `+` en el subindice, y una clave de JSON que no existe. Ambos los atrapo la auto-verificacion del notebook antes de escribirlo.

## La respuesta en tres partes

```text
la interaccion existe    +0.0657, el 17.41 por ciento de la senal de la palabra
pero no cabe             repaga el 17.4 por ciento de su coste de anadido
y la cobertura no es la via  min_df=1 sube al 99.5 por ciento y resta
```

## Estado

| | |
|---|---|
| experimentos | 5 de 8 principales, 0 de 4 reservas |
| registro | 344 filas, 344 identificadores unicos |
| pistas cerradas sin experimento | N, P y O |
| cumplimiento | 21 comprobaciones, 0 fallos |
| submissions | 0 |
| Kaggle | 0 |
| holdout | 0 |
| test | 0 |
| reproducibilidad | R2, sin R4 |

## Cierre

```text
no evidence of exploitable remaining model capacity in the linguistic representation

V03 remains the best verified leaderboard result.
```

No se afirma que V03 sea matematicamente optimo. Se afirma lo que las mediciones sostienen: el eje de la representacion esta cerrado en sus dos dimensiones naturales, la gramatica y la cobertura, ambas con numeros, y el limite que queda esta en el orden de los textos largos y ambiguos.

## Artefactos

* `reports/p62_cobertura_vocabulario.csv`
* `reports/p62_cobertura_vocabulario.json`
* `reports/p62_cobertura_vocabulario.md`
* `reports/p62_compliance.md`
* `reports/p62_final_summary.json`
* `reports/p62_hypothesis_gap_analysis.csv`
* `reports/p62_hypothesis_gap_analysis.md`
* `reports/p62_linguistic_composition.csv`
* `reports/p62_linguistic_composition.md`
* `reports/p62_linguistic_eda.json`
* `reports/p62_linguistic_gap.csv`
* `reports/p62_master_results.csv`
* `reports/p62_notebook_integration.json`
* `reports/p62_p61_audit.json`
* `reports/p62_pista_l_word_pos.json`
* `reports/p62_pista_l_word_pos.md`
* `reports/p62_pista_m_dependencias.json`
* `reports/p62_pista_m_dependencias.md`
* `reports/p62_pos_o_no_pos.csv`
* `reports/p62_pos_o_no_pos.json`
* `reports/p62_pos_o_no_pos.md`
* `reports/p62_presupuesto_canal.csv`
* `reports/p62_presupuesto_canal.json`
* `reports/p62_presupuesto_canal.md`
* `reports/p62_probability_analysis.csv`
* `reports/p62_reproducibility.md`
* `reports/p62_techo_cortos.csv`
* `reports/p62_techo_cortos.json`
* `reports/p62_techo_cortos.md`
* `figuras/p62/01_probability_distributions_reference.png`
* `figuras/p62/02_probability_distance_to_threshold.png`
* `figuras/p62/03_high_confidence_error_distribution.png`
* `figuras/p62/04_pos_composition_by_label.png`
* `figuras/p62/05_enrichment_HS.png`
* `figuras/p62/06_enrichment_TR.png`
* `figuras/p62/07_enrichment_AG.png`
* `figuras/p62/08_pos_sequence_enrichment.png`
* `figuras/p62/09_dependency_pattern_enrichment.png`
* `figuras/p62/10_pronoun_composition.png`
* `figuras/p62/11_function_words_and_negation.png`
* `figuras/p62/12_v03_error_by_linguistic_profile.png`

> V03 remains the best verified leaderboard result.
