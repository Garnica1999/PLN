# P63 - Representacion a nivel de registro, estructura del dataset y composicion linguistica

Informe final de la fase. Fecha 2026-09-29T06:59:40+00:00. Seis experimentos, dos auditorias, un dataset auxiliar. 0 submissions, 0 Kaggle, 0 holdout, 0 test, 0 leaderboard.

## Las dieciocho preguntas del encargo

**1. nV03 utiliza solo `text`?** Si, y de forma exhaustiva: word(1,2) min_df=2 sublinear mas char_wb(3,5) min_df=2 sublinear, tres regresiones logisticas C=2,0 y umbrales 0,44 / 0,51 / 0,46. No toca ninguna otra columna.

**2. nQue informacion adicional no textual existe realmente?** Muy poca, y esta medida. El unico recurso no textual es `id`, y P63-02 mostrn que es un indice de coleccion, no un contador. Fuera de el, el fichero no tiene user ID, ni fecha, ni perfil, y V15-00 ya habia cerrado la via de metadata. La estructura de casi duplicados existe pero esta 92.92 por ciento vacia.

**3. nEl id es verdaderamente unico?** Si dentro de train: 1731 valores distintos para 1731 filas, cero repeticiones, cero nulos. Y entre train y test la interseccion es 0.

**4. nEl id es estable?** Si, en el sentido de que es determinista: el mismo registro lleva el mismo id en train y en el fichero de grupos.

**5. nEl id se repite?** No, en ninguno de los dos ficheros.

**6. nEl id es secuencial?** No. Empieza en 20430 y no es monotono: solo el 51.21 por ciento de los pares consecutivos de filas crecen, un Kendall de +0.024277. El fichero esta barajado.

**7. nEl id contiene estructura?** Si. Vive en el bloque contiguo [20001, 24999] y train y test son muestras disjuntas de ese bloque.

**8. nEl id codifica procedencia?** Es la lectura mas probable: dice de que coleccion salio la fila y en que posicion ocupaba. La ocupacion es casi uniforme entre los diez tramos, de 0.5480 a 0.6180, y la proporcion de train entre los usados se mantiene entre 0.5401 y 0.6416, que es lo que se ve al partir un bloque unico al azar y no al muestrear subcolecciones.

**9. nEl id codifica tiempo?** No. La escala lo refuta por si sola: los valores llegan a 24999, unas cinco cifras, mientras que un esquema tipo Snowflake ocupa 63 bits y produce 18. La longitud binaria es constante en 15 bits y los huecos van de 1 a 31 sin cadencia. **No se creo `p63_id_temporal_hypothesis.md`** porque el encargo prohibe crear un informe afirmando una hipotesis que no se sostiene.

**10. nEl id tiene senal predictiva?** No. El bloque de 21 columnas da 0.4191359898 frente a 0.4240383000 de un control de igual dimension, y queda -0.0188186038 por debajo del maximo de las veinte permutaciones de etiqueta.

**11. nSobrevive a la segunda particion?** No se midio, y no por descuido: el encargo prohibe usar el holdout durante la exploracion, y con una brecha de -0.0049023102 contra el control no habia nada que verificar. Medirlo habria sido gastar una partida de validacion sobre un candidato que ya perdio en la primera.

**12. nEs legal?** Las reglas dicen que el objetivo es predecir HS, TR y AG **a partir del texto (text)**. El id no esta prohibido de forma explicita, pero el enunciado limita la tarea al texto, y una senal de procedencia no debe presentarse como un hallazgo linguistico.

**13. nLa estructura de grupos aporta algo mas alla de V03?** No, y por una razon que no es la que se suponia. El bloque esta casi vacio: 1378 de 1483 filas estan en un grupo singleton y el bloque entero tiene 7 perfiles distintos sobre 6 columnas. En solitario supera al aleatorio en +0,0484024301, luego hay senal en las 105 filas que no son singleton.

**14. nLa composicion linguistica agregada aporta informacion?** Si, y medido: el bloque de 42 proporciones vale +0.1055015148 frente al aleatorio en solitario, y +0.0083528268 al anadirse a V03 frente a un control de igual ancho.

**15. nEsa informacion mejora precisamente en long tweets?** No. En la region MUY_LARGA la brecha es -0.009437, es decir el bloque de posiciones es PEOR que un bloque aleatorio del mismo ancho precisamente donde se suponia que estaba la senal. La hipotesis de V30-01 queda refutada en su prediccion.

**16. nQue errores de alta confianza son rescatables?** En MUY_LARGA hay 48 errores de HS de los que 28 son de alta confianza, y el bloque conjunto rescata 13 y dana 10. Es decir, cambia muchas decisiones y las cambia a netas casi nulas: no es que acierte mas veces, es que acierta unas y falla otras.

**17. nQue errores siguen sin informacion disponible?** Los de la region larga, y la magnitud esta medida. El margen hasta el techo moviendo solo el corte es de +0.0000 en CORTA, +0.0093 en MEDIA, +0.0217 en LARGA y **+0.0890 en MUY_LARGA**, y la tasa de error sube de 6,59 a 22,55 por ciento en la misma direccion. Ademas el margen de V03 ya explica sus propios errores con AUC 0.757141, y ningun bloque lo mejora.

**18. nQue nueva representacion ha demostrado complementariedad?** Ninguna ha superado a V03. La de mayor brecha especifica es el conjunto de composicion y posicion, con +0.0085891400 frente a un control del ancho total, que es MARGINAL y no llega al gate de +0,0015.

## El resultado de los seis experimentos

| ID | track | corrected_cv | delta vs V03 | decision |
|---|---|---:|---:|---|
| P63-00 | ID forensics | - | - | D. STRUCTURED_IDENTIFIER |
| P63-00b | gap analysis | - | - | AUDITORIA |
| P63-02 | A id predictivo | 0.4191359898 | -0.3580754627 | CLOSED. El indice de coleccion no aporta senal especifica |
| P63-03 | B estructura relacional | 0.6727442905 | -0.1044671620 | CLOSED. La estructura de repeticion no aporta sobre V03 |
| P63-04 | D composicion en ratios | 0.7669427506 | -0.0102687019 | CLOSED. La composicion agregada no aporta sobre V03 |
| P63-05 | E1 estructura del tweet largo | 0.7671031904 | -0.0101082622 | CLOSED. La estructura interna no aporta sobre V03 |
| P63-06 | H integracion y techo | 0.7534060878 | -0.0238053648 | MARGINAL |
| P63-07 | K residuo de V03 | - | - | La senal es nueva y el limite es de representacion |
| P63-04C | correccion de escala de P63-04 | - | - | CORRECCION: confound de escala: concatenar columnas estandarizadas enc |
| P63-06C | correccion del techo de P63-06 | - | - | CORRECCION: el techo tomaba el maximo sobre tres etiquetas y no la med |

## La conclusion, separada en las seis clases de senal que pide el encargo

```text
1 senal lexical         no queda: V03 ya lleva el texto entero con word(1,2)
2 senal sublexical      no queda: el canal char_wb(3,5) de V03 ya es sublexical
3 senal agregada        SI existe, +0.1055015148 frente al aleatorio, y no cabe junto a V03
4 senal estructural     REFUTADA donde se suponia, -0.009437
5 senal relacional      el bloque esta 92.92 por ciento vacio en este corpus
6 senal de procedencia  el id tiene estructura y no porta senal, y se mide por que
```

## nV03 realmente carece de informacion, o la informacion existe y no se explota?

**La segunda, y es la respuesta que sostiene la fase.**

La composicion agregada **existe** y se puede medir: +0.1055015148 frente a un control de igual dimension. Los bloques de posicion **existen**: +0,0712736720 en solitario. El problema no es que falte informacion, es que no cabe y, mas exactamente, que lo que aportan ya esta implicito en la confianza de V03, que explica sus propios errores con AUC 0.757141.

Y hay un limite mas profundo que esta fase deja medido y que ningun bloque de los
probados ataca: en la region de texto largo, el margen hasta el techo moviendo solo
el corte es de +0.0890. Esa es la parte del residuo que es

**orden** y no **decision**, y para moverla haria falta algo que capture lo que
dice un tweet largo y no solo que palabras contiene. Los transformers y los LLM
estan prohibidos por las reglas de la competencia, y las tecnicas de semantica
contexto sin modelo preentrenado no se han probado en este repositorio.

## Estado

| | |
|---|---|
| experimentos | 6 de 25 principales, 0 de 7 reservas |
| registro | 354 filas, 354 identificadores unicos |
| correcciones propias | 5, dos de ellas con fila propia en el registro |
| cumplimiento | 29 comprobaciones, 0 fallos |
| figuras | 21, ninguna oculta un negativo |
| dataset auxiliar | 73 columnas en tres familias, sin etiquetas |
| submissions | 0 |
| Kaggle | 0 |
| holdout | 0 |
| test | 0 |
| reproducibilidad | R2, sin R4 |

## Cierre

```text
no evidence of exploitable remaining model capacity in the record-level and
aggregate-composition dimensions of this representation

V03 remains the best verified leaderboard result.
```

No se afirma que V03 sea matematicamente optimo, ni que la competencia tenga un
techo alcanzable con los recursos permitidos. Se afirma exactamente lo que las
mediciones sostienen: las cuatro dimensiones que V03 no usa, el indice de
coleccion, la estructura relacional, la composicion agregada y la estructura
posicional, se han medido con controles de capacidad y de permutacion, y ninguna
supera a V03. Y el residuo que queda esta en el orden de los textos largos, con un
margen medido de +0,0890 frente al techo de decision.

## Artefactos

* `reports/p63_compliance.md`
* `reports/p63_composicion_linguistica.csv`
* `reports/p63_composicion_linguistica.json`
* `reports/p63_composicion_linguistica.md`
* `reports/p63_error_intersection.csv`
* `reports/p63_estructura_relacional.csv`
* `reports/p63_estructura_relacional.json`
* `reports/p63_estructura_relacional.md`
* `reports/p63_estructura_tweet_largo.csv`
* `reports/p63_estructura_tweet_largo.json`
* `reports/p63_estructura_tweet_largo.md`
* `reports/p63_final_research.md`
* `reports/p63_final_summary.json`
* `reports/p63_hypothesis_gap_analysis.csv`
* `reports/p63_hypothesis_gap_analysis.json`
* `reports/p63_hypothesis_gap_analysis.md`
* `reports/p63_hypothesis_registry.csv`
* `reports/p63_id_auditoria_predictiva.csv`
* `reports/p63_id_auditoria_predictiva.json`
* `reports/p63_id_auditoria_predictiva.md`
* `reports/p63_id_forensics.csv`
* `reports/p63_id_forensics.json`
* `reports/p63_id_forensics.md`
* `reports/p63_id_forensics_blocks.csv`
* `reports/p63_id_forensics_ocupacion.csv`
* `reports/p63_integracion_y_techo.csv`
* `reports/p63_integracion_y_techo.json`
* `reports/p63_integracion_y_techo.md`
* `reports/p63_master_results.csv`
* `reports/p63_record_features.csv`
* `reports/p63_record_features.md`
* `reports/p63_record_structure.csv`
* `reports/p63_reproducibility.md`
* `reports/p63_residuo_de_v03.csv`
* `reports/p63_residuo_de_v03.json`
* `reports/p63_residuo_de_v03.md`
* `reports/p63_techo_por_rebanada.csv`
* `figuras/p63/p63_01_v03_vs_candidatos.png`
* `figuras/p63/p63_02_delta_vs_v03.png`
* `figuras/p63/p63_03_consistencia_pliegues.png`
* `figuras/p63/p63_04_delta_candidatos.png`
* `figuras/p63/p63_05_bloques_en_solitario.png`
* `figuras/p63/p63_06_v03_mas_bloques.png`
* `figuras/p63/p63_07_distribucion_id.png`
* `figuras/p63/p63_08_id_rango_vs_fila.png`
* `figuras/p63/p63_09_saltos_id.png`
* `figuras/p63/p63_10_multiplicidad_id.png`
* `figuras/p63/p63_11_distribucion_tamano_grupo.png`
* `figuras/p63/p63_12_grado_duplicado.png`
* `figuras/p63/p63_13_error_por_rebanada.png`
* `figuras/p63/p63_14_error_por_rebanada_y_etiqueta.png`
* `figuras/p63/p63_15_techo_por_rebanada.png`
* `figuras/p63/p63_16_rescate_vs_dano.png`
* `figuras/p63/p63_17_interseccion_errores.png`
* `figuras/p63/p63_18_confianza_por_rebanada.png`
* `figuras/p63/p63_19_margen_por_rebanada.png`
* `figuras/p63/p63_20_id_bloques.png`
* `figuras/p63/p63_21_evolucion_candidatos.png`
* `features/toxic_record_profile.parquet`
* `features/toxic_record_profile_schema.md`

> V03 remains the best verified leaderboard result.
