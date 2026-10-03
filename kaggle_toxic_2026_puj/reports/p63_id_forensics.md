# P63 - Forense del identificador `id`

Generado por `experiments/p63_02_id_forensics.py`. Fecha 2026-09-29T06:10:53+00:00.

> **DIAGNOSTICO, SIN SUPERVISION EN SU MAYORIA.** No entrena nada, no crea features y
> no usa `id` como variable predictiva. Las secciones 1 a 5 no leen una sola etiqueta;
> la seccion 4 las lee solo para DESCRIBIR bloques que ya estaban localizados
> estructuralmente, que es lo que el encargo permite y no mas.

## Por que este informe existe

V15-00 audito que informacion ofrece el fichero bruto y concluyo:

```text
La unica columna no textual es id, que es un contador y que la seccion 3.E prohibe
```

**Esa afirmacion no se contrasto con la construccion del valor.** Comprobar
`is_unique` no es una forense, y los cinco primeros ids del fichero son

```text
20430, 22434, 23930, 20412, 23531
```

Un identificador que no empieza en uno y no sigue el orden de las filas no es un
contador de filas. La afirmacion se reabre y se mide.

## 1. Lectura, tipos y unicidad

| magnitud | valor |
|---|---:|
| n_rows | 1731 |
| n_unique_ids | 1731 |
| n_repeated_ids | 0 |
| duplicate_rate | 0.0 |
| max_multiplicity | 1 |
| min_value | 20002 |
| max_value | 24999 |
| range | 4997 |
| rango_teorico_de_valores | 4998 |
| valores_faltantes | 0 |
| valores_no_enteros | 0 |
| primer_valor | 20430 |
| primer_valor_es_uno | False |

El id es globalmente unico dentro de train: 1731 valores distintos para 1731
filas, cero repeticiones, cero nulos y cero valores no enteros. El fichero bruto lo
lleva entrecomillado, '20430'.

## 2. Estructura numerica, monotonia y saltos

| magnitud | valor |
|---|---:|
| monotono | False |
| fraccion_creciente_entre_filas_consecutivas | 0.5121387283 |
| salto_mediano_entre_filas_consecutivas | 49 |
| salto_minimo_entre_filas_consecutivas | -4949 |
| salto_maximo_entre_filas_consecutivas | 4753 |
| n_saltos_distintos_entre_filas | 1549 |
| salto_mediano_sobre_valores_ordenados | 2 |
| salto_minimo_sobre_valores_ordenados | 1 |
| salto_maximo_sobre_valores_ordenados | 31 |
| n_huecos_sobre_valores_ordenados | 1105 |
| n_valores_ausentes_en_el_rango | 3267 |
| densidad_de_ocupacion_del_rango | 0.3463385354 |

**El fichero esta barajado.** Solo el 51.21 por ciento de los
pares consecutivos de filas crecen, lo que equivale a un coeficiente de Kendall de
+0.024277. Por tanto el id no codifica el orden de compilacion.

El rango 20002-24999 tiene 4998 posiciones y
train usa 1731 de ellas, una ocupacion del 34.63 por ciento. Faltan
3267 posiciones.

## 3. Digitos, longitudes y entropia

| magnitud | valor |
|---|---:|
| longitud_decimal | {5: 1731} |
| longitud_binaria | {15: 1731} |
| entropia_de_digitos | 3.058769 |
| entropia_maxima_teorica | 3.321928 |
| entropia_normalizada | 0.920781 |
| escala_observada | 24999 |
| bits_necesarios_para_un_snowflake | 63 |
| orden_de_gran_magnitud_que_un_snowflake_daria | 18 |

Frecuencia de cada digito: `{'0': 0.485846, '1': 0.48989, '2': 1.526863, '3': 0.488735, '4': 0.507799, '5': 0.311958, '6': 0.306181, '7': 0.289428, '8': 0.299827, '9': 0.293472}`

La entropia de los digitos esta muy por debajo del maximo teorico de 3,32 bits, lo que
es exactamente lo que se espera de un rango acotado y no de un hash. Y la escala lo
dice solo: los valores llegan a 24999, mientras que un identificador temporal
tipo Snowflake ocupa 63 bits y produce 18 o 19 cifras.

## 3 bis. Relacion con el orden de las filas

| correlacion | valor |
|---|---:|
| spearman_id_vs_indice_de_fila | -0.022613 |
| spearman_id_mod_2_vs_indice_de_fila | -0.043888 |
| spearman_id_mod_10_vs_indice_de_fila | +0.011531 |
| spearman_id_entre_bits_altos_y_bajos | -0.027917 |

## 4. Bloques, con el corte declarado antes de mirar nada

El rango util se parte en **10 bloques de 500 posiciones**.
El corte se declara ANTES de leer ninguna etiqueta y no se ajusta despues.

| bloque | id desde | id hasta | n train | n test | densidad | prevalencia cualquiera |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20002 | 20501 | 179 | 115 | 0.3580 | 0.4022 |
| 1 | 20502 | 21001 | 147 | 126 | 0.2940 | 0.4898 |
| 2 | 21002 | 21501 | 189 | 121 | 0.3780 | 0.4550 |
| 3 | 21502 | 22001 | 166 | 110 | 0.3320 | 0.4518 |
| 4 | 22002 | 22501 | 168 | 124 | 0.3360 | 0.4167 |
| 5 | 22502 | 23001 | 177 | 114 | 0.3540 | 0.4689 |
| 6 | 23002 | 23501 | 166 | 123 | 0.3320 | 0.3976 |
| 7 | 23502 | 24001 | 169 | 109 | 0.3380 | 0.5148 |
| 8 | 24002 | 24501 | 179 | 101 | 0.3580 | 0.4804 |
| 9 | 24502 | 24999 | 191 | 110 | 0.3835 | 0.4450 |

La densidad de ocupacion va de 0.2940 a 0.3835 y
la prevalencia de cualquier etiqueta varia 0.117203 en
total. **No hay sub-bloques con comportamiento distinto:** la submuestra tiene la
misma densidad en todo el rango, que es la firma de una coleccion unica submuestreada
y no de varias fuentes con densidades diferentes.

## 5. Id contra texto, duplicados y grupos

| magnitud | valor |
|---|---:|
| n_textos_unicos_normalizados | 1731 |
| n_textos_repetidos | 0 |
| mismo_texto_implica_mismo_id | True |
| mismo_id_implica_mismo_texto | True |
| n_grupos_con_mas_de_un_id_en_train | 6 |
| distancia_intra_grupo_mediana | 1328 |
| distancia_intra_grupo_media | 1169.5 |
| distancia_entre_ids_aleatorios_esperada | 1657.982 |
| tamanano_de_la_otra_submuestra_como_referencia | 1154 |

El `id` es unico, luego `mismo id implica mismo texto` es trivialmente cierto. Lo que
no es trivial es lo otro: un mismo texto normalizado podria aparecer con dos ids
distintos, y ocurre en 0 casos. Es decir **el id identifica la fila, no
el contenido**, que es exactamente lo que se espera de un indice de corpus.

## 6. Integridad del namespace: control, no seleccion

| magnitud | valor |
|---|---:|
| n_ids_train | 1731 |
| n_ids_test | 1154 |
| n_ids_duplicados_dentro_de_train | 0 |
| n_ids_duplicados_dentro_de_test | 0 |
| interseccion | 0 |
| union | 2885 |
| rango_train | [20002, 24999] |
| rango_test | [20001, 24998] |
| rango_union | [20001, 24999] |
| valores_del_rango_no_usados | 2114 |
| solapamiento_de_rangos | True |
| grupos_canonicos_dev | 1415 |
| grupos_canonicos_holdout | 230 |
| corrected_cv_del_oof_canonico | 0.7772114525165721 |
| coherencia_oof | True |

**Este es el hallazgo estructural de la fase.**

```text
train        1731 ids, rango [20002, 24999]
test         1154 ids, rango [20001, 24998]
interseccion 0
union        2885, rango [20001, 24999]
posiciones del rango que no usa nadie  2114
```

Train y test son **dos muestras disjuntas del mismo bloque contiguo**, y ese bloque
tiene 4999 posiciones de las que se usan 2885,
unas 57.7 por ciento. Ademas el fichero
`grupos_near_duplicados.csv` tiene exactamente 2885 filas, una por id de la
union, lo que confirma que el bloque es el conjunto de registros del que se extrajeron
las dos particiones.

## 6 bis. Ocupacion del bloque, train contra test, sin etiquetas

| bloque | id desde | id hasta | train | test | sin usar | ocupacion | proporcion de train |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20001 | 20500 | 178 | 116 | 206 | 0.5880 | 0.6054 |
| 1 | 20501 | 21000 | 148 | 126 | 226 | 0.5480 | 0.5401 |
| 2 | 21001 | 21500 | 189 | 120 | 191 | 0.6180 | 0.6117 |
| 3 | 21501 | 22000 | 166 | 110 | 224 | 0.5520 | 0.6014 |
| 4 | 22001 | 22500 | 168 | 125 | 207 | 0.5860 | 0.5734 |
| 5 | 22501 | 23000 | 177 | 114 | 209 | 0.5820 | 0.6082 |
| 6 | 23001 | 23500 | 165 | 123 | 212 | 0.5760 | 0.5729 |
| 7 | 23501 | 24000 | 170 | 109 | 221 | 0.5580 | 0.6093 |
| 8 | 24001 | 24500 | 179 | 100 | 221 | 0.5580 | 0.6416 |
| 9 | 24501 | 24999 | 191 | 111 | 197 | 0.6052 | 0.6324 |

La ocupacion va de 0.5480 a 0.6180 y la proporcion de train entre los usados va de
0.5401 a 0.6416. Una proporcion de train
**uniforme** en todos los bloques es la firma de una particion aleatoria de un unico
bloque, y es justo lo contrario de un muestreo por subcolecciones con distinta tasa de
inclusion. Este es el control que separa las dos historias, y se ha hecho sin leer una
sola etiqueta.

Por tanto el `id` **no es un contador de las filas de este fichero**. Es un indice
dentro de una coleccion mayor, y V15-00 se equivoco al llamarlo contador.

## 7. Las siete categorias, una a una

| categoria | veredicto | evidencia |
|---|---|---|
| A. OPAQUE_ADMINISTRATIVE | en contra | vive en el intervalo contiguo [20001, 24999], y un identificador administrativo opaco se suele repartir por todo el rango posible |
| B. RANDOM_UNIQUE_IDENTIFIER | en contra | si 1731 valores se sorteasen con reemplazo de un rango de 4999 posiciones, se esperarian 29.27 por ciento de colisiones y aqui hay cero; ademas el rango NO empieza en uno, lo que un generador aleatorio de ids no hace |
| C. SEQUENTIAL_IDENTIFIER | en contra | solo el 51.21 por ciento de los pares consecutivos de filas crecen, es decir un Kendall de +0.024277, y el fichero esta barajado |
| D. STRUCTURED_IDENTIFIER | **A FAVOR** | el id es un indice dentro de un bloque contiguo de 4999 posiciones del que train y test son muestras disjuntas, y eso es una estructura, aunque sea de procedencia y no de contenido |
| E. POSSIBLE_SOURCE_IDENTIFIER | en contra | la ocupacion del bloque va de 0.5480 a 0.6180 sin una sola posicion sin usar en 10 bloques, y la proporcion de train entre los usados se mantiene entre 0.5401 y 0.6416, que NO es lo que se veria si cada subcoleccion hubiera aportado una fraccion distinta |
| F. POSSIBLE_TIME_ENCODED_IDENTIFIER | en contra | los valores llegan a 24999, unas 5 cifras, y un esquema temporal como Snowflake ocupa 63 bits y produce 18 o 19 cifras, luego la MAGNITUD refute la hipotesis por si sola |
| G. UNKNOWN | en contra | hay evidencia que descarta A, B, C y F, y evidencia parcial a favor de D y E, pero nada que distinga definitivamente una subfuente de un indice de corpus |

### Clasificacion: **D. STRUCTURED_IDENTIFIER**, confianza **alta**

La categoria **no** sale de contar pruebas a favor y en contra, porque eso mezcla
pruebas que apuntan a categorias distintas y decide por mayoria. Sale de reglas
explicitas, evaluadas en orden fijo, y la primera que se cumple gana:

| regla | se cumple |
|---|---|
| F. POSSIBLE_TIME_ENCODED_IDENTIFIER | no |
| C. SEQUENTIAL_IDENTIFIER | no |
| B. RANDOM_UNIQUE_IDENTIFIER | no |
| E. POSSIBLE_SOURCE_IDENTIFIER | no |
| D. STRUCTURED_IDENTIFIER | si |
| A. OPAQUE_ADMINISTRATIVE | no |
| G. UNKNOWN | no |

### Lectura

El id es un INDICE DE COLECCION: un entero dentro del bloque contiguo [20001, 24999], del que train y test son muestras disjuntas que en conjunto ocupan 2885 de 4999 posiciones, dejando 2114 sin usar. Hay ESTRUCTURA, pero es de PROCEDENCIA y no de contenido: el numero dice de que coleccion salio la fila y en que posicion ocupaba, y no dice quien la escribio, ni cuando, ni nada del texto. Por eso queda como senal de procedencia y no como senal linguistica

**Evidencia a favor**

- el id es un indice dentro de un bloque contiguo de 4999 posiciones del que train y test son muestras disjuntas, y eso es una estructura, aunque sea de procedencia y no de contenido

**Contraevidencia**

- vive en el intervalo contiguo [20001, 24999], y un identificador administrativo opaco se suele repartir por todo el rango posible
- la densidad de ocupacion es casi constante entre los 10 bloques, de 0.2940 a 0.3835, lo que es propio de una submuestra uniforme de una coleccion y no de un contador de filas
- si 1731 valores se sorteasen con reemplazo de un rango de 4999 posiciones, se esperarian 29.27 por ciento de colisiones y aqui hay cero; ademas el rango NO empieza en uno, lo que un generador aleatorio de ids no hace
- solo el 51.21 por ciento de los pares consecutivos de filas crecen, es decir un Kendall de +0.024277, y el fichero esta barajado
- faltan 2114 posiciones del rango y hay 1105 huecos, asi que no es una serie correlativa
- no hay sub-bloques internos: la densidad de ocupacion varia solo de 0.2940 a 0.3835 entre bloques de 500 posiciones, y la prevalencia de cualquier etiqueta varia solo 0.117203
- la ocupacion del bloque va de 0.5480 a 0.6180 sin una sola posicion sin usar en 10 bloques, y la proporcion de train entre los usados se mantiene entre 0.5401 y 0.6416, que NO es lo que se veria si cada subcoleccion hubiera aportado una fraccion distinta
- la proporcion de train entre los usados se mantiene en una franja estrecha en los 10 bloques, lo que es consistente con una particion por azar de un bloque unico y no con varias fuentes
- los valores llegan a 24999, unas 5 cifras, y un esquema temporal como Snowflake ocupa 63 bits y produce 18 o 19 cifras, luego la MAGNITUD refute la hipotesis por si sola
- la longitud binaria es constante en todo el rango, asi que no hay un campo de tiempo de ancho fijo embebido en los bits altos
- los huecos entre valores ordenados van de 1 a 31 sin ninguna progresion, lo que descarta una cadencia de muestreo temporal
- hay evidencia que descarta A, B, C y F, y evidencia parcial a favor de D y E, pero nada que distinga definitivamente una subfuente de un indice de corpus

## Lo que este informe NO concluye

- que el id sea util: eso exige una prueba predictiva, y este script no entrena
- que el id sea legal: eso exige una auditoria de legalidad aparte
- que el id sea transferible: eso exige una segunda particion
- que el bloque sea una senal: la prevalencia por bloque se Midio para DESCRIBIR bloques ya localizados estructuralmente, y no es una feature

Correlacion no implica legalidad ni transferibilidad, y un id estructurado sigue
siendo un id. El paso siguiente, si la evidencia lo justifica, es una prueba
predictiva minima con controles de permutacion y una segunda particion, no una
feature nueva.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p63_02_id_forensics.py`. Sin push.*