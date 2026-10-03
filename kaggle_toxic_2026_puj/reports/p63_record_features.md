# P63 - Dataset auxiliar de registros

Generado por `experiments/p63_09_dataset_auxiliar.py`. Fecha 2026-09-29T06:53:48+00:00.

> **NO ES UN EXPERIMENTO.** No entrena, no crea candidatos, no usa test, holdout ni
> leaderboard. Construye el artefacto que el encargo pide y deja la procedencia de
> cada columna escrita, no solo contada.

## Las tres familias, separadas a proposito

| familia | columnas | que es | permitido como entrada |
|---|---:|---|---|
| RECORD | 2 | propiedades de la fila que NO leen el texto | **NO**: P63-03 lo mostro peor que un control de capacidad |
| RELATIONAL | 5 | el agrupador canonico de casi duplicados | las de recuento SI, con el protocolo de alcance |
| TEXT_DERIVED | 64 | todo lo que lee el texto | SI, con escala corregida y control de capacidad |

El gap analysis showed que el bloque historico de registro era 45 de 46
text-derived, luego esta distincion es justo el punto de construir el fichero: una
columna que se llama de registro y en realidad lee el texto no es un rasgo de
registro.

**El dataset no contiene HS, TR, AG, ni OOF, ni test, ni target encoding.** El
script lo comprueba con un assert antes de escribir.

## El alcance de las columnas relacionales

La unica columna RELATIONAL que se escribe es `rel_tamano_grupo_entrenamiento`, y su
alcance se calcula **dentro de cada pliegue de entrenamiento**. Una columna con el
mismo nombre y alcance sobre todo el dataset seria fuga, y la diferencia entre ambas
es exactamente lo que el track B midio: -0,1206 de corrected_cv.

Las columnas `rel_contradice_*` se escriben marcadas como NO usables. Un grupo con
la etiqueta contradictoria no puede acertarla todas, luego es un techo y no un
rasgo, y la razon queda en el esquema en vez de desaparecer.

## El esquema completo

Todas las columnas estan en `features/toxic_record_profile_schema.md` y en
`reports/p63_record_features.csv`, con tipo, familia, definicion, origen, regla de
seguridad de pliegue y uso permitido. Resumen por familia:

| familia | columnas | n distintos en RECORD y RELATIONAL |
|---|---:|---|
| CLAVE | 1 | 1 |
| RECORD | 2 | 2 |
| RELATIONAL | 5 | 5 |
| TEXT_DERIVED | 65 | - |

## Las figuras

21 figuras, y ninguna oculta un resultado negativo. La de candidatos
colorea en rojo todo lo que pierde contra V03 y la de delta tiene la linea de cero a
la vista.

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

## La lectura que las figuras sostienen

1. La ocupacion del bloque de ids es casi uniforme, luego no hay subfuentes.
2. El margen de V03 explica sus propios errores con AUC 0,7571, y los bloques nuevos
   lo empeoran, luego lo que saben ya estaba en la confianza.
3. El techo moviendo solo el corte crece de +0,0000 a +0,0890 con la longitud.
4. El error crece de 6,59 a 22,55 por ciento en la misma direccion.

Las cuatro lnneas apuntan al mismo sitio y ninguna la ataca.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p63_09_dataset_auxiliar.py`. Sin push.*