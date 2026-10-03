# G14 - auditoria de informacion de entrada disponible

## La pregunta

> ¿Existe algun campo de entrada legal, presente en los datos disponibles, que ningun modelo de AGE haya utilizado realmente, y que pueda contener senal predictiva independiente?

## El esquema, medido sobre los ficheros

El ZIP de la competicion contiene **exactamente tres ficheros**: `age_train.csv`, `age_test.csv` y `age_submission.csv`. No hay campos ocultos detras de un fichero que no se haya mirado.

| Campo | Split | Tipo | Filas | Nulos | Cardinalidad |
|---|---|---|---|---|---|
| `id` | train | int64 | 169,336 | 0 | 169,336 |
| `text` | train | object | 169,336 | 0 | 168,091 |
| `age_range` | train | category | 169,336 | 0 | 6 |
| `id` | test | int64 | 112,892 | 0 | 112,892 |
| `text` | test | object | 112,892 | 0 | 112,245 |

## La auditoria del campo `id`, que es donde estaba la duda

Un id de tweet de Twitter es un *snowflake* de 18-19 digitos cuya parte alta codifica una marca de tiempo. Si el campo `id` fuera eso, seria **una fuente de informacion real**: la fecha de publicacion, presente en train y en test, legal y no textual.

**No lo es.**

| | Train | Test |
|---|---|---|
| rango | 1 a 169,336 | 1 a 112,892 |
| monotono | True | True |
| unico | True | True |
| digitos del maximo | 6 | 6 |

**Interseccion train-test: 112,892, es decir el 100% del test.** Los ids van de 1 a N dentro de cada split y vuelven a empezar: el `id` 1 es un tweet distinto en train y en test.

Consecuencia: **un campo cuyo valor se repite entre train y test para filas distintas no puede aportar informacion sobre la fila: la misma entrada corresponderia a dos salidas distintas en el 100 % del test**.

Y aunque fuera un snowflake, la correlacion con la clase es **-0.000838**: nada.

### Lo que el brief prohibe aqui, y se respeta

correlacion con la clase -0.000838: nada. El brief anticipa este caso y prohibe rescatarlo con modulo, hash, posicion, patrones de digitos, buckets o sinonimos, porque eso seria pescar correlaciones espurias sobre un numero de fila

## Traza de la representacion del texto

La pregunta no es solo que columnas hay, sino **que informacion de `text` no llega a los modelos**. Cada dimension se contrasta con lo que ya se probo:

| Dimension | Se conserva | Que la captura | Probado en | Estado |
|---|---|---|---|---|
| contenido lexico | si, integro | TF-IDF de palabra (1-2) y char_wb (3-5) | desde el baseline E-001 | **EN USO** |
| case (mayusculas y minusculas) | no en el TF-IDF, que aplica lowercase | canal `ortografia`: bloques CAPS, ratio de mayusculas, CAPS por palabra | V31-03 (+0,001993 en banco) y V31-06 (sin aporte en la integracion) | **PROBADO, SIN APORTE CERRADO** |
| puntuacion | no en el TF-IDF, que tokeniza con \w+ | canal `ortografia`: repeticiones, y canal `discurso`: preguntas, exclamaciones, suspension | V31-03 y V31-06 | **PROBADO, SIN APORTE CERRADO** |
| elongaciones y caracteres repetidos | no | canal `ortografia`: REPETICION, REPVOCAL, REPCONSON, REPLONG | V31-03 y V31-06 | **PROBADO, SIN APORTE CERRADO** |
| URLs | si, como token de la propia URL | canal `url` (presencia, tipo, longitud, posicion) y la taxonomia de P32/P33 | V31-02, V31-03, P33-P02 | **PROBADO, SIN APORTE CERRADO** |
| menciones y hashtags | si como token | el propio TF-IDF de palabra, y el canal `discurso` para la posicion | features estructuradas de V20 y V31 | **EN USO** |
| longitud | si | bloque estructurado de 279 features (LENBUCKET, NTOKENS) y los slices de longitud de V31-F | V20 y V31 | **EN USO** |
| posicion de los elementos | no | canales `posicion` y `forma_palabra` de V31-03 | V31-03: **negativos** (-0,002303 y -0,002984, 0/5 folds) | **PROBADO Y CERRADO NEGATIVO** |
| acentos y ortografia | si, en el texto original | el TF-IDF de palabra sin strip_accents, y las listas de marcadores de V31 | V31-03 | **EN USO** |

Ninguna dimension queda en el limbo: las que no llegan al TF-IDF fueron cubiertas por los diez canales de V31-03, y su integracion se cerro sin aporte en V31-06. Eso es el **caso C** del brief: informacion dentro de `text` que ya esta contenida o ya fue probada. No es un canal nuevo.

## Matriz campo por modelo

| Campo | Estado | Veces |
|---|---|---|
| `text` | **USED** | 7 de 9 familias |
| `text` | **DERIVED** | 2 de 9 familias |
| `id` | **NOT_USED** | 9 de 9 familias |
| `age_range` | **PROHIBITED** | 9 de 9 familias |

## Lectura

- `text`: **USED** o **DERIVED** en las nueve familias de modelo. Es la entrada y esta explotada.
- `id`: **NOT_USED** en las nueve, y no por descuido. Es un indice de fila.
- `age_range`: **PROHIBITED** como entrada en las nueve, y correctamente: es el target.

## El grupo A, como informacion secundaria

El gap analysis establecio que **27,007** de los 65.403 errores de V30 no los corrige ninguna de las 26 fuentes disponibles, y **38,396** los corrige alguna. G14 anade un dato a esa lectura: **tampoco hay un campo de entrada sin explotar que pudiera corregirlos**. Las dos vias estan cerradas por motivos distintos, y conviene no confundirlas: el grupo A no esta cerrado por falta de explotacion, esta cerrado porque la informacion no existe en la entrada disponible.
