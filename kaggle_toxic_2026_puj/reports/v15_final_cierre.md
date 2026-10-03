# V15: cierre de la frontera final

Generado por `experiments/v15_02_informe_cierre.py` (V15-02 (informe de cierre de la frontera final)).

## A. Estado del proyecto

| medida | valor |
| --- | --- |
| V03 en el externo | **0.78306** |
| V03 en el desarrollo | 0.7772114525165721 |
| mejor resultado local de la historia | 0.8465 en S08 |
| suerte de ese mejor local | NO ADOPTADA en LB (0.78163 < 0.78306): la senal del meta no transfiere; se m |
| experimentos en el registro | 242 |
| filas con adopcion positiva | 24 |
| filas con adopcion negativa explicita | 59 |
| familias cerradas | 18 |

**El mejor resultado local de toda la historia no se adopto**, y el motivo es el dato mas instructivo del proyecto: 0.8465 en local frente a 0.78163 en el externo, es decir que la mejor senal local medida **perdio en la competencia**. Por eso V03 sigue siendo la mejor solucion reproducible, y no por falta de intentos sino por evidencia en contra.

### Familias cerradas

| familia | cierre |
| --- | --- |
| Representacion basica TF-IDF | V03 y todas las variantes de word y char |
| Subpalabras | cubierta exactamente por char_wb(3,5) |
| Embeddings desde cero | track v5, cerrado sin candidata |
| Pooling, secuenciales, GRU y fine-tuning | cerrados en el track de embeddings |
| Meta-modelos y stacking | E2-02 y E2-18, +0.0102 local y rechazo por transferencia |
| Routing duro | H2, -0.004187, la llave falla en 143 filas |
| Factorizacion condicional | V6-B, B1 -0.0107, B2 -0.0008, B3 -0.0115 |
| Label powerset | V4-E, 0.771289 frente a 0.777211, NO_SIGNAL |
| Objetivo de ranking o por pares | H3, -0.002469, y V12-01, +0.002738 sostenido por un pliegue |
| Perdidas a medida | H2a, GCE +0.001616, marginal bajo el gate exploratorio |
| Perdida enmascarada o partially-label | V14, identica a H2-M con diferencia 0.000e+00 |
| Instance weighting y downweight | V4-B05, V4-C, V6-03, V6-07, V8-07, todos negativos |
| Rasgos estructurales | E2-18, -0.0052 y -0.0051 en dos estimaciones |
| Correccion de la estimacion de HS | V13, la correccion deja TR y AG con delta exactamente cero |
| Inferencia estructurada de etiquetas | V13, cerrada formalmente |
| Bagging y ensembles de repeticiones | cerrados, no superan la referencia |
| Prior shift y calibracion | v11, cerrada |
| Transduccion | cerrada en su momento |

## B. Diagnostico del cuello de botella

El techo del oraculo es **+0.046271**, y El oraculo no cambia ni una sola celda de la columna HS: pone TR y AG a cero en las 835 filas donde el HS verdadero es cero, borrando 196 falsos positivos, 78 de TR y 118 de AG. Es un efecto aguas abajo de la condicion semantica.

**V13 lo midio: corregir la etiqueta de HS deja TR y AG con delta exactamente cero, porque el meta produce una probabilidad para HS y no altera lo que el sistema decide sobre las otras dos etiquetas.**

El techo del oraculo procede de conocer la condicion semantica HS al evaluar TR y AG. Mejorar HS por si sola no produce beneficio. Y anadir la condicion como llave tampoco, porque la llave falla en 143 filas y da -0.004187

## C. Inventario de senal

Se auditaron **12 clases** de senal, con las
cinco preguntas del enunciado y con la comprobacion de que una columna ausente no es
informacion nueva.

| categoria | senal | veredicto |
| --- | --- | --- |
| E identidad o metadata | user ID, autor, perfil | NO EXISTE. El fichero bruto tiene solo id, text, HS, TR y AG |
| E identidad o metadata | fecha o timestamp | NO EXISTE como campo. Hay fechas dentro del texto, y ahi si llegan a V03 |
| E identidad o metadata | id de fila | PROHIBIDA por la seccion 3.E. Es un contador, con valores 20002, 20003, 2000 |
| D no supervisada permitida | corpus externo mc4_es y mixto | NO UTILIZABLE. La seccion 3.D prohibe asumir que no este etiquetada implica  |
| C supervisada | margenes, probabilidades y scores OOF | CERRADA. El meta de E2-18 dio 0.7874 y se rechazo por transferencia, con el  |
| C supervisada | relaciones entre HS, TR y AG | CERRADA en cinco familias distintas. V13 demostro que corregir HS deja TR y  |
| A textual directa | n-gramas de palabra y de caracter | ES EL NUCLEO DE V03. No es senal residual, es la senal que ya se usa |
| A textual directa | subpalabras implicitas | CUBIERTA por la representacion actual |
| B estructura del texto | longitud, menciones, hashtags, URLs, comilla | MEDIDA Y NEGATIVA DOS VECES. META+CH+ESTR-RIDGE 0.7822 frente a META-BASES-R |
| B estructura del texto | proporciones y densidades de las anteriores | MEDIDA en el track de emocion, ortografia y NBSVM, que quedo cerrado |
| A textual directa | posicion de la mencion en el texto | IMPLEMENTADA como feature en dos canales, ambos cerrados. No es nueva, pero  |
| A textual directa | orden entre palabras distantes | NO REPRESENTADO por V03, porque la ventana de word(1,2) es de dos y un model |

## D. Senal residual

Queda **una sola clase**: posicion en el documento. TfidfVectorizer extrae n-gramas sin registrar la posicion en el documento, y char_wb hace lo mismo dentro de cada palabra.

Y esa clase **si es complementaria**, que es el matiz que hay que preservar:

| rasgo | AUC global | AUC en estratos de V03 | p empirico | veredicto |
| --- | --- | --- | --- | --- |
| primer_token_es_mencion | 0.5598 | 0.5547 | 0.0000 | COMPLEMENTARIA |
| posicion_primera_mencion | 0.5490 | 0.5491 | 0.0000 | COMPLEMENTARIA |
| mencion_en_primeros_dos | 0.5593 | 0.5547 | 0.0000 | COMPLEMENTARIA |
| proporcion_antes_de_la_mencion | 0.4391 | 0.4453 | 1.0000 | YA VISIBLE EN V03 |

El null se construye permutando el rasgo dentro de cada estrato 500 veces, conservando su
distribucion y destruyendo su relacion con la etiqueta. Un AUC de 0.5547 con valor p
empirico por debajo de 1 entre 500 significa que **el rasgo ve algo que V03 no ve**.

**Y aun asi no sirve: Las tres complementarias dan deltas negativos y, sobre todo, acierto CERO entre las celdas modificadas, frente al umbral de detectabilidad de 0.5166.**

## E. Resultado de V15

- Experimentos: **3**, de los cuales 2 miden y el tercero es este informe de cierre.
- Candidatas evaluadas: **4**, de las cuales **3** pasaron la prueba de complementariedad.
- Candidatas anadidas a V03: **3**.
- Candidatas adoptadas: **0**.

| rasgo anadido a V03 | delta | celdas modificadas | acierto entre las modificadas | IC95 inferior | IC95 superior | supera el gate |
| --- | --- | --- | --- | --- | --- | --- |
| primer_token_es_mencion | -0.002868 | 50 | 0.0000 | -0.007718 | +0.001579 | no |
| posicion_primera_mencion | -0.001352 | 14 | 0.0000 | -0.003348 | +0.000541 | no |
| mencion_en_primeros_dos | -0.003037 | 57 | 0.0000 | -0.007855 | +0.001423 | no |

El umbral de detectabilidad de V8-02 es 0.5166 y el acierto observado es
**0.0000 en las tres**. Es decir, de todas las celdas que el rasgo cambia, **ninguna** acierta. No es una mejora que no llega al gate: es un cambio que danan sin ambiguedad.

**Reproducibilidad**: Control de identidad de V03 exacto en cada ejecucion, con diferencia 0.000e+00, y las dos verificaciones del null por permutacion con 500 remuestreos y semilla fija.

**Gastos**: 0 submissions, 0 de 3 evaluaciones de holdout2, 0 consultas de leaderboard, 0 segunda particion.

## F. Decision

**CLOSED**

> CLOSED. V03 continua siendo la mejor solucion reproducible bajo las restricciones actuales. Las principales familias de representacion, objetivo, arquitectura, estructura de etiquetas, routing y adaptacion ya fueron exploradas. No queda una fuente de senal permitida y genuinamente nueva identificada en el input disponible.

**El matiz que no se omite**: La senal residual existe y es demostrable: la posicion discrimina dentro de los estratos de V03 con valor p empirico inferior a 1 entre 500. Lo que no existe es una fuente de senal que sea EFFECTIVA, y esa distincion es la que cierra la fase.

**No se abre una V16 para buscar microganancias. La seccion 14 prohibe la dinamica de experimento negativo seguido de reformulacion, y las tres adiciones de V15 son negativas con acierto cero, que es el caso mas limpio de ese patron.**


Fecha: 2026-09-27T00:43:43+00:00. Duracion: 0.0 s.
