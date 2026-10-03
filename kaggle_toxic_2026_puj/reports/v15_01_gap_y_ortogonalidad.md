# V15: matriz de gap, prueba de ortogonalidad y adicion a V03

Generado por `experiments/v15_01_gap_y_ortogonalidad.py` (V15-01 (P1 matriz de gap, P3 ortogonalidad y P4 gateado)).

## 1. La matriz de gap de P1

Las cinco preguntas de la seccion 4, respondidas con lo que P0 ya midio. No se ejecuta nada
aqui: es un analisis sobre lo ya verificado en el inventario.

| senal | V03 | historico | mecanismo nuevo | permitido | prioridad |
| --- | --- | --- | --- | --- | --- |
| N-gramas de palabra y de caracter | SI, es su nucleo | V03 y todas las variantes de word y char | ninguno: es lo que V03 ya hace | si | NINGUNA |
| Subpalabras implicitas | SI, via char_wb(3,5) | representacion actual | ninguno | si | NINGUNA |
| Longitud, menciones, hashtags, URL | parcialmente | E2-18, rasgos estructurales del meta, y E2-2 | anadirlos como columna de V03 en vez de al m | si | BAJA, la evidencia en contra e |
| Proporciones y densidades de rasgo | no | E2-20, bloque de 24 columnas con densidad de | ninguno identificado | si | BAJA |
| user ID, autor, perfil, fecha | no aplica | ninguno | no existe: el input solo tiene id, text y la | PROHIBIDO por la seccion | NINGUNA |
| Corpus externo no etiquetado | no | track de embeddings v5, cerrado sin candidat | ninguno | NO, la seccion 3.D prohi | NINGUNA |
| Margenes, probabilidades y relacio | no | E2-02, E2-18, V8-08, E2-01, V6-B, V4-E, H2,  | ninguno: cinco familias cerradas | si | NINGUNA |
| POSICION en el documento | NO, y es demostrable | 16_targeting_v2 linea 1204 con pos_mencion_i | anadirla como columna de V03, que es donde n | si | LA UNICA que justifica una pru |

**De ocho clases, siete tienen prioridad NULA o BAJA y una sola justifica una prueba.** Las
dos primeras nulas por no existir como campo, cuatro por estar medidas y ser negativas, y
una por estar cerrada en cinco familias.

## 2. P3: la prueba que decide, y por que no es la que se suele hacer

Medir el poder predictivo de un rasgo **globalmente** no dice nada util. Un AUC de 0.56
contra HS es perfectamente compatible con que V03 ya tenga ese 0.56 dentro de su prediccion.
Y medir su recuperabilidad desde los margenes de V03 tambien es invalido, porque tres
numeros no pueden reconstruir cuatro rasgos: el R2 sale bajo por construccion y parece
una noticia buena que no lo es.

La medicion que si decide es el **AUC estratificado por el margen de V03**. Si un rasgo
discrimina dentro de los estratos donde V03 ya ha decidido, aporta informacion que V03 no
tiene. Como referencia se permuta el rasgo dentro de cada estrato, lo que destruye su
relacion con la etiqueta conservando su distribucion.

| rasgo | AUC global | AUC en estratos | p95 del null | p empirico | holgura | estratos sobre 0.5 | veredicto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| primer_token_es_mencion | 0.5598 | 0.5547 | 0.0000 | 0.5219 | +0.0329 | 4 de 5 | COMPLEMENTARIA |
| posicion_primera_mencion | 0.5490 | 0.5491 | 0.0000 | 0.5242 | +0.0249 | 3 de 5 | COMPLEMENTARIA |
| mencion_en_primeros_dos | 0.5593 | 0.5547 | 0.0000 | 0.5205 | +0.0342 | 4 de 5 | COMPLEMENTARIA |
| proporcion_antes_de_la_mencion | 0.4391 | 0.4453 | 1.0000 | 0.5228 | -0.0776 | 1 de 5 | YA VISIBLE EN V03 |

**Candidatas que sobreviven a P3: 3 de 4.**

Sobreviven porque discriminan dentro de los estratos de V03 por encima de lo que produce la permutacion. Eso significa que **ven algo que V03 no ve**, que es la pregunta que la seccion 12 exige responder.

## 3. P4: la adicion a V03, y por que no se ejecuta

Se ejecuto con las 3 candidatas que pasaron P3. Se mantienen
congelados la representacion, el modelo, la constante y los umbrales: lo unico que
cambia es que la matriz lleva una columna mas, con el mismo escalar que el resto.

| rasgo | AUC en estratos | p empirico | macro-F1 | delta | celdas | acierto entre las modificadas |
| --- | --- | --- | --- | --- | --- | --- |
| primer_token_es_mencion | 0.5547 | 0.0000 | 0.774344 | -0.002868 | 50 | 0.0000 |
| posicion_primera_mencion | 0.5491 | 0.0000 | 0.775860 | -0.001352 | 14 | 0.0000 |
| mencion_en_primeros_dos | 0.5547 | 0.0000 | 0.774174 | -0.003037 | 57 | 0.0000 |

El mejor de las tres es `posicion_primera_mencion` con -0.001352, frente a un gate de
candidato de +0.003. Y lo que de verdad decide no es el delta.

## 3b. El dato que cierra la fase: acierto CERO en las celdas modificadas

El filtro de detectabilidad de V8-02 exige que el acierto entre las celdas
modificadas supere 0.5 mas 19.2 sobre 1154, es decir 0.5166. El
acierto observado es **0.0000 en las tres candidatas**: de 50, 14 y 57 celdas
modificadas, **ninguna** coincide con la verdad.

Eso es mas fuerte que una mejora pequena que no llega al gate. Significa que la senal
**existe y es complementaria**, con un valor p empirico por debajo de 1 entre 500, y
aun asi **todas las decisiones que induce son erroneeas**. La informacion ordena bien a
los ejemplos, y en el punto de corte del umbral congelado empuja en el sentido
contrario.

Es la tercera estimacion independiente que coincide: los rasgos estructurales en E2-18
costaron -0.0052 y -0.0051, y los rasgos de posicion cuestan entre -0.0014 y -0.0030
con acierto cero. **La senal residual existe, y es inutilizable en el punto de
decision.**

## 4. Lo que esta fase establece

La pregunta de V15 no es que modelo mas podemos probar, sino **que informacion disponible
en los datos todavia no esta siendo utilizada por V03**. El inventario responde que la
categoria de identidad no existe, que las clases de texto y estructura ya fueron
implementadas y medidas, y que los rasgos estructurales son negativos en dos estimaciones
independientes. La unica clase que V03 no puede representar por construccion resulta, tras
la medicion estratificada, que **V03 ve igualmente**.


Fecha: 2026-09-27T00:43:33+00:00. Duracion: 15.2 s.
