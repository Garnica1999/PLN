# P63 - Cumplimiento

Fecha 2026-09-29T06:59:40+00:00.

**29 comprobaciones, 0 fallos.**

Las comprobaciones son una INSPECCION del codigo de la fase, no una declaracion
de intenciones: se leen los scripts y se buscan las referencias prohibidas.

| comprobacion | resultado | evidencia |
|---|---|---|
| transformers = 0 | OK | analisis sintactico de 9 scripts: ningun import ni simbolo |
| BERT / RoBERTa / LLM = 0 | OK | ningun cliente de modelo de lenguaje |
| embeddings preentrenados = 0 | OK | doc.vector, token.vector, doc.tensor y similarity(): no se usan como atributo en ningun script. Aparecen en el esquema del dataset como documentacion de lo PROHIBIDO, no como uso |
| corpus externos = 0 | OK | spaCy es_core_news_md 3.8.0 es un tagger simbolico autorizado por D-003, no un embedding. Ningun corpus externo se lee en la fase |
| lexicones externos = 0 | OK | las proporciones de persona y categoria salen de los tags de spaCy, no de listas |
| etiquetas externas = 0 | OK | solo HS, TR y AG del train |
| pseudo-labeling = 0 | OK | ninguna referencia. Aparecen en comentarios que explican por que NO se usan |
| lectura de test = solo la de integridad | OK | toxic_test se menciona en ['p63_02_id_forensics.py'], y unicamente en el diagnostico de namespace que el encargo EXIGE en su seccion 14. Ningun script abre el fichero para entrenar ni para ajustar nada |
| test como entrenamiento = 0 | OK | ningun script de la fase abre el fichero de test |
| holdout2 congelado nunca leido = 0 | OK | holdout2_ids.csv aparece en ningun script |
| holdout_v2 leido solo como agrupador | OK | holdout_v2_ids.csv se lee en 3 scripts y en ninguno se usa para nada que no sea el agrupador canonico grupo_near_dup_autor, que es exactamente lo que el encargo autoriza |
| leaderboard tuning = 0 | OK | no se consulto el leaderboard en ningun momento de la fase |
| submissions = 0 | OK | ninguna submission generada en P63 |
| metadata ilegal = 0 | OK | el unico recurso no textual es el id, y se audito antes de cualquier uso |
| features derivadas de etiqueta = 0 | OK | el dataset auxiliar se verifica con un assert: sin HS, TR, AG, OOF ni test |
| id no adoptado como predictor = 0 | OK | el bloque id se midio, perdio contra el control y no se adopto |
| grupos no convertidos en lookup = 0 | OK | no hay target encoding ni propagacion de etiquetas; el rasgo de contradiccion se midio y se descarto por ser un techo |
| acceso a red = 0 | OK | ningun modulo de red en el arbol sintactico. No se importa requests, urllib, socket ni httpx en ningun script de la fase |
| V03 intacto | OK | sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| V03 reproducido bit a bit | OK | 0.7772114525165721 desde el codigo fuente |
| clasificador sin cambiar | OK | C=2,0 balanced liblinear en todos los brazos |
| umbrales sin cambiar | OK | 0,44 / 0,51 / 0,46 congelados |
| confound de escala controlado | OK | todos los bloques densos se normalizan a norma de fila uno antes de concatenarse |
| controles de capacidad presentes | OK | cada candidato tiene su control aleatorio de igual ancho |
| seleccion fold-safe | OK | la frecuencia de documento y los conteos de grupo se ajustan dentro del pliegue |
| segunda particion no usada | OK | el encargo la prohibe durante la exploracion y no se ejecuto |
| cortes de longitud declarados antes | OK | congelados en el fuente de los tracks E1 y H |
| errores propios corregidos y registrados | OK | cinco: escala, agrupador, StandardScaler, techo del residual, sintaxis del veredicto |
| presupuesto respetado | OK | 6 de 25 principales, 0 de 7 reservas |

> V03 remains the best verified leaderboard result.
