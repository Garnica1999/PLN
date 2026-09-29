# P64 - Cumplimiento

Fecha 2026-09-29T07:31:17+00:00.

**26 comprobaciones, 0 fallos.**

Las comprobaciones son un ANALISIS DEL ARBOL SINTAXTICO de los scripts de la fase,
no una declaracion de intenciones. Se recogen imports, atributos, nombres y
llamadas, de modo que un simbolo cuenta solo si se USA y no si aparece en una
cadena que explica por que no se usa. El script de cumplimiento se excluye de ese
analisis por autobusqueda, y la exclusion se declara aqui.

| comprobacion | resultado | evidencia |
|---|---|---|
| transformers = 0 | OK | analisis AST de 5 scripts |
| BERT / RoBERTa / LLM = 0 | OK | ningun cliente de modelo de lenguaje en el arbol |
| embeddings preentrenados = 0 | OK | doc.vector, token.vector y similarity(): no se usan como atributo. Aparecen en cadenas que documentan lo prohibido, que no es uso |
| corpus externos = 0 | OK | spaCy no se usa en esta fase: la segmentacion es por reglas, como exige el encargo |
| etiquetas externas = 0 | OK | solo HS, TR y AG del train |
| clasificador externo = 0 | OK | el unico modelo es el de V03, reentrenado dentro de cada pliegue |
| pseudo-labeling = 0 | OK | ningun segmento recibe etiqueta ni se usa como objetivo. Es descomposicion de un modelo que ya existe |
| test nunca leido = 0 | OK | toxic_test no se abre en ningun script de la fase |
| holdout2 congelado nunca leido = 0 | OK | holdout2_ids.csv no aparece en ningun script |
| holdout_v2 no se usa | OK | holdout_v2_ids.csv tampoco se lee en esta fase, que no necesita el agrupador |
| leaderboard tuning = 0 | OK | no se consulto el leaderboard en ningun momento |
| submissions = 0 | OK | ninguna submission generada en P64 |
| etiquetas de segmento derivadas de validacion = 0 | OK | los documentos parciales se puntuan con modelos ajustados SOLO en entrenamiento |
| metadata ilegal = 0 | OK | el unico recurso no textual, el id, quedo cerrado en P63 y no se ha reabierto |
| V03 intacto | OK | sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| V03 reproducido bit a bit | OK | 0.7772114525165721 desde el codigo fuente |
| clasificador sin cambiar | OK | C=2,0 balanced liblinear en todos los brazos |
| umbrales sin cambiar | OK | 0,44 / 0,51 / 0,46 congelados |
| confound de escala controlado | OK | los bloques densos se normalizan a norma de fila uno antes de concatenarse |
| controles de capacidad presentes | OK | cada candidato tiene su control aleatorio de igual anchura |
| control del margen presente | OK | cada candidato tiene V03 mas sus margenes, que es el control que decidio las tres lineas anteriores |
| cortes de longitud heredados | OK | los mismos cuatro de P63, congelados y no ajustados |
| segmentacion determinista | OK | cuatro expresiones regulares sobre el texto, sin modelo, sin lexicon y sin spaCy |
| errores propios corregidos | OK | seis: perfil duplicado en el subconjunto, captura por cierre, division por cero en la entropia, AUC con signo invertido, prefijos de columna, y una conclusion de la EDA escrita por segments |
| presupuesto respetado | OK | 4 de 22 principales, 0 de 6 reservas |
| acceso a red = 0 | OK | ningun modulo de red en el arbol sintactico |

> V03 remains the best verified leaderboard result.
