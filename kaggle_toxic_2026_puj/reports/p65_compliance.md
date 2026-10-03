# P65 - Cumplimiento

Fecha 2026-09-29T14:55:51+00:00.

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
| embeddings preentrenados = 0 | OK | doc.vector, token.vector y similarity(): no se usan como atributo |
| clasificador externo = 0 | OK | el unico modelo es el LogisticRegression de V03, reentrenado por pliegue |
| etiquetas externas = 0 | OK | solo HS, TR y AG del train |
| corpus externo = 0 | OK | ningun corpus ajeno al train |
| pseudo-labeling = 0 | OK | ninguna codificacion por objetivo ni busqueda de etiqueta |
| test labels = 0 | OK | toxic_test no se ABRE en ningun script. Mencionarlo no cuenta: el inspector mira los literales que llegan a read_csv, no el texto |
| test tuning = 0 | OK | y por la misma razon tampoco se usa para diagnosticar ni para medir |
| holdout reuse = 0 | OK | ni holdout2 ni holdout_v2 se ABRE en ningun script |
| leaderboard tuning = 0 | OK | no se consulto el leaderboard en ningun momento |
| submissions = 0 | OK | ninguna submission generada en P65 |
| V03 intacto | OK | sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| V03 reproducido | OK | 0.7772114525165721 desde el OOF canonico |
| clasificador sin cambiar | OK | C=2,0 balanced liblinear en todos los brazos |
| umbrales sin cambiar | OK | 0,44 / 0,51 / 0,46 congelados |
| control de capacidad en cada candidato | OK | todo bloque nuevo lleva su control aleatorio de igual anchura |
| control del margen en cada candidato | OK | todo bloque nuevo lleva V03 mas sus margenes |
| confound de escala controlado | OK | los bloques se normalizan a la norma media del bloque base |
| puerta del especialista no optimizada | OK | 16 tokens heredada de los cortes de P63, no elegida por F1 |
| rejillas evitadas | OK | ordenes de n-gramas y rangos de hueco en secuencia, no en rejilla |
| cortes de longitud heredados | OK | los cuatro de P63, congelados |
| segmentacion determinista | OK | expresiones regulares sobre el texto, sin modelo ni lexico |
| errores propios corregidos | OK | cinco: mascara booleana con .sum, salto de linea que movio un parentes, scipy.sparse.random llamada como modulo, prefijos de columna duplicados, y un veredicto partido entre dos literales |
| acceso a red = 0 | OK | ningun modulo de red en el arbol |
| presupuesto respetado | OK | 4 de 25 principales, 0 de 7 reservas |

> V03 remains the best verified leaderboard result.
