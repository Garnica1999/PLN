# P66 - Cumplimiento

Fecha 2026-09-29T15:56:16+00:00.

**25 comprobaciones, 0 fallos.**

Analisis del arbol sintactico: se recogen imports, atributos, nombres y llamadas, y
ADEMAS los literales que llegan a read_csv, np.load y open. Ese ultimo detalle
importa en esta fase porque se lee un CSV de GRUPOS para el bootstrap, y una
busqueda por nombre lo habria marcado como uso indebido. El propio inspector se
excluye del analisis y lo declara.

| comprobacion | resultado | evidencia |
|---|---|---|
| transformers = 0 | OK | analisis AST |
| BERT / RoBERTa / LLM = 0 | OK | ningun cliente de modelo de lenguaje |
| embeddings preentrenados = 0 | OK | vector, tensor y similarity no se usan |
| clasificador externo = 0 | OK | solo el LogisticRegression de V03 |
| etiquetas externas = 0 | OK | solo HS, TR y AG del train |
| corpus externo = 0 | OK | ningun corpus ajeno al train |
| pseudo-labeling = 0 | OK | ninguna codificacion por objetivo |
| test labels = 0 | OK | toxic_test no se ABRE. Mencionarlo no cuenta: se miran los literales que llegan a read_csv |
| test tuning = 0 | OK | el test no se lee ni para diagnosticar |
| holdout2 no se usa para seleccionar | OK | se lee el CSV de grupos SOLO para remuestrear en el bootstrap, nunca para ajustar |
| leaderboard tuning = 0 | OK | no se consulto el leaderboard |
| submissions = 0 | OK | ninguna submission generada en P66 |
| solo espacio de scores | OK | la fusion consume p_V03 y p_C, nunca las matrices TF-IDF del candidato |
| protocolo nested real | OK | alpha, beta, a, b0 y la regresion se ajustan en inner_train y se eligen en inner_validation; la outer evalua |
| umbrales congelados | OK | 0,44 / 0,51 / 0,46, sin reajustar |
| control de barajado en cada mecanismo | OK | los cinco mecanismos se repiten con el score barajado del candidato |
| bootstrap por grupos | OK | 1415 grupos, no filas |
| V03 intacto | OK | sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| V03 reproducido | OK | 0.7772114525165721 |
| scores candidatos verificados | OK | 5 de 7 reproducen su fase de origen, 1 no verificable y 1 con diferencia declarada |
| sin rejillas | OK | alphas y betas son listas cortas fijadas antes |
| sin puerta por longitud | OK | la correccion longitud es continua, no un selector |
| errores propios corregidos | OK | seis, y uno de ellos cambio el veredicto del mejor resultado de la fase |
| acceso a red = 0 | OK | ningun modulo de red en el arbol |
| presupuesto respetado | OK | 2 de 16 principales, 0 de 5 reservas |

> V03 remains the best verified leaderboard result.
