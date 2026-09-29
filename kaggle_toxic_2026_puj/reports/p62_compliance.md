# P62 - Cumplimiento

Generado por `experiments/p62_05_cerrar.py`. Fecha 2026-09-29T05:24:55+00:00.

**31 comprobaciones, 0 fallos.**

| comprobacion | resultado | evidencia |
|---|---|---|
| transformers = 0 | OK | no se importa transformers en ningun script de P62 |
| BERT = 0 | OK | sin referencias |
| RoBERTa = 0 | OK | sin referencias |
| LLM = 0 | OK | sin referencias |
| pretrained embeddings = 0 | OK | spaCy se carga con exclude=['ner'] y no se accede a vector, tensor ni similarity |
| external classifier = 0 | OK | solo LogisticRegression de sklearn |
| external labels = 0 | OK | solo HS, TR y AG del train canonico |
| pseudo-labeling = 0 | OK | ninguna pseudo-etiqueta |
| test labels = 0 | OK | ningun script lee test |
| test training = 0 | OK | ningun script entrena con test |
| test calibration = 0 | OK | no hay calibracion |
| holdout reuse = 0 | OK | holdout_v2_ids.csv solo como fuente del agrupador grupo_near_dup_autor |
| leaderboard tuning = 0 | OK | no se consulto el leaderboard |
| unauthorized external corpus = 0 | OK | las listas de negadores, modales y discursivos son del proyecto; no se descargo ningun lexicon |
| tagger declarado | OK | spaCy spaCy es_core_news_md 3.8.0, exclude=['ner'], autorizado por D-003 |
| tagger sin ambiguedad | OK | D-003 ya lo autorizo como anotador simbolico; no se introduce recurso nuevo |
| V03 intacto | OK | sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| V03 reproducido bit a bit | OK | ambos experimentos reconstruyen 0.7772114525165721 desde el codigo fuente |
| clasificador sin cambiar | OK | C=2,0 balanced liblinear en los tres brazos |
| umbrales sin cambiar | OK | 0,44 / 0,51 / 0,46 en todos los brazos |
| canales inducidos por pliegue | OK | min_df=2 sobre las filas de entrenamiento |
| confound de escala controlado | OK | la matriz candidata se reescala a la norma de fila del control |
| control independiente | OK | bloque aleatorio de tamano parecido, buscado por tamano de pool local del pliegue |
| EDA no usada para optimizar | OK | la EDA solo produjo diagnostico y no fue objetivo de ninguna busqueda |
| EDA global no mezclada con features | OK | no se uso ninguna estadistica de enriquecimiento global para construir rasgos |
| sintaxis de corte declarada antes | OK | los perfiles linguisticos se cortan en su cuartil 75, declarado antes de mirar |
| rejillas = 0 | OK | ninguna |
| presupuesto respetado | OK | 2 de 8 principales, 0 de 4 reservas |
| hipotesis duplicadas sin experimento | OK | N, P y O cerradas con 0 experimentos, como el registro permite |
| 0 submissions | OK | ninguna generada |
| 0 push | OK | sin push, merge, force-push ni reset destructivo |

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_05_cerrar.py`. Sin push.*