# P62 - Cumplimiento

Fecha 2026-09-29T05:50:51+00:00.

**21 comprobaciones, 0 fallos.**

| comprobacion | resultado | evidencia |
|---|---|---|
| transformers/BERT/RoBERTa/LLM = 0 | OK | ninguna referencia |
| pretrained embeddings = 0 | OK | spaCy con exclude=['ner'], sin vectores |
| external classifier/labels/corpus = 0 | OK | solo sklearn y las tres etiquetas del train |
| pseudo-labeling = 0 | OK | ninguna |
| test labels/training/calibration/features = 0 | OK | ningun script lee test |
| holdout reuse = 0 | OK | holdout_v2_ids.csv solo como fuente del agrupador grupo_near_dup_autor |
| leaderboard tuning = 0 | OK | no se consulto |
| tagger declarado y autorizado | OK | spaCy es_core_news_md 3.8.0, D-003 |
| V03 intacto | OK | sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| V03 reproducido bit a bit | OK | 0.7772114525165721 desde el codigo fuente |
| clasificador sin cambiar | OK | C=2,0 balanced liblinear en todos los brazos |
| umbrales sin cambiar | OK | 0,44 / 0,51 / 0,46 |
| confound de escala controlado | OK | todos los brazos reescalados |
| controles independientes | OK | bloques aleatorios de tamano parecido |
| seleccion fold-safe | OK | enriquecimiento solo con filas de entrenamiento |
| rejillas = 0 | OK | K es variable independiente de una medicion |
| guardia de control permanente | OK | verificacion del F1 de V03 en el pliegue 1 antes de cada bloque |
| errores propios corregidos y registrados | OK | acumulacion de probabilidades, desajustes de sintaxis, lemmas multipalabra, desempaquetado |
| lectura erronea corregida en el origen | OK | el perfil de longitud se renombro y el informe de la EDA avisa del cambio de signo |
| presupuesto respetado | OK | 5 de 8 principales, 0 de 4 reservas |
| 0 push | OK | sin push, merge, force-push ni reset |

> V03 remains the best verified leaderboard result.
