# P61 - Auditoria de cumplimiento

Generado por `experiments/p61_09_cerrar.py`. Fecha 2026-09-28T21:12:26+00:00.

**34 comprobaciones, 0 fallos.**

| comprobacion | resultado | evidencia |
|---|---|---|
| transformers = 0 | OK | no se importa transformers en ningun script de P61 |
| BERT = 0 | OK | sin referencias a BERT |
| RoBERTa = 0 | OK | sin referencias a RoBERTa |
| LLM = 0 | OK | sin referencias a LLM |
| embeddings = 0 | OK | solo se usa TF-IDF, sin embeddings |
| external labels = 0 | OK | las unicas etiquetas son las del train canonico |
| external classifiers = 0 | OK | solo LightGBM, autorizado por el encargo |
| pseudo-labeling = 0 | OK | sin pseudo-etiquetas en ningun script |
| test labels = 0 | OK | ningun script lee etiquetas de test |
| test training = 0 | OK | ningun script entrena con test |
| test calibration = 0 | OK | ningun script calibra con test |
| test feature selection = 0 | OK | la seleccion de miembros nunca ve test |
| holdout reuse = 0 | OK | holdout_v2_ids.csv solo como fuente del agrupador grupo_near_dup_autor |
| leaderboard tuning = 0 | OK | el LB se lee de CHAMPION.md por regex y no se usa para elegir nada |
| submissions = 0 | OK | ninguna submission generada en P61 |
| kaggle consultations = 0 | OK | 0 llamadas de red a Kaggle |
| V03 intacto | OK | sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e |
| V03 corrected_cv exacto | OK | 0.7772114525165721 verificado al inicio de cada script |
| grupos canonicos intactos | OK | 1415 grupos, 0 partidos, verificado en P61-00 |
| anidamiento de pesos | OK | los pesos se derivan dentro de cada pliegue, con las filas internas |
| anidamiento de pertenencia | OK | los miembros se eligen dentro de cada pliegue, con las filas internas |
| EDA no usada para optimizar | OK | el artefacto de la EDA declara su naturaleza diagnostica en su encabezado y ningun script posterior lee sus numeros como objetivo |
| calibracion no reabierta | OK | Brier, log loss y ECE se reportan como diagnostico y no se usan para recalibrar |
| control negativo ejecutado | OK | test 3 con los tres peores canales, delta +0.0059889238 |
| rejillas = 0 | OK | una busqueda de Optuna y una de pesos, ninguna rejilla |
| una familia nueva solo | OK | LightGBM, autorizada y unica |
| MLP no reabierto | OK | no se reejecuta, el criterio era diversidad ajustada y el test 3 la descarta |
| un experimento = un script nuevo | OK | p61_00, p61_01, p61_02, p61_03, p61_05, p61_08, p61_09, ninguno edita a otro |
| ejecucion real con exit code | OK | cada script se ejecuto y su exit code se capturo SIN pipe |
| artefactos parseables | OK | todos los JSON y CSV se releen al consolidarlos |
| ASCII puro | OK | auditoria byte a byte al final de este script |
| 3 filas de P61 en el registro | OK | 3 presentes: P61-01, P61-02, P61-03; paso de 339 a 339 |
| identificadores unicos | OK | 339 ids para 339 filas |
| 0 empujones | OK | sin push, sin merge, sin force-push, sin reset destructivo |

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p61_09_cerrar.py`. Sin push.*