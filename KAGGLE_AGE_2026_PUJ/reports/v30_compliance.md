# V30 - Auditoria de compliance

Se distingue **uso** de **mencion** con el tokenizer: el codigo se mide tras eliminar literales de cadena y comentarios, de modo que lo que se cuenta es exactamente lo que Python ejecuta. Un informe que declara por que una familia esta descartada no infringe la prohibicion al escribirla.

| Comprobacion | Uso en codigo | Menciones en prosa | Cumplida |
|---|---|---|---|
| transformers | 0 | 0 | si |
| BERT | 0 | 0 | si |
| RoBERTa | 0 | 0 | si |
| LLM | 0 | 0 | si |
| embeddings | 0 | 0 | si |
| external_classifier | 0 | 0 | si |
| external_labels | 0 | 0 | si |
| pseudo_labeling | 0 | 0 | si |
| test_labels | 0 | 0 | si |
| test_training | 0 | 0 | si |
| test_calibration | 0 | 0 | si |
| test_threshold_tuning | 0 | 1 | si |
| holdout_reuse | 0 | 0 | si |
| leaderboard_tuning | 0 | 0 | si |
| toxic_test | 0 | 0 | si |

```text
transformers = 0
BERT = 0
RoBERTa = 0
LLM = 0
embeddings = 0
external_classifier = 0
external_labels = 0
pseudo_labeling = 0
test_labels = 0
test_training = 0
test_calibration = 0
test_threshold_tuning = 0
holdout_reuse = 0
leaderboard_tuning = 0
toxic_test = 0
```

**Comprobacion positiva:** ningun script de V30 abre un fichero cuyo nombre contenga `holdout`. Hallazgos: ninguno.

## Excepciones y limitaciones, declaradas

- El script de compliance y sus tres artefactos quedan excluidos de la comprobacion que realizan: `experiments/v30_06_cierre.json, reports/v30_compliance.md, reports/v30_submission_audit.md, scripts/v30_06_cierre.py`. Sus patrones son literalmente las palabras que buscan.

- La compliance busca **firmas**. No puede demostrar que una practica no ocurra de otra forma, y **no se declara R4**.

- El termino *holdout* merece una aclaracion: el holdout interior de las pistas de Optuna es un **split de validacion dentro de outer_train**, que el punto 11 del brief exige, y no el holdout reservado de AGE, que V30 ni siquiera tiene.

- `leaderboard_tuning` busca el **uso de la API**. El leaderboard no se ha consultado para decidir nada: la unica lectura prevista es el score de la sonda, que se registra como evidencia externa.
