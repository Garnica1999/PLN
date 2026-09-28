# V29 - Auditoria de compliance

Medida distinguiendo **uso** de **mencion** con el tokenizer: el codigo se mide tras eliminar los literales de cadena. Un informe que declara por que una familia esta descartada no infringe la prohibicion al escribirla.

| Comprobacion | Uso en codigo | Menciones en prosa | Cumplida |
|---|---|---|---|
| transformers | 0 | 3 | si |
| BERT | 0 | 2 | si |
| RoBERTa | 0 | 2 | si |
| LLM | 0 | 2 | si |
| embeddings | 0 | 2 | si |
| external_classifier | 0 | 2 | si |
| external_labels | 0 | 2 | si |
| pseudo_labeling | 0 | 2 | si |
| test_labels | 0 | 2 | si |
| test_training | 0 | 0 | si |
| holdout_reuse | 0 | 10 | si |
| leaderboard_tuning | 0 | 0 | si |
| toxic_test | 0 | 2 | si |

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
holdout_reuse = 0
leaderboard_tuning = 0
toxic_test = 0
```

**Las 13 comprobaciones en cero usos.**

## Excepciones y limitaciones, declaradas

- El script de compliance y sus dos artefactos quedan excluidos de la comprobacion que realizan: `experiments/v29_08_auditoria.json, reports/v29_compliance.md, scripts/v29_08_auditoria.py`. Sus patrones son literalmente las palabras que buscan.

- La compliance busca **firmas**. No puede demostrar que una practica no ocurra de otra forma, y no se declara R4.

- El termino **holdout** merece una aclaracion: el "holdout interior" de las pistas de Optuna es un **split de validacion dentro de outer_train**, que el punto 10 del brief exige, y no el holdout reservado de AGE, que V29 ni siquiera tiene. Por eso la comprobacion mide uso en codigo con literales y comentarios eliminados, y se acompana de una **comprobacion positiva**: ningun script de V29 abre un fichero cuyo nombre contenga `holdout`.

- `test_training` y `leaderboard_tuning` buscan uso de la API y llamadas de ajuste sobre el test, no la palabra. El test no se abrio en ningun punto de V29.
