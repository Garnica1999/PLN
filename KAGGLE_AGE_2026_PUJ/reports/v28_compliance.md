# V28 - Auditoria de compliance

Medida por `scripts/v28_05_compliance_auditoria.py` con `grep` dirigido, distinguiendo **uso** de **mencion**: el codigo se mide tras eliminar los literales de cadena con el tokenizer, y en un informe toda mencion cuenta como prosa. Un informe que declara por que una familia esta descartada no infringe la prohibicion al escribirla.

Ficheros auditados: **19**.

| Comprobacion | Uso en codigo | Menciones en prosa | Cumplida |
|---|---|---|---|
| transformers | 0 | 2 | si |
| BERT | 0 | 0 | si |
| RoBERTa | 0 | 0 | si |
| LLM | 0 | 2 | si |
| embeddings | 0 | 2 | si |
| external_classifier | 0 | 0 | si |
| external_labels | 0 | 0 | si |
| pseudo_labeling | 0 | 0 | si |
| test_labels | 0 | 0 | si |
| test_calibration | 0 | 0 | si |
| test_feature_selection | 0 | 0 | si |
| test_threshold_tuning | 0 | 0 | si |
| holdout_reuse | 0 | 3 | si |
| leaderboard_tuning | 0 | 0 | si |

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
test_calibration = 0
test_feature_selection = 0
test_threshold_tuning = 0
holdout_reuse = 0
leaderboard_tuning = 0
```

**Las 14 comprobaciones en cero usos.**

## Higiene del repositorio, verificada al cierre

| Directorio | Sigue existiendo | Versionado | Fuera de la rama |
|---|---|---|---|
| `home` | True | False | **True** |
| `kaggle_toxic_2026_puj/tools` | True | False | **True** |

Ninguno se commiteo, ninguno se borro y ninguno se modifico.

## Excepciones y limitaciones, declaradas

- Este script y sus dos artefactos quedan excluidos de la compliance que realiza: `experiments/v28_05_auditoria_final.json, reports/v28_compliance.md, scripts/v28_05_compliance_auditoria.py`. Sus patrones son literalmente las palabras que busca, asi que sin la exclusion se detectaria a si mismo.

- La compliance busca **firmas** de las practicas prohibidas. No puede demostrar que no ocurran de otra forma, y no se declara R4 por esto.

- `leaderboard_tuning` busca el **uso de la API**, no la palabra. El leaderboard se cita en los informes como cifra y no se ha consultado para decidir nada.

- No se genero ninguna submission en V28 y no se toco ningun artefacto de V20.
