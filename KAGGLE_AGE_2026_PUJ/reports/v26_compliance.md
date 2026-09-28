# V26 · Auditoria de compliance

El punto 37 del brief pide **verificar codigo y artefactos**, no limitarse a declarar los valores. Cada comprobacion busca su patron en los tres scripts de V26 y en los artefactos de V26, y una comprobacion que encuentra el patron **solo en la frase que la describe** se considera mencion, no uso.

| Comprobacion | Uso en codigo | Menciones en artefactos | Cumplida |
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
| test_feature_selection | 0 | 0 | si |
| test_threshold_tuning | 0 | 0 | si |
| holdout_reuse | 0 | 0 | si |
| leaderboard_tuning | 0 | 0 | si |

**Todas cumplidas: True**

## Lo que cada comprobacion cubre

- **test labels, test training, test calibration, test feature selection y test threshold tuning**: el unico conjunto de datos que se carga en V26 es el train, mediante `datos.load_train()`. No hay ninguna llamada al test en los tres scripts.
- **holdout reuse**: la palabra holdout no aparece en codigo ejecutable de V26.
- **leaderboard tuning**: no hay llamada a la API de Kaggle. Las submissions de V26 son **0**.

## Lo que NO se verifica aqui y se declara como limitacion

**El script de auditoria y sus dos artefactos quedan excluidos de su propia compliance a proposito**: contienen literalmente los terminos prohibidos porque los busca, y un auditor que se encontraria a si mismo no seria un auditor. Su lint si se ejecuta y se reporta.

La comprobacion es sobre el **codigo y los artefactos de V26**. No es una auditoria del proyecto entero, que ya tiene su propia via en `scripts/v13_p0a_auditoria_idioma.py` y en la seccion 33 de V25.
