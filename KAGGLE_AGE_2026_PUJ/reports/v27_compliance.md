# V27 - Auditoria de compliance

Verificacion por `grep` dirigido sobre el codigo y los artefactos de V27. Los valores se miden, no se declaran.

Ficheros auditados: **18**.

| Comprobacion | Uso en codigo | Menciones en prosa | Cumplida |
|---|---|---|---|
| transformers | 0 | 1 | si |
| BERT | 0 | 0 | si |
| RoBERTa | 0 | 0 | si |
| LLM | 0 | 1 | si |
| embeddings | 0 | 1 | si |
| external_classifier | 0 | 0 | si |
| external_labels | 0 | 0 | si |
| pseudo_labeling | 0 | 0 | si |
| test_labels | 0 | 0 | si |
| test_training | 0 | 0 | si |
| test_calibration | 0 | 0 | si |
| test_feature_selection | 0 | 0 | si |
| test_threshold_tuning | 0 | 0 | si |
| holdout_reuse | 0 | 2 | si |
| leaderboard_tuning | 0 | 0 | si |

**Todas cumplidas: True**

Baseline V20: **0.6097782**, hash de submission `2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d`, data hash `32393abca28db4c9`. El hash no cambia en V26 ni en V27: no se genero ninguna submission nueva.

## Excepcion declarada

Este script y sus dos artefactos quedan **excluidos** de la compliance que realiza, porque si no detectaria sus propias menciones a las palabras prohibidas: `experiments/v27_09_auditoria_final.json, reports/v27_compliance.md, scripts/v27_09_compliance_auditoria.py`. Es la misma excepcion deliberada de la compliance de V26, y se declara en vez de esconderse.

## Alcance y limitacion

La comprobacion es sobre el codigo y los artefactos de **V27**, y se apoya en la compliance ya hecha de V26 para el resto del proyecto. Es una auditoria deausente: busca las firmas de las practicas prohibidas y no puede demostrar que no ocurran de otra forma. No se declara R4 por esto.
