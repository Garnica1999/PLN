# Auditoria de reglas de la fase v4

Fecha: 2026-09-25. Generado por `experiments/37_auditoria_v4.py`.

## 1. Reglas duras verificadas

| Regla | Verificacion | Resultado |
|---|---|---|
| NO_TRANSFORMERS | busqueda de `transformers` y `from_pretrained` en los scripts v4 | OK |
| NO_PRETRAINED_EMBEDDINGS | busqueda de `sentence_transformers`, `.vector`, `.tensor`, `similarity(` | OK |
| NO_EXTERNAL_CORPUS | busqueda de `kaggle`, `requests.get` y escrituras con `open(` | OK |
| NO_PSEUDO_LABELING | busqueda de `pseudo_label` y de entrenamiento con etiquetas del test | OK |
| NO_TEST_FITTING | el test no tiene etiquetas; los scripts que lo leen lo usan para `transform`/`predict` o para analisis no supervisado | OK |
| V03_INMUTABLE | sha256 de los artefactos congelados | OK |

## 2. Uso del conjunto de test por script

| Script | Lee el test | Uso declarado |
|---|---|---|
| 26_scaffold_v4.py | no | no lee el test |
| 27_label_consistency_v4.py | si | lee el test para analisis no supervisado (sin labels) |
| 28_hardness_v4.py | no | no lee el test |
| 29_reweighting_v4.py | no | no lee el test |
| 30_supervised_weighting_v4.py | no | no lee el test |
| 31_joint_label_v4.py | no | no lee el test |
| 32_meta_shift_v4.py | no | no lee el test |
| 33_class_conditional_lm_v4.py | no | no lee el test |
| 34_structural_sparse_v4.py | no | no lee el test |
| 35_discovery_v4.py | si | lee el test para analisis no supervisado (sin labels) |
| 36_candidatos_v4.py | no | no lee el test |
| 37_auditoria_v4.py | si | lee el test solo para transform/predict y escribe submission |
| 38_refit_final_v4.py | si | lee el test solo para transform/predict y escribe submission |

## 3. Referencias de la base congelada

| Artefacto | sha256 |
|---|---|
| model_v03.joblib | `263ed5bd0fc0de7cabfbaa85e88fca6fa65cf0a1b4e60a6e796b5120e26d13ca` (OK) |
| oof_v03_historical.npz | `cfb393f03b753dd3a2e750aca4bec3bfe80310055ef89581e6b671e7ea8cc896` (OK) |
| submission_v03.csv | `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e` (OK) |
| ../../models/mod03b_lr_union.joblib | `263ed5bd0fc0de7cabfbaa85e88fca6fa65cf0a1b4e60a6e796b5120e26d13ca` (OK) |
| ../../submissions/toxic_submission_v03_lr_union_thr.csv | `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e` (OK) |

## Veredicto

**CUMPLIDA**: 0 violaciones de reglas en los 12 scripts de la fase v4 y la referencia V03 permanece intacta.
