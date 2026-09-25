# Validacion de submissions v2 (V03 y candidatos experimentales)

Fecha: 2026-09-25. Script: `experiments/25_submissions_candidatos.py`.
Los archivos se validan contra `submissions/toxic_submission_v00_plantilla.csv`
(1154 filas; ids identicos y en el mismo orden). V03 y v08 solo se leen.

## Contrato de formato

| archivo | sha256 (12) | columnas | filas | ids v00 (mismo orden) | valores {0,1} | sin NaN | sin duplicados | dtype | resultado |
|---|---|---|---|---|---|---|---|---|---|
| submissions/toxic_submission_v09_candidate_a.csv | 8e5a95d3bd6a | id,HS,TR,AG | 1154 | OK | OK | OK | OK | int64 | OK |
| submissions/toxic_submission_v10_candidate_b.csv | 92caa41e379f | id,HS,TR,AG | 1154 | OK | OK | OK | OK | int64 | OK |
| submissions/toxic_submission_v11_candidate_c.csv | 066d46ca12e9 | id,HS,TR,AG | 1154 | OK | OK | OK | OK | int64 | OK |
| submissions/toxic_submission_v03_lr_union_thr.csv | 7d7478f271e6 | id,HS,TR,AG | 1154 | OK | OK | OK | OK | int64 | OK (hash congelado intacto) |

## Umbrales por etiqueta y positivos predichos

El screening de `experiments/23` congelo un unico umbral del meta (0.50) para las tres etiquetas; no hay umbrales optimizados por etiqueta en estos candidatos.

| submission | HS | TR | AG | positivos HS/TR/AG | tasa HS/TR/AG |
|---|---|---|---|---|---|
| C-EMO (toxic_submission_v09_candidate_a.csv) | 0.50 | 0.50 | 0.50 | 462/351/427 | 0.4003/0.3042/0.3700 |
| C-SYN (toxic_submission_v10_candidate_b.csv) | 0.50 | 0.50 | 0.50 | 460/351/432 | 0.3986/0.3042/0.3744 |
| C-EMO-SYN (toxic_submission_v11_candidate_c.csv) | 0.50 | 0.50 | 0.50 | 456/348/428 | 0.3951/0.3016/0.3709 |

Referencias de tasa de positivos: train 0.4367/0.2998/0.3824; test estimado E00 0.406/0.279/0.356.

## Criterio de datos y reproducibilidad

- Criterio verificado en `experiments/18_p7_holdout_y_submission_v08.py`: el meta se ajusta con el OOF del desarrollo (1483 filas) y las features del test se construyen con el train completo (1731 = desarrollo + holdout). El test solo se predice.
- Verificacion previa de definiciones: C-EMO corrected_cv 0.780941 (documentada 0.780941, dif 2.2e-07), C-SYN corrected_cv 0.779313 (documentada 0.779313, dif 1.5e-08), C-EMO-SYN corrected_cv 0.779661 (documentada 0.779661, dif 8.3e-08).
- Reproducibilidad: se guardan los sha256 en `reports/submission_hashes_v2.json`; una segunda ejecucion reescribe los mismos bytes. Ultima comprobacion: True.
- V03 intacta (sha256 7d7478f271e6...); la sonda v08 no se toca. Ninguna submission se sube a Kaggle en esta fase.
