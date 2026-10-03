# Auditoría del estado real — Fase A (2026-09-25)

Generado por `scripts/phase12_auditoria_estado.py`. Sin escritura sobre artefactos previos.

## 1. Alineación

- Train 169,336 · test 112,892
- Etiquetas vs OOF: idénticas
- Clústeres: 282,228 filas (esperadas 282,228) · coincidencia de normalizado 100.0%

## 2. Matriz del meta (train)

- Reconstruida [169336, 273] vs caché [169336, 273]
- Diferencia máxima absoluta: 0.0 · exacta: True

## 3. Firmas de bloque

| Bloque | Cols | exp-suma = 1 | Varianza cero | z máx train<->test |
|---|---|---|---|---|
| oof_E013_margenes | 6 | — | 0 | 7.17 |
| oof_E016_margenes | 6 | — | 0 | 6.92 |
| oof_E-009 | 6 | 9e-08 | 0 | 13.46 |
| oof_E-011 | 6 | 1e-07 | 0 | 11.31 |
| oof_E-010 | 6 | 9e-08 | 0 | 11.3 |
| oof_E014_margenes | 6 | — | 0 | 6.87 |
| oof_E015_ordinal | 1 | — | 0 | 1.03 |
| oof_E015_softmax | 6 | 1e-07 | 0 | 5.4 |
| oof_E017_probabilidades | 6 | 1e-07 | 0 | 12.2 |
| oof_E019b_densas_svd | 6 | 1e-07 | 0 | 23.78 |
| oof_E021a_ordinal | 6 | 1e-07 | 0 | 14.5 |
| oof_E021b_rocchio | 6 | 9e-08 | 0 | 1.58 |
| oof_E028a_pares | 5 | — | 0 | 7.47 |
| oof_ADV05_nbsvm | 6 | — | 0 | 5.64 |
| densas | 35 | — | 0 | 3.73 |
| estructuradas | 87 | — | 0 | 3.09 |
| supervisadas | 73 | — | 1 | 136.77 |

## 4. Reproducción del CV LOFO v12

- Referencia 0.6005 · reproducido 0.6004 ± 0.0042 · por fold [0.6007, 0.605, 0.6032, 0.6004, 0.5928]
- Dentro de tolerancia: True

## 5. End-to-end (sustrato de test)

- Acuerdo con `age_submission_v11.csv`: **100.0000%**
- Acuerdo v11 vs v12: 84.3957%

## 6. Holdout congelado

- 25,204 filas (14.884%) · determinista: True

## 7. Submission v12

- Checks: {'columns_ok': True, 'columns_match_sample': True, 'rows_ok': True, 'ids_equal_test': True, 'ids_unique': True, 'ids_same_order': True, 'no_nan': True, 'labels_valid': True}
- Distribución: {'13-17': 17673, '18-24': 20715, '25-34': 16543, '35-49': 17531, '50-64': 20763, '65-xx': 19667}

## 8. Envíos en Kaggle

- registrados: 15 · completos: 12 · con puntaje público: 12
- mejores: [{"ref": "56532366", "date": "2026-09-24 21:57:33.920000", "status": "SubmissionStatus.COMPLETE", "description": "", "file_name": "age_submission_v12.csv", "public_score": "0.62182", "private_score": ""}, {"ref": "56531121", "date": "2026-09-24 20:32:44.850000", "status": "SubmissionStatus.COMPLETE", "description": "", "file_name": "age_submission_v11.csv", "public_score": "0.61744", "private_score": ""}, {"ref": "56526066", "date": "2026-09-24 16:15:08.497000", "status": "SubmissionStatus.COMPL