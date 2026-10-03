# V9-02 (H4): correccion de prior shift por EM

## Prevalencias (train vs EM en test, sin etiquetas)

| etiqueta | train | EM test | factor |
|---|---|---|---|
| HS | 0.4370 | 0.4092 | 0.936 |
| TR | 0.3001 | 0.2865 | 0.955 |
| AG | 0.3823 | 0.3587 | 0.938 |

## Ganancia simulada (F1 ponderada: umbrales re-optimizados - congelados)

| factor de shift | HS | TR | AG |
|---|---|---|---|
| 1.0 | +0.0056 | +0.0037 | +0.0030 |
| 0.95 | +0.0046 | +0.0031 | +0.0024 |
| 0.93 | +0.0042 | +0.0029 | +0.0022 |
| 0.9 | +0.0036 | +0.0028 | +0.0022 |
| 0.85 | +0.0032 | +0.0024 | +0.0022 |
| 0.8 | +0.0042 | +0.0020 | +0.0022 |
| 0.75 | +0.0052 | +0.0016 | +0.0036 |

- Umbrales congelados: {'HS': 0.44, 'TR': 0.51, 'AG': 0.46}
- Umbrales corregidos (EM): {'HS': 0.37, 'TR': 0.37, 'AG': 0.38}
- Ganancia simulada al shift EM (lax): {'HS': 0.004364, 'TR': 0.003186, 'AG': 0.002286}
- Validacion anidada (leave-one-fold-out): ganancia media +0.000574 -> GANANCIA NESTED PEQUENA
- Detalle anidado: {'HS': {'f1_fijo_ponderado': 0.749901, 'f1_nested_ponderado': 0.75274, 'ganancia_nested': 0.002839}, 'TR': {'f1_fijo_ponderado': 0.810658, 'f1_nested_ponderado': 0.809817, 'ganancia_nested': -0.000841}, 'AG': {'f1_fijo_ponderado': 0.744686, 'f1_nested_ponderado': 0.744412, 'ganancia_nested': -0.000275}}
- Submission sonda: `submissions/toxic_submission_v17_prior_shift_em.csv` (62 celdas cambian frente a V03).

Lectura: la correccion aporta una ganancia pequena y creciente con el shift
simulado; la sonda queda pre-registrada para decision de Carlos.

Fecha: 2026-09-26. Duracion: 2.26 s.