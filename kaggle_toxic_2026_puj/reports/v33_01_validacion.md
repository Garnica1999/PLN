# V33-01: validacion completa de V32-05

Generado por `experiments/v33_01_validacion.py` (V33-01 (validacion completa de V32-05 antes de la submission)).

**Estado `VALIDATED`. Alcance: fases 1 a 8 de la directiva V33. No se genera submission en este script**

## 1. Fase 1: reproducibilidad

| magnitud | valor |
|---|---:|
| corrected_cv | 0.7817210522 |
| delta vs V03 | +0.0045095996 |
| rescates | 77 |
| danios | 59 |
| neto | +18 |
| celdas modificadas | 136 |
| hash predicciones | `c0624458e7cedc34` |
| identico en el mismo proceso | True |
| identico con semillas distintas | True |

## 2. Fase 2: el fold 4 y la heterogeneidad

| pliegue | filas | neto | rescates | danios | con URL | tramos |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 297 | +3 | 14 | 11 | 90 | 76/102/119 |
| 2 | 297 | -8 | 13 | 21 | 87 | 99/102/96 |
| 3 | 296 | +8 | 16 | 8 | 91 | 92/104/100 |
| 4 | 296 | +9 | 19 | 10 | 87 | 87/106/103 |
| 5 | 297 | +6 | 15 | 9 | 103 | 93/113/91 |

## 3. Fase 3: descomposicion de la ganancia

| paso | delta |
|---|---:|
| V03_a_URL_aditivo | +0.0002085130 |
| URL_aditivo_a_URL_mas_interacciones | +0.0039580233 |
| V03_a_URL_mas_interacciones | +0.0041665363 |
| V03_a_V32_05_tres_bins | +0.0045095996 |
| V32_05_menos_V32_C | +0.0003430633 |

## 4. Fase 4: los 3 bins

Limites por pliegue, calculados dentro del training fold:

- pliegue 1: [11.0, 19.0]
- pliegue 2: [11.0, 20.0]
- pliegue 3: [11.0, 20.0]
- pliegue 4: [11.0, 19.70620000000008]
- pliegue 5: [11.0, 20.0]

## 5. Fase 5: holdout2

los scripts de V31 y V32 solo leen la particion dev de holdout_v2_ids.csv; las 248 filas de holdout nunca se usaron para seleccionar modelos, interacciones ni bins.

| magnitud | valor |
|---|---:|
| filas | 248 |
| V03 macro-F1 | 0.828468 |
| V32-05 macro-F1 | 0.827949 |
| delta | -0.000519 |
| rescates | 14 |
| danios | 13 |
| neto | +1 |

## 6. Fase 6: V32-04 contra V32-05

Filas distintas: 76.
Rescates 43 | danios 42 | neto +1.

## 7. Fase 7: V32-05 contra V03

| label | rescates | danios | neto | celdas |
|---|---:|---:|---:|---:|
| HS | 29 | 22 | +7 | 51 |
| TR | 6 | 9 | -3 | 15 |
| AG | 42 | 28 | +14 | 70 |

## 8. Lo que este script no hace

- No genera submission.
- No modifica V03 ni V32-05.
- No usa el test para seleccionar nada.

Coste: 28.0 s, memoria 229 MB, 6 modelos, 0 submissions.

