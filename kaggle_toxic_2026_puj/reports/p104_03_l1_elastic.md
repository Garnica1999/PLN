# P104-03 (R7-1) - L1 / Elastic Net por etiqueta (nested)

Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba 0.00e+00; ccv 0.7772114525165721); particion 47 0 celdas distintas.
Piso de deteccion (pre-declarado): se bootstrap 0.005216; delta minimo detectable 95% +0.010222.

| Metrica | Canonica | Particion 47 |
|---|---|---|
| corrected_cv | 0.778129 | 0.775214 |
| delta vs V03 | +0.000917 | -0.000418 |
| pliegues mejorados | 3/5 | - |
| celdas cambiadas | 257 | 259 |
| IC95 delta | [-0.009404,+0.011041] | - |

## Configuracion elegida y nnz por pliegue (canonica)

| Pliegue | HS (pen,l1,C) | nnz HS | TR | nnz TR | AG | nnz AG |
|---|---|---|---|---|---|---|
| 1 | ['elasticnet', 0.5, 4.0] | 2000 | ['elasticnet', 0.8, 4.0] | 488 | ['elasticnet', 0.8, 4.0] | 700 |
| 2 | ['elasticnet', 0.2, 4.0] | 7214 | ['l1', None, 2.0] | 107 | ['l1', None, 2.0] | 142 |
| 3 | ['elasticnet', 0.2, 4.0] | 7091 | ['elasticnet', 0.5, 4.0] | 1418 | ['l1', None, 4.0] | 299 |
| 4 | ['elasticnet', 0.5, 4.0] | 2099 | ['elasticnet', 0.2, 4.0] | 5068 | ['elasticnet', 0.8, 4.0] | 688 |
| 5 | ['elasticnet', 0.2, 2.0] | 4391 | ['elasticnet', 0.5, 4.0] | 1402 | ['elasticnet', 0.5, 2.0] | 936 |

## Estabilidad (conteo de elecciones, canonica)

- HS: ["elasticnet", 0.5, 4.0] x2
- HS: ["elasticnet", 0.2, 4.0] x2
- HS: ["elasticnet", 0.2, 2.0] x1
- TR: ["elasticnet", 0.5, 4.0] x2
- TR: ["elasticnet", 0.8, 4.0] x1
- TR: ["l1", null, 2.0] x1
- TR: ["elasticnet", 0.2, 4.0] x1
- AG: ["elasticnet", 0.8, 4.0] x2
- AG: ["l1", null, 2.0] x1
- AG: ["l1", null, 4.0] x1
- AG: ["elasticnet", 0.5, 2.0] x1

## Features no nulas (nnz) por etiqueta

- HS: media 4559.0 (min 2000, max 7214) de 20441 rasgos
- TR: media 1696.6 (min 107, max 5068) de 20441 rasgos
- AG: media 553.0 (min 142, max 936) de 20441 rasgos

Gate A3: GATE_FAILED | {"delta_can_ge_0003": false, "pliegues_ge_4": false, "ic95_excluye_cero": false, "delta_47_positivo": false, "celdas_ge_200": true}
No convergencias (max_iter 3000): canonica 0; 47 0

Conclusion: sin senal detectable: IC95 del delta [-0.009404,+0.011041] incluye cero; piso de deteccion +0.010222.

Sin test; sin holdouts; sin leaderboard; umbrales V03 congelados.
