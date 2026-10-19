# P103-01 (N5) - Vectorizadores por etiqueta (seleccion anidada)

Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba 0.00e+00; ccv 0.7772114525165721); particion 47 0 celdas distintas.
Piso de deteccion (pre-declarado): se bootstrap 0.002463; delta minimo detectable 95% +0.004828.

| Metrica | Canonica | Particion 47 |
|---|---|---|
| corrected_cv | 0.779601 | 0.776994 |
| delta vs V03 | +0.002389 | +0.001362 |
| pliegues mejorados | 4/5 | - |
| celdas cambiadas | 64 | 114 |
| IC95 delta | [-0.002405,+0.007250] | - |

## Configuracion elegida por pliegue (canonica)

| Pliegue | HS | TR | AG |
|---|---|---|---|
| 1 | [3, 0.9, [1, 3], [3, 5]] | [3, 0.9, [1, 2], [3, 5]] | [2, 1.0, [1, 2], [2, 5]] |
| 2 | [2, 1.0, [1, 2], [3, 6]] | [3, 0.9, [1, 2], [3, 5]] | [2, 1.0, [1, 2], [2, 5]] |
| 3 | [2, 0.9, [1, 3], [3, 5]] | [2, 0.9, [1, 3], [3, 5]] | [2, 1.0, [1, 2], [2, 5]] |
| 4 | [2, 0.9, [1, 3], [3, 5]] | [2, 0.9, [1, 2], [3, 5]] | [3, 0.9, [1, 2], [3, 5]] |
| 5 | [2, 0.9, [1, 3], [3, 5]] | [3, 0.9, [1, 3], [3, 5]] | [3, 0.9, [1, 2], [3, 5]] |

## Estabilidad (conteo de elecciones, canonica)

- HS: [2, 0.9, [1, 3], [3, 5]] x3
- HS: [3, 0.9, [1, 3], [3, 5]] x1
- HS: [2, 1.0, [1, 2], [3, 6]] x1
- TR: [3, 0.9, [1, 2], [3, 5]] x2
- TR: [2, 0.9, [1, 3], [3, 5]] x1
- TR: [2, 0.9, [1, 2], [3, 5]] x1
- TR: [3, 0.9, [1, 3], [3, 5]] x1
- AG: [2, 1.0, [1, 2], [2, 5]] x3
- AG: [3, 0.9, [1, 2], [3, 5]] x2

Gate A3: GATE_FAILED | {"delta_can_ge_0003": false, "pliegues_ge_4": true, "ic95_excluye_cero": false, "delta_47_positivo": true, "celdas_ge_200": false}

Conclusion: sin senal detectable: IC95 del delta [-0.002405,+0.007250] incluye cero; piso de deteccion +0.004828.

Sin test; sin holdouts; sin leaderboard; umbrales V03 congelados.
