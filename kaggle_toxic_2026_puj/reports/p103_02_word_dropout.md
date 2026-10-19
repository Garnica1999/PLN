# P103-02 (N6) - Word dropout conservador (seleccion anidada)

Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba 0.00e+00; ccv 0.7772114525165721); particion 47 0 celdas distintas.
Piso de deteccion (pre-declarado): se bootstrap 0.004398; delta minimo detectable 95% +0.008619.

| Metrica | Canonica | Particion 47 |
|---|---|---|
| corrected_cv | 0.770339 | 0.771367 |
| delta vs V03 | -0.006873 | -0.004265 |
| pliegues mejorados | 1/5 | - |
| celdas cambiadas | 173 | 178 |
| IC95 delta | [-0.015445,+0.001793] | - |

## Configuracion elegida por pliegue (canonica)

| Pliegue | p | copias |
|---|---|---|
| 1 | 0.1 | 2 |
| 2 | 0.05 | 1 |
| 3 | 0.15 | 1 |
| 4 | 0.05 | 2 |
| 5 | 0.1 | 2 |

Gate A3: GATE_FAILED | {"delta_can_ge_0003": false, "pliegues_ge_4": false, "ic95_excluye_cero": false, "delta_47_positivo": false, "celdas_ge_200": false}

Conclusion: sin senal detectable: IC95 del delta [-0.015445,+0.001793] incluye cero; piso de deteccion +0.008619.

Sin test; sin holdouts; sin leaderboard; umbrales V03 congelados.
