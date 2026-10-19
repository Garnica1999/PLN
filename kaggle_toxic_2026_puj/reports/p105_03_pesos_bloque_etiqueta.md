# P105-03 (S7c) - Pesos de bloque por etiqueta (nested)

Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba 0.00e+00; ccv 0.7772114525165721); particion 47 0 celdas distintas.
Piso de deteccion (pre-declarado): se bootstrap 0.002425; delta minimo detectable 95% +0.004752.

| Metrica | Canonica | Particion 47 |
|---|---|---|
| corrected_cv | 0.776087 | 0.775454 |
| delta vs V03 | -0.001125 | -0.000178 |
| pliegues mejorados | 2/5 | - |
| celdas cambiadas | 67 | 91 |
| IC95 delta | [-0.005996,+0.003508] | - |

## Peso char elegido por pliegue (canonica)

| Pliegue | HS | TR | AG |
|---|---|---|---|
| 1 | 1.5 | 1.0 | 1.25 |
| 2 | 1.25 | 1.25 | 1.5 |
| 3 | 1.5 | 0.75 | 1.5 |
| 4 | 1.5 | 1.0 | 1.0 |
| 5 | 1.25 | 0.5 | 1.0 |

## Estabilidad (conteo de elecciones, canonica)

- HS: peso 1.5 x3
- HS: peso 1.25 x2
- TR: peso 1.0 x2
- TR: peso 1.25 x1
- TR: peso 0.75 x1
- TR: peso 0.5 x1
- AG: peso 1.5 x2
- AG: peso 1.0 x2
- AG: peso 1.25 x1

Gate A3: GATE_FAILED | {"delta_can_ge_0003": false, "pliegues_ge_4": false, "ic95_excluye_cero": false, "delta_47_positivo": false, "celdas_ge_200": false}

Conclusion: sin senal detectable: IC95 del delta [-0.005996,+0.003508] incluye cero; piso de deteccion +0.004752.

Sin test; sin holdouts; sin leaderboard; umbrales V03 congelados.
