# P102-01 - Variantes char sin 'wb' (control = V03)

Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba 0.00e+00; ccv 0.7772114525165721).

| Variante | corrected_cv | delta V03 | pliegues | celdas | IC95 vs V03 | delta 47 | Gate |
|---|---|---|---|---|---|---|---|
| control_M2 | 0.777211 | +0.000000 | 0/5 | 0 | [+0.000000,+0.000000] | +0.000000 | - |
| V1_char_no_wb | 0.777247 | +0.000035 | 2/5 | 128 | [-0.007346,+0.007820] | +0.000235 | GATE_FAILED |
| V2_triple | 0.781678 | +0.004466 | 3/5 | 91 | [-0.001664,+0.011226] | +0.002446 | GATE_FAILED |
| V3_char_no_wb_case | 0.754837 | -0.022374 | 1/5 | 332 | [-0.035641,-0.008659] | -0.024704 | GATE_FAILED |

Gate A3: delta >= +0.003; >=4/5 pliegues; IC95 excluye cero;
delta 47 > 0; >= 200 celdas cambiadas.
Sin test; sin holdouts; sin leaderboard; umbrales V03 congelados.
