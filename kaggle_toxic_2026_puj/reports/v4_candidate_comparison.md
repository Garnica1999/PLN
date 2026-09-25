# Comparativa de candidatos finales (v4)

Generado por `experiments/36_candidatos_v4.py` el 2026-09-25. La comparacion es **descriptiva y no un ranking**: la spec (§99) prohibe ordenar mecanicamente y el historial del proyecto demuestra que el corrected_cv no predice el leaderboard (D-014, D-020, D-024).

## 1. Evidencia por candidato

| Candidato | corrected_cv (umbrales V03) | delta vs V03 | pliegues+ | IC95 pareado | F1 HS/TR/AG | nuevos aciertos | nuevos errores | neto |
|---|---|---|---|---|---|---|---|---|
| A_cost_sensitive_downweight | 0.7854 | +0.0082 | 5/5 | [-0.0002, +0.0117] | 0.779/0.817/0.761 | 44 | 34 | +10 |
| B_meta36 | 0.7828 | +0.0056 | 4/5 | [+0.0031, +0.0166] | 0.771/0.812/0.765 | 55 | 33 | +22 |
| C_meta36_downweight | 0.7882 | +0.0110 | 4/5 | [+0.0054, +0.0200] | 0.782/0.817/0.766 | 73 | 39 | +34 |
| D_svd300_rbf | 0.7904 | +0.0132 | 5/5 | [-0.0011, +0.0148] | 0.777/0.828/0.766 | 77 | 47 | +30 |
| V03_referencia | 0.7772 | +0.0000 | 0/5 | [+0.0000, +0.0000] | 0.762/0.816/0.754 | 0 | 0 | +0 |

## 2. F1 por pliegue

| Candidato | pliegues | V03 en los mismos pliegues |
|---|---|---|
| A_cost_sensitive_downweight | 0.7983;0.7956;0.7822;0.7622;0.7894 | 0.7960;0.7940;0.7789;0.7386;0.7791 |
| B_meta36 | 0.7989;0.8014;0.7839;0.7533;0.7751 | 0.7960;0.7940;0.7789;0.7386;0.7791 |
| C_meta36_downweight | 0.7878;0.8059;0.7985;0.7609;0.7870 | 0.7960;0.7940;0.7789;0.7386;0.7791 |
| D_svd300_rbf | 0.8013;0.7946;0.7870;0.7696;0.8010 | 0.7960;0.7940;0.7789;0.7386;0.7791 |
| V03_referencia | 0.7960;0.7940;0.7789;0.7386;0.7791 | 0.7960;0.7940;0.7789;0.7386;0.7791 |

## 3. Criterios y riesgo de transferencia

| Candidato | nivel >= +0.005 | >= 3/5 pliegues | IC95 > 0 | cumple D-016 | Riesgo de transferencia |
|---|---|---|---|---|---|
| A_cost_sensitive_downweight | True | True | False | False | bajo: mecanismo propio, coste marginal, no es stacking |
| B_meta36 | True | True | True | True | alto: misma familia que el meta v2, que rindio 0.78163 en el LB |
| C_meta36_downweight | True | True | True | True | medio-alto: combines la familia de stacking con el mecanismo nuevo |
| D_svd300_rbf | True | True | False | False | medio: modelo no lineal sobre compresion; el precedente DL-01 (0.7949) nunca se sondeo, y SVD + RBF puede depender de la estructura de los 300 componentes |
| V03_referencia | False | False | False | False | referencia (LB 0.78306) |

## 4. Decision de la fase

Candidatos que superan **todos** los gates de D-016: `B_meta36`, `C_meta36_downweight`. La decision de gastar la ultima evaluacion del holdout y una sonda de leaderboard corresponde a Carlos; este script no envia nada y no consume el holdout.
