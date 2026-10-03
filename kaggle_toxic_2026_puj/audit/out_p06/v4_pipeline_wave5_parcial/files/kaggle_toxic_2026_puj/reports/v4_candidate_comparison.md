# Comparativa de candidatos finales (v4)

Generado por `experiments/36_candidatos_v4.py` el 2026-09-26. La comparacion es **descriptiva y no un ranking**: la spec (§99) prohibe ordenar mecanicamente y el historial del proyecto demuestra que el corrected_cv no predice el leaderboard (D-014, D-020, D-024).

## 1. Evidencia por candidato

| Candidato | corrected_cv (umbrales V03) | delta vs V03 | pliegues+ | IC95 pareado | F1 HS/TR/AG | nuevos aciertos | nuevos errores | neto |
|---|---|---|---|---|---|---|---|---|
| C_meta36_downweight | 0.7882 | +0.0110 | 4/5 | [+0.0054, +0.0200] | 0.782/0.817/0.766 | 73 | 39 | +34 |
| V03_referencia | 0.7772 | +0.0000 | 0/5 | [+0.0000, +0.0000] | 0.762/0.816/0.754 | 0 | 0 | +0 |

## 2. F1 por pliegue

| Candidato | pliegues | V03 en los mismos pliegues |
|---|---|---|
| C_meta36_downweight | 0.7878;0.8059;0.7985;0.7609;0.7870 | 0.7960;0.7940;0.7789;0.7386;0.7791 |
| V03_referencia | 0.7960;0.7940;0.7789;0.7386;0.7791 | 0.7960;0.7940;0.7789;0.7386;0.7791 |

## 3. Criterios y riesgo de transferencia

| Candidato | nivel >= +0.005 | >= 3/5 pliegues | IC95 > 0 | cumple D-016 | Riesgo de transferencia |
|---|---|---|---|---|---|
| C_meta36_downweight | True | True | True | True | medio-alto: combines la familia de stacking con el mecanismo nuevo |
| V03_referencia | False | False | False | False | referencia (LB 0.78306) |

## 4. Decision de la fase

Candidatos que superan **todos** los gates de D-016: `C_meta36_downweight`. La decision de gastar la ultima evaluacion del holdout y una sonda de leaderboard corresponde a Carlos; este script no envia nada y no consume el holdout.
