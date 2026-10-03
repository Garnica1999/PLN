# V30 - Reproducibilidad

El determinismo **se mide**, no se declara. Y hay un dato que va en contra de lo que el propio brief advertia: la reoptimizacion de los pesos del ensemble de V29 **reprodujo exactamente**, con desviacion 0,000000.

| Comprobacion | Resultado |
|---|---|
| V20 reproduce bit a bit | **True** |
| corrected_cv de V20 recomputado | 0.6097782 |
| LightGBM de V29 reproduce | **True** |
| Ensemble de V29 re-optimizado, desviacion | **0.0** |
| Optuna, trials por fold | 15 (75 en total) |
| Optuna, sampler | `TPESampler(seed=20300930 + fold) con trial 0 encolado` |
| Techo de estimadores | 1500 con early stopping de 60 rondas |
| Mejor trial outlier | 0 de 5 |
| Segunda particion, semilla | 987654 |
| Segunda particion re-optimizada | False |

## Que se declara y que no

Optuna con TPESampler y semilla fija resulto **exactamente reproducible** en la reoptimizacion de los pesos del ensemble de V29 (desviacion 0,000000), lo que demuestra que en esta maquina y con estas versiones de libreria el metodo es determinista. Aun asi no se declara exactitud bit a bit entre versiones distintas de LightGBM u Optuna.

## Lo que se **no** declara

- No se declara R4. No hubo reproduccion independiente por un tercero.

- No se declara que Optuna*dara* la misma solucion con otras versiones de LightGBM o de la propia Optuna. La reproduccion medida es dentro de este entorno.
