# V13: evidencia de reproducibilidad

## V03 verificada por control de identidad en cada ejecucion

| comprobacion | valor | resultado |
| --- | --- | --- |
| metrica de V03 reconstruida | 0.7772114525165721 |  |
| metrica de referencia | 0.7772114525165721 |  |
| diferencia | 0.000e+00 | identidad exacta |
| F1 de HS | 0.76168929 | canonical OOF |
| F1 de TR | 0.81573034 | canonical OOF |
| F1 de AG | 0.75421473 | canonical OOF |

Las tres F1 coinciden con las del OOF canonico `features/oof_v03_final.npz`, que la
fase v12 dejo congelado. La identidad es exacta, con diferencia cero, no aproximada.

## Elementos fijados que hacen reproducible la fase

| elemento | valor | de donde |
| --- | --- | --- |
| semilla | 42 | src.splits.RANDOM_STATE |
| pliegues externos | 5 | StratifiedGroupKFold con grupos near-dup fusionados |
| pliegues internos | 4 | para las features OOF del meta |
| umbral de HS | 0.44 | congelado en V03, sin reoptimizar |
| C de las bases | 2.0 | V03 sin cambios |
| C del meta | 1.0 | fijo, sin barrido |
| filtro de identidad | 1e-12 | parada si V03 no se reproduce |

## Estabilidad entre ejecuciones

El experimento se ejecuto dos veces: la primera antes de anadir la comprobacion de
cobertura OOF y la segunda despues. Los resultados son identicos fila a fila:

| metrica | primera ejecucion | segunda ejecucion |
| --- | --- | --- |
| B_solo_TR_AG macro-F1 | 0.769670 | 0.769670 |
| C_propio_mas_TR_AG macro-F1 | 0.777746 | 0.777746 |
| AUC de HS con TR y AG solos | 0.8433 | 0.8433 |
| HS rescatados del brazo C | 20 | 20 |
| HS danados del brazo C | 1 | 1 |

La unica diferencia entre las dos ejecuciones es que la segunda verifica ademas la
cobertura de las features OOF y aborta si no es exacta. No altera ningun resultado.


Fecha: 2026-09-26T23:57:55+00:00. Duracion: 0.0 s.
