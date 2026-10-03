# Segunda particion de A_cost_sensitive_downweight

Generado por `experiments/p69_02_segunda_particion.py`. Fecha 2026-09-29T17:51:33+00:00.

## Por que se ejecuta

La reproduccion primaria salio **R3 exacta**: el score, el delta, los cinco deltas por
pliegue, los 99 rescates y los 74 danos, todos identicos. Lo que no salio compatible con
mejora es el bootstrap pareado, cuyo intervalo toca el cero por dos diezmilesimas.

Eso deja dos lecturas distintas:

```text
el efecto es real y el intervalo es ancho por tener 1483 filas
el efecto es un artefacto de ESTA particion
```

Solo se distinguen cambiando la particion y congelando todo lo demas.

## Lo que queda congelado

| elemento | valor |
|---|---|
| pesos | error_score del OOF propio, z sobre el IQR, clip(1-0.5z,0.5,1.5) |
| clasificador | LogisticRegression(C=2.0, max_iter=1000, solver=liblinear) |
| representacion | TF-IDF word(1,2) min_df=2 + char_wb(3,5) min_df=2, por pliegue |
| umbrales | [0.44, 0.51, 0.46] |
| barrido_de_clase | 3 rondas x 3 etiquetas x (0.8, 1.0, 1.2, 1.5) |
| protocolo_de_pliegues | StratifiedGroupKFold sobre grupos fusionados |

**Lo unico que cambia**: la semilla de particion, `20260928`, la misma que uso
P61 para su prueba de robustez. Ningun grupo queda partido entre pliegues.

### Lo que SI se recomputa, y por que no es una modificacion

La regla de pesos lee el error OOF **del propio modelo**, asi que bajo una particion
nueva el OOF es otro vector y los pesos son otros numeros. Eso no es cambiar la regla:
es aplicarla. Congelar los pesos mientras se cambia la particion seria la distorsion
real, porque emparejaria la estimacion de error de una particion con la estructura de
pliegues de otra.

## El resultado

| | particion canonica | segunda particion |
|---|---:|---:|
| V03 de su propia particion | 0.7772114525 | 0.7674405718 |
| candidato | 0.7854160387 | 0.7779658848 |
| delta | +0.0082045862 | +0.0105253130 |
| pliegues positivos | 5 de 5 | 5 de 5 |
| deltas por pliegue | 0.002243 | 0.001592 | 0.003378 | 0.023607 | 0.010356 | 0.026302 | 0.004643 | 0.002975 | 0.020868 | 0.003209 |
| rescate | 99 | 86 |
| damage | 74 | 74 |
| neto | +25 | +12 |

### Los pesos en la particion nueva

- IQR `0.020355`, peso medio `0.8601`, fraccion de peso retirada `0.139944`.
- Ejemplos con peso mayor que 1: `0` (la regla no sube ninguno, igual que en la canonica).

## La resolucion

```text
CONFIRMED

la segunda particion reproduce la direccion con al menos 4 de 5 pliegues. El efecto no es un artefacto de la particion canonica
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_02_segunda_particion.py`. Sin push.*