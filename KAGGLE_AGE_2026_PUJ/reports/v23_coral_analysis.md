# V23 · CORAL / cumulative-link con pesos compartidos

## A. El control V20

- Reproduccion: **0.6097782** frente a 0.6097782 | coincide: **True**.
- Leaderboard del champion: **0.62689**. No se ha modificado V20.
- Violaciones de monotonia de sus cinco CDF: **18,927** de 169,336 (11.18%).

## B. La formulacion

```
P(Y <= k | x) = sigmoid(theta_k - w' x),   k = 0..4
theta_k = gamma + suma_{j<k} softplus(delta_j)
```

Un unico vector de pesos y cinco umbrales **ordenados por construccion**. Es distinto de V20 (cinco binarios independientes) y distinto del modelo compartido de D-057 (multiclass descompuesto por suma, sin umbrales ordinales).

**Pre-registrado, sin tocar AGE**: `lambda_w = 1.0`, `lambda_umbral = 0.0`. Los fija el TEST B de `scripts/v23_00_tests_coral.py`, que demuestra que a la escala de AGE no hay separacion.

## C. Resultado

| Modelo | macro-F1 | Delta | Folds + | monotonia | rescate | dano | neto | cambios |
|---|---|---|---|---|---|---|---|---|
| V20 (control A) | 0.6097782 | +0.000000 | 3/5 | 18,927 | 0 | 0 | +0 | 0 |
| cinco binarios lineales (control B) | 0.4690055 | -0.140773 | 0/5 | no medido | 5,468 | 28,811 | -23,343 | 49,212 |
| CORAL pesos compartidos | 0.4837017 | -0.126077 | 0/5 | 0 | 12,234 | 34,853 | -22,619 | 67,763 |

### Scores por fold

| Modelo | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | media |
|---|---|---|---|---|---|---|
| V20 (control A) | 0.6132028 | 0.6120810 | 0.6104461 | 0.6088426 | 0.6043510 | 0.609785 |
| cinco binarios lineales (control B) | 0.4737085 | 0.4712285 | 0.4705763 | 0.4710854 | 0.4584631 | 0.469012 |
| CORAL pesos compartidos | 0.4842829 | 0.4854819 | 0.4887768 | 0.4850558 | 0.4746802 | 0.483656 |

### Delta por fold

| Modelo | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | positivos |
|---|---|---|---|---|---|---|
| V20 (control A) | +0.003425 | +0.002303 | +0.000668 | -0.000936 | -0.005427 | 3/5 |
| cinco binarios lineales (control B) | -0.136070 | -0.138550 | -0.139202 | -0.138693 | -0.151315 | 0/5 |
| CORAL pesos compartidos | -0.125495 | -0.124296 | -0.121001 | -0.124722 | -0.135098 | 0/5 |

## D. F1 por clase

| Modelo | 13-17 | 18-24 | 25-34 | 35-49 | 50-64 | 65-xx |
|---|---|---|---|---|---|---|
| V20 (control A) | 0.6818 | 0.5170 | 0.4965 | 0.5687 | 0.6637 | 0.7310 |
| cinco binarios lineales (control B) | 0.6617 | 0.4961 | 0.4738 | 0.5397 | 0.6428 | 0.0000 |
| CORAL pesos compartidos | 0.6267 | 0.4342 | 0.3288 | 0.3641 | 0.4923 | 0.6560 |

## E. Monotonicidad: el problema que dice resolver

- **V20**: 18,927 filas con la serie de CDF decreciente.
- **CORAL**: **0** filas, y no por ajuste sino **por construccion**, porque `softplus(delta_j) > 0` obliga a `theta` a crecer y con el una sola `w` las cinco CDF heredan el orden.

**La monotonia queda resuelta de verdad. La pregunta es si eso cambia el macro-F1.**

## F. Intersecion de errores

| Par | V20 | CORAL |
|---|---|---|
| 18-24->25-34 | 4,022 | 5,284 |
| 25-34->18-24 | 5,926 | 9,010 |
| 25-34->35-49 | 3,349 | 5,973 |
| 35-49->25-34 | 3,655 | 6,118 |
| 35-49->50-64 | 3,742 | 7,008 |
| 50-64->35-49 | 2,466 | 5,745 |
| 13-17->50-64 | 842 | 547 |
| 50-64->13-17 | 459 | 78 |
| 18-24->65-xx | 1,039 | 250 |
| 65-xx->18-24 | 794 | 676 |
| 13-17->65-xx | 544 | 75 |
| 65-xx->13-17 | 337 | 107 |

Masa en la diagonal (aciertos): V20 **103,331**, CORAL **80,712** (delta -22,619). Celdas de la matriz de confusion que cambian: **36/36**, con distancia L1 **57,650**.

## G. Decision

- Delta de CORAL frente a V20: **-0.126077** (gate: >= 0.001) -> **False**.
- Folds positivos: **0/5** (gate: >= 4) -> **False**.
- Bootstrap: **no se calcula**, porque el gate de delta no se cumple y el brief lo reserva al caso en que CORAL lo supera.

**CORAL = CLOSED**

## H. Atribucion

- El control B (cinco binarios **independientes** y lineales) da **-0.140773** frente a V20.
- CORAL (peso **compartido**, lineal) da **-0.126077** frente a V20.
- La diferencia entre CORAL y el control B es el efecto **puro** de compartir pesos. La diferencia entre el control B y V20 es el efecto de la no linealidad.
