# P59-E - Regularizacion por grafo sobre el objetivo de V03

Generado por `experiments/p59_10_grafo_regularizacion.py`. Fecha 2026-09-28T14:31:40+00:00.

## Hipotesis

La estructura de vecindad en el espacio de V03 puede usarse para **regularizar el entrenamiento**, no para predecir localmente. Los documentos cercanos deberian recibir coeficientes similares, luego la decision de uno toma fuerza estadistica de sus vecinos sin que ninguno sea nunca etiquetado como evidencia al predecir.

## E1 gap check: NUEVA

| eje | filas del registro |
|---|---:|
| regularizacion por grafo | **0** |
| manifold regularization | **0** |
| termino cuadratico sobre un grafo de documentos | **0** |
| PageRank o paseo aleatorio | **0** |
| red neuronal sobre grafo | **0** |

El precedente mas cercano, `V4-F01-F04`, usa la vecindad para **propagar etiquetas**, que es predecir, y cerro con techo nulo por 0 solapamientos literales train-test. `V4-A01`, `V4-A02`, `E2-20-ESPECIALISTAS` y `V6-07` usan vecindad para predecir o para ponderar muestras. Ninguno regulariza un objetivo.

**Previo descriptivo que baja la priori y conviene no esconder**: `V4-F01-F04` midio que un vecino discrepante aporta solo +0,018 de error con IC95 [-0,013, +0,049], no significativo. Un grafo puede regularizar sin que los vecinos compartan error, asi que no cierra la pista.

## Configuracion, fijada antes de mirar validation

```text
grafo        coseno kNN con k = 10 sobre el TF-IDF de V03, filas de ENTRENAMIENTO
afinidad     simetrizada, cotos validos solo, reescalada a norma espectral unitaria
objetivo     0,5 w'w + (lambda/2) w'X'WX w + C sum_i omega_i perdida(s_i)
lambda       0.1, un solo valor, sin rejilla
clasificador LogisticRegression C=2.0 balanced, la escala de V03
umbrales     0,44 / 0,51 / 0,46 congelados
rol           el grafo SOLO regulariza; nunca predice, nunca vota, nunca recupera
```

El grafo se escapa por construccion: el mayor autovalor de `X'WX` se pone a 1, luego lambda = 0.1 significa una curvatura de suavizado de al menos diez veces menor que la del L2. Que ese sea el magnitud es un juicio declarado, no una magnitud derivada.

## Control: el brazo 0 es lo que hace legible al brazo 1

```text
brazo 0 (lambda = 0, mi optimizador)  corrected_cv 0.7772647318762402
           delta vs V03 +0.0000532794 | max |proba - canonica| 6.908e-03 | celdas distintas 3
```

Sin ese brazo, la diferencia entre el candidato y V03 seria una afirmacion sobre mi optimizador y no sobre el grafo.

## Resultados

```text
brazo 0 corrected_cv        0.7772647319
brazo 1 corrected_cv        0.7773636094
delta GRAFO PURO (1 - 0)    +0.0000988776
delta vs V03 (1 - V03)      +0.0001521569  incluye la tolerancia del optimizador
pliegues que mejoran 2/5
folds brazo 1 0.7866 0.7882 0.7910 0.7469 0.7738
folds brazo 0 0.7960 0.7955 0.7789 0.7363 0.7800
rescate 35 | damage 40 | neto -5 | cambian 75
```

| etiqueta | F1 brazo 1 | F1 brazo 0 | rescate | damage |
|---|---:|---:|---:|---:|
| HS | 0.7625 | 0.7623 | 12 | 14 |
| TR | 0.8158 | 0.8153 | 10 | 9 |
| AG | 0.7537 | 0.7542 | 13 | 17 |

## Convergencia

```text
brazo 0 todas convergen True | gradiente max 1.027e-05 | iteraciones 27-47
brazo 1 todas convergen True | gradiente max 2.577e-06 | iteraciones 29-45
warnings 0
```

## Gate

| criterio | valor | superado |
|---|---|---|
| delta del grafo >= +0,003 | +0.000099 | **False** |
| al menos 4 de 5 pliegues | 2/5 | **False** |
| gate completo | | **False** |

Bootstrap NO ejecutado: no se supero el gate.

## Decision

```text
decision  MARGINAL, y en la practica indistinguible del ruido. La regularizacion por grafo mueve la metrica en +0.0000988776 frente a un gate de +0,003, y el efecto puro del grafo es +0.0000988776 y la dispersion entre pliegues es 0.0441, luego el efecto es 446 veces mas pequeno que el ruido del propio instrumento. El recuento de celdas es rescate 35 contra damage 40, neto -5, con acierto 0.4667 en las 75 celdas que cambian, es decir una moneda al aire. NO se demuestra, NO se convierte en voto kNN para rescatarlo (parrafo 22), y NO se hace rejilla de lambda ni de k. Bootstrap NO ejecutado
```

> V03 remains the best verified leaderboard result.

*Seccion generada por `experiments/p59_10_grafo_regularizacion.py`. Sin push.*