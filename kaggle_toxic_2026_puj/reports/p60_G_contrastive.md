# P60-G - Contraste supervisado por solapamiento de etiquetas

Generado por `experiments/p60_01_contrastive.py`. Fecha 2026-09-28T15:09:11+00:00.

## Motivacion e hipotesis

Es el unico eje de los doce de P60 que no modifica la funcion de perdida, ni los pesos, ni la
capa de decision, ni la representacion de entrada, sino la **geometria aprendida**. P59 cambio
que celdas importan, como se agrega la evidencia y que funcion se minimiza, y nunca la forma
del espacio en el que se aprende.

## Adaptacion legal

```text
dominio original  multi-etiqueta de texto con encoder grande
adaptacion        TF-IDF de V03 -> SVD 128 (filas de training) -> proyeccion lineal 32
que cambia        el encoder; aqui no hay transformer ni embedding preentrenado
que es invariante el positivo es un par definido por solapamiento de etiquetas
```

## Configuracion, fijada antes de mirar validation

```text
SVD 128 | proyeccion 32 | tau 0.1 | lambda 1.0
Adam lr 0.001 | weight decay 0.0001 | 200 epocas | full batch
positivos  instancias que comparten al menos una etiqueta, de las de TRAINING
rejilla     ninguna
```

## Control

Los dos brazos tienen **misma arquitectura, misma capacidad y mismos hiperparametros**. Lo
unico que difiere es el termino contrastivo, luego la diferencia entre ellos aisla el
contraste y nada mas.

| pliegue | positivos por instancia | pares positivos | loss control | loss contraste |
|---:|---:|---:|---|---|
| 1 | 239.41 | 20.22 por ciento | 0.9682 -> 0.4548 | 9.9517 -> 7.3721 |
| 2 | 239.27 | 20.21 por ciento | 0.9729 -> 0.4591 | 9.9570 -> 7.3862 |
| 3 | 240.0 | 20.26 por ciento | 0.9741 -> 0.4593 | 9.9935 -> 7.3801 |
| 4 | 239.1 | 20.18 por ciento | 0.9669 -> 0.4462 | 9.9135 -> 7.3513 |
| 5 | 238.53 | 20.15 por ciento | 0.9732 -> 0.4584 | 9.9706 -> 7.3711 |

## Resultados

```text
brazo 0 (control)                corrected_cv 0.7446595197
brazo 1 (control + contraste)   corrected_cv 0.4497522457
delta CONTRASTE PURO            -0.2949072740
delta vs V03                    -0.3274592068
V03 referencia                  0.7772114525
pliegues donde mejora           0/5
rescate 278 | dano 868 | neto -590 | cambian 1146
convergencia                    True
```

| etiqueta | F1 contraste | F1 control |
|---|---:|---:|
| HS | 0.7008 | 0.7218 |
| TR | 0.0000 | 0.7870 |
| AG | 0.6485 | 0.7252 |

## Por que el gate NO compara contra V03

Los umbrales de V03 son umbrales sobre la escala de probabilidad de V03. Un modelo entrenado
con otro optimizador y otra arquitectura tiene otra escala, luego aplicarle los umbrales de
V03 es un handicap, y ajustarlos seria un barrido de umbrales, familia que V8-09 cerro de raiz
al medir que V03 esta en el maximo de la curva de tasa de positivos. Por eso el gate de esta
pista es el efecto del contraste sobre su propio control, que si es un experimento
controlado, y la comparacion con V03 se reporta como contexto, no como veredicto.

## Gate

| criterio | valor | superado |
|---|---|---|
| efecto contraste >= +0,003 | -0.294907 | **False** |
| al menos 4 de 5 pliegues | 0/5 | **False** |
| gate completo | | **False** |

## Decision

```text
decision  CLOSED. El termino contrastivo no aporta sobre el control de misma arquitectura y capacidad
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_01_contrastive.py`. Sin push.*