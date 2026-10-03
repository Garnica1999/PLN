# P60-J - PolyLoss, forma polinomica de la perdida

Generado por `experiments/p60_09_polyloss.py`. Fecha 2026-09-28T17:57:52+00:00.

## Motivacion e hipotesis

El objetivo de V03 es la perdida logistica, y el proyecto ya muestreo las dos formas
obvias de cambiarla: focal con una unica potencia, cerrada en `V7-01` con -0,0024, y
GCE L_q, que es justamente la familia (1 - p_y^q) / q, probada en `V12-02` con cuatro
valores de q y un mejor de +0,001616 que no paso el gate.

Lo que PolyLoss anade a esa familia **no es otra potencia**. Es un polinomio en el error
con **coeficientes independientes** en el termino lineal y en el cuadratico. Y la razon de
que sean dos y no uno es precisa: una sola potencia mueve el reparto del gradiente entre
regiones de error a lo largo de una curva de un parametro, mientras que dos coeficientes
pueden pesar de forma diferente la region lineal del error y la cuadratica.

## Configuracion, fijada antes de mirar validation

```text
L = CE + 1.0 (1 - p_y) + 1.0 (1 - p_y^2) / 2
los tres terminos canonicos, con los dos primeros ordenes no nulos, a peso unidad
rejilla  ninguna sobre eps_1 ni sobre eps_2
```

El valor no informativo es el que **iguala** los dos ordenes, y por eso es el que se usa:
una potencia ya la barre `V12-02`, luego el eje de esta pista es el peso relativo entre
los dos ordenes, y cualquier otro valor seria una eleccion hecha para mover el numero.

## Derivada y su verificacion

Con s = y f, q = p_y y dq/df = y q (1 - q):

```text
dL/df = (p - y) - y q (1 - q) (1 + eps_2 q)
```

| regimen | escala de pesos | error relativo w | error relativo b |
|---|---:|---:|---:|
| moderado | 0.3 | 4.48e-07 | 1.98e-06 |
| extremo | 2.0 | 1.75e-06 | 6.86e-08 |

## Control

```text
brazo 0 (logistica)  corrected_cv 0.7772647318762402
  delta vs V03 +0.0000532794 | celdas distintas 3
```

## Resultados

```text
brazo 0 corrected_cv 0.7772647319
brazo 1 corrected_cv 0.7755546945
delta FORMA DE PERDIDA PURO -0.0017100374
delta vs V03                -0.0016567581
pliegues que mejoran        2/5
rescate 25 | dano 23 | neto +2 | cambian 48
convergencia True
V03 referencia              0.7772114525
```

| etiqueta | F1 PolyLoss | F1 control |
|---|---:|---:|
| HS | 0.7613 | 0.7623 |
| TR | 0.8149 | 0.8153 |
| AG | 0.7504 | 0.7542 |

## Gate

| criterio | valor | superado |
|---|---|---|
| efecto de la forma >= +0,003 | -0.001710 | **False** |
| al menos 4 de 5 pliegues | 2/5 | **False** |
| gate completo | | **False** |

Bootstrap NO ejecutado: el gate no se supero.

## Decision

```text
decision  CLOSED. La forma polinomica de la perdida no mejora sobre la logistica con el mismo C, y V7-01 y V12-02 ya habian cerrado la rama de la perdida. El confound de escala declarado limita la lectura del lado positivo
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_09_polyloss.py`. Sin push.*