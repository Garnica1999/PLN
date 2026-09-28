# P59-C - Confident learning y estimacion formal de ruido de etiqueta

Generado por `experiments/p59_03_confident_learning.py`. Fecha 2026-09-28T04:19:15+00:00.

**Diagnostico. 0 modelos, 0 submissions, 0 Kaggle, 0 holdout, 0 test. Ninguna etiqueta observada se cambia (parrafo 23).** No se registro fila en `experiments.csv`.

## 1. Auditoria de hueco

> la tabla conjunta segura usa la distribucion global de las etiquetas observadas dentro de cada conjunto seguro, que es una propiedad que V4-B01 y V6-07 no computan

V30-00 cerro la familia `label noise` con cuatro experimentos negativos, pero todos eran **identificadores de dificultad**. Un identificador de dificultad pregunta que probabilidad tiene una celda de fallar. Un estimador de consistencia latente pregunta si la etiqueta observada concuerda con la evidencia. No son el mismo objeto, y esa es la razon por la que este track sigue abierto tras el gate de novedad.

## 2. Metodo

```text
implementacion         confident joint escrito desde cero con numpy; sin cleanlab
umbral_seguro          media de p sobre las instancias predichas de cada clase
conjunto_seguro        predice c y p_c >= umbral de c
tabla_conjunta         Q[c][j] = miembros del conjunto seguro de c con etiqueta j
pureza_calibrada       Q[c][c] / suma_j Q[c][j]
perdidos               |y = c| - Q[c][c], la clase observada que su conjunto no cubre
```

Implementado con numpy, sin `cleanlab`, como pide el parrafo 21.

## 3. Tablas conjuntas por etiqueta

| etiqueta | Q[0][0] | Q[0][1] | Q[1][0] | Q[1][1] | seguro 0 | seguro 1 | pureza 0 | pureza 1 | perdidos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HS | 420 | 33 | 29 | 269 | 453 | 298 | 0.9272 | 0.9027 | 415 / 379 |
| TR | 630 | 14 | 19 | 226 | 644 | 245 | 0.9783 | 0.9224 | 408 / 219 |
| AG | 470 | 44 | 28 | 248 | 514 | 276 | 0.9144 | 0.8986 | 446 / 319 |

`perdidos` es la magnitud que mas importa: cuantas celdas cuya etiqueta observada es `c` **no aparecen en el conjunto seguro de `c`**. Ese es el conteo duro de celdas potencialmente mal etiquetadas, y no depende de ningun umbral elegido a mano.

## 4. Control: el score clasico

| etiqueta | correlacion de 1 - p_true con el error | referencia de V4-B01 |
|---|---:|---:|
| HS | 0.7952 | -0.795 |
| TR | 0.8098 | -0.81 |
| AG | 0.8006 | -0.801 |

> el signo es el inverso porque el score es 1 - p_true; la magnitud es la que tiene que reproducirse

El control sirve para dos cosas: demostrar que la instrumentacion es correcta, y marcar la vara que el estimador nuevo tiene que superar. El parrafo 62 es explicito: sospechar ruido que es solo error de alta confianza no cuenta.

## 5. Resultado

| objetivo | n | AUC | IC95 |
|---|---:|---:|---|
| s_conf: error de alta confianza | 36 | **0.9836** | [0.9785, 0.9884] |
| s_joint: error de alta confianza | 36 | **0.5620** | [0.4725, 0.6511] |
| s_resid: error de alta confianza | 36 | **0.9827** | [0.9772, 0.9879] |
| s_conf: no rescatable entre los 757 errores | 42 | **0.9741** | [0.9653, 0.9828] |
| s_joint: no rescatable entre los 757 errores | 42 | **0.5824** | [0.4945, 0.6643] |
| s_resid: no rescatable entre los 757 errores | 42 | **0.9730** | [0.9637, 0.9822] |

```text
control  1 - p_true   AUC 0.974143
nuevo     s_resid     AUC 0.973019
supera al control por mas de 0,01 de AUC   False
```

## 6. Decision

```text
decision  MARGINAL. El estimador alcanza el umbral de 0,60 pero no supera al control de dificultad, luego no aporta mas que medir que tan dificil es un ejemplo. Ponderar no tendria justificacion mechanica y no se hace
```

## 7. Formulacion

> V03 remains the best verified leaderboard result.

*Seccion generada por `experiments/p59_03_confident_learning.py`. Sin push.*
