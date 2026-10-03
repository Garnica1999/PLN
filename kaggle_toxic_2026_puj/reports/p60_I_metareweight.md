# P60-I - Meta-reweighting por alineacion de gradiente

Generado por `experiments/p60_10_metareweight.py`. Fecha 2026-09-28T17:58:49+00:00.

## Motivacion, y por que no es uno de los ponderados ya cerrados

El proyecto ha cerrado tres familias de peso por ejemplo, y las tres lo derivan de una
**regla heuristica fija** computada antes de entrenar: `V4-B03` y `V4-B05` por
dificultad, `V6-07` por sospecha, `P59-06` por tamano del residuo. Lo que Ren et al.
proponen es de otra naturaleza: el peso sale de preguntar que gradientes estan
**alineados con un objetivo meta** evaluado sobre un conjunto retenido, luego es
endogeno a la optimizacion y no prescrito por una regla.

## El peso tiene forma cerrada, y para un modelo lineal eso importa

```text
eps_i = <g_i, G> / <G, G>     G = gradiente del objetivo meta
```

Ren et al. evaluan esto en la inicializacion, luego con w = 0 el gradiente de un
ejemplo esta disponible en forma cerrada, sin retropropagacion, porque el gradiente de
la regularizacion se anula en el origen y el de la perdida queda siendo
`C * omega_i * (sigma(0) - y_i) * x_i`.

El producto interno `<g_i, G>` necesitaria normalmente una matriz de Gram de tamano
`d` por `d`, con `d = 27.431`. No hace falta: escribiendo `G = C X' (omega dlogit)` sobre
las filas meta, se obtiene `<g_i, G> = C omega_i dlogit_i (X G)_i`, y `X G` es un unico
producto disperso de forma `n` por `d`. Eso es lo que hace la pista viable con 1483
filas y 27.431 rasgos, y es la razon por la que no hace falta meta-aprendizaje.

## El anidamiento, que es la parte facil de hacer mal

```text
outer_train  ->  inner_train + inner_meta, partidos POR GRUPO
             ->  los pesos salen solo del gradiente de inner_meta
             ->  se ajusta sobre el outer_train completo
outer_validation  ->  se puntua una vez, sin ajustar nada
```

La particion interna se hace **por grupo near-duplicate**, no por fila, y el script
comprueba de forma explicita que ningun grupo queda partido entre las dos partes
internas. La validacion exterior no participa en ningun paso.

## La limitacion honesta, declarada antes de ejecutar

Ren et al. necesitan un conjunto retenido **limpio**. No hay evidencia de que ningun
subconjunto de las etiquetas de TOXIC lo sea, luego el objetivo meta se evalua aqui
sobre un conjunto tan ruidoso como el resto. Esa es la debilidad real de esta pista en
este corpus y no es un detalle de configuracion.

## Configuracion, fijada antes de mirar validation

```text
momento        en la inicializacion, con w = 0, como en el trabajo original
fraccion meta  0.25 de los grupos internos, un solo valor
forma           eps_i = <g_i, G> / <G, G>, forma cerrada
normalizacion   la masa del ponderado se reescala a la del control, como en la pista L
rejilla         ninguna
```

## Control

```text
brazo 0 (pesos unitarios)  corrected_cv 0.7772647318762402
  delta vs V03 +0.0000532794 | celdas distintas 3
```

## Resultados

```text
brazo 0 corrected_cv 0.7772647319
brazo 1 corrected_cv 0.4761602463
delta META PURO        -0.3011044856
delta vs V03           -0.3010512063
pliegues que mejoran   0/5
rescate 318 | dano 890 | neto -572 | cambian 1208
convergencia True
V03 referencia         0.7772114525
```

| etiqueta | F1 meta | F1 control |
|---|---:|---:|
| HS | 0.3697 | 0.7623 |
| TR | 0.6208 | 0.8153 |
| AG | 0.4380 | 0.7542 |

## Diagnostico del mecanismo

| pliegue | etiqueta | eps mediana | eps p10 | eps p90 | razon p90/abs(p10) | fraccion negativa |
|---:|---|---:|---:|---:|---:|---:|
| 1 | HS | 2.259e-03 | -2.401e-03 | 5.440e-03 | 2.2657 | 24.11 por ciento |
| 1 | TR | 2.040e-03 | -7.020e-04 | 7.092e-03 | 10.1029 | 16.19 por ciento |
| 1 | AG | 1.972e-03 | -2.013e-03 | 5.411e-03 | 2.6879 | 20.99 por ciento |
| 2 | HS | 2.053e-03 | -1.648e-03 | 5.001e-03 | 3.0343 | 22.77 por ciento |
| 2 | TR | 2.262e-03 | -6.157e-04 | 5.429e-03 | 8.8177 | 14.33 por ciento |
| 2 | AG | 2.146e-03 | -1.821e-03 | 4.816e-03 | 2.6452 | 20.24 por ciento |
| 3 | HS | 1.702e-03 | -1.583e-03 | 5.331e-03 | 3.3682 | 23.59 por ciento |
| 3 | TR | 1.601e-03 | -7.887e-04 | 7.728e-03 | 9.7981 | 19.29 por ciento |
| 3 | AG | 1.710e-03 | -1.447e-03 | 6.096e-03 | 4.2133 | 20.72 por ciento |
| 4 | HS | 1.744e-03 | -1.538e-03 | 5.576e-03 | 3.6256 | 24.35 por ciento |
| 4 | TR | 1.504e-03 | -9.095e-04 | 8.104e-03 | 8.9101 | 21.15 por ciento |
| 4 | AG | 1.695e-03 | -1.723e-03 | 6.232e-03 | 3.6168 | 23.50 por ciento |
| 5 | HS | 2.325e-03 | -2.291e-03 | 5.227e-03 | 2.2817 | 25.80 por ciento |
| 5 | TR | 2.074e-03 | -5.186e-04 | 6.706e-03 | 12.9307 | 14.08 por ciento |
| 5 | AG | 2.336e-03 | -2.501e-03 | 4.874e-03 | 1.9486 | 20.40 por ciento |

Fraccion negativa media **20.77 por ciento**, razon intercuartil media
**5.3498**. Una fraccion negativa alta significa que el criterio premia y castiga
de verdad; una razon cercana a uno significa que apenas distingue entre ejemplos.

## Gate

| criterio | valor | superado |
|---|---|---|
| efecto del ponderado >= +0,003 | -0.301104 | **False** |
| al menos 4 de 5 pliegues | 0/5 | **False** |
| gate completo | | **False** |

Bootstrap NO ejecutado: el gate no se supero.

## Decision

```text
decision  CLOSED. El ponderado derivado de la alineacion de gradiente no mejora sobre el control, y los pesos que produce estan casi degenerados
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_10_metareweight.py`. Sin push.*