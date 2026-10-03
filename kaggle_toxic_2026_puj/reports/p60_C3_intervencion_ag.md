# P60-C3 - Una intervencion conservadora sobre la region indefinida de AG

Generado por `experiments/p60_06_intervencion_ag.py`. Fecha 2026-09-28T15:48:27+00:00.

## Que midio C2, y por que esta intervencion apunta a AG y solo a AG

| etiqueta | AUC del control | mejor senal | AUC incremental | ganancia |
|---|---:|---|---:|---:|
| HS | 0,9917 | rango | 0,5547 | +0,0252 |
| TR | **1,0000** | centroide | 0,5338 | +0,0288 |
| AG | 0,9961 | estructural | 0,6627 | **+0,1129** |

Dos de esas tres lineas no son utilizables. Para **TR el control tiene AUC exactamente
1,0000**: la confianza out-of-fold ordena perfectamente todos los errores de TR, luego no
queda margen residual y una ganancia de 0,03 medida contra ese techo es ruido. Para **HS**
no existe region indefinida, luego el indicador estructural vale cero siempre. AG es la
unica etiqueta donde una segunda senal aporta informacion residual real.

## La region, medida

```text
filas en la region          1038 de 1483
tasa de error dentro        0.196532
tasa de error fuera         0.164045
enriquecimiento             1.1980 veces
AUC del control global      0.9961
AUC del control en la region 0.1814
```

## El factor, DERIVADO y no barrido

```text
peso_agregado = peso_de_clase * (1 - e_indefinida) / (1 - e_definida)
las dos tasas se miden con las filas de ENTRENAMIENTO del pliegue
```

| pliegue | filas indefinidas | error dentro | error fuera | factor derivado |
|---:|---:|---:|---:|---:|
| 1 | 829 | 0.0169 | 0.0196 | 1.002774 |
| 2 | 831 | 0.0181 | 0.0113 | 0.993140 |
| 3 | 831 | 0.0168 | 0.0253 | 1.008652 |
| 4 | 831 | 0.0156 | 0.0197 | 1.004100 |
| 5 | 830 | 0.0193 | 0.0140 | 0.994693 |

## Por que no es V14-00

`V14-00` puso peso **exactamente cero** sobre la region, lo que demostro equivalente a
descartar las filas, y dio -0,116581. Ahi el factor era 0 por construccion. Aqui es un
ratio de fiabilidad derivado, distinto de cero, luego es una intervencion distinta y no un
rescate de la anterior.

## Resultados

```text
brazo 0 (control)          corrected_cv 0.7772647319
  delta vs V03 +0.0000532794 | celdas distintas 3
brazo 1 (reescalado suave) corrected_cv 0.7772647319
delta INTERVENCION PURA   +0.0000000000
delta vs V03               +0.0000532794
pliegues que mejoran       0/5
rescate 2 | dano 1 | neto +1 | cambian 3
cambios SOLO en AG 0 celdas | en las otras dos 0
factor derivado entre 0.993140 y 1.008652 (media 1.000672)
convergencia True
```

| etiqueta | F1 intervencion | F1 control |
|---|---:|---:|
| HS | 0.7623 | 0.7623 |
| TR | 0.8153 | 0.8153 |
| AG | 0.7542 | 0.7542 |

## Gate

| criterio | valor | superado |
|---|---|---|
| efecto de la intervencion >= +0,003 | +0.000000 | **False** |
| al menos 4 de 5 pliegues | 0/5 | **False** |
| gate completo | | **False** |

Bootstrap NO ejecutado: el gate no se supero.

## Decision

```text
decision  CLOSED. La intervencion resulto ser un no-op exacto: el factor derivado quedo entre 0,993 y 1,009 y no cambio ni una celda. V14-00 ya habia cerrado la version dura de la misma idea
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_06_intervencion_ag.py`. Sin push.*