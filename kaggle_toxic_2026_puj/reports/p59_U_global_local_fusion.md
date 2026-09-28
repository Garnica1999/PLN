# P59-U - Fusion continua del score global y el local

Generado por `experiments/p59_09_fusion_local.py`. Fecha 2026-09-28T14:27:27+00:00.

## Hipotesis

Aunque el score local aislado sea peor que V03, puede contener evidencia complementaria. Una fusion continua conserva la representacion global y aprovecha la senal local sin sustituirla.

## U1 gap check: base nueva, mecanismo ya explorado

Lo nuevo es la **base**, no el mecanismo. `ENS-01` a `ENS-04` combinan cuatro bases globales (voto igual, rejilla del simplex, stacking por etiqueta, mejor par o trio anidado); `V4-H01` y `S08` usan un meta de 36 rasgos; `FIN-08`, `FIN-09` y `FIN-10` enviaron candidatos de meta al leaderboard y los tres quedaron por debajo de V03. **Ninguna fusion previa uso un score calculado sobre una unidad de representacion distinta.**

Por que U no esta cerrada como T: una reparametrizacion monotona no cambia el ranking, luego T cae en el cierre de V8-09; la fusion si cambia el ranking siempre que los dos scores discrepen.

## Configuracion, fijada antes de mirar validation

```text
p_global   V03, el OOF canonico
p_local    B1 crudo, el max sobre ventanas
fusion     p = (1 - alpha) * p_global + alpha * p_local
alpha      UN valor, compartido por las tres etiquetas
regla      maximizar el macro-F1 a los umbrales congelados, sobre las predicciones
           inner-OOF del TRAINING FOLD EXTERIOR. Validation nunca se toca
rejilla    alpha de 0.00 a 1.00 en pasos de 0.05, 21 valores
nulo       alpha = 0 esta en la rejilla, luego la regla puede volver a V03
umbrales   0,44 / 0,51 / 0,46 congelados
```

## Control

```text
V03 0.7772114525165721 reproduce exacto: True
B1 crudo 0.7765868373082134
```

## Resultados

```text
corrected_cv U     0.7786914600
corrected_cv V03   0.7772114525
delta_vs_V03       +0.0014800075
pliegues que mejoran 2/5
folds U     0.7942 0.7969 0.7750 0.7386 0.7897
folds V03   0.7960 0.7940 0.7789 0.7386 0.7791
alpha por pliegue  [0.3, 0.45, 0.55, 0.0, 0.75]
rescate 17 | damage 11 | neto +6 | cambian 28
```

| etiqueta | F1 U | F1 V03 | rescate | damage |
|---|---:|---:|---:|---:|
| HS | 0.7666 | 0.7617 | 9 | 3 |
| TR | 0.8157 | 0.8157 | 2 | 2 |
| AG | 0.7538 | 0.7542 | 6 | 6 |

## Complementariedad por estrato

| estrato | n tweets | rescate | damage | neto |
|---|---:|---:|---:|---:|
| multi_ventana | 122 | 3 | 0 | +3 |
| una_sola_ventana | 1361 | 14 | 11 | +3 |

## Diagnostico oracle, NO es candidato

El alpha que habria maximizado el score **externo** es 0.65, con corrected_cv 0.7778556033 y delta +0.0006441507. Mide si hay senal complementaria en principio, que es la pregunta que la regla anidada no puede responder. Nunca se puntua como candidato.

## Gate

| criterio | valor | superado |
|---|---|---|
| delta >= +0,003 | +0.001480 | **False** |
| al menos 4 de 5 pliegues | 2/5 | **False** |
| gate completo | | **False** |

Bootstrap NO ejecutado: no se supero el gate.

## Decision

```text
decision  MARGINAL. La fusion mejora la metrica pero no alcanza el gate delta >= +0,003. NO se demuestra. Bootstrap NO ejecutado
```

> V03 remains the best verified leaderboard result.

*Seccion generada por `experiments/p59_09_fusion_local.py`. Sin push.*