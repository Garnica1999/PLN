# P64 - Los bloques de evidencia por estrato, y si el margen ya lo sabia

Generado por `experiments/p64_05_especializacion_larga.py`. Fecha 2026-09-29T07:26:45+00:00.

## Por que esta pregunta es separada

P64-03 y P64-04 fallaron en el promedio sobre 1483 filas, y ese promedio lo dominan
las cortas: CORTA y MUY_LARGA tienen 167 filas cada una. Un numero global dice casi
nada de ningun extremo.

P63 midio donde esta el margen residual, y la respuesta fue especifica:

```text
CORTA      margen hasta el techo de umbral  +0,0000
MEDIA      +0,0093
LARGA      +0,0217
MUY_LARGA  +0,0890
```

En los tweets muy largos hay casi nueve centesimas de macro F1 sobre la mesa, y estan
de una forma que ningun umbral alcanza. Si el bloque de evidencia lleva la estructura que
la bolsa de n-gramas aplana, ahi es donde tiene que pagar, y un promedio global es
exactamente la estadistica disenada para esconderlo.

## La metrica primaria NO cambia

El corrected_cv **global** sigue siendo la metrica primaria, y la decision **no** se
toma sobre la mejor rebanada. Adoptar algo que solo ayuda en un estrato es como se entra
en una especializacion que no transfiere. Las rebanadas se informan para que el lector
vea la forma.

## Los dos mecanismos, que son distintos

| bloque | columnas | mecanismo |
|---|---:|---|
| G1 concentracion | 12 | un segmento domina la decision |
| G2 conflicto | 12 | la decision descansa en un reparto contradictorio |

G1 **no** es el max-pooling de P59-07: aqui el maximo es un RASGO que un modelo lineal
pesa, no un agregador que sustituye la prediccion, luego el sesgo al alza que hundio a
P59 no aplica. Los dos bloques se **reutilizan** de P64-03 y P64-04, ya calculados con
los coeficientes de V03 del pliegue, luego este script no introduce ningun modelo nuevo.

## Resultados

| brazo | corrected_cv | delta global | delta C | delta M | delta L | delta M |
|---|---:|---:|---:|---:|---:|---:|
| V03_G1 | 0.7608159383 | -0.016396 | -0.0082 | -0.0160 | -0.0148 | -0.0049 |
| V03_G2 | 0.7687139414 | -0.008498 | -0.0106 | -0.0022 | -0.0138 | -0.0063 |
| V03_ambos | 0.7587037656 | -0.018508 | -0.0101 | -0.0164 | -0.0223 | -0.0019 |
| V03_random | 0.7691613890 | -0.008050 | -0.0103 | -0.0096 | +0.0018 | -0.0312 |
| V03_margen | 0.7661312520 | -0.011080 | -0.0101 | -0.0140 | -0.0028 | -0.0010 |
| **V03** | 0.7772114525 | +0.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

### F1 absoluto por estrato

| bin | n | V03 | G1 | G2 | juntos | margen |
|---|---:|---:|---:|---:|---:|---:|
| CORTA | 167 | 0.9480 | 0.9398 | 0.9373 | 0.9378 | 0.9379 |
| MEDIA | 602 | 0.7964 | 0.7804 | 0.7942 | 0.7800 | 0.7825 |
| LARGA | 547 | 0.6834 | 0.6685 | 0.6695 | 0.6610 | 0.6806 |
| MUY_LARGA | 167 | 0.5606 | 0.5556 | 0.5543 | 0.5587 | 0.5595 |

## La pregunta, en el mismo estrato

Un bloque que ayuda a los tweets largos tiene que superar no a V03 sino a la
**confianza que V03 ya expone en esos mismos tweets**, porque si el margen ya lleva la
informacion, cualquier ganancia por rebanada es contabilidad.

```text
CORTA      G1 -0.0082 (+0.0019 contra el margen) | G2 -0.0106 (-0.0006) | juntos -0.0101 (-0.0001)
MEDIA      G1 -0.0160 (-0.0020 contra el margen) | G2 -0.0022 (+0.0118) | juntos -0.0164 (-0.0024)
LARGA      G1 -0.0148 (-0.0120 contra el margen) | G2 -0.0138 (-0.0110) | juntos -0.0223 (-0.0195)
MUY_LARGA  G1 -0.0049 (-0.0039 contra el margen) | G2 -0.0063 (-0.0053) | juntos -0.0019 (-0.0008)
```

## Veredicto

```text
ni V03_ambos ni los demas ganan en la region larga por encima del margen: el mejor es V03_ambos con -0.001881 frente a -0.001046 del margen en el mismo estrato. La hipotesis de la fase queda confirmada como DESCRIPCION y cerrada como REPRESENTACION, y la region larga no es una excepcion que la resuelva: ahi la estructura interna es mas repartida y mas contradictoria, y V03 ya la lleva dentro de su confianza

LECTURA: NADA_EN_EL_ESTRATO
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p64_05_especializacion_larga.py`. Sin push.*