# Tercera particion de A_cost_sensitive_downweight

Generado por `experiments/p69_04_tercera_particion.py`. Fecha 2026-09-29T18:03:23+00:00.

## El protocolo estaba escrito antes

La semilla, el metodo de pliegues, la regla de decision y el diseno de la permutacion se
fijaron en `docs/p69_especificacion_congelada.md`, commit `915991b`, **antes** de que
existiera ningun resultado de esta particion.

Ese orden es el punto entero. Una particion ejecutada **despues** de ver el resultado de
otra no es evidencia independiente, por mucho que la semilla parezca arbitraria. Lo unico
que la hace confirmatoria en vez de adaptativa es que la especificacion vino antes y se
puede comprobar.

```text
semilla de la tercera particion   20261001
escrito y commiteado en           docs/p69_especificacion_congelada.md, 915991b
```

## Las tres particiones

| particion | V03 | candidato | delta | folds+ | rescate | damage | neto |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonica | 0.7772114525 | 0.7854160387 | +0.0082045862 | 5/5 | 99 | 74 | +25 |
| segunda | 0.7674405718 | 0.7779658848 | +0.0105253130 | 5/5 | 86 | 74 | +12 |
| tercera | 0.7739813065 | 0.7787759768 | +0.0047946703 | 4/5 | 78 | 69 | +9 |

```text
media     +0.0078415232
mediana   +0.0082045862
minimo    +0.0047946703
maximo    +0.0105253130
positivas 3 de 3
```

### Deltas por pliegue

```text
canonica   +0.002243 +0.001592 +0.003378 +0.023607 +0.010356
```
```text
segunda    +0.026302 +0.004643 +0.002975 +0.020868 +0.003209
```
```text
tercera    +0.001880 -0.006834 +0.012343 +0.012575 +0.004440
```

## La tercera particion, en detalle

- V03 de su propia particion: `0.7739813065`.
- Candidato: `0.7787759768`. Delta `+0.0047946703`.
- Pliegues positivos: `4` de 5.
- Rescate `78`, dano `69`, neto `+9`.

| etiqueta | V03 ok / cand ko | V03 ko / cand ok | ambos ok | ambos ko | rescate | damage | neto |
|---|---:|---:|---:|---:|---:|---:|---:|
| HS | 27 | 43 | 1126 | 287 | 43 | 27 | +16 |
| TR | 14 | 18 | 1298 | 153 | 18 | 14 | +4 |
| AG | 28 | 17 | 1181 | 257 | 17 | 28 | -11 |

## La regla de decision, la pre-registrada

```text
CASO A
PROMISING_UNCONFIRMED

tres de tres particiones positivas y la tercera con 4 de 5 pliegues. Se pasa al test de permutacion
```

Procede el test de permutacion: **True**.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_04_tercera_particion.py`. Sin push.*