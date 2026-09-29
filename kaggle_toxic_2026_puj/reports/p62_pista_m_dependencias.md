# P62 - Pista M, triples lexicos de dependencia

Generado por `experiments/p62_04_pista_m.py`. Fecha 2026-09-29T05:21:32+00:00.

## En que se diferencia de la pista L

Un token `palabra__POS` dice que una palabra es un sustantivo. Un token
`cabeza__relacion__dependiente` dice que EN ESTE TWEET ese sustantivo es el objeto
directo de ESE verbo. Lo segundo es lo que distingue un insulto dirigido a alguien
de un adjetivo suelto, y ningun unigrama ni bigrama del texto crudo lo dice.

Las relaciones se declaran antes de ejecutar, y no se eligen por como discriminan.

```text
relaciones  obj, amod, obl, obl:arg, nsubj, iobj, nmod, acomp, attr, advmod, compound, appos, parataxis, ccomp, xcomp, acl:relcl
```

## Los tres brazos

| brazo | corrected_cv | delta vs V03 |
|---|---:|---:|
| control, V03 sin canal | 0.7772114525 | +0.0000000000 |
| pista M, + triple lexico | 0.7659752040 | -0.0112362485 |
| control aleatorio | 0.7767681701 | -0.0004432824 |

## La lectura en tres partes

```text
(1) coste de anadir el bloque     +0.0004432824
(2) valor de la estructura        -0.0107929661
(3) saldo                         -0.0112362485
```

## Control de vocabulario

| brazo | rasgos por pliegue |
|---|---|
| control | [20441, 21044, 20802, 21038, 20982] |
| pista_M | [20784, 21405, 21148, 21375, 21335] |
| control_aleatorio | [20633, 21202, 20988, 21262, 21167] |

Relaciones que sobrevivieron a min_df=2, con el numero de rasgos:

```text
  obj             486
  advmod          331
  nsubj           299
  amod            179
  iobj            177
  nmod            103
  obl              53
  ccomp            33
  xcomp            28
  appos            26
  compound         22
  _appos            3
```

## Gate

```text
delta -0.0112362485 | pliegues 0/5 | superado False
bootstrap: NO EJECUTADO
```

## Decision

```text
CLOSED. El triple lexico de dependencia no aporta sobre V03
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_04_pista_m.py`. Sin push.*