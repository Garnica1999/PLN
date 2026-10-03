# A_cost_sensitive_downweight - Evidencia final

Generado por `experiments/p69_07_cierre_final.py`. Fecha 2026-09-29T19:49:10+00:00.

## La conclusion, congelada

> A_cost_sensitive_downweight presenta una mejora reproducible, consistente entre tres
> particiones y altamente improbable bajo la permutacion de la asignacion de pesos.
> Sin embargo, el bootstrap por grupos continua incluyendo cero, por lo que la magnitud
> del efecto no queda resuelta con suficiente precision para clasificarlo como ADOPTED.

La formulacion precisa es: **evidencia fuerte de direccion y mecanismo, pero
incertidumbre residual sobre la magnitud poblacional**.

### Lo que NO se afirma

```text
no 'demostrado definitivamente'
no 'es ruido'
no 'V03 ha sido superado definitivamente'
```

## Los seis estados, que no se sustituyen entre si

| estado | valor | nota |
|---|---|---|
| `ORIGINAL` | +0.0082045862 | 2026-09-25, nunca enviado, nunca auditado |
| `REPRODUCED` | +0.0082045862 | R3, diferencia 3,2e-11 |
| `SECOND_PARTITION` | +0.0105253130 | 5 de 5 pliegues |
| `THIRD_PARTITION_CORRECTED` | +0.0048825807 | 4 de 5, tras corregir el defecto |
| `PERMUTATION` | p = 0.000999 | 0 de 1000 superan al observado |
| `FINAL_STATUS` | PROMISING_UNCONFIRMED | 5 de 6 componentes |

## El candidato, congelado

```text
margen_favor = (p_V03 - umbral) * (2*y - 1)
violacion_l  = clip(-margen_favor, 0, None)      por etiqueta
error_score  = violacion_l.mean(axis=1)          por ejemplo, media de 3
z            = (error_score - mediana) / IQR
peso         = clip(1 - 0.5*z, 0.5, 1.5)
```

Cantidades de la particion canonica, que no se tocan:

```text
441 ejemplos con peso < 1
1042 ejemplos con peso = 1
0 ejemplos con peso > 1
13,66 por ciento del peso total retirado
```

La descripcion correcta del mecanismo es **cost-sensitive margin-violation
downweighting**. P58 lo etiquetaba como *'meta con downweight coste-sensible'*, y el
codigo no construye ningun meta-modelo: entrena `LogisticRegression` directamente sobre
la representacion de V03. El informe historico no se modifica.

## Las tres particiones

| particion | V03 | candidato | delta | folds+ | rescate | damage | neto |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonica | 0.7772114525 | 0.7854160387 | +0.0082045862 | 5/5 | 99 | 74 | +25 |
| segunda | 0.7674405718 | 0.7779658848 | +0.0105253130 | 5/5 | 86 | 74 | +12 |
| tercera | 0.7739813065 | 0.7788638872 | +0.0048825807 | 4/5 | 80 | 70 | +10 |

```text
media     +0.0078708266
mediana   +0.0082045862
minimo    +0.0048825807
maximo    +0.0105253130
positivas 3 de 3
```

### Deltas por pliegue

```text
canonica   +0.002243 +0.001592 +0.003378 +0.023607 +0.010356
segunda    +0.026302 +0.004643 +0.002975 +0.020868 +0.003209
tercera    +0.001880 -0.006834 +0.012343 +0.012575 +0.004440
```

## Rescue, damage y neto por etiqueta

| particion | etiqueta | rescate | damage | neto |
|---|---|---:|---:|---:|
| canonica | HS | 42 | 23 | +19 |
| canonica | TR | 14 | 19 | -5 |
| canonica | AG | 43 | 32 | +11 |
| segunda | HS | 37 | 30 | +7 |
| segunda | TR | 21 | 25 | -4 |
| segunda | AG | 28 | 19 | +9 |
| tercera | HS | 45 | 27 | +18 |
| tercera | TR | 18 | 15 | +3 |
| tercera | AG | 17 | 28 | -11 |

### La lectura mecanica

```text
HS   positivo en las TRES particiones   +19, +7, +18
TR   negativo en dos de tres            -5, -4, +3
AG   positivo en dos de tres            +11, +9, -11
```

La mejora **no depende de una sola etiqueta**: HS aporta de forma consistente en las
tres, y el neto global es positivo en las tres. TR es el mas debil, y AG es el mas
inestable, con un cambio de signo entre la segunda y la tercera. Se describe como
estabilidad mecanica, y no se optimiza sobre estas etiquetas despues.

## El test de permutacion

| estadistico | valor |
|---|---:|
| delta real | +0.0082045862 |
| media nula | -0.0013209910 |
| desviacion nula | 0.0020508746 |
| maximo nulo | +0.0045971670 |
| percentil 99 nulo | +0.0030856337 |
| permutaciones >= observado | 0 de 1000 |
| p empirico | 0.000999 |

## LIMITACION DE PRECISION ESTADISTICA

Este apartado existe para describir el limite, no para resolverlo.

### Lo que la evidencia sostiene

```text
la direccion del efecto es consistente
la permutacion respalda la asociacion especifica de los pesos
las tres particiones son positivas
```

### Lo que la evidencia no resuelve

```text
el bootstrap por grupos mantiene suficiente incertidumbre
para que el cero permanezca dentro del IC95
```

IC95 por grupos: `[-0.0013209910, 0.0118175641]` en su version de
delta de acierto por ejemplo; el intervalo del candidato canonico es
`[-0.0006906975, +0.0118175641]`.

**La limitacion es de precision, y la precision esta limitada por el tamano efectivo
del train.** No se intenta resolver con mas particiones, porque repetir particiones
hasta que un intervalo excluya el cero no seria evidencia, seria seleccion.

## Las tres pruebas responden a tres preguntas

| prueba | pregunta | NO responde |
|---|---|---|
| particiones | estabilidad frente al cambio de reparto | la magnitud del efecto |
| permutacion | especificidad de la correspondencia peso-ejemplo | la magnitud |
| bootstrap por grupos | incertidumbre sobre la magnitud | la especificidad |

No se utiliza una prueba para afirmar lo que responde otra.

## El defecto de class_weight, conservado como evidencia de auditoria

```text
class_weight='balanced' interactuaba incorrectamente con sample_weight
sklearn recalcula 'balanced' CON los sample_weight; un dict explicito no
los coeficientes difieren hasta 1,4e-2
```

La reproduccion valida usa el mecanismo historico exacto: `class_weight` explicito mas
`sample_weight`. La tercera particion corregida da `+0.0048825807`, y **el signo y la
conclusion no cambiaron**. El codigo historico no se modifica.

## Analisis mecanico

El resultado es coherente con la regla observada:

```text
violacion alta de margen  ->  peso menor
```

Y difiere de lo que historicamente empeoro:

```text
aumentar el peso de ejemplos dificiles   V4-B03: medium_up -0,0096, mild_up -0,0021
reducir el peso de la violacion alta     V4-B03: down_violacion +0,0036, hasta +0,0082
```

Se presenta como **evidencia empirica especifica de este dataset**, no como una teoria
universal sobre ponderacion de ejemplos.

## Estado de la investigacion

```text
research_status = CANDIDATE_PENDING_GOVERNANCE

V03                          = FROZEN_REFERENCE
A_cost_sensitive_downweight  = PROMISING_UNCONFIRMED
todos los demas mecanismos   = CLOSED / INVALIDATED / DUPLICATE
```

## Resumen ejecutivo

```text
A_cost_sensitive_downweight fue reconstruido desde sus artefactos
y reproducido con R3.

El efecto canonico es +0.0082045862.
La segunda particion confirma +0.0105253130.
La tercera particion confirma +0.0048825807 tras corregir
un defecto real de class_weight/sample_weight.
Las tres particiones mantienen la direccion positiva.

El test de permutacion de 1000 iteraciones produce
p empirico = 0.000999.

El bootstrap correcto por grupos mantiene el cero dentro del IC95.

Por ello el candidato es PROMISING_UNCONFIRMED,
no ADOPTED y no INVALIDATED.
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_07_cierre_final.py`. Sin push.*