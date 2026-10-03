# Post-mortem: CV interno frente a transferencia al leaderboard

Generado por `experiments/p70_01_sensibilidad_tamano.py` y `experiments/p70_02_postmortem.py`. Fecha 2026-09-29T20:48:42+00:00.

## Los tres hechos, que no se colapsan en uno

| hecho | valor |
|---|---|
| el weighting mejora el protocolo OOF controlado | `+0.0082045862` |
| la submission congelada obtiene LB por debajo de V03 | `0.77522 contra 0.78306, delta -0.00784` |
| los dos hechos no permiten afirmar que el weighting sea falso | `el CV mejora en las tres particiones y la permutacion es p = 0,000999` |

## El resultado externo, congelado

```text
V03 LB        0.78306
candidato LB  0.77522
delta LB      -0.00784
clase         EXTERNAL_TRANSFER_RESULT
transferencia NEGATIVA
```

## El hecho estructural que hay que explicar

```text
gap de V03        LB - CV = +0.00585   (el LB esta POR ENCIMA)
gap del candidato LB - CV = -0.01020   (el LB esta POR DEBAJO)
diferencia de gaps        = -0.01604
```

No es solo que el candidato pierda en el leaderboard: es que **los dos gaps van en
direcciones opuestas**. Esa asimetria es lo que hay que explicar, y es mas informativa
que el delta entre ambos.

## Item 4: el confounder de tamano, confirmado

| elemento | valor |
|---|---:|
| V03_enviado_filas | `1731` |
| candidato_enviado_filas | `1483` |
| holdout_excluido_filas | `248` |
| training_ids_hash | `4edbfd82b1fb4a23a53a544467efad5a` |
| excluded_holdout_ids_hash | `3b0c4726d3a9358f1ba50b2f243ed4b9` |

```text
V03 enviado         1731 filas
candidato enviado   1483 filas
diferencia          248 filas, 14.3 por ciento menos
```

**Las labels del holdout no se leen.** El holdout permanece congelado y esta auditoria
se completa sin el.

## Items 7 y 8: cuanto mueve el CV reducir el entrenamiento

Submuestreo **por grupo** del desarrollo: un grupo near-duplicate entero entra o sale.
Tres repeticiones por tamano, con semillas fijadas antes de ejecutar.

| tamano | filas | CV de V03 medio | desviacion | rango | delta medio |
|---|---:|---:|---:|---:|---:|
| 100% | 1483 | 0.777211 | 0.000000 | 0.000000 | +0.006765 |
| 90% | 1334 | 0.768916 | 0.007214 | 0.014086 | +0.006978 |
| 80% | 1185 | 0.762201 | 0.003470 | 0.006516 | +0.010913 |
| 70% | 1038 | 0.765370 | 0.002017 | 0.003927 | +0.011470 |

```text
desviacion maxima entre repeticiones   0.007214
rango maximo dentro de un tamano       0.014086
caida de V03 del 100 al 70 por ciento  +0.011842
el efecto que hay que explicar         +0.008205
RATIO desviacion / efecto              0.88
```

### La lectura, sin afirmar causalidad

La magnitud esta **medida**, no supuesta: reducir el entrenamiento produce variaciones del
mismo orden que el efecto. Y al 90 por ciento las tres repeticiones van de `0,7628` a
`0,7769`, un rango de `0,0141`, **mayor que el efecto entero**.

El control es limpio: al 100 por ciento las tres repeticiones son **identicas**, porque el
submuestreo no cambia nada. La variabilidad de los otros tamanos es del procedimiento, no
del ruido de la semilla.

### Y una observacion que va en contra de la explicacion facil

```text
al 100% el candidato bate a V03 por +0.006765
al 90% el candidato bate a V03 por +0.006978
al 80% el candidato bate a V03 por +0.010913
al 70% el candidato bate a V03 por +0.011470
```

**El candidato gana en todos los tamanos medidos.** El weighting no se degrada al reducir
los datos, luego la parte B de las causas candidatas no queda demostrada.

## Items 9 y 10: la regla de pesos es estable

| tamano | mediana | IQR | peso<1 | peso=1 | peso medio | retirado |
|---|---:|---:|---:|---:|---:|---:|
| 100% | 0,0000 | 0,016266 | 441 | 1042 | 0,8634 | 0,1366 |
| 90% | 0,0000 | 0,016021 | ~422 | ~912 | 0,8581 | 0,1419 |
| 80% | 0,0000 | 0,021352 | ~379 | ~807 | 0,8576 | 0,1424 |
| 70% | 0,0000 | 0,019329 | ~323 | ~715 | 0,8602 | 0,1398 |

La mediana del error_score es 0 en todos los tamanos, ningun peso supera 1, y la fraccion
retirada se mueve entre 0,1366 y 0,1424. **La misma formula produce poblaciones de pesos
practicamente iguales.** La causa E no aplica.

Sobre la estabilidad entre submuestras: solo 3 de los 66 pares son comparables, y son las
tres corridas del 100 por ciento, que dan correlacion perfecta. Las demas **no son
comparables** porque el submuestreo por grupo cambia los ids en cada corrida. Se declara
como limite de identificacion, no se esconde.

## Item 16: la prevalencia apenas se mueve

| etiqueta | real | V03 predicha | candidato predicha | cambio |
|---|---:|---:|---:|---:|
| HS | 0.4370 | 0.4572 | 0.4673 | +0.010115 |
| TR | 0.3001 | 0.3001 | 0.3210 | +0.020904 |
| AG | 0.3823 | 0.3776 | 0.3688 | -0.008766 |

El weighting **no altera el balance de decisiones** de forma apreciable. La causa D no
aplica: la mejora interna no viene de predecir mas o menos positivos.

## Item 12: de donde viene la mejora interna

| etiqueta | F1 V03 | F1 candidato | delta | precision | recall |
|---|---:|---:|---:|---|---|
| HS | 0.7617 | 0.7785 | +0.016834 | 0.7448 -> 0.7532 | 0.7793 -> 0.8056 |
| TR | 0.8157 | 0.8165 | +0.000773 | 0.8157 -> 0.7899 | 0.8157 -> 0.8449 |
| AG | 0.7542 | 0.7612 | +0.007006 | 0.7589 -> 0.7751 | 0.7496 -> 0.7478 |

Confirma el patron de rescate historico, sobre la evidencia reproducida:

```text
HS = estable     aporta de forma consistente
TR = debil       el mas flojo de los tres
AG = inestable   cambia de signo entre particiones
```

No se optimiza ninguna etiqueta. Es lectura descriptiva.

## Item 13: los margenes

| etiqueta | mediana V03 | mediana candidato | fraccion < 0,10 del umbral |
|---|---:|---:|---|
| HS | 0.2292 | 0.2401 | 0.2111 -> 0.2158 |
| TR | 0.3778 | 0.3804 | 0.0904 -> 0.0883 |
| AG | 0.2506 | 0.2625 | 0.1935 -> 0.1807 |

El weighting **no reescribe la poblacion entera**: la mediana del margen se mueve poco y
la fraccion cerca del umbral tambien. El cambio es de frontera, no de escala.

**Esto no es evidencia de generalizacion al test.** Es una descripcion de lo que hace la
regla sobre OOF.

## Item 14: interseccion de errores

| etiqueta | rescue | damage | neto |
|---|---:|---:|---:|
| HS | 42 | 23 | +19 |
| TR | 14 | 19 | -5 |
| AG | 43 | 32 | +11 |
| **GLOBAL** | **99** | **74** | **+25** |

## Items 17 y 27: la clasificacion

| causa candidata | aplica |
|---|---|
| A_diferencia_de_datos | `True` |
| B_el_weighting_no_generaliza | `NO_DEMOSTRABLE` |
| C_variabilidad_entre_particiones | `True` |
| D_cambio_en_la_distribucion_de_clases | `False` |
| E_sensibilidad_de_la_regla_de_pesos | `False` |
| H_no_identificable_con_datos_permitidos | `True` |

### A_diferencia_de_datos

- **magnitud_medida**: reducir el entrenamiento un 10 por ciento mueve el CV de V03 con desviacion 0.007214, que es 0.88 veces el efecto
- **la_diferencia_real**: 14.3 por ciento menos filas
- **lectura**: la magnitud es del orden correcto para explicar el deficit. NO se puede medir V03 con 1731 sin el holdout

### C_variabilidad_entre_particiones

- **magnitud_medida**: el rango dentro de un mismo tamano llega a 0.014086
- **lectura**: comparable al efecto, y refuerza A

### H_no_identificable_con_datos_permitidos

- **razon**: el holdout esta congelado y las labels del test estan prohibidas, luego no se puede separar 'menos datos' de 'no transfiere' con certeza

```text
MIXED_EXPLANATION

el confounder de tamano es plausible por magnitud, la regla de pesos y la prevalencia son estables, y la separacion final no es identificable sin datos prohibidos
```

## Item 20: la clasificacion del candidato NO cambia

```text
external_transfer   NEGATIVE
candidate_status    PROMISING_UNCONFIRMED    (no se modifica)
```

Un leaderboard negativo **no es** un experimento invalido. Solo se cambiaria a
`INVALIDATED` si apareciera una contradiccion metodologica real, y no ha aparecido: la
reproduccion sigue siendo R3, las tres particiones siguen positivas y la permutacion sigue
en `p = 0,000999`.

## Item 18: lo que NO se hace con el resultado externo

```text
no se usa el numero de filas del test (1154) para proyectar nada
no se analiza cuantas filas cambiaron como prediccion del leaderboard
no se inspeccionan salidas del test para elegir una hipotesis nueva
no se genera una submission de V03 con 1483 filas
no se genera ninguna variante del candidato
```

El resultado de leaderboard es una **observacion externa ya realizada**, no un nuevo
conjunto de entrenamiento.

## Item 17: el limite de identificacion, dicho explicitamente

```text
CAUSE_NOT_IDENTIFIABLE_WITHOUT_PROHIBITED_DATA
```

No se puede separar con certeza:

```text
'el candidato entreno con 248 filas menos que V03'
de
'el weighting no transfiere'
```

El holdout esta congelado y las labels del test estan prohibidas. **No se intenta inferir
la causa con el test**, y no se usa la magnitud de -0,00784 para inventar una explicacion.

## Item 28: la regla de parada

El post-mortem se detiene aqui. No se abre una hipotesis nueva, no se genera otra
submission, y no se usa el leaderboard para tuning.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p70_02_postmortem.py`. Sin push.*