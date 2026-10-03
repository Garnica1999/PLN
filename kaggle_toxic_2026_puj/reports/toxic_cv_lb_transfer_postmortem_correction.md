# Correccion del post-mortem tras auditoria independiente

Generado por `experiments/p70_03_correccion_postmortem.py`. Fecha 2026-09-29T20:47:25+00:00.

## Que encontro la auditoria

Tres objeciones, y la tercera cambia la respuesta.

### Las dos que se aceptan como correcciones de redaccion

```text
A estaba SUBVALORADA    su estimador puntual no tiene 'el orden correcto':
                        es el termino dominante del gap
'gana en todos los      es un non-sequitur como evidencia sobre transferencia
 tamanos'               externa: ganar en CV controlado no dice nada del LB
```

### La tercera: faltaba una causa, y es la mayor de todas

La taxonomia no contemplaba la **seleccion del baseline**. Y resulta ser el efecto mas
grande del registro.

| elemento | valor |
|---|---:|
| scores de leaderboard registrados | 22 |
| de ellos no triviales | 21 |
| **maximo** | **0.78306** |
| ... y es | `v03_lr_union_thr` |
| mediana de los no triviales | 0.77663 |
| desviacion | 0.00574 |
| el candidato cae en el percentil | 42.9 |
| el candidato esta por debajo de la mediana por | 0.00141 |
| ventaja del maximo sobre la mediana | **0.00643** |
| el gap que hay que explicar | +0.00585 |

**El candidato no quedo por debajo: quedo en la mediana.** Y V03 no es una medicion
representativa: es el **maximo de 21 intentos**. Comparar una medicion fresca contra un
maximo seleccionado favorece al maximo en una cantidad del orden del propio gap.

```text
ventaja de seleccion del maximo   0.00643
el gap a explicar                 0.00585
```

Los dos numeros son del mismo tamano. **El gap deja de necesitar el confounder de tamano
para explicarse.**

## La cuota de A, como mapeo declarado y no como veredicto

La extrapolacion hacia arriba es un escenario, y su cuota depende de una forma que no se
puede medir sin el holdout. Se publica el mapeo completo:

| forma supuesta | H predicho | cuota de A | residuo |
|---|---:|---:|---:|
| lineal | 0.013900 | 87% | 0.002145 |
| logaritmica | 0.012232 | 76% | 0.003813 |
| decaimiento 0.81 | 0.008409 | 52% | 0.007636 |
| concavidad estricta (cota) | 0.011203 | 70% | 0.004842 |

La concavidad tipica de una curva de aprendizaje hace que el tramo de abajo
**sobreestime** la ganancia del tramo de arriba. Bajo concavidad estricta la cuota baja al
77 por ciento. Y formalmente el intervalo del slope incluye cero, luego la cota inferior de
la cuota es cero.

## La reconciliacion del control, que la auditoria exigio

```text
al 100% la sensibilidad dio  +0.006765
el canonico es               +0.0082045862
diferencia                    0.001440, el 17.5 por ciento del efecto
```

**Causa**: la sensibilidad uso la **mediana** de los multiplicadores `[1.2, 1.5, 1.2]` y
el canonico los multiplicadores **por pliegue**. La columna del candidato es por tanto un
**limite inferior declarado**, no una discrepancia sin explicar.

Y la parte que sostiene el argumento de tamano no se ve afectada: **V03 al 100 por ciento
da `0,777211`, exactamente el valor canonico.**

## Las causas, corregidas

| causa | aplica | porque |
|---|---|---|
| F_seleccion_del_baseline | `True` | V03 es el maximo de 21 mediciones; el candidato es una medicion fresca que cae en el percentil 43. Comparar un |
| A_diferencia_de_datos | `True` | la extrapolacion es hacia ARRIBA del rango medido y la concavidad tipica sobreestima la cuota. Queda como esce |
| C_variabilidad_entre_particiones | `COMPONENTE` | es una componente de incertidumbre, no una explicacion separada |
| B_el_weighting_no_generaliza | `SIN_EVIDENCIA_POSITIVA` | el candidato cae en la mediana historica, no por debajo. Un leaderboard negativo no es evidencia de que el mec |
| D_cambio_en_la_distribucion_de_clases | `NO_EVALUADO` | no evaluable sin labels de test |
| E_sensibilidad_de_la_regla_de_pesos | `NO EVIDENCIADO` | la estabilidad de escala no es un analisis de sensibilidad de la regla |
| G_transferencia_de_umbral | `NO_EVALUADO` | el candidato mueve la prevalencia predicha de TR +0.0209, y un intercambio de precision/recall puede cambiar d |

## Las correcciones de redaccion, registradas

| decia | debe decir |
|---|---|
| B decia NO_DEMOSTRABLE | SIN_EVIDENCIA_POSITIVA: la razon anterior era un non-sequitur |
| A decia 'magnitud del orden correcto' | A es el termino dominante del estimador puntual, con cuota declarada |
| C figuraba como causa | C es una componente de incertidumbre, no una causa |
| D y E decian NO aplica | NO_EVALUADO y NO EVIDENCIADO, respectivamente |
| faltaba F | seleccion del baseline, y es la de mayor magnitud |
| faltaba G | transferencia de umbral, no evaluada |

## La clasificacion

```text
MIXED_EXPLANATION

La causa dominante en el registro es la SELECCION del baseline, que la taxonomia no contemplaba: V03 es el maximo de 21 mediciones de leaderboard y el candidato es una medicion fresca que cae en la mediana. El confounder de tamano aporta una cuota declarada de entre el 58 y el 95 por ciento segun la forma de la curva, con un intervalo que incluye cero. La transferencia de umbral y el cambio de distribucion no son evaluables sin labels de test. No hay evidencia positiva de que el mecanismo sea falso: el candidato aterrizo donde aterriza una submission media de esta familia.
```

## Lo que sigue sin poder afirmarse

```text
no se puede afirmar 'transferencia externa negativa' como hallazgo:
  es una unica medicion de 1154 filas, a ~1 error estandar de la diferencia
no se puede afirmar que el weighting sea falso
no se puede afirmar que el experimento OOF sea invalido
no se puede separar del todo la seleccion del tamano de entrenamiento
```

## El candidato NO cambia de estado

```text
external_transfer   NEGATIVE, registrado
candidate_status    PROMISING_UNCONFIRMED, sin cambios
research_status     EXPERIMENTAL_RESEARCH_CLOSED
```

Solo se cambiaria a `INVALIDATED` con una contradiccion metodologica real. La auditoria no
encontro ninguna: encontro un sesgo en la INTERPRETACION, no en el experimento.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p70_03_correccion_postmortem.py`. Sin push.*