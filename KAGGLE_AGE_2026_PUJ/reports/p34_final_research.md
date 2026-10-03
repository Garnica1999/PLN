# P34 - correccion adyacente por pares sobre V30

## Hipotesis

> La senal residual es demasiado debil para corregir las seis clases a la vez, pero puede bastar para resolver algunas confusiones entre bandas adyacentes.

## Correccion historica de la hipotesis, y por que importa

**La cifra sobre la que se apoyo P34 era falsa.** V31 reporto que el 90 % de las confusiones de V30 eran entre bandas adyacentes. Ese numero salio de mirar las **diez confusiones mas frecuentes**, que por construccion son casi todas adyacentes. Contado sobre los **65.403 errores completos**:

| | Filas | % |
|---|---|---|
| adyacentes | 39.202 | **59,94 %** |
| no adyacentes | 26.201 | **40,06 %** |

La conclusion correcta es: **las confusiones adyacentes son una mayoria, pero no dominan el 90 % del error total**. La hipotesis sigue siendo valida en su forma, pero su techo es considerablemente mas bajo. Los informes de V31 no se modifican; la correccion vive aqui.

## Protocolo

- `d_V30(i,j) = log((P_V30[i]+1e-6)/(P_V30[j]+1e-6))` sobre **probabilidades reales**.
- `d_C(i,j) = score_C[i] - score_C[j]` sobre el **decision_function CRUDO**, nunca dividido por la suma de su fila.
- La correccion **solo intercambia `top1` y `top2` de V30, y solo si son adyacentes**. No puede llevar la fila a una tercera clase ni a una no adyacente. Eso la distingue de V28-03, que **sustituia** el multiclase y perdio 0,067553.
- `a` y `b` de `d_C = a + b * d_V30` en `inner_train`; `beta` en `inner_val`; `outer_val` intacto. Semilla 20261003.

Filas donde `top1` y `top2` de V30 son adyacentes: **130.106 de 169.336 (76,83 %)**. Son la puerta por la que puede actuar la correccion.

## Resultado

| Brazo | corrected_cv | delta | folds+ | rescate | danio | neto |
|---|---|---|---|---|---|---|
| real | 0.6132432 | -0.0002039 | 1/5 | 701 | 735 | -34 |
| barajado | 0.6130362 | -0.0004109 | 0/5 | 641 | 707 | -66 |
| aleatorio | 0.6133229 | -0.0001242 | 3/5 | 679 | 705 | -26 |
| v30 | 0.6134471 | +0.0000000 | 0/5 | 0 | 0 | +0 |

## La atribucion falla, y por que control manda

- `real - barajado = +0.0002070`: positivo, asi que la correspondencia fila-residuo parece aportar algo.
- `real - aleatorio = -0.0000797`: **negativo**. El ruido aleatorio de la misma media y desviacion **rinde mas que la senal real**.

La regla del brief es explicita: `real <= random` implica **INVALIDATED**. El control de ruido es el mas fuerte de los tres porque responde a la pregunta que de verdad importa: **¿gana porque hay senal, o solo porque se permite sumar un termino mas?** Aqui gana la segunda.

## P13 explica por que el residuo no puede funcionar

| Par | V30 acierta | Candidato acierta | AUC del residuo | AUC de d_V30 |
|---|---|---|---|---|
| A1 | 0.6048 | 0.6641 | 0.5136 | 0.6555 |
| A2 | 0.5126 | 0.6322 | 0.4942 | 0.5052 |
| A3 | 0.5067 | 0.6718 | 0.4946 | 0.4333 |
| A4 | 0.6221 | 0.7343 | 0.5032 | 0.4200 |
| A5 | 0.7328 | 0.7866 | 0.4946 | 0.4569 |

El candidato **supera a V30** en la decision binaria de la pareja en los cinco casos. Pero el residuo se queda en **0,49-0,51**, y las correlaciones `d_C/d_V30` van de 0,57 a 0,79. La lectura es una sola: **esa ventaja es la misma informacion de V30 reescalada con ruido, no informacion independiente**. Por eso, al quitarle lo que V30 ya explica, no queda nada.

## P09 y P12: donde la correccion no toca

| Par | Filas adyacentes | Rescate | Danio | Neto |
|---|---|---|---|---|
| A1 | 50,608 | 377 | 400 | -23 |
| A2 | 48,606 | 379 | 411 | -32 |
| A3 | 45,678 | 246 | 232 | +14 |
| A4 | 43,117 | 141 | 168 | -27 |
| A5 | 33,820 | 78 | 103 | -25 |

**P12: cero rescates de alta confianza y 735 danados.** El mecanismo no toca precisamente lo que se le pedia: las confusiones adyacentes en las que V30 estaba seguro. Y de las cinco parejas, solo A3 (25-34/35-49) tiene neto positivo, y es **+14 sobre 45.678 filas**.

## Bootstrap

Delta **-0.0002207**, IC95 **[-0.0006645, +0.0002135]**, contiene cero: **True**. 2000 remuestreos pareados, unidad la fila. La mejora no esta demostrada ni en signo ni en magnitud.

## Interpretacion

**P34 INVALIDATED.** La senal residual no aporta informacion suficientemente especifica para corregir de forma robusta las decisiones entre bandas adyacentes.

Lo que se **no** dice: que la informacion linguistica no exista. P33 midio AUC 0,5411 en el residuo global. Lo que se dice es mucho mas estrecho: **esa informacion residual no ha podido convertirse en una mejora predictiva robusta mediante los mecanismos global y local evaluados.**

## Limitaciones

- **La correccion del 90 % baja el techo de la hipotesis**, pero no la invalida: el 59,94 % sigue siendo mayoria.
- Un solo candidato linguistico. Los nueve restantes tienen AUC de residuo similar.
- No se ejecuto particion alternativa: con el veredicto ya invalidado por el control de ruido, ejecutarla confirmaria algo que no va a cambiar.

## Reproducibilidad

- Brazo real reconstruido desde cero: **0.6132310**, frente a 0.6132432 en el artefacto. Coincide: False.
- `score_type` declarado por candidato en `reports/p34_score_inventory.csv`: V30 es **probability**, los diez candidatos son **decision_function**.

## Figuras

- `p33p34_01_brazos.png`: V30 frente a los cuatro brazos de P33 y los tres de P34
- `p33p34_02_delta_por_fold.png`: corrected_cv por fold
- `p33p34_03_rescate_danio.png`: rescate, danio y neto, con las cinco parejas
- `p33p34_04_auc_por_pareja.png`: AUC del residuo frente a AUC de d_V30
