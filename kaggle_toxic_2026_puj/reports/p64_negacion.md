# P64 - Alcance de negacion

Generado por `experiments/p64_07_negacion.py`. Fecha 2026-09-29T14:24:46+00:00.

## Que se prueba, con precision

El encargo advierte que la hipotesis NO es una bandera de negacion. Son tres cosas a
la vez:

```text
un MARCADOR      uno de una lista cerrada de negadores, por regla
una DISTANCIA     a quantos tokens esta la palabra afectada: 1, 2 o 3
una EVIDENCIA     lo que V03 cree sobre el texto en ese alcance
```

Y la afirmacion es sobre la INTERACCION: cambia el marcador la evidencia de las palabras
que alcanza, de una forma que V03, que ve la bolsa, no puede representar.

## Los tres mecanismos

| mecanismo | que es |
|---|---|
| NEG_alcance | canal sparse de los pares negador y palabra alcanzada, con el alcance en el nombre |
| NEG_efecto | el cambio en la evidencia de V03 al quitar el segmento con negador |
| NEG_ambos | los dos juntos, con el presupuesto L2 recalibrado |

Lista cerrada de marcadores: `no, nunca, nadie, nada, ni, sin, tampoco, ningun, ninguna, ninguno, jamas`. Sin lexico externo, sin corpus
externo y sin nada aprendido de etiquetas, luego no hay nada que pueda filtrar.

## La EDA de la negacion

| magnitud | valor |
|---|---:|
| filas con negador | 389 (26.2%) |
| error con negador | 21.77% |
| error sin negador | 15.33% |
| diferencial | +0.0644 |
| filas que reciben marcas | 389 |
| marcas de alcance emitidas | 3132 |
| columnas del canal sparse | 353 |

Negadores presentes en el corpus: `nada, nadie, ni, ningun, ninguna, no, nunca, sin, tampoco`.

## Resultados

| brazo | corrected_cv | delta vs V03 | pliegues | neto |
|---|---:|---:|---:|---:|
| NEG_alcance | 0.7750287751 | -0.0021826774 | 2/5 | -6 |
| NEG_efecto | 0.7537595332 | -0.0234519193 | 0/5 | -93 |
| NEG_ambos | 0.7675657054 | -0.0096457472 | 1/5 | -36 |
| V03_random | 0.7548721605 | -0.0223392920 | 0/5 | -79 |
| V03_margen | 0.7661312520 | -0.0110802005 | 1/5 | -90 |

### Las brechas

```text
alcance_contra_aleatorio         +0.0201566146
alcance_contra_el_margen         +0.0088975231
efecto_contra_aleatorio          -0.0011126273
efecto_contra_el_margen          -0.0123717188
ambos_contra_el_margen           +0.0014344533
```

### Por estrato de longitud

| bin | n | F1 NEG_alcance | F1 NEG_efecto | F1 NEG_ambos | F1 V03_random | F1 V03_margen |
|---|---:|---:|---:|---:|---:|---:|
| CORTA | 167 | 0.9480 | 0.9301 | 0.9417 | 0.9395 | 0.9379 |
| MEDIA | 602 | 0.7998 | 0.7768 | 0.7940 | 0.7762 | 0.7825 |
| LARGA | 547 | 0.6733 | 0.6675 | 0.6773 | 0.6547 | 0.6806 |
| MUY_LARGA | 167 | 0.5490 | 0.4826 | 0.4983 | 0.5183 | 0.5595 |

## Veredicto

```text
Este es el resultado mas positivo de la fase y hay que decirlo con precision, porque es distinto de todos los anteriores. El canal de alcance de negacion supera al control de anchura en +0.0201566146 y al margen de V03 en +0.0088975231. Las dos brechas son positivas: la informacion existe, no es capacidad y no es lo que V03 ya sabe de si mismo. Y a la vez el global es -0.0021826774 con 2 de 5 pliegues, luego no mejora a V03.

La lectura honesta es que el cuello de botella NO es la informacion, es la INTEGRACION. Un canal sparse de 350 columnas que compite por el presupuesto L2 de un espacio de 20000 dimensiones no logra entrar aunque lleve dentro algo que el resto del modelo no tiene. Y eso es una afirmacion sobre el ARQUITECTURA de la adicion y no sobre la senal. El efecto sobre la evidencia, en cambio, si es redundante: -0.0123717188 contra el margen, luego quitar el segmento con negador no le dice a V03 nada que no sepa ya.

La negacion si esta asociada al error, con +0.0644 puntos, y emite 3132 marcas de alcance sobre 389 filas. La estructura existe. Lo que falta no es saberla, sino un sitio donde caber

LECTURA: INFORMACION_SI_INTEGRACION_NO
DECISION: COMPLEMENTARY
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p64_07_negacion.py`. Sin push.*