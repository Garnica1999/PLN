# P65 - n-gramas de palabra de orden superior anadidos a V03

Generado por `experiments/p65_01_ngrams.py`. Fecha 2026-09-29T14:55:47+00:00.

## Que hay de nuevo aqui, frente al precedente

**REP-04** midio TF-IDF word (1,3) **suelto**, con **C=1.0** y **sin bloque de
caracteres**, y lo registro como descartado. Eso es una sustitucion con otra fuerza de
regularizacion. Esto es una **adicion**: el bloque de caracteres, C, los umbrales, los
pliegues y la semilla se quedan exactamente como estan, y lo unico que cambia es que el
bloque de palabras tambien emite trigramas.

La distincion no es cosmetica. Una sustitucion pregunta si basta un bloque de palabras
mas grande por si solo, y REP-04 ya respondio eso. Una adicion pregunta si la evidencia
lexica de largo alcance contiene algo que el bloque actual mas el de caracteres no
contiene, y eso no lo ha respondido nadie.

## El control de capacidad es obligatorio

Los trigramas multiplican el vocabulario. Sea cual sea el resultado, un bloque que anade
decenas de miles de columnas a un espacio que ya tiene veinte mil va a mover la funcion
ajustada por si mismo. La unica manera de separar informacion de capacidad es anadir un
bloque aleatorio de la misma anchura y comparar.

## Resultados

| brazo | corrected_cv | delta vs V03 | pliegues | neto | ancho | segundos |
|---|---:|---:|---:|---:|---:|---:|
| palabras_1_3 | 0.7732593837 | -0.0039520688 | 1/5 | -15 | 3210 | 0.4 |
| palabras_1_3_sin_char | 0.7547741314 | -0.0224373211 | 0/5 | -88 | 3210 | 0.1 |
| V03_aleatorio | 0.6743436992 | -0.1028677533 | 0/5 | -349 | 0 | 0.0 |
| V03_aleatorio_palabras | 0.7135922067 | -0.0636192458 | 0/5 | -203 | 0 | 0.0 |
| V03_margen | 0.7661312520 | -0.0110802005 | 1/5 | -90 | 0 | 0.0 |

### Las brechas, que son las unicas preguntas

| brecha | valor |
|---|---:|
| palabras_1_3_contra_aleatorio | +0.0989156845 |
| palabras_1_3_contra_aleatorio_de_palabras | +0.0596671770 |
| palabras_1_3_contra_margen | +0.0071281317 |
| palabras_1_3_sin_char_contra_aleatorio | +0.0804304322 |

### Por estrato de longitud

| bin | n | V03 | palabras_1_3 | palabras_1_3_sin_char | V03_aleatorio | V03_aleatorio_palabras | V03_margen |
|---|---:|---:|---:|---:|---:|---:|---:|
| CORTA | 167 | 0.9480 | 0.9449 | 0.9314 | 0.9006 | 0.9030 | 0.9379 |
| MEDIA | 602 | 0.7964 | 0.7911 | 0.7651 | 0.7147 | 0.7439 | 0.7825 |
| LARGA | 547 | 0.6834 | 0.6847 | 0.6768 | 0.5046 | 0.5838 | 0.6806 |
| MUY_LARGA | 167 | 0.5606 | 0.5491 | 0.5208 | 0.3856 | 0.4389 | 0.5595 |

## Veredicto

```text
palabras_1_3 lleva informacion que no es capacidad ni confianza: +0.0989156845 contra el control de anchura y +0.0071281317 contra el margen. Y el global es -0.0039520688 con 1 de 5 pliegues, luego no entra. Es el tercer caso de este patron en el proyecto, despues de la negacion en P64-07 y de la composicion tematica en P64-08: la senal existe y la integracion la descarta

LECTURA: INFORMACION_SI_INTEGRACION_NO
DECISION: COMPLEMENTARY
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p65_01_ngrams.py`. Sin push.*