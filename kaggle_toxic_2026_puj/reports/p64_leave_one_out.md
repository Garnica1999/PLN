# P64 - Leave-one-segment-out y el perfil de contradiccion

Generado por `experiments/p64_04_leave_one_out.py`. Fecha 2026-09-29T07:23:30+00:00.

## Que es genuinamente nuevo aqui

P64-03 midio la **distribucion** de las probabilidades por segmento, que describe lo que
el modelo dice de cada pieza AISLADA. No puede decir de que pieza depende la DECISION
una vez junto el resto del documento, porque un segmento puede puntuar alto por si solo
y seguir siendo irrelevante para el total.

Eso es lo que responde el leave-one-out, y **P59 nunca lo computo**:

```text
contribucion_i = score del documento completo menos el score sin el segmento i
```

La contribucion es **con signo**, y esa es la clave: un segmento que empuja el score
hacia la clase positiva aporta un numero positivo y uno que lo frena aporta un negativo,
y la separacion entre ambos es una propiedad del DOCUMENTO que ni las probabilidades
por segmento ni el score agregado expresan.

**751 filas** tienen dos o mas segmentos y reciben perfil. Las
**732 de un solo segmento no lo reciben**: su leave-one-out seria la
cadena vacia, que no es evidencia de nada, y no se rellenan con un cero porque un cero
pareceria una medida cuando es una ausencia.

El bloque tiene **36 columnas**, 12 por etiqueta.

## Los tres brazos

| brazo | corrected_cv | delta vs V03 | pliegues | neto |
|---|---:|---:|---:|---:|
| V03_loo | 0.7656662317 | -0.0115452208 | 0/5 | -63 |
| V03_random | 0.7679574277 | -0.0092540248 | 1/5 | -33 |
| V03_margen | 0.7661312520 | -0.0110802005 | 1/5 | -90 |

```text
contra el control de capacidad: -0.0022911960
contra el margen de V03:      -0.0004650203
```

### Por rebanada de longitud

| bin | n | F1 de V03 | F1 de V03_loo | delta |
|---|---:|---:|---:|---:|
| CORTA | 167 | 0.9480 | 0.9345 | -0.0134 |
| MEDIA | 602 | 0.7964 | 0.7924 | -0.0040 |
| LARGA | 547 | 0.6834 | 0.6660 | -0.0174 |
| MUY_LARGA | 167 | 0.5606 | 0.5534 | -0.0072 |

## La contradiccion, que es el objetivo de P6

| estrato | n | `loo_concentracion` | `loo_conflicto` | `loo_cambia_de_signo` | `loo_fraccion_implicita` | `loo_entropia` |
|---:|---:|---:|---:|---:|---:|---:|
| todas multi | 751 | 0.7080 | 0.0182 | 0.7168 | 0.6602 | 0.6635 |
| V03 correcto | 480 | 0.7262 | 0.0156 | 0.6764 | 0.6880 | 0.6390 |
| V03 incorrecto | 271 | 0.6757 | 0.0227 | 0.7884 | 0.6109 | 0.7068 |
| error de alta confianza | 176 | 0.6881 | 0.0201 | 0.7538 | 0.6184 | 0.6861 |
| CORTA | 50 | 0.7554 | 0.0202 | 0.6133 | 0.6900 | 0.6108 |
| MEDIA | 247 | 0.7414 | 0.0202 | 0.6977 | 0.6639 | 0.6551 |
| LARGA | 325 | 0.7114 | 0.0176 | 0.7108 | 0.6502 | 0.6639 |
| MUY_LARGA | 129 | 0.6170 | 0.0148 | 0.8088 | 0.6666 | 0.6989 |

## La contribucion contra la confianza

| rasgo | AUC para predecir el ACIERTO |
|---|---:|
| concentracion | 0.449595 |
| conflicto | 0.380911 |
| cambia_de_signo | 0.398551 |
| fraccion_implicita | 0.454946 |
| margen_de_V03 | 0.770875 |

## Veredicto

```text
la contribucion con signo NO supera al margen de V03, con una brecha de -0.0004650203. El mejor rasgo alcanza 0.454946 de AUC contra el acierto frente a 0.770875 del margen. Y P64-03 ya habia encontrado lo mismo con la distribucion de probabilidades. La descomposicion con signo es MAS informativa que ninguna de las dos, y aun asi no aporta, porque V03 lleva la estructura interna dentro de su confianza

LECTURA: CONTRIBUCION_REDUNDANTE
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p64_04_leave_one_out.py`. Sin push.*