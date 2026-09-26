# V10-04: calibracion fuerte por etiqueta (P1, primera linea de la via A)

Generado por `experiments/v10_04_calibracion_fuerte.py` (V10-04 (P1, calibracion fuerte)).

## 1. Como se mide

La regla 3 de la fase prohibe optimizar nada sobre el mismo OOF que se reporta, y la
auditoria anterior midio que hacerlo infla +0.0036. Por eso **el calibrador y el umbral
se ajustan los dos dentro de cada pliegue de entrenamiento**, se aplican al pliegue de
validacion, y el proceso se repite con cinco semillas de particion.

La referencia es V03 con sus umbrales congelados. **No** se le reoptimiza el umbral:
eso seria mover el punto de partida y comparar dos cosas distintas.

## 2. Calidad de calibracion

| metodo | ECE HS | ECE TR | ECE AG | ECE medio |
|---|---|---|---|---|
| sin_calibrar | 0.051207 | 0.063331 | 0.060344 | 0.058294 |
| platt | 0.028615 | 0.020722 | 0.025065 | 0.024801 |
| isotonica_recortada | 0.019980 | 0.014801 | 0.017343 | 0.017375 |
| beta_tres_parametros | 0.082335 | 0.034021 | 0.058705 | 0.058354 |

Reduccion del ECE frente a Platt, con el ECE medio de las tres etiquetas:

| metodo | reduccion | supera el gate del 20 por ciento |
|---|---|---|
| platt | +0.00 por ciento | no |
| isotonica_recortada | +29.94 por ciento | si |
| beta_tres_parametros | -135.29 por ciento | no |

El mejor es **isotonica_recortada** con +29.94 por ciento.

**Gate 1 superado: True.**

## 3. Ganancia anidada de F1 de la competencia

Cada celda es el delta de la candidata contra V03. Con asterisco, la particion en la que
la candidata mejora.

| metodo | 42 | 7 | 123 | 2024 | 5150 | media | mejoran |
|---|---|---|---|---|---|---|---|
| sin_calibrar | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0/5 |
| platt | +0.000082* | -0.004974 | +0.006487* | +0.000690* | -0.001545 | +0.000148 | 3/5 |
| isotonica_recortada | -0.001201 | -0.005019 | +0.003353* | +0.001221* | -0.003132 | -0.000956 | 2/5 |
| beta_tres_parametros | -0.000392 | -0.005212 | +0.006449* | -0.000644 | -0.000765 | -0.000113 | 1/5 |

**Gate 2, no empeorar la F1 anidada: True.**

## 4. Intervalos pareados sobre la particion canonica

| metodo | diferencia observada | IC95 | incluye el cero |
|---|---|---|---|
| platt | +0.000082 | [-0.005221, +0.005932] | si |
| isotonica_recortada | -0.001201 | [-0.006393, +0.004703] | si |
| beta_tres_parametros | -0.000392 | [-0.005189, +0.004521] | si |

## 5. Gate 3 del addendum de protocolo

- metodo evaluado: **platt**
- delta medio de al menos 0.002: **False**
- al menos 4 de 5 particiones mejoran: **False**
- intervalo pareado que no incluye el cero: **False**
- **gate 3 superado: False**

## 5 bis. Una salvedad sobre la calibracion beta

La calibracion beta sale con **peor** ECE que no calibrar en HS, lo que es raro en un
modelo de tres parametros que en principio contiene a Platt. Se comprobo que no es un
fallo de la teoria sino de la rejilla usada:

| etiqueta | verosimilitud negativa de Platt | de beta |
|---|---|---|
| HS | 672.0 | 702.1 |
| TR | 422.4 | 426.4 |
| AG | 637.8 | 654.1 |

Platt alcanza igual o mejor verosimilitud en las tres etiquetas: **True**.

La causa es que la rejilla beta limita el parametro de asimetria a [0.7, 1.3] mientras que
el desplazamiento de Platt es libre, de modo que **la rejilla no contiene a Platt**.
Lo medido es por tanto esta rejilla estrecha, y no la familia beta en general. El gate 1
no depende de beta, asi que su veredicto sigue siendo valido, y no se abre una via nueva
para ampliar la rejilla porque el criterio de cierre ya se cumplio con la isotonica.

## 6. Veredicto

**NO ACCIONABLE.** La calibracion cumple los gates 1 y 2 pero no el 3 del
addendum, que exige delta medio de +0.002, 4 de 5 particiones que mejoran e
intervalo que no incluye el cero. Es una mejora real de calibracion que **no se
traduce en mejora de la metrica de la competencia**, y eso es exactamente lo que se
documenta.

Fecha: 2026-09-26T07:50:19+00:00. Duracion: 13.9 s.
