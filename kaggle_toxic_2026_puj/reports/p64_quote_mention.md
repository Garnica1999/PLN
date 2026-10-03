# P64 - Frontera de cita y frontera de mencion

Generado por `experiments/p64_06_quote_mention.py`. Fecha 2026-09-29T14:17:17+00:00.

## La EDA va primero y decide si alguno de los dos tracks corre

El encargo dice que la linea de citas es prioritaria SOLO si la EDA demuestra que las
citas son frecuentes o diferencialmente asociadas a errores. Esa condicion se comprueba
antes de entrenar nada, y la regla esta escrita de antemano:

```text
frecuente   mas del 5% de las filas con al menos dos partes
diferencial la tasa de error entre las dos partes difiere mas de 5%
```

| frontera | filas | porcentaje | error dentro | error fuera | diferencial | justificada |
|---|---:|---:|---:|---:|---:|---|
| cita cerrada | 1 | 0.1% | 0.00% | 17.03% | +0.0000 | False |
| mencion | 702 | 47.3% | 14.86% | 18.95% | -0.0409 | True |

## Que se modela, y por que no es un recuento

Un recuento dice cuantas menciones tiene un tweet. Esto dice **lo que el modelo cree a
cada lado de la frontera**, y para una cita la asimetria entre el interior y el exterior
es la unica forma de responder a la pregunta del encargo: esta V03 puntuando una cita
como si fueran sus propias palabras? Una bandera no lo responde y un recuento tampoco.

**No se usa la identidad del usuario**: solo la posicion del simbolo `@`.

## Resultados

| brazo | corrected_cv | delta vs V03 | pliegues | neto |
|---|---:|---:|---:|---:|
| V03_mencion | 0.7585094002 | -0.0187020523 | 0/5 | -49 |
| V03_margen | 0.7661312520 | -0.0110802005 | 1/5 | -90 |
| V03_random | 0.7677487020 | -0.0094627505 | 2/5 | -36 |

### Las brechas

```text
V03_mencion_contra_aleatorio             -0.0092393017
V03_mencion_contra_el_margen             -0.0076218518
```

### Por estrato de longitud

| bin | n | F1 V03_mencion | F1 V03_margen | F1 V03_random |
|---|---:|---:|---:|---:|
| CORTA | 167 | 0.9515 | 0.9379 | 0.9371 |
| MEDIA | 602 | 0.7817 | 0.7825 | 0.7927 |
| LARGA | 547 | 0.6378 | 0.6806 | 0.6670 |
| MUY_LARGA | 167 | 0.5532 | 0.5595 | 0.5423 |

## Veredicto

```text
ninguna frontera aporta: el mejor es V03_mencion con -0.0187020523 global, -0.0092393017 contra el control de anchura y -0.0076218518 contra el margen

LECTURA: SIN_APORTE
DECISION: CLOSED
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p64_06_quote_mention.py`. Sin push.*