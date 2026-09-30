# Dominio Twitter frente a general, con los modelos que SI entrenaron

Generado por `experiments/p71_09_downstream_lrfix.py`. Fecha 2026-09-29.

## Por que se reejecuta

Los embeddings del downstream anterior eran practicamente aleatorios: el learning rate no
estaba escalado por el lote, y la loss se quedaba clavada en el valor del estado cero. Estos
vienen de la corrida corregida:

```text
twitter   loss 1,272136 -> 1,174189 -> 1,132058 -> 1,108284 -> 1,095145
general   loss 1,254160 -> 1,140164 -> 1,094067 -> 1,069379 -> 1,055625
```

## Los numeros

| | twitter | general |
|---|---:|---:|
| vocabulario | 77,657 | 100,097 |
| embedding-only | 0.5514425602 | 0.5574677438 |
| fusion_estandarizada | 0.6579898386 | 0.6658165404 |
| fusion_normalizada | 0.5797391785 | 0.5782642824 |

V03 de referencia: `0.7772114525`.

## El control de dominio, que es la pregunta del encargo

| comparacion | twitter | general | delta | gana |
|---|---:|---:|---:|---|
| embedding_only | 0.551443 | 0.557468 | -0.006025 | **GENERAL** |
| fusion_estandarizada | 0.657990 | 0.665817 | -0.007827 | **GENERAL** |
| fusion_normalizada | 0.579739 | 0.578264 | +0.001475 | **TWITTER** |

**El resultado es INCONSISTENTE entre las tres formas.** Dos favorecen al general
(`-0,006025` en embedding-only y `-0,007827` en fusion estandarizada) y una favorece a twitter
por `+0,001475`, que es una milesima.

Eso no es una senal de dominio. Es lo que se ve cuando no hay senal: el signo cambia segun
como se mida.

## Los gates

| corpus | delta fusion | pliegues | neto | pasa |
|---|---:|---:|---:|---|
| twitter | -0.197472 | 0/5 | -560 | False |
| general | -0.198947 | 0/5 | -554 | False |

**Ninguno pasa.** El delta de fusion es de `-0,20`, los pliegues positivos son 0 de 5, y la
complementariedad es negativa en los dos.

VEREDICTO: `MARGINAL`

## La anomalia que queda, y hay que declararla

**La fusion sigue destruyendo la representacion de V03, y no deberia.**

```text
V03 sola                    0,777211
V03 + densas estandarizadas 0,665817 (general) y 0,657990 (twitter)
V03 + densas normalizadas   0,578264 (general) y 0,579739 (twitter)
```

Con la normalizacion por filas el bloque denso queda practicamente anulado frente a las
24.646 dimensiones dispersas, asi que la fusion deberia quedarse cerca de V03. **No lo hace:**
cae a `0,578`, que es 0,20 por debajo. Eso apunta a un segundo defecto de implementacion que
**no he aislado**.

Los dos defectos son mios, no del embedding ni del corpus. Y significan que **el numero de
fusion todavia no se puede leer como "el embedding no aporta"**.

## Lo que SI se puede afirmar

```text
el corpus de Twitter esta validado: periodo cerrado en 2020, cero solapamiento con train
los dos corpus estan preparados con 10M tokens y tokenizacion identica
el skip-gram ahora SI entrena: la loss baja monotonicamente en los dos
```

## Lo que NO se puede afirmar todavia

```text
si el dominio Twitter aporta o no, porque el signo cambia segun la forma de fusion
si el embedding aporta a V03, porque la fusion tiene un defecto no aislado
```

La hipotesis **sigue sin ponerse a prueba de forma limpia**, y ahora por un motivo distinto
del anterior: antes fallaba el entrenamiento, ahora falla la fusion.

> V03 remains the best verified leaderboard result.

*Sin push.*