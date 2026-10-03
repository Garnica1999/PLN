# P84-04: SGNS corregido y anclado (EXP-04)

- Fecha: 2026-10-02
- Script: experiments/p84_04_sgns_corregido_anclado.py
- Especificacion: docs/p84_spec.md seccion 5; preregistro docs/p84_preregistro.md
- Referencia V03: corrected_cv 0.7772114525165721; V03@0.5 0.771778476256122
- V03 remains the best verified leaderboard result.

## Ancla de implementacion

- Veredicto: ANCLA LOGRADA
- Algebraica: error relativo N 1.2763992653873705e-16, loss 1.1587846379097492e-16 (tol 1e-09)
- Gradiente (diferencias finitas): error max 4.4209981849124934e-08 (tol 1e-05)
- Ejecutable: pureza oficial 1.0 vs nuestra 1.0 (tol 0.05)

## Corpus y vocabulario

- Documentos 2847 | tokens 10002400 | vocabulario 100097 | cobertura dev 0.658542

## Downstream (lectura primaria: umbrales congelados 0.44/0.51/0.46)

| brazo | F1 congelada | por semilla |
|---|---|---|
| S | 0.626091 | 20261016: 0.613333, 20261017: 0.610643, 20261018: 0.613621 |
| M | 0.622888 | 20261016: 0.611190, 20261017: 0.602817, 20261018: 0.607611 |
| R | 0.651825 | 20261016: 0.616519, 20261017: 0.614104, 20261018: 0.607417 |

## Comparaciones pareadas (S menos referencia)

| comparacion | delta | pliegues | IC95 | rescate/dano |
|---|---|---|---|---|
| S_vs_M | +0.003204 | 3/5 | [-0.012606, +0.018844] | 334/318 |
| S_vs_R | -0.025734 | 1/5 | [-0.046887, -0.004500] | 463/596 |
| S_vs_V03 | -0.151120 | 0/5 | [-0.175340, -0.127571] | 259/894 |

## ensemble_gain (media de probabilidades con V03 a 0.5)

- S: gain -0.049004 | IC95 [-0.066411, -0.031799]
- M: gain -0.059147 | IC95 [-0.077785, -0.041325]
- R: gain -0.037883 | IC95 [-0.054244, -0.022010]

Nota de lectura: loss_suma y loss_media NO son comparables entre si (distinto funcional y distinta magnitud del gradiente negativo con el mismo lr); la comparacion S vs M confunde la correccion del objetivo con la magnitud del gradiente. Solo se comparan metricas downstream.

## D-057 (brecha train-validacion)

- brecha S 0.1383568371 | V03 0.204721418 | R 0.1422080099 | M 0.1375367204 | max +0.05 | ok True

## Downstream secundario (exploratorio)

- Fusion TF-IDF escala 1.0: F1 congelada 0.749247 | delta vs V03 -0.027964 | delta vs S +0.123156
- Media de probabilidades (S+V03): ver ensemble_gain de S (diagnostico).

## Gate

- Decision: CERRAR la linea SGNS corregido con el resultado verificado (sin rejilla nueva)
- S vs M NO: delta +0.003204, pliegues 3/5, IC95 [-0.012606, +0.018844]
- S vs R NO: delta -0.025734, pliegues 1/5, IC95 [-0.046887, -0.004500]
- S vs V03 standalone NO: delta -0.151120 (min 0.005), IC95 [-0.175340, -0.127571]
- complementariedad NO: ensemble_gain -0.049004 (min 0.0015), IC95 [-0.066411, -0.031799]

- Tiempo total: 15511.17 s | device: cuda (AMD Radeon RX 9070 XT)
