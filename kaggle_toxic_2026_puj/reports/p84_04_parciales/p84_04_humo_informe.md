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

- Documentos 64 | tokens 205014 | vocabulario 7652 | cobertura dev 0.466884

## Gate

- Decision: CERRAR la linea SGNS corregido con el resultado verificado (sin rejilla nueva)
- S vs M NO: delta +0.009541, pliegues 3/5, IC95 [-0.004852, +0.023682]
- S vs R NO: delta +0.002664, pliegues 2/5, IC95 [-0.018870, +0.023774]
- S vs V03 standalone NO: delta -0.215076 (min 0.005), IC95 [-0.243806, -0.186834]
- complementariedad NO: ensemble_gain -0.075604 (min 0.0015), IC95 [-0.096224, -0.055910]

- Tiempo total: 25.97 s | device: cuda (AMD Radeon RX 9070 XT)
