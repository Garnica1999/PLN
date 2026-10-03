# P55 - Auditoria estrategica del pipeline final de V03

Generado por `experiments/v55_00_auditoria_pipeline.py` (V55-00 (auditoria estrategica del pipeline final de V03)).

Fecha 2026-09-28T02:49:03+00:00. **0 modelos, 0 submissions, 0 consultas a Kaggle, 0 uso de holdout o test. V03 no se modifica.**

## 1. El pipeline real, leido del codigo y no del docstring

```text
raw train (1731)
  -> particion congelada dev 1483 / holdout 248
  -> pliegues: DOS protocolos distintos, y ahi esta el problema

A) 04_refit_submissions_v02_v03.py
      grupos = near-duplicado (D-004), 5 pliegues
      vectorizador y modelo DENTRO de cada pliegue
      -> oof_mod03b.npz
      -> umbrales 0,44 / 0,51 / 0,46 maximizando F1 sobre ESE OOF
      -> refit sobre el 100 por ciento del train con ESOS umbrales
      -> submission V03

B) 14_oof_corregido_v2.py
      grupos = near-duplicate UNION pseudo-autor (D-016), 5 pliegues
      vectorizador y modelo DENTRO de cada pliegue
      -> v2_oof_base_corrected.npz
      -> corrected_cv = 0,7772114525 con LOS UMBRALES DE A)
```

> los umbrales y la metrica reportada provienen de protocolos de agrupacion distintos, y D-004 es el mas debil

- umbrales adoptados de: experiments/04_refit_submissions_v02_v03.py sobre features/oof_mod03b.npz con grupos D-004
- corrected_cv reportado de: experiments/14_oof_corregido_v2.py sobre v2_oof_base_corrected.npz con grupos D-016
- vectorizador dentro del pliegue: **True**, luego no hay leakage de representacion
- modelo final: refit sobre el 100 por ciento del train con los mismos hiperparametros, y umbrales los mismos del OOF D-004

## 2. Transferencia de umbrales: in-sample, anidados y estabilidad

| metodo | HS | TR | AG | macro-F1 | delta vs V03 |
|---|---:|---:|---:|---:|---:|
| umbrales de V03 | 0.44 | 0.51 | 0.46 | 0.7772114525 | +0,0000000000 |
| optimos in-sample D-016 | 0.41 | 0.48 | 0.45 | 0.7808400798 | +0.0036286273 |
| **anidados (honesto)** | ver CSV | ver CSV | ver CSV | **0.7781367700** | **+0.0009253174** |

> el delta in-sample NO es una mejora esperable, porque el umbral se eligio sobre el mismo OOF que se puntua; el delta anidado si es una estimacion honesta

### Los umbrales anidados por pliegue

| pliegue | anidado HS | anidado TR | anidado AG | V03 HS | V03 TR | V03 AG |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.40 | 0.50 | 0.47 | 0.44 | 0.51 | 0.46 |
| 1 | 0.41 | 0.48 | 0.45 | 0.44 | 0.51 | 0.46 |
| 2 | 0.40 | 0.50 | 0.45 | 0.44 | 0.51 | 0.46 |
| 3 | 0.41 | 0.48 | 0.45 | 0.44 | 0.51 | 0.46 |
| 4 | 0.41 | 0.48 | 0.45 | 0.44 | 0.51 | 0.46 |

## 3. Tasa positiva, que es la consecuencia operativa

| umbrales | HS | TR | AG |
|---|---:|---:|---:|
| de V03 | 0.4572 | 0.3001 | 0.3776 |
| optimos in-sample | 0.4943 | 0.3095 | 0.3844 |
| anidados | 0.4970 | 0.3075 | 0.3810 |

## 4. La brecha OOF a leaderboard

| magnitud | valor |
|---|---:|
| corrected_cv | 0.7772114525 |
| LB | 0.78306 |
| delta | **+0.00584855** |
| sigma de pliegues del OOF | 0.023095 |
| delta en sigma | 0.25 |

- **lo que se sabe**: el LB fue mas alto que el OOF
- **lo que se desconoce**: la causa, y ninguna consulta a Kaggle es legal en esta fase
- **lo que es plausible y NO esta probado**: variacion de muestreo entre dev y test; el OOF subestima por umbrales heredados de un protocolo mas debil

## 5. Decision

**`P55_PROCEDURAL_GAP = B`.**

