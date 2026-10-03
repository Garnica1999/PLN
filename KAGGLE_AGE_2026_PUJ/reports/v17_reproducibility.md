# V17 — Reproducibilidad

- **Rama**: `kaggle-age-2026-puj-v17`
- **Commit base**: `a818349`
- **Commit de la fase**: `66be95da0f0821f8a66bfd55f3a6113c622f979f`
- **Semilla**: `cfg.SEED`
- **Interprete**: `/home/carlos/venv/bin/python` (Python 3.14.7)
- **Hardware**: AMD Ryzen 9 5950X (32 hilos), 31,2 GB de RAM, RX 9070 XT. **CPU pura**: ninguna fase de V17 usa GPU, porque solo hay un LogReg o un LinearSVC por fold.

## Script de cada fase

| Fase | Script |
|---|---|
| P0 auditoria | `scripts/v17_auditoria.py (documental, sin codigo de metricas)` |
| P0' referencia y A2 | `scripts/v17_p0_reproduccion.py` |
| P1 y P1.1 fronteras | `scripts/v17_p1_fronteras.py` |
| P1b curva de compuerta | `scripts/v17_p1b_curva_compuerta.py` |
| P1c regla conjunta | `scripts/v17_p1c_regla_conjunta.py` |
| P8 perfil de confianza | `scripts/v17_p8_perfil_confianza.py` |
| P9 arbitro | `scripts/v17_p9_arbitro.py` |
| P5 combinador local | `scripts/v17_p5_combinador_local.py` |
| Cierre | `scripts/v17_cierre.py` |

## Artefactos de entrada (todos ya existentes en el proyecto, ninguno se reconstruye)

| Artefacto | Que es |
|---|---|
| `features/oof_ensemble_v12.npy` | OOF de produccion de V12, 169.336 x 6 |
| `features/v16_oof_combinador_logistico.npy` | A2 de V16 |
| `features/oof_E028a_pares.npy` | 5 margenes de pares adyacentes de E-028a |
| `features/oof_E028b_grupos.npy` | 3 margenes de grupo grueso de E-028b |
| `features/dup_clusters.parquet` | agrupacion de casi duplicados para los folds |
| `features/oof_*.npy` (13 miembros) | OOF de los miembros del ensamble de V12 |

## Artefactos de salida

| Artefacto | Contenido |
|---|---|
| `features/v17_p9_oof.npy` | OOF de la candidata con arbitro |
| `features/v17_p5_oof.npy` | OOF de la candidata con combinador local |
| `reports/v17_master_results.csv` | tabla maestra, 20 filas |
| `reports/v17_hypothesis_registry.csv` | registro de hipotesis, 18 filas |
| `reports/v17_member_boundary_matrix.csv` | matriz por miembro y frontera |
| `reports/v17_error_intersection.csv` | interseccion de errores |
| `reports/v17_candidate_comparison.csv` | comparativa de candidatos |

## Puntos donde la reproduccion depende de una convencion

1. **El estimador de la referencia.** Todos los deltas de V17 se calculan sobre el pooled de `metricas_globales`, que redondea a cuatro decimales. Para medir la discrepancia con la media por fold se recalcula el pooled sin redondear.
2. **El margen del especialista.** Columna 1 del bloque de 5 margenes para `18-24|25-34` y columna 2 para `25-34|35-49`, con margen positivo hacia la clase superior. El orden de pares esta en `scripts/experimento_e028_ordinal_fino.py`.
3. **La region de frontera.** Se define como las filas en las que las dos clases mas probables de V12 son las del par. Sin esa condicion el especialista se aplica a filas lejanas y el delta cae a -0,28 o -0,48.

