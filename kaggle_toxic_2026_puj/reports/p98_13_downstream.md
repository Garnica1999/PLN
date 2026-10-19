# P98-13 (D98-1): evaluacion downstream de las representaciones C98-1

Pre-registro: `p98_architecture.md` seccion 6. Reps congeladas (concat capa2 LTR+RTL, media sobre tokens, 2048-d); umbrales V03.

| Regla | corrected_cv | delta vs V03 | F1 HS | F1 TR | F1 AG |
|---|---|---|---|---|---|
| V03 frozen | 0.7772114525 | +0.000000 | 0.7617 | 0.8157 | 0.7542 |
| embedding-only (LR C=1) | 0.6522376845 | -0.124974 | 0.6522 | 0.6859 | 0.6186 |
| fusion v5.1-style (TF-IDF + denso) | 0.7822662599 | +0.005055 | 0.7647 | 0.8172 | 0.7649 |

- Fusion vs V03: pliegues mejorados 4/5; IC95 clusters [-0.004058, 0.014721]; neto de celdas +11.
- Referencia estatica v5.1 (citada): embedding-only 0.6523; fusion 0.7818.

**Veredicto: MARGINAL_SENAL** (gate: delta >= +0.003, >=4/5 pliegues, IC sin cero).
