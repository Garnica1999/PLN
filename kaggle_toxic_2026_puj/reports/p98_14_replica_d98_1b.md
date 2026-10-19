# P98-14 (D98-1b): replica de robustez en segunda particion agrupada

Pre-registro: `p98_14_preregistro.json` (semilla 43; tamanos [297, 297, 297, 296, 296]; sin fuga de grupos: True). Todo congelado de D98-1: reps, configuracion de fusion y umbrales V03; sin blendings.

| Regla | corrected_cv | delta | F1 HS/TR/AG |
|---|---|---|---|
| V03-equivalente (part47) | 0.7756319869 | - | 0.7566/0.8247/0.7456 |
| Fusion contextual (part47) | 0.7767830429 | +0.001151 | 0.7609/0.8151/0.7544 |

- Pliegues positivos: 2/5; neto de celdas: +2 (91 rescatadas / 89 rotas); IC95 clusters [-0.007491, 0.009919] (pct_mejora 0.5975).
- Referencia canonica D98-1: fusion 0.782266, delta +0.005055, 4/5, IC95 [-0.004058, +0.014721].

**Veredicto: FALSIFICADA** (criterios pre-registrados: robusta = delta >= +0.003, >=4/5, IC sin cero, neto > 0; marginal = positivo sin todos; falsificada = delta <= 0 o <4/5 o neto <= 0).

Sin test, sin holdouts, sin leaderboard; ningun artefacto de D98-1 fue modificado.
