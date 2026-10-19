# P97-02 (H97-B): Group DRO minimax sobre grupos del protocolo

Pre-registro en `p97_independent_research.md` seccion 5 (T=15, eta=0.5, pesos q_g/n_g, umbrales V03 congelados).

- Control A0 (pesos uniformes): corrected_cv 0.7772114525 (reproduce V03 exacto: True), std pliegues 0.02309.
- DRO: corrected_cv 0.7720416249, std pliegues 0.02598.
- Delta pooled -0.005170; pliegues mejorados 3/5; IC95 clusters [-0.015025, 0.005359] (positivos 0.1545).
- Error por grupo (celda agregada): control {'p50': 0.2222222222222222, 'p90': 0.37777777777777777, 'p99': 0.4444444444444444, 'n_grupos': 17} | dro {'p50': 0.2857142857142857, 'p90': 0.37777777777777777, 'p99': 0.4444444444444444, 'n_grupos': 17}.

**Veredicto pre-registrado: FALSIFICADA**

## Nota de ejecuciones (trazabilidad)

Antes de esta ejecucion valida hubo una corrida con `sample_weight` sin
normalizar (liblinear escala la perdida frente a C): colapso de escala
preservado en `p97_02_group_dro_escala_invalida.*` y
`features/p97_02_oof_dro_escala_invalida.npz`. La presente corrida normaliza
los pesos a media 1 (suma n_train).
