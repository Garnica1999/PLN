# P105-01 (S2/S7a) - Normalizacion por bloque vs conjunta

Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba 0.00e+00; ccv 0.7772114525165721); particion 47 0 celdas distintas.
Piso de deteccion del arnes (pre-declarado): se ~0.0035 mediano; minimo detectable 95% ~+0.0069.

| Variante | corrected_cv | delta V03 | pliegues | celdas | IC95 vs V03 | delta 47 | diff vs control | Gate |
|---|---|---|---|---|---|---|---|---|
| V1_l2_por_bloque | 0.777211 | +0.000000 | 0/5 | 0 | [+0.000000,+0.000000] | +0.000000 | 0.00e+00 | GATE_FAILED |
| V2_l2_conjunta | 0.759694 | -0.017517 | 1/5 | 282 | [-0.028724,-0.006328] | -0.018905 | 3.78e-01 | GATE_FAILED |
| V3_norma_media | 0.742843 | -0.034369 | 1/5 | 373 | [-0.048696,-0.020160] | -0.036038 | 7.98e-01 | GATE_FAILED |

Nota: V1_l2_por_bloque es identidad con V03 (diff 0.0 esperado; no compite).

Conclusion: V1_l2_por_bloque: identidad con V03 confirmada (diff 0.00e+00; no compite). V2_l2_conjunta: efecto NEGATIVO detectable (IC95 [-0.028724,-0.006328] excluye cero por abajo): la variante perjudica; no es candidata.

Sin test; sin holdouts; sin leaderboard; umbrales V03 congelados.
