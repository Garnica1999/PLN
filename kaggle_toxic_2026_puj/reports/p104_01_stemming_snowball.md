# P104-01 (R4) - Stemming Snowball espanol

Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba 0.00e+00; ccv 0.7772114525165721); particion 47 0 celdas distintas.
Piso de deteccion del arnes (pre-declarado): se ~0.0030; delta minimo detectable 95% ~+0.0059.

| Variante | corrected_cv | delta V03 | pliegues | celdas | IC95 vs V03 | delta 47 | Gate |
|---|---|---|---|---|---|---|---|
| S1_word_stem | 0.765319 | -0.011893 | 1/5 | 202 | [-0.021755,-0.002301] | -0.009679 | GATE_FAILED |
| S2_stem_mas_raw | 0.769493 | -0.007719 | 1/5 | 119 | [-0.014381,-0.000603] | -0.007150 | GATE_FAILED |

Conclusion: S2_stem_mas_raw: efecto NEGATIVO detectable (IC95 [-0.014381,-0.000603] excluye cero por abajo): la variante perjudica; no es candidata.

Sin test; sin holdouts; sin leaderboard; umbrales V03 congelados.
