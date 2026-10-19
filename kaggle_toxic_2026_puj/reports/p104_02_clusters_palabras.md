# P104-02 (R5) - Clusters de palabras (k-means sobre embeddings propios)

Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba 0.00e+00; ccv 0.7772114525165721); particion 47 0 celdas distintas.
Piso de deteccion del arnes (pre-declarado): se ~0.0030; delta minimo detectable 95% ~+0.0059.

| Variante | corrected_cv | delta V03 | pliegues | celdas | IC95 vs V03 | delta 47 | Gate |
|---|---|---|---|---|---|---|---|
| K100 | 0.761455 | -0.015756 | 0/5 | 302 | [-0.027868,-0.004064] | -0.019338 | GATE_FAILED |
| K300 | 0.762410 | -0.014801 | 1/5 | 279 | [-0.026471,-0.003431] | -0.012558 | GATE_FAILED |
| K1000 | 0.759350 | -0.017862 | 1/5 | 293 | [-0.030264,-0.005551] | -0.022933 | GATE_FAILED |

Conclusion: K300: efecto NEGATIVO detectable (IC95 [-0.026471,-0.003431] excluye cero por abajo): la variante perjudica; no es candidata.

Sin test; sin holdouts; sin leaderboard; umbrales V03 congelados.
