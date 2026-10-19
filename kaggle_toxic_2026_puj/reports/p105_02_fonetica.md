# P105-02 (S4/S7b) - Folding fonetico como canal char adicional

Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba 0.00e+00; ccv 0.7772114525165721); particion 47 0 celdas distintas.
Ejemplos de folding: {'vacaciones': 'bakasiones', 'quiero': 'kiero', 'caballo': 'kabayo', 'huevo': 'uebo', 'mucha': 'mucha'}
Piso de deteccion del arnes (pre-declarado): se ~0.0035 mediano; minimo detectable 95% ~+0.0069.

| Variante | corrected_cv | delta V03 | pliegues | celdas | IC95 vs V03 | delta 47 | Gate |
|---|---|---|---|---|---|---|---|
| F_fonetica_canal | 0.776848 | -0.000363 | 1/5 | 96 | [-0.006092,+0.005504] | +0.000414 | GATE_FAILED |

Conclusion: F_fonetica_canal: sin senal detectable (IC95 [-0.006092,+0.005504] incluye cero; piso de deteccion +0.005798).

Sin test; sin holdouts; sin leaderboard; umbrales V03 congelados.
