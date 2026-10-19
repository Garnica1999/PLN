# P102-02 - Promedio V03+LinearSVC+ComplementNB con umbrales recalibrados

Fecha: 2026-10-14. Control reproduce V03 exacto (diff 0.00e+00). Umbrales recalibrados: {'HS': 0.41, 'TR': 0.38, 'AG': 0.37}.

| Metrica | Valor |
|---|---|
| corrected_cv (recalibrado) | 0.773370 |
| corrected_cv (umbrales V03 frozen) | 0.743469 |
| delta vs V03 | -0.003841 |
| pliegues mejorados | 3/5 |
| celdas cambiadas | 163 |
| IC95 vs V03 | [-0.011717,+0.004407] |
| particion 47 (recal transferido) | 0.766183 (-0.009448 vs v03) |
| gate | GATE_FAILED |

Gate A3: delta >= +0.003; >=4/5 pliegues; IC95 excluye cero; delta 47 > 0; >= 200 celdas.
