# P102-04 - Micro-barrido de umbrales sobre OOF congelado

Fecha: 2026-10-14. Rejilla 0.3-0.6 paso 0.005; uno-a-uno por etiqueta.

| Etiqueta | umbral V03 | argmax canonica | argmax 47 |
|---|---|---|---|
| HS | 0.44 | 0.410 | 0.365 |
| TR | 0.51 | 0.475 | 0.505 |
| AG | 0.46 | 0.450 | 0.490 |

Combo argmax canonico: delta +0.003895; celdas 83; IC95 [-0.001274,+0.009407]; delta 47 -0.002002.

Gate: GATE_FAILED {'delta_can_ge_0003': True, 'pliegues_ge_4': True, 'ic95_excluye_cero': False, 'delta_47_positivo': False, 'celdas_ge_200': False}

Nota: el argmax sobre el OOF es seleccion sobre el mismo OOF; sin gate completo no hay candidato.
