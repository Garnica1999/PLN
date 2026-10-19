# P102-03 - Regla dura TR=0 si HS=0 (aislada sobre V03)

Fecha: 2026-10-14. Invariante en train: TR=1 & HS=0 -> 0 casos; AG=1 & HS=0 -> 26 casos.

| Particion | violaciones V03 | celdas cambiadas | delta vs V03 |
|---|---|---|---|
| Canonica | 20 | 20 | -0.000646 |
| Particion 47 | 21 | 21 | -0.004097 |

Bootstrap vs V03 (canonica): IC95 [-0.003901,+0.002672]; pliegues 3/5.
Submission V03: la regla cambiaria 15 celdas (contexto, sin decision).

Gate: GATE_FAILED {'delta_can_ge_0003': False, 'pliegues_ge_4': False, 'ic95_excluye_cero': False, 'delta_47_positivo': False, 'celdas_ge_200': False}
