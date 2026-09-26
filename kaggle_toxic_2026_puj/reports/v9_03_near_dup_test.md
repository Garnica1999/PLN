# V9-03 (H2): propagacion train->test por near-duplicados

- Pares nivel A (>= 0.85): 8; tuits de test afectados: 4
- Pares nivel B (0.8-0.85): 3
- Acuerdo celdas nivel A (etiqueta de la copia vs V03): 75.00%
- Celdas que cambian en la sonda v18: 0

## Pares nivel A

| id_train | id_test | sim | y_train | V03 | acuerdo |
|---|---|---|---|---|---|
| 23919 | 20568 | 1.000 | 0/0/0 | 0/0/0 | 3/3 |
| 21262 | 20568 | 1.000 | 0/0/0 | 0/0/0 | 3/3 |
| 23245 | 22448 | 1.000 | 1/0/1 | 1/0/1 | 3/3 |
| 22860 | 20568 | 1.000 | 0/0/0 | 0/0/0 | 3/3 |
| 22883 | 20496 | 0.948 | 1/1/1 | 0/0/0 | 0/3 |
| 22883 | 21178 | 0.912 | 1/1/1 | 1/1/1 | 3/3 |
| 20862 | 20496 | 0.871 | 1/1/1 | 0/0/0 | 0/3 |
| 21150 | 20496 | 0.853 | 0/0/0 | 0/0/0 | 3/3 |

Submission sonda: `submissions/toxic_submission_v18_near_dup_propagacion.csv` (integridad: {'filas': 1154, 'columnas': ['id', 'HS', 'TR', 'AG'], 'ids_iguales': True, 'binario': True}).
Lectura: el efecto maximo posible esta acotado por el numero de celdas que cambian;
la decision de usar la copia como etiqueta depende del acuerdo observado.

Fecha: 2026-09-26. Duracion: 0.15 s.