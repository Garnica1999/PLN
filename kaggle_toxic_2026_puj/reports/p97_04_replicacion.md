# P97-04 (H97-C replicacion): nnPU/PNU para TR en segunda particion

Pre-registro en el encabezado del script (particion agrupada nueva, seed 43; prediccion: delta > 0 y >= 4/5 pliegues).

| Regla | corrected_cv | F1 TR | std pliegues |
|---|---|---|---|
| propios control | 0.7720980489 | 0.821586 | 0.01376 |
| propios nnpu | 0.7703599750 | 0.816372 | 0.01018 |
| frozen control | 0.7756319869 | 0.824719 | 0.00968 |
| frozen nnpu | 0.7632054910 | 0.787440 | 0.01394 |

Comparacion principal (nnpu - control, regla propia): delta -0.001738, pliegues 1/5, delta F1 TR -0.005214, pliegues TR 1/5, IC95 clusters [-0.004401, 0.000683].

Zona no definida (HS=0): tasa de TR=1 predicha - V03 0.0934 | control 0.1030 | nnpu 0.1018.

**Veredicto pre-registrado: FALSIFICADA**
