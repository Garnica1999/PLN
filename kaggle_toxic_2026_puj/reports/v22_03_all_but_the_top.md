# V22-03 (P4): All-But-The-Top sobre el embedding de V5

Generado por `experiments/v22_03_all_but_the_top.py` (V22-03 (P4: All-But-The-Top sobre el embedding de V5)).

**Ningun modelo entrenado. Ninguna submission.**

## 1. Control de cordura

Con `D = 0` no se elimina ninguna direccion y la reconstruccion es la identidad. Da **0.781766** frente a **0.781766** publicados (desviacion 0.000000), dentro de tolerancia: **True**.

Ese control verifica a la vez la carga de vectores, el centrado, la SVD y la
reconstruccion. Si falla, ningun `D` es atribuible a la eliminacion de componentes.

## 2. Donde se aplica el PCA

sobre la matriz de los 6.059 vectores de tipo que el pipeline consulta, antes del pooling, que es donde los autores colocan el procedimiento. La eleccion no es libre: el pipeline solo tiene vectores de esos tipos.

Renormalizacion posterior: ninguna, siguiendo a los autores.

## 3. El espectro

Varianza explicada acumulada por las primeras direcciones principales:

| direccion | varianza acumulada |
|---|---:|
| PC1 | 0.0531 |
| PC2 | 0.0863 |
| PC3 | 0.1160 |
| PC4 | 0.1449 |
| PC5 | 0.1684 |
| PC6 | 0.1881 |
| PC7 | 0.2073 |
| PC8 | 0.2239 |
| PC9 | 0.2385 |
| PC10 | 0.2524 |

## 4. Las tres configuraciones

| D | var. eliminada | corrected_cv | delta V03 | **delta control** | emb solo | celdas | rescates | danos | neto |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0531 | 0.779935 | +0.002724 | **-0.001831** | 0.659548 | 133 | 67 | 66 | +1 |
| 2 | 0.0863 | 0.775645 | -0.001566 | **-0.006121** | 0.661255 | 114 | 51 | 63 | -12 |
| 5 | 0.1684 | 0.778144 | +0.000933 | **-0.003622** | 0.658931 | 108 | 53 | 55 | -2 |

El control (D=0) queda en **0.781766** con delta frente a V03 de **+0.004555**.

## 5. Veredicto de P4: ALL_BUT_THE_TOP_NO_APORTA

Ninguna eliminacion de componentes principales mejora la fusion por encima del control, que ya es el embedding sin post-procesar. La hipotesis de que los componentes dominantes ocultan senal recuperable no se sostiene en este pipeline

Completado en 11.3 s.

