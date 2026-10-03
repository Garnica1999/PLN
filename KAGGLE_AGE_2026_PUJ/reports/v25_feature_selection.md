# V25 — Seleccion supervisada de features por informacion mutua

**Una sola comparacion.** No se corre ninguna grid, no se prueba otro K y no se prueba otro selector.

## 1. Hipotesis y por que no es V15

> V20 puede mejorar eliminando features ruidosas mediante una seleccion supervisada realizada correctamente dentro de cada fold.

| | V15 | V25 |
|---|---|---|
| Metodo | VIF, correlacion, clustering | `mutual_info_classif`, 6 categorias |
| Usa la variable objetivo | **no** | **si**, solo del training fold |
| Resultado | poda -0,004, 0 de 5 folds | se mide en este informe |
| Conclusion | la redundancia es funcional | - |

V15 concluyo que podar **empeora**, y de ahi salio el bloque de 73 variables label-informed que V19 si explotó. V25 no repite esa poda: usa la etiqueta.

## 2. Procedencia de las 73 label-informed

Verificadas por codigo, por comparacion de arrays y por separacion empirica en `v25_00_inventario.py`: se construyen con `y_train` del training fold, se escriben solo en `valid_idx`, y usan **el mismo array de grupos y los mismos folds** que V20. **LEGAL**.

## 3. Por que se estandariza antes del mutual information

`mutual_info_classif` estima con vecino mas cercano y **la metrica de distancia no es invariante a escala**. Las 73 supervisadas llegan a magnitud absoluta **59.439** frente a las probabilidades de miembro, que viven en [0, 1]. El estandarizado se aplica dentro de cada training fold y **nunca ve la validacion**.

## 4. Control T0

- V20 reconstruido: **0.6097782** frente a 0.6097782 | reproduce: **True**.
- Si no reprodujera, la fase se detiene sin entrenar T1.

- Blend con cuatro cuartos: **0.6097782**.

## 5. Resultado

| Modelo | corrected_cv | Delta | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Media |
|---|---|---|---|---|---|---|---|---|
| T0 V20 sin seleccion | **0.6097782** | +0.000000 | 0.613203 | 0.612081 | 0.610446 | 0.608843 | 0.604351 | 0.609785 |
| T1 seleccion MI top 50% | **0.6091492** | -0.000629 | 0.611552 | 0.611332 | 0.610107 | 0.609584 | 0.603205 | 0.609156 |

### F1 por clase

| Modelo | 13-17 | 18-24 | 25-34 | 35-49 | 50-64 | 65-xx |
|---|---|---|---|---|---|---|
| T0 V20 sin seleccion | 0.6818 | 0.5170 | 0.4965 | 0.5687 | 0.6637 | 0.7310 |
| T1 seleccion MI top 50% | 0.6823 | 0.5169 | 0.4954 | 0.5680 | 0.6625 | 0.7299 |

### Complementariedad y error por fold

| Modelo | rescue | damage | net | changed rows | folds + |
|---|---|---|---|---|---|
| T0 V20 sin seleccion | 0 | 0 | +0 | 0 | 0/5 |
| T1 seleccion MI top 50% | 1,042 | 1,143 | -101 | 3,545 | 1/5 |

| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| T0 V20 sin seleccion | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| T1 seleccion MI top 50% | -0.001651 | -0.000749 | -0.000339 | +0.000742 | -0.001146 |

## 6. La seleccion, fold a fold

K = floor(279 x 0.5) = **139**, aplicado **independientemente en cada training fold** y **sin ajustarlo observando el resultado**.

| Fold | K | MI max | MI min |
|---|---|---|---|
| 1 | 139 | 0.4926 | 0.0001 |
| 2 | 139 | 0.4965 | 0.0004 |
| 3 | 139 | 0.4959 | 0.0000 |
| 4 | 139 | 0.4973 | 0.0000 |
| 5 | 139 | 0.5011 | 0.0000 |

## 7. Estabilidad y error por bloque

La frecuencia de seleccion es **diagnostico**. No se usa para construir una segunda seleccion, que es justo lo que V15 hizo mal.

| Bloque | Columnas | Frecuencia media | MI media | Nunca seleccionadas |
|---|---|---|---|---|
| `miembro_ensamble` | 78 | 1.000 | 0.21538 | 0 |
| `nbsvm_palabra` | 6 | 1.000 | 0.20831 | 0 |
| `label_informed` | 73 | 0.523 | 0.09925 | 34 |
| `denso_textometrico` | 35 | 0.143 | 0.01110 | 30 |
| `estructurado` | 87 | 0.136 | 0.01190 | 74 |

## 8. Decision

- Delta de T1 frente a T0: **-0.000629** (gate >= 0.001) -> **False**
- Folds positivos: **1/5** (gate >= 4) -> **False**
- Bootstrap: **no se calcula**, porque el gate de delta no se cumple y el brief lo reserva a ese caso.

**SUPERVISED_FEATURE_SELECTION = CLOSED**

No se genera submission y no se abre ninguna otra hipotesis.
