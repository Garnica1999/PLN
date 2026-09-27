# V20 · P2 — Pesos ordinales: un solo parametro cambiado

**V20 queda congelado.** No se toca un solo clasificador de los cinco: solo la **perdida** de uno.

## 1. Por que aqui y no en otra parte

La auditoria P6 degrada cada frontera una a la vez y mide cuanto cae el blend de V20:

| Frontera degradada | Caida del blend |
|---|---|
| `<= 13-17 vs >= 18-24` | **-0,0129** |
| `<= 18-24 vs >= 25-34` | **-0,0072** |
| `<= 25-34 vs >= 35-49` | -0,0040 |
| `<= 35-49 vs >= 50-64` | -0,0018 |
| `<= 50-64 vs >= 65-xx` | -0,0024 |

La primera frontera pesa **siete veces mas** que la cuarta, y es ademas la de **menor exactitud balanceada** (0,7632). Hay una asimetria medida: **la frontera mas debil es la que mas importa**, y ninguna de las otras cuatro justifica un cambio.

## 2. La hipotesis

El discriminador del primer umbral esta desbalanceado por construccion (28.863 filas de `13-17` contra 140.473 del resto, un 17/83) y LightGBM con `min_child_samples = 200` tiende a podar los cortes que separan al grupo pequeno. **Subir el peso del lado minoritario de ese umbral concretamente** deberia mejorar su exactitud balanceada sin tocar las otras cuatro fronteras.

**Lo que NO se hace**: no se abre un grid de `C` por frontera, ni `class_weight` por frontera, ni `threshold` por frontera. **Una sola configuracion**, con un solo parametro cambiado.

## 3. Resultado

| Medida | V20 | Con el peso |
|---|---|---|
| macro-F1 del blend | 0.6097782 | **0.593202** |
| delta contra V20 | - | **-0.0166** |
| folds positivos | - | **0/5** |
| IC95 contra V20 | - | [-0.0178, -0.0153] |
| exactitud balanceada del primer umbral | 0,7632 | **0.843** (+0.0797) |
| frontera 1 | 0,5075 | 0.4117 |
| frontera 2 | 0,5029 | 0.5088 |
| rescates | - | 5,234 |
| danos | - | 6,574 |
| neto | - | -1,340 |
| predicciones modificadas | - | 18,164 |

## 4. Delta por fold

**[-0.014157, -0.017618, -0.017164, -0.015906, -0.018048]**
