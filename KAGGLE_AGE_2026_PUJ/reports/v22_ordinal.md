# V22 — Forense de V20 y un modelo ordinal con representacion compartida

**V20 queda congelado**: LB **0,62689**, OOF **0,6097782**.

## 1. La inconsistencia del informe, resuelta contra el codigo

El informe de V21 decia `LightGBMClassifier(objective="multiclass")`, objetivo binario `clase <= 0` contra `clase >= 1`, y `scores = predict_proba[:, 1] = P(lado bajo)`. Las tres cosas son ciertas **a la vez**, y el codigo es este:

```python
objetivo = (clase <= k - 1).astype(int)   # target BINARIO
modelo = lgb.LGBMClassifier(**HIERBA)  # HIERBA declara multiclass, num_class 6
modelo.fit(matriz[entrenamiento], objetivo[entrenamiento])
acumuladas[validacion, k - 1] = modelo.predict_proba(matriz[validacion])[:, 1]
```

| Pregunta | Respuesta del codigo |
|---|---|
| 1. Labels de cada frontera | **binarios**: `clase <= k-1` contra `clase >= k` |
| 2. Clases del target | **2** en cada entrenamiento, porque el vector solo contiene 0 y 1 |
| 3. `objective` real | HIERBA **declara** `multiclass` con `num_class = 6`, pero LightGBM construye un booster de **seis** clases y las columnas 2 a 5 son **cero exacto**. El objetivo *declarado* es multiclase y el *efectivo* es binario: ambas cosas son ciertas |
| 4. Columnas de `predict_proba` | **seis**, no dos. La 0 es `P(clase >= k)` y la 1 es `P(clase <= k-1)`; las otras cuatro son cero estructural |
| 5. Como se obtiene cada CDF | **directamente**: `acumuladas[:, k-1] = P(Y <= k-1)`, porque el target binario es exactamente esa probabilidad |
| 6. Cinco binarios independientes | **SI**, el bucle `for k in range(1, 6)` ajusta cinco instancias distintas |
| 7. Multiclase transformado | **NO**. No existe ningun modelo multiclase del que se deriven |
| 8. V20 es una cadena de cinco | **CASO A, confirmado** |
| 9. Probabilidades acumuladas | **ACUMULADAS**, `P(Y <= k)`, no de clase |
| 10. Monotonia garantizada | **NO**. Cada modelo es independiente y el **11,18 %** de las filas la viola. V20 la impone despues, con el maximo acumulado |

## 2. La inconsistencia cuantificada

No es solo una cuestion de etiquetas. Sobre las mismas 279 features y los mismos cinco folds:

| Especificacion | macro-F1 del blend | Violaciones de monotonia |
|---|---|---|
| **declarada** (multiclass 6, target binario) | **0,6097782** | 18.927 (11,18 %) |
| **binaria correcta** (`objective="binary"`) | 0,6091232 | 17.099 (10,10 %) |

La correlacion entre las dos CDF es de **0,998** en las cinco fronteras, luego el *ranking* es el mismo, pero la **calibracion** difiere hasta **0,33**. Y la especificacion declarada es **0,00066 mejor**. Luego **no es un bug que haya que arreglar**: es una version que funciona mejor, y se documenta y conserva.

## 3. El experimento: representacion compartida

V20 usa cinco modelos independientes con hiperparametros identicos. El comparador que ademas **elimina el problema de monotonia por construccion** es un **unico LightGBM multiclase de seis clases** del que se derivan las cinco CDF **por suma**: `P(Y <= k) = p_0 + ... + p_k`. Al ser una suma de probabilidades no negativas, **las cinco CDF son monotonicas siempre**.

| Candidato | macro-F1 | Delta vs V20 | Folds | IC95 | F1 13-17 | F1 18-24 | Rescates | Danos | Neto | Cambios | Estado |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A: V20, cinco independientes** | 0.6097782286309541 | - | - | - | 0,6818 | 0,5170 | - | - | - | - | CONTROL |
| **B: un modelo compartido** | 0.6096150258934702 | -0.0002 | 3/5 | [-0.0008, +0.0004] | 0.6823 | 0.5177 | 1,375 | 1,388 | -13 | 4,451 | **WRONG_OBJECTIVE** |

## 4. Monotonia

| | V20 | B compartido |
|---|---|---|
| violaciones | 18.927 (11,18 %) | **0** |
| garantizada | no, se impone despues | **si, por suma de probabilidades no negativas** |

## 5. Delta por fold

**[-0.002028, 0.000678, 0.000345, 0.001039, -0.00087]**
