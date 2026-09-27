# V19 · B1 — Boosting como tercer miembro de V12

## 1. Por que esta y las otras tres no

| Familia | Estado | Numero |
|---|---|---|
| Retrieval | **descartada sin ejecutar** | cobertura de cluster in-fold **0,00 %** por diseno del CV agrupado, luego no hay numero CV certificable; la etiqueta de duplicado exacto acierta **61,5 %** y el techo del pool literal es **+0,0010** (PH10, D-026) |
| Label cleaning | **descartada sin ejecutar** | PH11 dio **+0,0016** y PH18 **+0,0009**, ambos bajo el gate, y el criterio propuesto selecciona filas que el modelo no puede acertar, no etiquetas falsas |
| Cascada ordinal | **descartada sin ejecutar** | es D-037 con otro nombre: la formulacion probabilistica ya se midio en ORD-1 (**-0,0118**) y ORD-B (**-0,3057**) |
| **Boosting** | **esta es la que se ejecuta** | E-022 probo meta LightGBM sobre **102** columnas de la era V10 y dio aporte **0,0000**; nunca se probo sobre las **279** de la era V12, y V12 no contiene ningun arbol |

**La premisa de partida esta falsificada, y lo dice un artefacto del propio proyecto.** En P0' el especialista de frontera sobre las mismas 279 variables saca **0,7136 con `LogisticRegression`** y **0,7075 con `LinearSVC(0.2)`**: dos clases **lineales** con el mismo resultado. Si un modelo lineal extrae la frontera, un arbol no tiene ventaja mecanica sobre el, solo otra clase de funcion sobre el mismo input. El argumento que queda en pie es mas estrecho y es el que se mide: la era de las 279 variables nunca se probo con boosting.

## 2. Diseno

- **Arbol pre-registrado, sin busqueda**: {"objective": "multiclass", "num_class": 6, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 8, "max_depth": 3, "min_child_samples": 200, "colsample_bytree": 0.5, "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 1.0, "n_jobs": 32, "random_state": 42, "verbose": -1}
- **Target**: las **seis clases verdaderas**. Las 279 variables son features auxiliares. No se sustituye ninguna etiqueta por una prediccion.
- **OOF real**: el meta se ajusta en los cuatro folds de entrenamiento y predice el retenido. Nunca sus propias filas.
- **Pesos pre-registrados**: {"V12 (referencia)": [0.5, 0.5, 0.0], "sustituye MLP por arbol": [0.5, 0.0, 0.5], "tercero a partes iguales": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], "arbol con peso bajo": [0.4, 0.4, 0.2]}. No se ajustan sobre la validacion exterior.
- **Control de ruido**: el mismo arbol con las **filas de las variables barajadas**, de modo que la matriz deja de corresponder con las etiquetas. Todo lo demas identico. Si el arbol real no supera a su gemelo, el delta es ruido de seleccion.

**Fidelidad de la reconstruccion**: las dos cabezas propias dan 0.6011711 frente a los 0.6005017 del artefacto congelado, una brecha de 0.0006694. La matriz de 279 se tomo de `meta_features_train_v11.npy`, verificada bit a bit identica a `build_features()`.

## 3. Resultado

| Combinacion | macro-F1 | Delta | Media folds | Folds pos. | IC95 | 18-24\|25-34 | 25-34\|35-49 | F1 25-34 | Rescate | Dano | Neto | Adopta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V12 (referencia) | 0.6011711 | +0.0007 | 0.60117 | 2/5 | [-0.0002, +0.0016] | 0.4929 | 0.4965 | 0.4831 | 3,149 | 2,978 | 171 | **no** |
| sustituye MLP por arbol | 0.607221 | +0.0067 | 0.607221 | 5/5 | [+0.0055, +0.0079] | 0.5065 | 0.5012 | 0.495 | 6,014 | 4,847 | 1,167 | si |
| tercero a partes iguales | 0.6079388 | +0.0074 | 0.607945 | 5/5 | [+0.0064, +0.0085] | 0.5038 | 0.5013 | 0.4935 | 4,622 | 3,311 | 1,311 | si |
| arbol con peso bajo | 0.6059151 | +0.0054 | 0.605915 | 5/5 | [+0.0045, +0.0064] | 0.5004 | 0.5004 | 0.4904 | 3,900 | 2,930 | 970 | si |
| **CONTROL** filas barajadas | 0.5909687 | -0.0095 | 0.590962 | 0/5 | [-0.0108, -0.0083] | 0.4864 | 0.4876 | 0.4756 | 4,782 | 6,395 | -1,613 | - |
| **V12 (referencia)** | 0.6005017 | - | - | - | - | 0.4963 | 0.4974 | 0.4858 | - | - | - | - |

## 4. El control, que es la comparacion que decide

- **Arbol aislado**: 0.6072312
- **Su gemelo de ruido**: 0.1576842
- **Ventaja del arbol real**: **+0.4495470**
- **Delta de la mejor combinacion sobre su control equivalente**: **+0.016970**

Una ventaja negativa significa que el arbol real no es mejor que un arbol que no puede ver la relacion entre la fila y su etiqueta, y que por tanto **el delta no es informacion**.

## 5. Criterio de adopcion

| Criterio | Estado |
|---|---|
| supera_control | si |
| efecto_minimo | si |
| folds_positivos | si |
| ic95_excluye_cero | si |

**Decision: ADOPTAR.**
