# V22 · Frontier audit — el espacio restante despues de V20

**V20 queda congelado**: LB **0,62689**. No se toca su configuracion, sus features, sus thresholds, su artefacto, su submission ni su hash.

## A. Baselines

| Modelo | macro-F1 | LB | SHA256 (16) |
|---|---|---|---|
| V12 | 0,6005017 | 0,62182 | ad09e4725c1e3066 |
| V19 | 0,6079388 | 0,62578 | 877908a745bbc01e |
| **V20** | **0,6096150** | **0,62689** | 374b3d31eabbe707 |

## B. Limitaciones de la evaluacion

**No existe evaluacion independiente disponible.** El proyecto si tiene holdout (`advanced_03_holdout_candidato.py`, 15 %, semilla 2026), pero **ha sido usado en V11, V12, V13 y confirmaciones posteriores**, y la fase 36 lo declara agotado. Luego toda seleccion se hace con el OOF de cinco folds, y eso hay que decirlo.

## C. Errores conjuntos

| Interseccion | Filas |
|---|---|
| V19 wrong & V20 wrong | **64.825** |
| V19 wrong & V20 right | 1.474 |
| V20 wrong & V19 right | 1.193 |
| nadie acierta | **64.825** |

V20 rescata 1.474 errores de V19 e introduce 1.193: **neto +281**. Y **64.825 filas (38,3 % del train) las equivocan ambos**, luego **esa es la masa sobre la que cualquier mejora futura tiene que trabajar**.

| Clase | ambos wrong | al menos uno |
|---|---|---|
| 13-17 | 9.653 | 66.476 |
| 18-24 | 12.703 | 66.562 |
| 25-34 | 14.913 | 66.514 |
| 35-49 | 12.813 | 66.545 |
| 50-64 | 8.255 | 66.459 |
| 65-xx | 6.488 | 66.431 |

## D. Diversidad real

| Modelo | Errores | Jaccard vs V20 | Rescata a V20 |
|---|---|---|---|
| LogReg 279 | 69.098 | 0,826 | 4.917 |
| MLP(64) 279 | 67.781 | 0,820 | 5.745 |
| arbol V19 | 66.443 | 0,892 | 3.562 |
| V12 ensamble | 67.610 | 0,867 | 3.982 |
| **NB-SVM word** | 80.434 | **0,611** | **10.462** |

**El modelo mas complementario con V20 es NB-SVM word, que rescata 10.462 de sus 66.018 errores (15,85 %) — pero ya es un miembro de V20.** Los modelos buenos son altamente redundantes: los tres primeros comparten mas del 82 % de sus errores con V20. **No hay un modelo historico con informacion complementaria suficiente como para justificar una combinacion nueva.**

## E. Literatura 2024-2026

| Paper | Ano | Metodo | Legal en AGE |
|---|---|---|---|
| Joulin et al., Bag of Tricks | 2016 | fastText, n-gramas con hashing | **ILLEGAL**: usa vectorizacion aprendida/hashing |
| Zhang et al., char-CNN | 2015 | CNN a nivel de caracter | **ILLEGAL**: deep learning |
| Conneau et al., VDCNN | 2016 | CNN muy profunda a caracteres | **ILLEGAL**: deep learning |
| O'Connor et al. | 2024 | revision de metodos de edad y genero desde Twitter | si, como referencia de mecanismos |
| Suh | 2024 | prediccion difusa de diferencia de edad | si, como adaptacion conceptual |
| TextAge | 2024 | dataset curado de clasificacion de edad | si, como referencia de representaciones |

La literatura confirma que **las representaciones a nivel de caracter y las convolutional siguen siendo el estandar no-transformer**, pero ambas son **ilegales en AGE**. Los mecanismos conceptuales (n-gramas, rasgos lexicos, morfologia) si son legales y **ya estan probados**.

## F. El filtro legal

**ILLEGAL en AGE**: embeddings, transformers, LLM, corpus externos, modelos preentrenados, vectorizaciones aprendidas, **hashing**, deep learning. No se asume ninguna excepcion.

## G. La respuesta a la pregunta central

> **¿Que informacion todavia no esta siendo utilizada por V20, y que evidencia tenemos de que esa informacion existe realmente en los datos?**

**Tres fuentes, y solo una tiene evidencia de existir:**

1. **Las 64.825 filas que V19 y V20 equivocan juntas (38,3 % del train).** Es la unica masa grande y medida. **Pero ya esta agotada**: cuatro subfamilias de la ordinal (reconstruccion, pesos, B1, compartida) fallaron ahi, y la razon estructural es que **las cinco CDF comparten las mimas 279 features**.
2. **La descalibracion de las cinco CDF. Evidencia de que NO existe**: entre -0,0003 y +0,0002. No hay nada que recalibrar.
3. **Las representaciones. Evidencia de que NO estan agotadas, pero son ilegales**: char-level, byte-level, subword. **No se pueden usar.**

Luego la respuesta honesta es: **no se ha encontrado una fuente nueva con evidencia de existir.** La unica fuente grande medida (los errores compartidos) esta agotada, y las que quedan son estructuralmente ilegales.
