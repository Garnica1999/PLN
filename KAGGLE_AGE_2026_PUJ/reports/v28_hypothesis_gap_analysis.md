# V28 - Gap analysis de las seis tecnicas

Clasificacion por **equivalencia funcional**, no por coincidencia de nombre. Cada veredicto lleva al lado la busqueda que lo sostiene y, cuando existe, el numero ya medido.

**Excepcion declarada:** este script queda excluido de su propia busqueda, porque sus patrones son literalmente los nombres que busca. Sin la exclusion, `OneVsOneClassifier` y `VotingClassifier` aparecerian con una coincidencia y se declararia existentes familias que no existen.

## Veredicto

| ID | Tecnica | Veredicto | Experimentos | Motivo en una linea |
|---|---|---|---|---|
| A | Meta-model bagging | **PARTIALLY_TESTED** | 1 | el bagging de filas ya esta activo dentro de cada arbol; A anade un segundo nivel |
| B | Compact consensus geometry | **ALREADY_CLOSED** | 0 | el bloque de incertidumbre se implemento y se midio: delta 0.0001 |
| C | Pairwise class decomposition | **GENUINELY_NEW** | 1 | one-vs-one no existe en el proyecto y los pares adyacentes concentran 60% de los errores |
| D | Dirichlet / evidence fusion | **FUNCTIONALLY_DUPLICATED** | 0 | es un cambio de combinador, y H04 y H06 estan cerrados con numero |
| E | Meta-model con incertidumbre explicita | **CLOSED_BY_DUPLICATION** | 0 | las mismas tres magnitudes que B, con la integracion de H06 |
| F | Randomized meta-subspace | **FUNCTIONALLY_DUPLICATED** | 0 | `colsample_bytree` ya es aleatorizacion de subespacio sobre todas las columnas |

**De seis tecnicas, 2 pueden ejecutarse** y 4 se cierran sin gastar un solo experimento. El presupuesto autorizado es de 8 y esta fase consumira 2.

## Por que B esta cerrada con un numero, no con una opinion

`scripts/advanced_04_incertidumbre.py` construye por miembro la **confianza** (probabilidad maxima), el **margen** (top-1 menos top-2) y la **entropia** de Shannon; entre miembros, el reparto de votos por clase, la pluralidad, el desacuerdo y la desviacion de scores por clase; y cuatro estadisticos agregados. Eso es un **superconjunto** del bloque compacto que B propone.

Y esta medido en `reports/tables/advanced_04_variantes.csv`:

| Variante | Features | macro-F1 |
|---|---|---|
| v09 (base) | 267 | 0.5897 |
| + bloque de incertidumbre completo | 318 | 0.5898 |

Delta: **0.0001**. Anadir 51 columnas de geometria del consenso no cambio el macro-F1. Un bloque mas pequeno no puede dar mas.

## Por que C es genuinamente nueva, y por que tiene un mecanismo medido

- `OneVsOneClassifier`: **0 ficheros**.

- `VotingClassifier`: **0 ficheros**.

- `ClassifierChain`: **0 ficheros**.

- `OneVsRestClassifier`: 8 ficheros, pero **siempre sobre modelos base**, nunca sobre el meta-modelo, y one-vs-rest no es one-vs-one.

La distincion con lo ya cerrado es de estructura, no de nombre. V20 entrena **cinco binarios anidados** de `clase <= k-1`, en los que cada uno mezcla todas las clases por debajo del umbral. C entrenaria **cinco binarios disjuntos**, uno por par adyacente, sin solape y sin restriccion de orden. No es CORAL, que comparte pesos e impone umbrales ordenados, ni cumulative-link.

Y el objetivo mecanico esta medido en la confusion del control V20:

| Par adyacente | Confusiones mutuas |
|---|---|
| 13-17|18-24 | 10,793 |
| 18-24|25-34 | 9,948 |
| 25-34|35-49 | 7,004 |
| 35-49|50-64 | 6,208 |
| 50-64|65-xx | 5,603 |

Los cinco pares adyacentes concentran **39,556 errores, el 59.9%** de los 66,005 errores de V20. La confusion es claramente bandada y C apunta exactamente a donde esta el error.

**Criterio de seleccion de pares, fijado antes de ejecutar:** pares con al menos 5000 confusiones mutuas en la confusion del control V20, fijada antes de ejecutar C. Pares que entran: `13-17|18-24`, `18-24|25-34`, `25-34|35-49`, `35-49|50-64`, `50-64|65-xx`. Se calcula sobre la confusion **out-of-fold del control**, que es informacion de train, y **no se toca** despues.

## Por que A es solo parcialmente nueva

`HIERBA` ya tiene `subsample=0.8` con `subsample_freq=1`. En LightGBM eso significa que **cada uno de los 300 arboles se ajusta sobre una muestra aleatoria del 80% de las filas**: el meta-modelo de V20 ya esta embolsado a nivel de arbol.

Lo que A anade es un **segundo nivel**: N ajustes independientes, promediados despues. Es un estimador distinto, asi que se ejecuta, pero el brief obliga a declarar el confundidor: el numero de estimadores pasa de 300 a 300 por N, luego parte de cualquier mejora podria ser capacidad y no reduccion de varianza.

## Por que F ya esta implementado

`HIERBA` ya tiene **`colsample_bytree=0.5`**. Cada arbol ve un subconjunto aleatorio de la mitad de las 279 columnas, **manteniendo todas las disponibles**. La distincion que el brief traza frente a V25 (V25 eliminaba columnas, F aleatoriza la exposicion) esta ya en la configuracion del champion. Y V27-04 lo midio como control: quitar `colsample_bytree` no cambia el numero de features que el modelo abre.

## Las cuatro cerradas, en una frase cada una

- **B**: el bloque se implemento y se midio, con delta 0.0001.

- **D**: `dirichlet` en el proyecto es el prior de Monroe para el EDA, no fusion de evidencia; y un pool es un combinador, que es H04 mas H06, cerrados.

- **E**: las mismas tres magnitudes que B, y la integracion es H06. La regla critica del punto 23 lo prohibe explicitamente.

- **F**: `colsample_bytree` ya aleatoriza el subespacio sobre todas las columnas.
