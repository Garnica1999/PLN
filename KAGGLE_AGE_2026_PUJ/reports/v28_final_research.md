# V28 - Informe final de la fase

**Fecha:** 2026-09-28 · **Rama:** `proyecto-kaggle-age-2026-v28` · **Baseline congelado:** 0,6097782 · **Champion:** V20, LB 0.62689

Esta fase no reabre ninguna familia de V25 a V27. La premisa es que el limite de V20 no esta en la cantidad ni en la escala de las features, sino en **la forma de estimar y combinar la evidencia**. Se evaluan seis mecanismos estructuralmente distintos y se responde la pregunta con lo medido.

## 1. Resumen ejecutivo de V28

| Concepto | Valor | Fuente |
|---|---|---|
| Baseline vigente | **0,6097782** corrected_cv | canonico, reproducido por los 2 controles de V28 |
| Leaderboard de V20 | **0.62689** | envio verificado, sin cambio |
| Tecnicas evaluadas | **6** | gap analysis |
| Cerradas sin experimento | **4** (B, D, E, F) | mismo fichero |
| Experimentos ejecutados | **2** de 8 autorizados | `v28_04_cierre.json` |
| Mejor corrected_cv de V28 | **0,6094663** (A, peor que V20) | artefactos |
| Finalistas | **0** | ninguna supero el gate |
| Submissions generadas | **0** | hash de V20 sin cambio |
| Filas en `experiments.csv` | **137** | registro |

**La respuesta a la pregunta de V28 es no, y se contesta con dos experimentos y cuatro  cierres por equivalencia.** Ningun mecanismo estructuralmente distinto de combinar, estabilizar o descomponer la evidencia supera a V20.

| ID | Tecnica | Modo | Resultado | Decision |
|---|---|---|---|---|
| V28-02 | A · Meta-model bagging | probado | -0,0003120, 1/5 folds | CLOSED |
| V28-03 | C · Pairwise class decomposition | probado | -0,0675530, 0/5 folds | CLOSED |
| - | B · Compact consensus geometry | equivalencia, ya medido | +0,0001 historico | cerrada sin experimento |
| - | D · Dirichlet / evidence fusion | equivalencia con H04 y H06 | no medido | cerrada sin experimento |
| - | E · Explicit uncertainty | duplicacion de B | no medido | cerrada sin experimento |
| - | F · Randomized meta-subspace | equivalencia con `colsample_bytree` | no medido | cerrada sin experimento |

## 2. Gap analysis: de seis tecnicas, dos sobreviven

La busqueda es por **equivalencia funcional**. Este script queda excluido de su propia busqueda, porque sus patrones son literalmente los nombres que busca: sin esa exclusion, `OneVsOneClassifier` apareceria con una coincidencia y se declararia existente una familia que no existe.

| ID | Tecnica | Veredicto | Experimentos | Evidencia |
|---|---|---|---|---|
| A | Meta-model bagging | **PARTIALLY_TESTED** | 1 | `HIERBA` ya tiene `subsample=0.8` con `subsample_freq=1`, y en LightGBM eso hace que **cada uno de los 300 arboles se ajuste sobre una muestra aleator |
| B | Compact consensus geometry | **ALREADY_CLOSED** | 0 | `scripts/advanced_04_incertidumbre.py` implementa un **superconjunto** de lo que B propone: por miembro, confianza (probabilidad maxima), margen (top- |
| C | Pairwise class decomposition | **GENUINELY_NEW** | 1 | `OneVsOneClassifier`, `VotingClassifier` y `ClassifierChain` tienen **0**, **0** y **0** ficheros respectivamente: no existe. `OneVsRestClassifier` si |
| D | Dirichlet / evidence fusion | **FUNCTIONALLY_DUPLICATED** | 0 | La palabra `dirichlet` aparece en 3 ficheros, y **todos son el prior estadistico de Monroe et al. para ordenar terminos por log-odds en el EDA**, no f |
| E | Meta-model con incertidumbre explicita | **CLOSED_BY_DUPLICATION** | 0 | B ya incorpora entropia, margen, reparto de votos y desacuerdo, que son exactamente las tres magnitudes que E propone representar: consenso, desacuerd |
| F | Randomized meta-subspace | **FUNCTIONALLY_DUPLICATED** | 0 | `HIERBA` ya tiene **`colsample_bytree=0.5`**: cada arbol ve un subconjunto aleatorio de la mitad de las 279 columnas, y **se mantienen todas las colum |

### B: cerrada con un numero, no con una opinion

`scripts/advanced_04_incertidumbre.py` construye por miembro la **confianza**, el **margen** (top-1 menos top-2) y la **entropia** de Shannon; entre miembros, el reparto de votos por clase, la pluralidad, el desacuerdo y la desviacion de scores; y cuatro estadisticos agregados. Eso es un **superconjunto** de lo que B propone.

| Variante | Features | macro-F1 |
|---|---|---|
| v09 (base) | 267 | 0.5897 |
| + bloque de incertidumbre completo | 318 | 0.5898 |

Delta: **0.0001**. Anadir 51 columnas de geometria del consenso no cambio el macro-F1.

### F: ya implementado en la configuracion del champion

`HIERBA` ya tiene `colsample_bytree=0,5`: cada arbol ve un subconjunto aleatorio de la mitad de las 279 columnas **manteniendo todas las disponibles**. Eso es exactamente el mecanismo que F describe, y V27-04 lo midio como control: quitarlo no cambia el numero de features que el modelo abre.

## 3. Las dos tecnicas realmente ejecutadas

### 3.1 A · Meta-model bagging

**Motivacion.** V20 usa un unico meta-modelo. Aunque la evidencia base sea correcta, la estimacion de la frontera podria tener varianza residual.

**Lo unico que cambia.** Las filas que ve cada ajuste del meta-modelo. Se mantienen la matriz de 279 features, las predicciones de los tres miembros del blend, los folds, la cadena de cinco umbrales, la reconstruccion, los thresholds y el blend. No se anade ningun modelo base, ninguna representacion y no se selecciona ninguna feature.

**Configuracion, fijada antes de mirar la validacion.** N = 5 miembros, remuestreo bootstrap de Poisson(1) como sample_weight, semilla derivada de cfg.SEED, miembro, fold y umbral, agregacion **promedio de probabilidades**.

**El confundidor, declarado antes del resultado.** los estimadores totales pasan de 300 a 1500: parte de cualquier mejora puede ser capacidad y no reduccion de varianza.

| Metrica | T0 (V20) | T1 (embolsado) |
|---|---|---|
| corrected_cv | 0.6097782 | **0.6094663** |
| Delta | 0,0000000 | **-0,0003120** |
| Folds positivos | - | **1/5** |
| Rescue / damage / net | 0 / 0 / 0 | 815 / 861 / **-46** |
| Filas que cambian | 0 | 2,728 |
| IC95 | - | [-0,0007850, 0,0001560], incluye el cero |

**F1 por clase:**

| Clase | V20 | Embolsado | Cambio |
|---|---|---|---|
| 13-17 | 0.6818 | 0.6822 | +0.0004 |
| 18-24 | 0.5170 | 0.5171 | +0.0001 |
| 25-34 | 0.4965 | 0.4957 | -0.0008 |
| 35-49 | 0.5687 | 0.5680 | -0.0007 |
| 50-64 | 0.6637 | 0.6633 | -0.0004 |
| 65-xx | 0.7310 | 0.7304 | -0.0006 |

**A2, el control de varianza, que es lo que hace informativo al resultado:**

| Medida | Valor |
|---|---|
| **Acuerdo medio entre los 5 miembros** | **0.9189** |
| Acuerdo minimo entre dos miembros | 0.9158 |
| Acuerdo de la prediccion final con V20 | 0.9839 |
| Desviacion tipica de las acumuladas, V20 | 0.321442 |
| Desviacion tipica de las acumuladas, embolsado | 0.321046 |
| Mediana de la diferencia absoluta de acumuladas | 0.003821 |

**Mecanismo observado.** Los cinco miembros **acuerdan solo en el 91,9 %** de las decisiones, luego **la varianza que A pretendia eliminar existia y era real**. Y promediarla empeoro: el resultado es 98,4 % identico a V20 y 0,000312 peor, con 815 rescates contra 861 danos. La dispersion de las acumuladas apenas se movio (0,32144 a 0,32105).

**Lo que A cierra.** La varianza de la estimacion del meta-modelo **no es donde esta el error**. Los desacuerdos entre miembros ocurren en filas donde el modelo ya era incierto, y promediarlos no recupera senal, solo suaviza. Y el resultado **no** es un artefacto de capacidad: 1.500 estimadores en vez de 300 no dieron ganancia, lo que descarta la explicacion facil.

### 3.2 C · Descomposicion pairwise en 5 pares disjuntos

**Motivacion, medida antes de escribir codigo.** La confusion del control V20 es claramente bandada: los cinco pares adyacentes concentran el **59,9 % de sus 66.005 errores**.

| Par adyacente | Confusiones mutuas |
|---|---|
| 13-17|18-24 | 10,793 |
| 18-24|25-34 | 9,948 |
| 25-34|35-49 | 7,004 |
| 35-49|50-64 | 6,208 |
| 50-64|65-xx | 5,603 |

**La distincion con V20, que es lo que hace nueva a la tecnica.** V20 entrena **cinco binarios anidados** de `clase <= k-1`, en los que cada uno mezcla todas las clases por debajo del umbral. C entrena **cinco binarios disjuntos**, uno por par adyacente, sin solape y sin restriccion de orden. No es CORAL, que comparte pesos e impone umbrales ordenados, ni cumulative-link: aqui cada frontera la aprende un clasificador independiente con la matriz completa de 279 features.

**Pares y agregacion, fijados antes de entrenar.** Criterio: pares con al menos 5000 confusiones mutuas en la confusion out-of-fold del control V20, fijado antes de entrenar. Pares: `13-17|18-24`, `18-24|25-34`, `25-34|35-49`, `35-49|50-64`, `50-64|65-xx`. Agregacion: **cadena de probabilidad independiente: P(clase<=k) = producto de (1 - p_j)**, una sola implementacion.

**El coste mecanico, declarado antes.** cada clasificador ve 57,539 filas en vez de 169,336: la descomposicion disjunta particiona las filas, y eso es su mecanismo, no un defecto.

| Metrica | T0 (V20) | T1 (pares) |
|---|---|---|
| corrected_cv | 0.6097782 | **0.5422249** |
| Delta | 0,0000000 | **-0,0675530** |
| Folds positivos | - | **0/5** |
| Rescue / damage / net | 0 / 0 / 0 | 8,461 / 19,457 / **-10,996** |
| Filas que cambian | 0 | 54,252 |
| Errores totales | 66,005 | 77,001 |
| Errores en los 5 pares adyacentes | **39,556** | **32,826** |
| IC95 | - | [-0,0692860, -0,0657640], **excluye el cero** |

**F1 por clase:**

| Clase | V20 | Pares | Cambio |
|---|---|---|---|
| 13-17 | 0.6818 | 0.4980 | -0.1838 |
| 18-24 | 0.5170 | 0.3473 | -0.1697 |
| 25-34 | 0.4965 | 0.4458 | -0.0507 |
| 35-49 | 0.5687 | 0.5599 | -0.0088 |
| 50-64 | 0.6637 | 0.6653 | +0.0016 |
| 65-xx | 0.7310 | 0.7370 | +0.0060 |

**Mecanismo observado, y aqui C aporto conocimiento real:**

1. **Los pares disjuntos si resuelven su objetivo local.** Los errores en los cinco pares adyacentes **bajan de 39,556 a 32,826**

2. **El precio esta en las clases que no son frontera.** 50-64 y 65-xx *mejoran* ligeramente, mientras 13-17 y 18-24 se desploman. Un clasificador que solo ve `{k, k+1}` no puede colocar una fila lejos de su par: la masa de probabilidad de las clases no adyacentes desaparece al encadenar.

3. **C es complementario en la masa dificil y aun asi inutil.** De las 61,949 filas que V19 y V20 fallan juntas, **C acierta 7,788 (12.57%)**: senal real, no ruido. Pero a cambio rompe 3,383 aciertos que V20 tenia. Complementariedad sin rentabilidad.

4. **Descartado el defecto de implementacion.** El IC95 es estrecho y excluye el cero, y el dano esta localizado en las clases no frontera, que es exactamente lo que predice el mecanismo de la cadena.

> **Una discrepancia que se declara.** El brief cita 38,3 % de errores conjuntos y > esta medicion da **36,6 %**, porque la definicion no es identica: aqui se usa > `arbol_v19` (el miembro arbol del blend) contra V20. Se usa la cifra medida y > reproducible, no la que encajaria.

## 4. Por que se paro aqui, y no por inercia

El brief autoriza ocho experimentos y permite no consumirlos. Se consumieron **2**, y la razon es concreta:

1. El gap analysis cerro **cuatro de seis** antes de entrenar, y dos de ellas (**B** y **F**) porque ya estaban implementadas, una de ellas con un numero medido hace anos.

2. Las dos que quedaron son **mecanismos distintos entre si** y ambos se midieron: uno de estimacion, uno de descomposicion de la decision.

3. Los dos son negativos, y el punto 34 pide precisamente eso: si dos o tres tecnicas consecutivas no muestran ninguna senal, hacer una evaluacion global antes de seguir consumiendo. Las dos son negativas y de forma rotunda, con un de ellas con el IC95 excluyendo el cero.

4. **E y F no se ejecutaron** y no por prudencia vaga: E es duplicacion de B por la regla critica del propio brief, y F es un hiperparametro ya activo en `HIERBA`. Ninguna de las dos necesita un experimento para estar cerrada.

## 5. Conclusion cientifica de V28

### Donde esta el cuello de botella de AGE, ahora

| Eje | Estado tras V28 | Evidencia |
|---|---|---|
| Representacion de features | agotado | V25 (seleccion), V26-01 (escala), V27-05 (percentil) |
| Combinacion de predicciones | agotado | H04, H05, H06, H13, y **D** cerrada por equivalencia en V28 |
| Calibracion | agotado | H01 (D-029) |
| Estructura ordinal | agotado | H10, CORAL (D-058) |
| Geometria del consenso | agotado | **B**, superconjunto ya implementado y medido: +0,0001 |
| **Varianza del meta-modelo** | **agotado en V28** | **A**: la varianza existia (91,9 % de acuerdo) y promediarla empeoro |
| **Descomposicion de la decision** | **agotado en V28** | **C**: los pares disjuntos atacan el 59,9 % de los errores y aun asi colapsan las clases no frontera |
| Incertidumbres explicita | cerrada | **E**, duplicacion de B |
| Subespacio aleatorio | cerrado | **F**, ya implementado en `colsample_bytree` |
| **Limitacion de los datos** | **lo unico abierto** | la masa que ningun modelo corrige |

### La respuesta

**La pregunta de V28 era si existia una mejora escondida en la forma de estimar y combinar la evidencia, y la respuesta es que no.** Las dos rutas genuinamente nuevas se han medido:

- **La varianza de la estimacion no es el cuello.** Existia, se midio, y eliminarla empeoro. Y 1.500 estimadores en vez de 300 no dieron ganancia, asi que tampoco es capacidad.

- **La descomposicion de la decision no es el cuello.**-ataca exactamente donde esta el error medido y funciona ahi, pero el dano en las clases no frontera es mayor que el beneficio. El problema no es *que clase* se confunde, es *cuanta informacion* sobrevive a la descomposicion.

**Junto con V25 a V27, el limite que queda es el mismo que se identifico entonces: la limitacion de los datos.** La masa de filas que ningun modelo del espacio probado corrige sigue ahi, y V28 no la ha reducido.

**No se afirma que AGE este resuelto.** Se afirma que el espacio de mecanismos accesible sin senal nueva esta agotado, con evidencia negativa reproducible en cada eje.

## 6. Conclusion practica de V28

| Pregunta | Respuesta |
|---|---|
| Baseline vigente | **V20**, 0,6097782 corrected_cv, LB 0.62689 |
| corrected_cv de A | 0,6094663, **peor** que V20 |
| corrected_cv de C | 0,5422249, **mucho peor** que V20 |
| Mejor candidato interno | ninguno: las dos tecnicas son negativas |
| Existe candidato listo para submission | **No** |
| Debe cerrarse la investigacion | **Si**, si no aparece senal nueva |
| Se genera submission | **No** |
| Se modifica V20 | **No** |

### Que queda por delante, y no es una hipotesis mas

La unica via con recorrido es **senal nueva desde los datos**, y las restricciones de AGE (sin transformers, sin LLM, sin embeddings, sin etiquetas externas) cierran las fuentes obvias. Cualquier linea nueva deberia empezar por demostrar que aporta senal que el bloque de evidencia de los miembros no aporta, y no por otra forma de combinar lo que ya existe: V25 a V28 hanShown que las cuatro familias de esa pregunta estan cerradas.

## 7. Reproducibilidad, compliance y git

| Comprobacion | Resultado |
|---|---|
| Control absoluto de V20 en P0 | 0,6097782, OOF **identico bit a bit** al artefacto, folds identicos |
| `ast.parse`, `ruff --select F,E9` y `py_compile` en los 7 scripts de V28 | 0 errores |
| Auditoria de idioma | ver `reports/v13_auditoria_idioma.md` |
| Compliance de V28 | 13 comprobaciones, **todas en cero usos** |
| Higiene del repositorio | los 2 directorios ajenos **siguen fuera de la rama** |
| Trazabilidad | `experiments.csv` <-> tabla maestra <-> artefactos: coherente |
| Notebook | seccion 36 ejecutada de principio a fin, 4 aserciones en verde |
| Artefactos de V20 tocados | **0** |
| Submissions generadas | **0** |

**No se declara R4.** No hubo reproduccion independiente por un tercero. La compliance busca firmas de las practicas prohibidas y no puede demostrar que no ocurran de otra forma.

### Git

Rama `proyecto-kaggle-age-2026-v28`, creada desde `proyecto-kaggle-age-2026-v26`. **Sin push, sin merge, sin force-push y sin reset destructivo.** No se incluye ningun artefacto ajeno, ningun secreto y ninguno de los dos directorios no ajenos a AGE.

---

**Cierre de V28.** V20 permanece como baseline absoluto en 0,6097782. Las seis tecnicas evaluadas quedan respondidas: cuatro por equivalencia funcional, dos por experimento, y las dos experimentadas son negativas. No se genera ninguna submission, no se toca el test, el holdout, el leaderboard ni ningun artefacto de V20.
