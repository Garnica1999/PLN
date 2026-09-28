# AGE - Informe completo de la investigacion, de V20 al cierre de V30

**Fecha:** 2026-09-30 · **Rama:** `proyecto-kaggle-age-2026-v30` · **Baseline congelado:** 0,6097782 · **Champion:** V20, LB 0.62689

Reconstruye toda la trayectoria de AGE desde V20 hasta el cierre de V30, sin omitir ninguna fase negativa ni ninguna positiva. Una sola cadena de fuentes: V20 a V27 las genera `v27_10_informe_completo.py`, V28 `v28_07_informes.py` y V29 y V30 este script.

## 1. Resumen ejecutivo

| Concepto | Valor | Fuente |
|---|---|---|
| Baseline final | **0,6097782** corrected_cv | canonico del proyecto, reproducido por los 4 controles |
| Mejor corrected_cv | **0,610051** (H03, MARGINAL) | `experiments/v27_05_h03_percentil.json` |
| Mejor score verificado en leaderboard | **0.62689** (V20) | envio verificado de V20 |
| Mejor candidato validado internamente | H03, MARGINAL, no finalista | su IC95 incluye el cero y esta 11 veces bajo el gate |
| Numero de experimentos | **139** filas en `experiments.csv` | registro |
| Familias evaluadas desde V20 | **15** | `v26_hypothesis_gap_analysis.csv` |
| Familias cerradas | **14** | mismo fichero, columna `status` |
| Familias marginales | **1** | idem |
| Familias promising / adopted | **0 / 0** | idem |
| Finalistas | **0** | ninguna hipotesis supero el gate |
| Submissions generadas en V26 y V27 | **0** | hash de V20 sin cambio |
| Features del meta-modelo | **279**, de las que el boosting abre **7** | `v27_04_concentracion.json` |

**Lo primero que hay que decir:** el unico experimento con resultado positivo de V26 y V27 es H03, con delta 0,000272 y un IC95 que **incluye el cero**. Ninguna hipotesis supero el gate de +0,003 y 4/5 folds. V20 sigue siendo el champion y no se ha generado ninguna submission nueva.

**Lo segundo, y es el hallazgo de la fase:** el meta-modelo de V20 divide por **7 de sus 279 features**, todas del bloque de evidencia de los miembros, y **0 de 35 densas, 0 de 87 estructuradas, 0 de 73 label-informed y 0 de 6 NB-SVM**. Eso explica de una sola vez por que fallaron V25, V26-01 y H11.

## 2. Cronologia de V20 a V27

### V20 - el champion que sigue vigente

corrected_cv **0,6097782**, leaderboard **0.62689**, hash de submission `2c964af78c90bf08...`. V20 **no es la cadena de umbrales sola**: es un blend de cuatro cuartos, `0.25*logreg + 0.25*mlp + 0.25*arbol_v19 + 0.25*cadena`. La cadena por si sola da **0,6046893**. Esta distincion confundio a V22, V23, V25 y V26, y queda registrada en `experiments/v25_00_inventario.json` como `cadena_sin_blend`.

### V21 - B1 no se puede mejorar cambiando de clasificador

Cierre de `B1_SPECIALIST` y del modelo compartido (D-056). La busqueda de un clasificador mejor para la parte especialista se agota: cambiar de familia de modelo no mejora.

### V22 - frontier audit y coste ordinal

El frontier audit revisa 15 familias por equivalencia funcional. El coste ordinal **destruye** el macro-F1 (LOFO agrupado,Decision `NO_SIGNAL`). La familia ordinal queda cerrada en sus subfamilias (D-057).

### V23 - CORAL y la cadena de generadores

CORAL comparte pesos entre las cinco fronteras ordinales. D-058 mide dos cosas a la vez: compartir pesos **funciona** (+0,014696 frente a un control lineal) y la no linealidad **domina** (-0,140773). Ademas CORAL resuelve de verdad la monotonia, 0 violaciones frente a 18.927 en V20, y aun asi el macro-F1 cae. La conclusion es dura: la consistencia ordinal no era el cuello de botella. En paralelo, D-059 establece que la fuente de verdad del notebook es una **cadena de 5 scripts con un eslabon por fase**, y V18 a V22 nunca tuvieron eslabon.

### V24 - la causa del notebook incompleto era estructural

D-059 reconcilia el conteo: 121 a 136 celdas, 33 secciones, V12 a V23 presentes. `experiments.csv` tenia **3 cabeceras**, dos espurias ocupando `experiment_id` dos veces: el conteo real era 128, no 130.

### V25 - la seleccion no elimina ruido, elimina senal

Mutual information con K = 139 sobre las 279 features: corrected_cv **0,6091492**, delta **-0,000629**, 1 de 5 folds, neto **-101**. V25 quedo cerrado y con el mapa de las 279 features por bloque, que es la base de todo lo que vino despues.

### V26 - la evidencia del ensemble es invariante a su escala

H02, z-score de las 84 columnas de evidencia: delta **-0,0000010**, 2 de 5 folds, 1.809 filas cambian, neto **+3** sobre 169.336. La invariancia teorica de los arboles habria cerrado H02 sin gastar presupuesto, y resulto ser **falsa**: un z-score afin cambia la prediccion del arbol bit a bit, con una diferencia maxima de 0,219 sobre 135.468 filas. Por eso se midio antes de decidir (D-061).

### V27 - las cuatro hipotesis que quedaban

- **H12** (fiabilidad por modelo y clase): delta **exactamente 0,000000**, **0 filas cambian**. CLOSED.

- **H11** (restricciones monotonas nativas): delta **-0,000355**, 2/5 folds, neto -55. CLOSED.

- **H14** (reescalar la salida del blend): **cerrada por equivalencia funcional**, sin entrenar. Las cuatro formas posibles son H01, H02, H03 y un cambio de combinador, y las cuatro estan cerradas con numero.

- **H03** (percentil): delta **0,000272**, 3/5 folds, neto +51, 1,764 filas cambian. **MARGINAL**, el unico resultado positivo de la fase, y por debajo del gate.

Y un hallazgo que no es experimento: el meta-modelo de V20 divide por **7 de 279 features**. Se controla que no sea efecto del submuestreo: sin `colsample_bytree` y sin `subsample` sigue siendo el mismo numero.

## 3. Tabla maestra historica de experimentos

Construida desde `reports/v26_master_results.csv` y los artefactos JSON, no escrita a mano. La columna `bootstrap` es el IC95 del delta contra el control.

| ID | Fecha | Hipotesis | corrected_cv | Delta | F1 folds | Bootstrap | Rescue | Damage | Net | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| V20 | 2026-09-27 | blend de cuatro cuartos sobre la cadena de c | 0,6097782 | +0.000000 | n/d | referencia | 0 | 0 | +0 | CHAMPION. LB 0.62689 |
| V26-01 | 2026-09-28 | H02 | 0,6097770 | -0.000001 | 0,61306 0.61217 0.61040 0.60950 0.60379 | no persistido | 555 | 552 | +3 | H02 CLOSED |
| V27-01 | 2026-09-28 | el meta-modelo se beneficia de conocer expli | 0,6097782 | +0.000000 | 0,61320 0.61208 0.61045 0.60884 0.60435 | [0,000000, 0,000000] | 0 | 0 | +0 | H12 CLOSED |
| V27-02 | 2026-09-28 | certainas relaciones entre evidencia y clase | 0,6094234 | -0.000355 | 0,61234 0.61150 0.61109 0.60898 0.60324 | [-0,000803, 0,000127] | 752 | 807 | -55 | H11 CLOSED |
| V27-05 | 2026-09-28 | la evidencia de los modelos base tiene una e | 0,6100505 | +0.000272 | 0,61253 0.61255 0.61165 0.60966 0.60390 | [-0,000109, 0,000639] | 564 | 513 | +51 | H03 MARGINAL |

Los **folds** de V20 no se listan porque es la referencia, no un candidato: sus cinco folds se comparan contra si mismo. Los cuatro candidatos sí tienen los cinco.

## 4. Familias: cerradas, marginales y abiertas

| ID | Familia | Estado | Motivo del cierre | Mejor resultado | Evidencia |
|---|---|---|---|---|---|
| H02 | probability normalization (afin) | CLOSED (tested) | z-score de las 84 columnas de evidencia: delta -0,000001, 2/5 folds, IC95 [-0,000375, +0,000397], 1.809 filas cambian ... | 0,6097770 | gap analysis: grep de `StandardScaler` devuelve 67 scr |
| H03 | rank transformation / percentil | MARGINAL (tested) | delta +0,000272, 3/5 folds, IC95 [-0,000109, +0,000639] que incluye el cero, y un gate de +0,003: el efecto es once ve... | 0,6100505 | gap analysis: los 8 scripts que contienen `percentile` |
| H12 | OOF reliability modeling | CLOSED (tested) | la fiabilidad por modelo y clase es una tabla constante por fila, y una columna constante no puede generar splits en n... | 0,6097782 (identico al control) | gap analysis: `reliability` y `calibracion_error` no e |
| H14 | rank transformation DENTRO del blend | CLOSED (closed by functional equivalence) | reescalar la salida del blend es, caso por caso, H02, H03, H01 o un cambio de combinador, y los cuatro estan cerrados ... | no se midio, por equivalencia | gap analysis: ningun script normaliza las salidas de l |
| H11 | monotone constraints en LightGBM | CLOSED (tested) | delta -0,000355, 2/5 folds, IC95 [-0,000803, +0,000127]. Las 42 features con relacion monotona defendible son probabil... | 0,6094234 | gap analysis: `monotone_constraints` devuelve **cero** |
| H01 | probability calibration | CLOSED | Medido y cerrado: la recalibracion de las salidas no aporta sobre el macro-F1 | no se midio en esta fase | gap analysis: grep de `isotonic|calibracion|temperatur |
| H04 | model weighting | CLOSED | Medido tres veces con tres familias distintas, ninguna mejora el global | no se midio en esta fase | gap analysis: v16_arquitectura.py midio `peso_aprendid |
| H05 | diversity weighting | CLOSED | La complementariedad es real y grande pero no identificable, segun D-047 | no se midio en esta fase | gap analysis: 29 scripts contienen `diversidad` o `com |
| H06 | stacking de segundo nivel | CLOSED | V20 ya ES el segundo nivel: las 84 columnas de miembros son la salida de un apilado previo | no se midio en esta fase | gap analysis: 72 scripts contienen `stacking` o `meta` |
| H07 | interaction terms | CLOSED | Medido: ampliar la matriz con interacciones no mejora | no se midio en esta fase | gap analysis: 12 scripts contienen `interaccion` o `in |
| H08 | nonlinear meta-model | CLOSED | Ninguna clase de funcion no lineal supero al boosting; ExtraTrees tambien quedo 0,0236 debajo (D-053) | no se midio en esta fase | gap analysis: 9 scripts contienen meta-modelos no line |
| H09 | class-specific meta models | CLOSED | Cerrado con numero en D-037 y D-047 | no se midio en esta fase | gap analysis: 66 scripts; D-037 cerro nueve hipotesis  |
| H10 | ordinal constraints | CLOSED | La familia ordinal esta cerrada en sus cinco subfamilias | no se midio en esta fase | gap analysis: 100 scripts; V20 las impone con maximo a |
| H13 | residual correction | CLOSED | La destilacion de V18 es la forma canonica de este mecanismo y cerro con 0 adopciones | no se midio en esta fase | gap analysis: 20 scripts contienen `residual` o `resid |
| H15 | supervised feature selection | CLOSED | Cerrado: la seleccion no elimina ruido, elimina senal, y las probabilidades de los miembros son lo que el MI ordena alto | no se midio en esta fase | gap analysis: V25 midio mutual information top 50 por  |

### Recuento por estado

- **CLOSED**: 10
- **CLOSED (tested)**: 3
- **MARGINAL (tested)**: 1
- **CLOSED (closed by functional equivalence)**: 1

**Ninguna familia queda abierta.** Las quince estaban abiertas al empezar V26 o ya estaban cerradas con numero; las cinco genuinamente abiertas se han medido o cerrado por equivalencia en V26 y V27.

### Una familia negativa no es investigacion fallida

Cada cierre responde a una pregunta concreta, y esa pregunta sigue cerrada:

- **H15, seleccion supervisada**: se pregunto si podar elimina ruido. No: la seleccion **elimina senal**, y sobreviven con frecuencia 1,000 las probabilidades del ensamble.

- **H02, escala de la evidencia**: se pregunto si parte de la senal perdida esta en las diferencias de escala entre miembros. No: la evidencia es invariante a su escala en todo lo que mide la macro-F1.

- **H11, monotonia dentro del modelo**: se pregunto si el boosting gasta capacidad en relaciones no plausibles. No: y el motivo es mas fuerte que el resultado, las features con relacion defendible no son las que el modelo abre.

- **H12, fiabilidad por miembro y clase**: se pregunto si el meta-modelo puede aprovechar conocer la fiabilidad. No, en esa formulacion: es una constante por fila.

## 5. La historia de V20: por que sigue siendo el baseline

### Que contiene

Un meta-modelo sobre **279 features**, de las que 78 son predicciones de los miembros del ensamble y 6 de NB-SVM, mas una cadena de cinco umbrales con maximo acumulado y un blend de cuatro cuartos con los tres miembros historicos.

### Evidencia fuerte frente a evidencia debil pero complementaria

Esta distincion gobierna todo lo que se ha intentado despues:

- **Evidencia fuerte**: las probabilidades por clase de los miembros del ensamble. Tienen el mutual information mas alto y sobreviven a la seleccion con frecuencia 1,000. Son las que el meta-modelo **efectivamente usa** para dividir.

- **Evidencia debil pero complementaria**: las 73 label-informed, las 87 estructuradas y las 35 densas. V19 se construyo sobre las 73, y aportaron su contribution, pero **el boosting de V20 no las abre para dividir**. Sobreviven porque el blend las pondera a traves de la prediccion del arbol de V19, no porque el meta-modelo las lea.

### Por que V25 no mejoro podandolo

Porque lo que se podo no es lo que se usaba. V25 quito 139 de las 279 features por mutual information, y de las eliminadas solo 7 estaban en uso. La seleccion no elimino ruido: elimino senal de la fila que el modelo ya no necesitaba porque la tiene mejor en la probabilidad de los miembros.

### Por que V26 no mejoro transformando la evidencia

Porque la evidencia ya **no estaba en la escala**: H02 cambio 1.809 filas y movio **neto +3** sobre 169.336. Y V27-04 explico el por: las columnas reescaladas y las que el boosting abre son las mismas 78 del bloque de miembros, y reordenar lo que ya se lee en orden correcto no puede ayudar.

## 6. Los hallazgos de V25

### Las 279 features, por bloque

| Bloque | Features | Usadas por el boosting de V20 |
|---|---|---|
| miembro_ensamble | 78 | 7 |
| estructurado | 87 | 0 |
| label_informed | 73 | 0 |
| denso_textometrico | 35 | 0 |
| nbsvm_palabra | 6 | 0 |
| **total** | **279** | **7** |

### La seleccion de mutual information

| Magnitud | Valor |
|---|---|
| K seleccionado (50% de 279) | 139 |
| corrected_cv con seleccion | 0,6091492 |
| Delta frente a V20 | **-0,000629** |
| Folds positivos | 1 de 5 |
| Rescue / damage / neto | 3.545 filas cambian, neto **-101** |
| Sobrevivientes con frecuencia 1,000 | las 78 del ensamble y 6 de NB-SVM |
| Nunca seleccionadas | 74 de las 87 estructuradas, 30 de las 35 densas |

### Por que la seleccion queda CLOSED

Porque el resultado no es ambiguo en su direccion: **seleccionar por mutual information elimina senal, no ruido**. Las features con mas mutual information son las probabilidades de los miembros, que son justamente las que el modelo usa. Y las que el mutual information descarta, las estructuradas y las densas, son las que el boosting ya no abria para dividir. Podar no quita nada que el modelo estuviera usando.

No se reintenta con otro K ni con otro selector: no se abre una nueva rejilla para rescatar un resultado negativo.

## 7. Los hallazgos de V26 y V27, hipotesis por hipotesis

Cada hipotesis aparece con su modo de cierre real. **A ninguna se le atribuye un resultado que no se midio.**

| ID | Hipotesis | Modo | Estado | Numero |
|---|---|---|---|---|
| H02 | reescalada afin de la evidencia | probado | CLOSED | -0,0000010 |
| H12 | fiabilidad por modelo y clase | probado | CLOSED | 0,0000000 |
| H11 | restricciones monotonas nativas | probado | CLOSED | -0,000355 |
| H14 | reescalar la salida del blend | **equivalencia funcional** | CLOSED | no medido |
| H03 | percentil de la evidencia | probado | MARGINAL | 0,000272 |

### H02 - cerrada por resultado

Delta -0,0000010, 2/5 folds, 1,809 filas cambian, neto +3. La transformacion **si** cambia el modelo, y **no** cambia la metrica. La evidencia del ensemble es invariante a su escala en todo lo que mide el macro-F1.

### H12 - cerrada por resultado y por equivalencia

Delta **exactamente 0,0000000** y **0 filas cambian**. La causa se verifico, no se supuesto: la fiabilidad por modelo y clase es una tabla constante por fila, y una columna constante no puede generar splits. Se midio ademas que LightGBM con columnas constantes devuelve predicciones **identicas bit a bit**. Su unica variante razonable es un producto fiabilidad por evidencia, que es una interaccion entre miembros, cerrada como H07.

### H11 - cerrada por resultado, y el mecanismo explica mas que el numero

Delta -0,000355, 2/5 folds, neto -55. Las restricciones se fijaron **antes** de entrenar, sobre 42 features con relacion monotona defendible: los 7 modelos con probabilidad genuina por clase. `E019b_densas_svd` quedo excluido aunque sus componentes sumen 1, porque es un TruncatedSVD de features densas y un SVD no genera probabilidades de clase.

El mecanismo es el hallazgo: **las 42 features restringidas no se usan para dividir**, porque el meta-modelo abre 7 de sus 279. La restriccion no puede actuar sobre features que el modelo no abre, y su unico efecto es el que produce el cambio de heuristica interna de LightGBM al activar `monotone_constraints`.

> Una correccion que se declara: el diagnostico de mecanismo de H11 reporto `0/0 splits` en > su primera version por contar una clave inexistente de `dump_model()`. Corregido en > `v27_03`, **el delta, los folds y la decision de H11 no se modificaron**. Lo que estaba > mal era la medida, no el experimento.

### H14 - cerrada sin experimento

«Reescalar las salidas del blend» admite cuatro formas, y las cuatro estan cerradas con numero:

| Forma | Familia que ya lo midio | Resultado |
|---|---|---|
## 8. Los resultados negativos, leidos como conocimiento

Cuatro de los cinco experimentos de V26 y V27 son negativos. No son fracasos: cada uno elimina una pregunta.

### Podar no quita ruido, quita senal

La seleccion supervisada de features empeoro en -0,000629 porque el mutual information alto de las probabilidades de los miembros **no es ruido, es la prediccion de los miembros que sostiene el modelo**. Y lo que el mutual information descarta ya no lo usaba el boosting. La leccion general: en un meta-modelo con evidencia redundante, seleccionar por importancia marginal es especialmente peligroso, porque la importancia marginal de una feature que colisiona con otras se parece mucho a la de un ruido.

### La evidencia del ensemble no tiene el problema de escala

H02 cambio 1.809 filas y movio neto +3. La conclusion no es «el z-score no sirve», que seria una lectura trivial: es que **la escala de la confianza de los miembros no es el cuello de botella**, porque el modelo lee orden, no magnitud.

### La fiabilidad agregada no puede ser feature de un arbol

H12 dio delta exactamente cero, y la causa es estructural: una medida agregada por (modelo, clase) es una constante por fila, y **una columna constante no puede dividir**. La leccion: pedirle a un modelo de arbol que conozca una propiedad de los *componentes* exige que esa propiedad **varie entre filas**, o hay que darla como sesgo del modelo, no como columna.

### La monotonia no se puede imponer donde el modelo no mira

H11 falló, y el motivo es mas fuerte que el fallo: las features con relacion monotona defendible son las probabilidades de clase, y **el meta-modelo no las abre para dividir**. La leccion: verificar que la restriccion puede actuar antes de medirla. Es la diferencia entre una hipotesis que responde una pregunta y una que gasta presupuesto para confirmar que el modelo no pasa por ahi.

### Un modelo de arbol no es invariante a transformaciones monotonas

La invariancia teorica de los arboles a transformaciones monotonas habria cerrado H02 sin entrenar nada. Es **falsa** en esta LightGBM: el z-score cambia la prediccion bit a bit, con diferencia maxima 0,219 sobre 135.468 filas. La leccion metodologica es general: una propiedad teorica de la familia de modelos no exime de medirla si la implementacion concreta puede no cumplirla.

## 9. Analisis de error

No se vuelve a analizar el test y no se usa el leaderboard para explicar errores internos. Las cifras vienen de artefactos ya existentes del train.

### Errores conjuntos: la masa grande y medida

| Magnitud | Valor | Fuente |
|---|---|---|
| Filas que **V19 y V20 fallan juntas** | **38,3 %** del train | interseccion de errores |
| Techo con enrutamiento en 18-24 / 25-34 | 0,7481 | `v17_error_intersection.csv` |
| Techo con enrutamiento en 25-34 / 35-49 | 0,7726 | `v17_error_intersection.csv` |
| Correccion neta posible en 18-24 / 25-34 | 10.837 filas | idem |
| Correccion neta posible en 25-34 / 35-49 | 12.699 filas | idem |

### Errores corregibles frente a no corregibles

- **Corregibles**: los que viven en las fronteras entre clases contiguas, donde un modelo mejor podria mover la decision sin cambiar el orden global. Ahi el techo con enrutamiento esta entre 0,748 y 0,773, muy por encima del 0,610 actual.

- **No corregibles con esta senal**: el 38,3 % que V19 y V20 fallan **juntas**. Ningun modelo del historico los separa, y por eso ninguna de las hipotenes de V22 a V27 los toco. Es la unica masa grande y medida, y ya esta agotada como via de mejora.

### Slices y complementariedad

La complementariedad por clase de los tres candidatos de V27 esta medida y en el artefacto. H03 es la mas illustrativa: mueve F1 en las seis clases en la cuarta decimal, con rescate y dano casi iguales (neto +51 sobre 1.764 filas cambiadas). Es la firma de un modelo que **redistribuye** decisiones entre clases que ya eran ambiguas, no de uno que corrige errores.

## 10. V28: mecanismos estructuralmente distintos


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
| Filas en `experiments.csv` | **139** | registro |

**La respuesta a la pregunta de V28 es no, y se contesta con dos experimentos y cuatro  cierres por equivalencia.** Ningun mecanismo estructuralmente distinto de combinar, estabilizar o descomponer la evidencia supera a V20.

| ID | Tecnica | Modo | Resultado | Decision |
|---|---|---|---|---|
| V28-02 | A · Meta-model bagging | probado | -0,0003120, 1/5 folds | CLOSED |
| V28-03 | C · Pairwise class decomposition | probado | -0,0675530, 0/5 folds | CLOSED |
| - | B · Compact consensus geometry | equivalencia, ya medido | +0,0001 historico | cerrada sin experimento |
| - | D · Dirichlet / evidence fusion | equivalencia con H04 y H06 | no medido | cerrada sin experimento |
| - | E · Explicit uncertainty | duplicacion de B | no medido | cerrada sin experimento |
| - | F · Randomized meta-subspace | equivalencia con `colsample_bytree` | no medido | cerrada sin experimento |


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

## 11. Las dos tecnicas realmente ejecutadas

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

## 12. V29: Optuna y busqueda de ensemble diverso

El eje de capacidad resulto ser el cuello real de botella. Un LightGBM sobre la misma matriz de 279 features alcanza 0,6121698 frente a los 0,6097782 de V20, y es el unico con rescate neto positivo de todo el inventario historico. El ensemble de tres o cuatro miembros llega a 0,6125960 con 5 de 5 folds y neto +480, un 94 % del gate, pero **no llega**. La diversidad de modelos historicos queda cerrada con numero: **0 de 47**. Detalle en `reports/V29_FINAL_RESEARCH_REPORT.md`.

## 13. V30: refinamiento local y la unica submission autorizada

## 14. Control absoluto: V30 no empieza sin que el punto de partida sea el que se dice

| Control | Esperado | Obtenido | Reproduce |
|---|---|---|---|
| V20, reconstruido y comparado **bit a bit** | 0,6097782 | 0,6097782 | **True** |
| LightGBM de V29, OOF en disco | 0,6121698 | 0,6121698 | **True** |
| Ensemble de V29, **re-optimizado** | 0,6125960 | 0,6125960 | **True** |

**La diferencia maxima entre el OOF reconstruido de V20 y su artefacto es 0.00e+00**, y los folds son los mismos: [33868, 33868, 33867, 33866, 33867].

Un resultado que merece staffed atencion: la reoptimizacion de los pesos del ensemble de V29 **reprodujo con desviacion 0.0**. El propio brief advertia que no debia declararse exactitud bit a bit entre reproducciones de Optuna; en esta maquina y con estas versiones de libreria, si lo es. Se registra lo medido, no lo temido.

## 15. V30: refinamiento local del LightGBM, sembrado con el mejor trial de V29

### Por que el espacio es local, y no una busqueda nueva

Los cinco mejores trials de V29 coinciden en una region estrecha, y **V20 estaba sub-parametrizado en casi todo**:

| Parametro | 5 mejores de V29 | `HIERBA` de V20 |
|---|---|---|
| `num_leaves` | 43 - 87 | **8** |
| `max_depth` | 5 - 9 | **3** |
| `learning_rate` | 0,021 - 0,043 | 0,05 |
| `min_child_samples` | 22 - 94 | **200** |
| `reg_lambda` | 1e-7 - 0,018 | **1,0** |

Ese contraste **es el mecanismo de la mejora de V29**, y por eso V30 centra su espacio ahi en vez de recorrer la region entera. Y la localidad no es solo un rango: el **trial 0 de cada fold se encola con los parametros exactos** del mejor trial de V29 para ese fold.

- Trials: **15 por fold, 75 en total**, el maximo que autoriza el brief.
- Sampler: `TPESampler(seed=20300930 + fold) con trial 0 encolado`.
- Techo de estimadores: **1500**, con early stopping de 60 rondas.

### Iteraciones reales, que es donde se ve el early stopping

| Fold | Mejor trial | Inner | Mediana | Sigma | Iteraciones / techo | Early stop | Outer |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 0.612670 | 0.610101 | 1.89 | 999 / 1011 | True | 0.615979 |
| 1 | 3 | 0.611625 | 0.610502 | 1.20 | 1173 / 1486 | True | 0.616113 |
| 2 | 5 | 0.617228 | 0.615592 | 1.72 | 644 / 1023 | True | 0.611859 |
| 3 | 11 | 0.618832 | 0.616098 | 2.12 | 1129 / 1477 | True | 0.610876 |
| 4 | 13 | 0.617436 | 0.616016 | 1.43 | 398 / 1000 | True | 0.606381 |

**El early stopping corta siempre muy por debajo del techo.** Es exactamente el comportamiento que el punto 38 del brief describe como correcto: dar margen alto de capacidad y dejar que el criterio interior decida donde parar, en vez de entrenar artificialmente hasta el limite.

**Mejor trial outlier: 0 de 5.** Ningun fold se gano por un trial anomalo, asi que el resultado no depende de una tirada afortunada.

### Resultado del refinamiento

- corrected_cv **0,6122387**
- frente a V20: **0,002461**
- frente al LightGBM de V29: **0,000069**
- folds que mejoran a V29: **3/5** ([0.001485, 0.000216, -0.000409, -0.001387, 0.000452])
- veredicto: **EQUIVALE_A_V29**

Criterio declarado de consolidacion: delta frente al LightGBM de V29 superior a +0,0005; entre -0,0005 y +0,0005 se considera equivalente y se decide por complementariedad en la fase de ensemble.

### ¿V30 aprende mejor, o solo reproduce a V29?

| Modelo | CV individual | corr. errores con V20 | Rescate unico | Dano unico | Neto |
|---|---|---|---|---|---|
| v20_f3b_oof_candidato | 0.6097782 | 1.0000 | 0 | 0 | +0 |
| v29_lgbm_optuna | 0.6121698 | 0.8636 | 5,689 | 5,288 | +401 |
| v30_lgbm_refinado | 0.6122387 | 0.8694 | 5,448 | 5,061 | +387 |

Y entre los dos LightGBM, que es la comparacion que decide si hay algo que combinar:

| Par | corr. errores | Discrepancia | Uno arregla al otro |
|---|---|---|---|
| V29 vs V30 | 0.9164 | 0.0626 | 3,365 |

Una correlacion de errores **muy alta** entre V29 y V30 significa que el refinamiento encontro la misma funcion con parametros distintos: es una confirmacion de que la region esta bien elegida, y tambien de que **no hay un tercer miembro que aportar** al ensemble. Por eso el pool de V30 se queda en tres: V20, V29 y V30.

## 16. El ensemble: tres miembros, no dieciseis

V29 ya contesto la pregunta del pool historico (**0 de 47** con rescate neto positivo, y los tres metodos negativos sin el modelo nuevo). V30 **no repite esa busqueda**: el pool es de **3 miembros** y cada uno esta justificado.

| Miembro | Papel |
|---|---|
| `v20_f3b_oof_candidato` | el champion, intacto |
| `v29_lgbm_optuna` | la capacidad que V29 demostro |
| `v30_lgbm_refinado` | el intento de consolidar esa capacidad |

Los pesos se optimizan **solo sobre el holdout interior**, con `w = softmax(theta)` para imponer `w_i >= 0` y `sum w_i = 1` sin restricciones. La validacion exterior no entra.

| Metodo | CV | Delta V20 | Delta V29-LGBM | Folds + | Rescue | Damage | Net |
|---|---|---|---|---|---|---|---|
| E1 Optuna macro-F1 | 0.6130051 | +0.003227 | +0.000835 | 5/5 | 3,768 | 3,238 | +530 |
| E2 Optuna macro-F1 menos lambda por entropia | 0.6121866 | +0.002408 | +0.000017 | 5/5 | 4,341 | 3,953 | +388 |

### El candidato principal: `E1 Optuna macro-F1`

- corrected_cv **0,6130051**, delta frente a V20 **0,003227**
- **5/5 folds** positivos, rescate 3,768 contra dano 3,238, neto **+530**
- pesos: `{'v20_f3b_oof_candidato': 0.3548, 'v29_lgbm_optuna': 0.3354, 'v30_lgbm_refinado': 0.3098}`
- filas que cambian frente a V20: 10,914

### Las dos banderas, separadas como pide el punto 20

| Bandera | Valor | Significado |
|---|---|---|
| Gate historico | **True** | exige delta >= +0,003 **y** >= 4/5 folds |
| STRONG_MARGINAL | **True** | 5/5 folds **y** delta entre +0,002 y +0,003 |

Son **dos preguntas distintas** y no se confunden: el gate es el criterio de finalista; la bandera marginal es la observacion de que la mejora es consistente pero no llega al umbral.

### Bootstrap del candidato principal

- delta **0,003227**
- IC95 **[0,002290, 0,004219]**
- proporcion de diferencias positivas: 1.0000
- **el IC95 incluye el cero: False**

calculado **solo** para el candidato principal, y no se usa para elegir hiperparametros ni pesos: es evidencia de decision de la submission, no senal de entrenamiento

## 17. Segunda particion, con la configuracion congelada

Los hiperparametros y los pesos se aplican **congelados** sobre una particion agrupada distinta (semilla 987654, mismos grupos, sin re-optimizar). Los solapes con la particion original fueron [7155, 7272, 7167, 7229, 7220], luego la particion cambio de verdad.

| Fold | V20 recomputado | Candidato | Delta |
|---|---|---|---|
| 0 | 0.6122675 | 0.6150008 | +0.002733 |
| 1 | 0.6139561 | 0.6186241 | +0.004668 |
| 2 | 0.6138090 | 0.6163075 | +0.002498 |
| 3 | 0.6107416 | 0.6143509 | +0.003609 |
| 4 | 0.6013498 | 0.6073941 | +0.006044 |

- V20 recomputado **0,6104285** | candidato **0,6143347**
- delta contra el recomputado: **0,003906** con **5/5 folds** positivos
- el gate usa el V20 **recomputado en esta particion**: el gate se evalua contra el V20 **recomputado en esta misma particion**, porque los folds son distintos; contra el canonico se informa tambien, pero no decide
- frente al delta de la particion original (0,003227) la diferencia es 0,000679: la mejora se sostiene
- gate historico en la segunda particion: **True** | STRONG_MARGINAL: **False**

## 18. Construccion del fichero y validacion

- Fichero: `age_submission_v30_final_candidate.csv`, **112,892 filas**, 1,356,504 bytes
- **SHA256 `f543a715a1881c6128a860301b84a7273c3db8f8a709545be1e6267822e713be`**
- Checks: **10/10**
- Candidato: `E1 Optuna macro-F1`, pesos congelados `{'v20_f3b_oof_candidato': 0.3548, 'v29_lgbm_optuna': 0.3354, 'v30_lgbm_refinado': 0.3098}`
- Distribucion: `{'13-17': 18141, '18-24': 20767, '25-34': 17135, '35-49': 16406, '50-64': 20794, '65-xx': 19649}`
- Frente al envio de V20 difieren **7,253 decisiones (6.42%)**

### Los diez checks

| Check | Resultado |
|---|---|
| `rows_ok` | **True** |
| `columns_ok` | **True** |
| `ids_equal_test` | **True** |
| `ids_same_order` | **True** |
| `ids_unique` | **True** |
| `no_nan` | **True** |
| `labels_valid` | **True** |
| `labels_width_ok` | **True** |
| `sin_filas_vacias` | **True** |
| `schema_verificado` | **True** |

### Como se uso el test

solo transform y predict. Ninguna etiqueta, ninguna estadistica, ningun ajuste de umbral. Los pesos y los hiperparametros estaban congelados antes de entrar aqui.

### La decision de los cinco modelos por miembro

Cada fold produjo su mejor configuracion y son distintas. Ajustar las cinco sobre todo el train y promediar usa toda la evidencia de la busqueda; elegir una sola seria un arbitrio sin base.

## 19. La subida

- Estado: **STOP_SIN_CREDENCIAL**
- Numero de submissions: **0**
- Subido: **False**

**Causa del STOP:** el fichero ~/.kaggle/kaggle.json existe pero solo contiene la clave `token`, que es el formato antiguo; el CLI 2.2.4 exige `username` y `key`, o bien autenticacion OAuth. Sin eso la API rechaza la llamada.

**Como se resuelve:** `kaggle auth login` (flujo OAuth, abre el navegador y cachea el token) o dejar un ~/.kaggle/kaggle.json con las claves `username` y `key`. Ambas vias las hace el usuario en su terminal; ninguna credencial tiene que pasar por esta conversacion.

> Esto es un **STOP de infraestructura**, no un resultado experimental. El > fichero esta construido, validado con sus diez checks y con su hash, y **no > necesita ninguna correccion**: en cuanto haya credencial valida, se sube tal > cual con `v30_04_upload.py`. No se pide ni se copia ningun secreto por esta via.

## 20. La evolucion de la pregunta, de V20 a V30

| Fase | Eje | Respuesta |
|---|---|---|
| V20 | construccion | el blend de cuatro cuartos | **LB 0,62689**, sigue champion |
| V21 a V24 | auditoria | por que V20 funciona y donde se rompe | estructura, no mejoras |
| V25 | feature selection | podar columnas elimina ruido | no: elimina senal |
| V26 | evidencia | reescalar cambia algo | no: es invariante a su escala |
| V27 | monotonia, fiabilidad, percentil | restricciones y transformaciones | cerradas |
| V28 | estructura | bagging y descomposicion de la decision | ambos negativos |
| V29 | **capacidad y diversidad** | mas capacidad, mas diversidad, mejor combinacion | **la capacidad si era el cuello**; primer modelo individual por encima de V20 |
| V30 | **refinamiento y transferencia** | el margen de V29 se consolida y se transfiere | ver las secciones 12 |

## 21. Compliance y reproducibilidad acumulados

Valores **medidos** por `scripts/v27_09_compliance_auditoria.py` con `grep` dirigido sobre el codigo y los artefactos de V27, y no declarados.

La medicion **distingue uso de mencion**: el codigo se mide tras eliminar los literales de cadena con el tokenizer, y en un informe toda mencion se cuenta como prosa. Un informe que declara por que una familia esta descartada no infringe la prohibicion al escribirla.

| Comprobacion | Uso en codigo | Menciones en prosa | Cumplida |
|---|---|---|---|
| transformers | 0 | 1 | si |
| BERT | 0 | 0 | si |
| RoBERTa | 0 | 0 | si |
| LLM | 0 | 1 | si |
| embeddings | 0 | 1 | si |
| external_classifier | 0 | 0 | si |
| external_labels | 0 | 0 | si |
| pseudo_labeling | 0 | 0 | si |
| test_labels | 0 | 0 | si |
| test_training | 0 | 0 | si |
| test_calibration | 0 | 0 | si |
| test_feature_selection | 0 | 0 | si |
| test_threshold_tuning | 0 | 0 | si |
| holdout_reuse | 0 | 2 | si |
| leaderboard_tuning | 0 | 0 | si |

### En la forma que pide el punto 28

```text
transformers = 0
BERT = 0
RoBERTa = 0
LLM = 0
embeddings = 0
external_classifier = 0
external_labels = 0
pseudo_labeling = 0
test_labels = 0
test_training = 0
test_calibration = 0
test_feature_selection = 0
test_threshold_tuning = 0
holdout_reuse = 0
leaderboard_tuning = 0
```

**Las quince comprobaciones en cero usos.** Hay 5 menciones en prosa en todo el conjunto de V27, y son de los propios informes, al enumerar las lineas descartadas y al declarar las excepciones. **No son usos.** El hash de la submission de V20 no cambia: no se genero ninguna submission en V26 ni en V27.

### Excepciones declaradas

- El script de auditoria y sus dos artefactos quedan **excluidos** de la compliance que realiza, porque si no detectaria sus propias menciones a las palabras prohibidas. Es la misma excepcion deliberada de V26 y se declara en vez de esconderse.

- La compliance de V27 se apoya en la ya hecha de V26 para el resto del proyecto. Es una auditoria de firmas: busca las practicas prohibidas y **no puede demostrar que no ocurran de otra forma**.

- El patron de `leaderboard_tuning` busca el **uso de la API**, no la palabra. Un informe puede citar el leaderboard como cifra, y eso no es ajustar por el.

### Lo que no se hizo, por regla

No se genero ninguna submission en V26 ni en V27. No se toco el test, ni para labels, ni para entrenamiento, ni para calibracion, ni para seleccion de features, ni para umbrales. No se toco el holdout. No se hizo push, ni merge, ni force-push, ni reset destructivo.


### Compliance de V30

```text
transformers = 0
BERT = 0
RoBERTa = 0
LLM = 0
embeddings = 0
external_classifier = 0
external_labels = 0
pseudo_labeling = 0
test_labels = 0
test_training = 0
test_calibration = 0
test_threshold_tuning = 0
holdout_reuse = 0
leaderboard_tuning = 0
toxic_test = 0
```

## 22. Figuras acumuladas

### Figuras maestras de la investigacion completa

| Figura | Que muestra |
|---|---|
| `v27_maestra_resultados.png` | corrected_cv por experimento, delta frente a V20, rescate frente a dano y F1 por clase |
| `v27_maestra_folds.png` | consistencia por fold y folds positivos de cada candidato |
| `v27_maestra_efecto_percentil.png` | el efecto de H03: delta por fold, mejora por clase y complementariedad por clase |

Presentes en disco: **4 de 3** esperadas (`v26_evolucion_corrected_cv.png, v27_maestra_efecto_percentil.png, v27_maestra_folds.png, v27_maestra_resultados.png`).

### Figuras de cada hipotesis

| Figura | Que muestra |
|---|---|
| `v27_concentracion_features.png` | las 7 features que V20 divide de 279, por bloque |
| `v27_h12_mapa_fiabilidad.png` | mapa de fiabilidad por modelo y clase (H12) |
| `v27_h12_delta_por_fold.png` | delta de H12 por fold, que es exactamente cero |
| `v27_h11_splits_por_feature.png` | reparto real de splits, marcando las restringidas |
| `v27_h11_delta_por_fold.png` | delta de H11 por fold |
| `v27_h03_delta_por_fold.png` | delta de H03 por fold |

Otras figuras de V25 a V27 en disco: **12**, entre ellas `v25_delta_por_fold.png, v25_frecuencia_seleccion.png, v26_comparacion_por_fold.png, v26_delta_por_experimento.png, v26_f1_por_clase.png, v26_rescate_vs_dano.png, v27_concentracion_features.png, v27_h03_delta_por_fold.png`.

**Total de figuras de V27: 16.** Ninguna oculta un resultado negativo: las tres figuras de delta por fold de H12, H11 y H03 estan las tres, y dos de ellas son negativas o marginales.

### Figuras de V30 (3)

| Figura | Que muestra |
|---|---|
| `v30_ensemble.png` | candidatos, pesos del ensemble, delta por fold y rescate/dano |
| `v30_optuna_refinamiento.png` | historia del refinamiento, distribucion de trials y outer validation frente a V29 |
| `v30_segunda_particion.png` | delta por fold en las dos particiones, con la configuracion congelada |

---

**Cierre de AGE.** V20 permanece como baseline absoluto en 0,6097782 (LB 0.62689). El leaderboard no se utilizo para seleccionar hiperparametros, pesos, thresholds ni variantes. El test solo se toco para transform y predict, con el candidato ya congelado. No se generaron mas de una submission, y no se genero ninguna. La investigacion termina porque las familias defendibles estan contestadas, y porque el margen que queda, medido y reproducible, es un orden de magnitud menor que el necesario para un candidato defendible.
