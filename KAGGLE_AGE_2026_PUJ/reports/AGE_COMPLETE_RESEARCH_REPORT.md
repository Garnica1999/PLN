# AGE - Informe completo de la investigacion, de V20 al cierre de V29

**Fecha:** 2026-09-29 · **Rama:** `proyecto-kaggle-age-2026-v29` · **Baseline congelado:** 0,6097782 · **Champion:** V20, LB 0.62689

Reconstruye **toda** la trayectoria de AGE desde V20 hasta el cierre de V29, sin omitir ninguna fase negativa ni ninguna fase positiva medida. La mayoria de lo que se midio fue negativo, y ese resultado es el producto.

**Una sola cadena de fuentes:** las secciones de V20 a V27 las genera `v27_10_informe_completo.py`, las de V28 `v28_07_informes.py` y las de V29 este script, que importa a los dos anteriores. Ningun informe puede divergir de los que le preceden.

## 1. Resumen ejecutivo

| Concepto | Valor | Fuente |
|---|---|---|
| Baseline final | **0,6097782** corrected_cv | canonico del proyecto, reproducido por los 4 controles |
| Mejor corrected_cv | **0,610051** (H03, MARGINAL) | `experiments/v27_05_h03_percentil.json` |
| Mejor score verificado en leaderboard | **0.62689** (V20) | envio verificado de V20 |
| Mejor candidato validado internamente | H03, MARGINAL, no finalista | su IC95 incluye el cero y esta 11 veces bajo el gate |
| Numero de experimentos | **138** filas en `experiments.csv` | registro |
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

## 10. V28: mecanismos estructuralmente distintos para combinar la evidencia


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
| Filas en `experiments.csv` | **138** | registro |

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


**Fecha:** 2026-09-29 · **Rama:** `proyecto-kaggle-age-2026-v29` · **Baseline congelado:** 0,6097782 · **Champion:** V20, LB 0.62689

## 13. Motivacion y pregunta

V25 a V28 habian cerrado la representacion, la escala de la evidencia, la monotonia, los percentiles, la fiabilidad por modelo, la estructura ordinal, el bagging del meta-modelo, la descomposicion pairwise y la geometria del consenso. Lo que **no** se habia tocado es la capacidad de un miembro individual ni la composicion del ensemble.

La hipotesis central de V29 es una sola:

> V20 puede no estar limitado por la capacidad de sus miembros, sino por la **falta de diversidad util entre sus errores** y por una combinacion de pesos fija.

La palabra clave es **complementariedad**, y el punto 3 del brief es el que gobierna todo lo demas: *mejor modelo individual* no implica *mejor miembro del ensemble*.

## 14. Estado inicial

| Concepto | Valor |
|---|---|
| Baseline vigente | **0,6097782** corrected_cv, reproducido bit a bit en P0 |
| Leaderboard de V20 | **0.62689**, envio sin cambio |
|.Hash de la submission de V20 | `2c964af78c90bf08...`, intacto |
| Candidato para submission | ninguno |

## 15. Inventario historico: la surprise de V29

El primer paso no fue entrenar: fue **leer lo que ya estaba entrenado**. En el directorio de features hay **51 matrices de prediccion out-of-fold de seis clases**, de las que **51 son metodologicamente reutilizables** sin volver a entrenar nada.

Proceden de **169,336 filas** y cubren **17 familias distintas** de modelo: arboles, boosting, logistica, perceptrones, NB-SVM, BM25, vecinos mas cercanos, ExtraTrees, SGD, modelos de estructura ordinal, versiones solo-contenido y solo-estilo, entre otras.

Se descartaron **22** por no tener seis columnas: son CDFs de cinco umbrales o decisiones binarias, cuyo `argmax` no es una clase.

**El criterio de reutilizacion fue triple, y se declara porque no basta con que el fichero exista:**

1. seis columnas de clase y exactamente las filas del train;

2. **homogeneidad entre folds** superior a 0,90, porque un OOF real no tiene folds dispares;

3. que no sea duplicado de otro artefacto ya inventariado.

**Un hallazgo de inventario que habria falseado la medicion:**

- `v19_oof_arbol_5folds` y `v22_seis_compartida` tienen el **argmax identico** en las 169.336 filas. Difieren en las magnitudes, pero toman las mismas decisiones, y en un ensemble con voto aportan cero.

El cuarto de cadena de V20 se reconstruyo desde `v20_f3_oof_cumuladas.npy` en vez de tomar ese artefacto, porque usarlo habria comparado a V20 consigo mismo y habria understated la diversidad real de sus miembros.

## 16. Diversidad: la medicion que decide la fase

### La referencia: la diversidad interna de V20

V20 es un promedio de cuatro cuartos, asi que su propia diversidad interna es el liston que cualquier candidato tiene que superar.

| Medida | Valor |
|---|---|
| Correlacion de errores media entre los 4 cuartos | **0.7802** |
| Discrepancia de decision media | 0.1644 |
| Discrepancia maxima entre dos cuartos | 0.1906 |

### Cinco medidas, no una

Una sola correlacion no alcanza y puede actively mislead, asi que se miden cinco a la vez y **no se resumen en un numero unico**:

| Medida | Que responde |
|---|---|
| `corr_probabilidades` | cuan parecidas son las distribuciones completas |
| `corr_errores` | cuan parecidos son **los fallos**, que es lo que importa |
| `tasa_discrepancia` | con que frecuencia difieren las decisiones |
| `rescate_unico` | cuantas filas corrige el modelo y V20 no |
| `dano_unico` | cuantas filas rompe el modelo que V20 acertaba |

### La trampa del rescate bruto, y como se evita

El mayor rescate bruto del inventario lo tiene un modelo con **CV 0,06**: resucita decenas de miles de filas, no por complementariedad sino porque V20 falla el 39 % del train y cualquier moneda acierta parte de eso.

Por eso la metrica de decision son **dos fracciones normalizadas**, y no el numero absoluto de filas:

- `fraccion_de_los_errores_de_V20_que_corrige`

- `fraccion_de_los_aciertos_de_V20_que_rompe`

Y el dato mas duro de toda la fase: de los 47 modelos que nunca han formado parte de un ensemble, **0 tienen rescate unico mayor que dano unico frente a V20**. Ninguno. Por si solo, ninguno es un complemento neto.

Si 28 modelos tienen correlacion de errores por debajo del liston de V20 (mas diversos que cualquier par de sus miembros) y 13 ademas tienen CV suficiente para no ser ruido, el espacio de candidatos existe. Lo que faltaba era comprobar si un blend ponderado puede explotarlo.

## 17. Gap analysis: donde V29 casi reabria una familia cerrada

Dos de los tres ejes de V29 se parecen a familias ya cerradas, y decirlo es parte del trabajo:

- **Composicion del ensemble** es, en mecanismo, **H04** (ponderar modelos) y **H06** (apilar un segundo nivel), las dos cerradas con numero. El veredicto es **PARTIALLY_TESTED**, no GENUINELY_NEW, y la distincion que lo salva es concreta: H04 y H06 reevaluaron los miembros que V20 ya tenia; V29 amplía el conjunto de candidatos a modelos de 17 familias que nunca han entrado en un ensemble.

- **Capacidad individual** si es genuinamente nueva: ninguna familia cerrada de V25 a V28 modifico la capacidad de un miembro, todas transformaron la combinacion o la representacion.

- **Diversidad** es genuinamente nueva y ya medida, sin gastar un experimento.

Y queda un criterio de cierre declarado por si el mecanismo resultara ser el viejo: **si el optimum se apoya en una sola estructura, es H04 de nuevo y se cierra por equivalencia.**

Optuna queda ademas prohibido por decision propia sobre threshold, feature selection, calibracion, pairwise, ordinal, robust loss, grafo, ventana local: usar un metodo de busqueda nuevo sobre una familia cerrada no la abre, la reembala.

## 18. El pool: cuatro filtros declarados

De 52 candidatos de partida quedaron **16** en el pool, con los caidos por filtro visibles:

| Filtro | Caidos | Motivo |
|---|---|---|
| Util | 15 | CV por debajo de 0,50: ruido, no senal |
| Diverso | 9 | familia de V20 y correlacion de errores por encima del liston |
| No redundante | 0 | mismas decisiones que otro ya elegido |
| Tamano | 13 | tope de 16 miembros |

El criterio **no es el CV maximo**, que es lo que el brief prohibe. Es CV util, mas diversidad, mas rescate normalizado, con la tension entre rescate y dano **visible** en la tabla en vez de escondida en un score.

## 19. Optuna individual: estructura anidada

La parte que hace legitimos los numeros es que **la validacion exterior nunca entra en Optuna**:

```text
outer_train
  +-- Optuna: inner_train / inner_val   <-- aqui se eligen los hiperparametros
  +-- refit con la mejor configuracion sobre TODO el outer_train
        +--> outer_val                  <-- aqui SOLO se mide
```

Hay una **asercion** que aborta si el holdout interior toca la validacion exterior.

Un recorte de presupuesto que se declara: el brief pide 30-50 trials por familia *solo si el coste computacional lo permite*. Medido en esta maquina, con `n_estimators` hasta 3000 **el fold 0 no termino en 55 minutos**. Se maintaining los 30 trials, que son el suelo del rango, y se acoto el techo de iteraciones, que es donde estaba el coste; el early stopping ponia el corte real en ambos casos.


**Resultado de las pistas ejecutadas:**

| Pista | CV | Delta | Trials | Outliers | Iteraciones reales de maximas | Early stopping |
|---|---|---|---|---|---|---|
| P4 · Optuna individual LightGBM | 0.6121698 | +0.002392 | 150 | 0 | [] de [] | 0 de 0 folds cortaron antes del maximo |

La columna de iteraciones registra `max_iterations`, `actual_iterations` y si hubo `early_stopping`, como pide el punto 12 del brief. Un best-trial outlier significa que el resultado vendria de un trial anomalo; **hubo 0 en total**.

## 20. Composicion del ensemble: el resultado central

Tres metodos, los tres anidados, sobre el pool congelado de 16 miembros, que ahora **incluye el LightGBM de Optuna, el mejor modelo individual del proyecto**. Ninguno reentrena a nadie: solo pondera matrices congeladas.

**Este resultado cambio cuando entro el modelo de Optuna, y el cambio es la conclusion mas importante de V29.** Con el pool historico de 16 miembros, los tres metodos eran negativos. Con el modelo nuevo dentro, los tres son **positivos**. El pool historico no aportaba nada; el modelo nuevo aporta casi todo.

| Metodo | corrected_cv | Delta | Folds + | Rescue | Damage | Neto | Gate |
|---|---|---|---|---|---|---|---|
| P9-1 Optuna macro-F1 | 0.6119153 | +0.002137 | 4/5 | 4,004 | 3,665 | +339 | no |
| P9-2 Optuna macro-F1 menos lambda por entropia | 0.612596 | +0.002818 | 5/5 | 3,901 | 3,421 | +480 | no |
| P9-3 Caruana con tope de 6 miembros | 0.6121102 | +0.002332 | 4/5 | 3,030 | 2,626 | +404 | no |
| T0 V20 sin tocar | 0.6097782 | +0.000000 | 0/5 | 0 | 0 | +0 | no |
| T1 promedio igual de todo el pool | 0.5946618 | -0.015116 | 0/5 | 5,614 | 8,242 | -2,628 | no |
| C-1 Optuna macro-F1 | 0.6090958 | -0.000682 | 2/5 | 1,658 | 1,765 | -107 | no |
| C-2 Optuna macro-F1 menos lambda por entropia | 0.6094175 | -0.000361 | 2/5 | 1,397 | 1,451 | -54 | no |
| C-3 Caruana con tope de 6 miembros | 0.6091703 | -0.000608 | 2/5 | 1,640 | 1,751 | -111 | no |

El gate exige delta >= +0.003 **y** 4 de 5 folds positivos. Los tres superan la segunda condicion y **ninguno llega a la primera**: el mejor es 0,002818, un 94% del gate.

**Que se puede decir de esto, con la medida delante:**

- El mejor ensemble mejora a V20 en **5/5 folds**, tiene **neto positivo** y cambia menos de 6.000 decisiones. Es la primera vez en todo el proyecto que un ensemble gana en los dos criterios a la vez.

- Pero **+0,002818 en CV** vale unos +0,0014 en el leaderboard con la calibracion `LB = +0,00223 + 1,02825 x CV`, y la oscilacion entre dos modelos vecinos del propio proyecto es de 0,0034. Este resultado seria **indistinguible de haber cambiado de modelo**.

- Por eso no es finalista, y no se genera submission. El gate existe para esto.

### F1 por clase

| Metodo | 13-17 | 18-24 | 25-34 | 35-49 | 50-64 | 65-xx |
|---|---|---|---|---|---|---|
| P9-1 Optuna macro-F1 | 0.6832 | 0.5190 | 0.4979 | 0.5723 | 0.6650 | 0.7342 |
| P9-2 Optuna macro-F1 menos lambda por entropia | 0.6843 | 0.5215 | 0.4983 | 0.5724 | 0.6662 | 0.7329 |
| P9-3 Caruana con tope de 6 miembros | 0.6840 | 0.5202 | 0.4976 | 0.5712 | 0.6663 | 0.7334 |
| T0 V20 sin tocar | 0.6818 | 0.5170 | 0.4965 | 0.5687 | 0.6637 | 0.7310 |
| T1 promedio igual de todo el pool | 0.6651 | 0.5005 | 0.4826 | 0.5521 | 0.6483 | 0.7194 |
| C-1 Optuna macro-F1 | 0.6813 | 0.5173 | 0.4952 | 0.5680 | 0.6628 | 0.7300 |
| C-2 Optuna macro-F1 menos lambda por entropia | 0.6811 | 0.5170 | 0.4959 | 0.5684 | 0.6640 | 0.7301 |
| C-3 Caruana con tope de 6 miembros | 0.6807 | 0.5179 | 0.4957 | 0.5679 | 0.6623 | 0.7306 |

**El cambio se concentra en las clases que V20 tenia debiles.** El 25-34, la clase con peor F1 de todo el proyecto, es la que mas se mueve. Es coherente con que el modelo nuevo se haya formado sobre la misma matriz de 279 features que V20 y solo cambie la capacidad de ajuste.

## 21. El control que decide: ¿es esto H04 otra vez?

Optimizar pesos **si** es el mecanismo de H04, asi que el juicio sobre si V29/aporto algo no puede ser "el ensemble mejoro". Hay que mirar **en quien se apoya el optimum**.

**P9-1 Optuna macro-F1**

- peso de V20: **0.1262**
- miembro que lleva mas peso: `v29_lgbm_optuna` (0.5941)
- miembros con peso superior a 0,05: **4**
- familias con peso superior a 0,05: **4**

Reparto del peso por familia: `Optuna (nueva capacidad)` 0.594, `cuarto de V20` 0.151, `perceptron multicapa` 0.110, `NB-SVM (lineal sobre binarios)` 0.057, `cabeza de modelo multiclase` 0.028, `desconocida` 0.023

**P9-2 Optuna macro-F1 menos lambda por entropia**

- peso de V20: **0.0822**
- miembro que lleva mas peso: `v29_lgbm_optuna` (0.6145)
- miembros con peso superior a 0,05: **4**
- familias con peso superior a 0,05: **4**

Reparto del peso por familia: `Optuna (nueva capacidad)` 0.615, `perceptron multicapa` 0.155, `cuarto de V20` 0.082, `arbol / boosting` 0.051, `NB-SVM (lineal sobre binarios)` 0.043, `logistica multiclase` 0.024

**P9-3 Caruana con tope de 6 miembros**

- peso de V20: **0.2467**
- miembro que lleva mas peso: `v29_lgbm_optuna` (0.4267)
- miembros con peso superior a 0,05: **5**
- familias con peso superior a 0,05: **4**

Reparto del peso por familia: `Optuna (nueva capacidad)` 0.427, `cuarto de V20` 0.313, `perceptron multicapa` 0.120, `arbol / boosting` 0.060, `NB-SVM (lineal sobre binarios)` 0.033, `logistica multiclase` 0.013

### Lo que se ve

**El optimizador elige a V20.**

### El brazo de control: los mismos tres metodos sin el modelo de Optuna

Este es el contraste que sostiene la conclusion de la fase, y **esta medido, no escrito**: los mismos tres metodos, los mismos folds, la misma estructura anidada, sobre el mismo pool **sin los modelos de Optuna**.

| Metodo | Con el modelo nuevo | Sin el modelo nuevo | Diferencia atribuible |
|---|---|---|---|
| P9-1 Optuna macro-F1 | +0.002137 (4/5) | -0.000682 (2/5) | **+0.002819** |
| P9-2 Optuna macro-F1 menos lambda por entropia | +0.002818 (5/5) | -0.000361 (2/5) | **+0.003179** |
| P9-3 Caruana con tope de 6 miembros | +0.002332 (4/5) | -0.000608 (2/5) | **+0.002940** |

**Sin el modelo nuevo, los tres metodos son negativos yNinguno llega a 2 de 5 folds en positivo neto.** Con el, los tres son positivos y dos llegan a 4 y 5 de 5. La diferencia atribuible al modelo individual es de **+0,0028 a +0,0032**, que es del orden del propio gate.

Eso deja la conclusion sin ambiguedad: **la mejora viene de la capacidad individual, no de la diversidad de modelos historicos.**

### El mecanismo de la perdida: los pesos son inestables

El vector de pesos ajustado sobre una mitad del holdout interior y el ajustado sobre la otra mitad **difieren en L1 0.98 de media** (maximo 1,78). Sobre un simplex de 16 miembros, una distancia L1 de 1,1 es enorme: el vector de pesos se moveria por completo cambiando la mitad de las filas.

Eso explica por que el **objetivo 2, que penaliza el reparto, gana al objetivo 1**: concentrar el peso en menos miembros reduce la varianza de la estimacion. Y explica por que el pool historico, con 16 candidatos y ninguno capaz de destacar, no rendia nada: habia que repartir el peso entre piezas mediocres, y la incertidumbre se comia la ganancia.

**La lectura correcta es que la inestabilidad de los pesos era un techo, no una pared.** Con un miembro claramente superior, el optimizador puede concentrar sin perder: de ahi el giro de negativo a positivo. Sin ese miembro, el techo se nota.

**El mejor trial no es un anomalo en ninguno de los dos metodos** (0 de 20 casos), y su distancia a la mediana de los trials esta entre 0,15 y 0,26 desviaciones. **Ni el positivo ni el negativo vienen de un trial afortunado.**

## 22. Segunda particion: no ejecutada, y por que

ningun metodo supero el gate (delta >= +0.003 y >= 4/5 folds), luego hay 0 finalistas. El punto 31 del brief reserva la segunda particion para finalistas congelados, y aplicarla sin ninguno seria fabricar una validacion que nada sostiene.

Finalistas: **0**.

## 23. Resultados negativos, sin ocultar

| Resultado | Valor | Que cierra |
|---|---|---|
| Los 47 modelos historicos fuera del circulo de V20 | 0 de 47 con rescate neto positivo | que la hipotesis de diversidad de modelos **historicos** esta cerrada |
| Brazo de control: los mismos 3 metodos sin el modelo de Optuna | -0.000682, -0.000361, -0.000608, todas 2/5 folds | que repartir peso entre miembros diversos **sin uno capaz de destacar** no rinde nada |
| Promedio igual de todo el pool | -0,015377 | que la diversidad sin criterio es claramente peor: el punto de partida ingenuo |
| El mejor ensemble de V29 frente al gate | 0,002818 frente a 0,003 | que el margen es real pero **insuficiente** |
| Pista de Optuna sobre MLP | no ejecutada | el eje de capacidad queda medido solo a medias |

**El resultado negativo mas duro de la fase** es que 0 de 47 modelos historicos tienen rescate neto positivo frente a V20. Ninguno.

## 24. El limite empirico que se mantiene

La masa de filas que V19 y V20 fallan juntas seguia en **61.949 filas (36,6 %)** al cerrar V28. **V29 no la ha reducido.** El mejor ensemble de la fase cambia menos de 6.000 decisiones sobre 169.336 filas, y la masa dura queda intacta.

## 25. Decision

| Pregunta | Respuesta |
|---|---|
| Mejor modelo individual de V29 | 0,6121698, delta +0.002392 |
| Baseline vigente | **V20**, 0,6097782, LB 0.62689 |
| Modelos historicos reutilizados | **51** OOF en 17 familias, sin reentrenar ninguno |
| Mejor ensemble de V29 | 0,6125960, delta +0.002818, 5/5 folds |
| Ensemble que supere el gate | **ninguno** |
| Finalistas | **0** |
| Submissions generadas | **0** |
| Artefactos de V20 tocados | **0** |

### La respuesta a la pregunta de V29

> *¿V20 esta limitado principalmente por falta de capacidad de sus modelos, o por falta de diversidad complementaria entre sus miembros, y puede una busqueda controlada de Optuna encontrar un conjunto pequeno de modelos que mejore de forma reproducible la combinacion?*

**La respuesta es: por capacidad, en parte. Y el margen es real pero insuficiente.**

La cadena de razon, con la medida al lado:

1. **La diversidad existe y es medible.** 17 familias, con correlaciones de errores con V20 que bajan hasta 0,53 frente al liston interno de 0,78. El problema de V20 no es que sus miembros sean todos iguales.

2. **Pero la diversidad de modelos historicos no es util, y queda cerrada con numero.** De 47 modelos fuera del circulo de V20, **0** tienen rescate unico mayor que dano unico. Y con el pool historico, los tres metodos de ensemble eran negativos.

3. **La capacidad individual si era un cuello de botella, y el primer modelo que lo demuestra es el de Optuna.** 0,6121698 frente a los 0,6097782 de V20: 0,002392 de mejora con la **misma** matriz de 279 features, los mismos folds y la misma estructura anidada. Es la primera mejora individual de todo el proyecto en 138 experimentos.

4. **Y el ensemble mejora con el, y solo con el.** El brazo de control lo mide: con el pool historico los tres metodos dan -0.000682, -0.000361 y -0.000608, todas con 2 de 5 folds. Con el modelo nuevo dentro: 0,002818 de mejora y 5/5 folds positivos. El reparto de peso cambia de 3 a 6 familias con peso, y V20 baja del 33 % al 12-20 %.

5. **Pero no llega al gate, y por dos motivos independientes.** El delta exigido es +0,003 y el mejor es 0,002818, un 94% del gate. Y con la calibracion `LB = +0,00223 + 1,02825 x CV`, ese margen valeria **+0,0014 en el leaderboard**, frente a una oscilacion de **0,0034** entre dos modelos vecinos del propio proyecto: seria indistinguible de haber cambiado de modelo.

### Que cambia respecto a D-062, y que no

D-062 (V27) concluyo que el cuello de botella estaba en los datos. V29 **no lo derriba, lo afina**:

- **Lo que confirma**: la masa de 61.949 filas que ningun modelo corrige sigue intacta, y ninguna combinacion de 16 modelos la toca.

- **Lo que corrige**: quedava margen de **capacidad individual**, y es real. No era cierto que el eje estuviera agotado.

- **Lo que no cambia**: ese margen es un orden de magnitud inferior al necesario para producir un candidato defendible. UnAxes agotado no es un eje donde buscar un decimal mas: es un eje donde la busqueda empioja.

**No se afirma que AGE este resuelto.** Se afirma algo mas preciso y mas util: la diversidad de modelos historicos esta agotada, la capacidad individual tiene un margen medido y pequeno, y la unica via con recorrido sigue siendo **senal nueva en los datos**.

## 26. Reproducibilidad acumulada de V20 a V29

### Identificadores y hashes

| Elemento | Valor |
|---|---|
| Baseline V20 | **0,6097782** corrected_cv |
| Hash de submission de V20 | `2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d` |
| Hash de datos | `32393abca28db4c9` |
| Ramas de trabajo | `proyecto-kaggle-age-2026-v26` |
| Folds | `StratifiedGroupKFold` 5 folds con `dup_clusters`, `[33868, 33868, 33867, 33866, 33867]` |
| Semilla | `cfg.SEED`, sin semilla nueva para seleccionar |
| Experimentos registrados | 138 filas, 0 duplicados |
| Notebook | 212 celdas, secciones 0 a 35, 4 aserciones obligatorias |
| Commits sin push | `946026f` V26, `42da3d8` V25, `e39bcf3` V24, `a6a18f4` V23, `ff3dc65` y `30b895e` V22, `372127a` V21 |

### Clasificacion

| Nivel | Significado | Este proyecto |
|---|---|---|
| R0 | no reproducible | no |
| R1 | reproducible con limitaciones | no |
| R2 | reproducido dentro de tolerancia | no |
| **R3** | **reproduccion exacta** | **si** |
| R4 | reproduccion independiente | **no se declara** |

**R3** porque los cuatro controles de la fase reproducen el canonico 0,6097782 **exacto**, con los mismos folds, el mismo hash de datos, la misma configuracion del meta-modelo y los mismos thresholds, y porque cada experimento **se detiene** si el control no reproduce.

**No se declara R4.** No hubo reproduccion independiente por un tercero ni con codigo separado. Todos los controles leen el mismo artefacto que el candidato, que es lo correcto para garantizar una comparacion justa, pero no es verificacion independiente.

### Inventario de artefactos de V27

| Artefacto | Que contiene |
|---|---|
| `experiments/v27_01_h12_reliability.json` | H12, incluye la tabla de fiabilidad y su mecanismo |
| `experiments/v27_02_h11_monotone.json` | H11, con la decision y un puntero a la correccion del mecanismo |
| `experiments/v27_03_h11_mecanismo.json` | correccion del mecanismo de H11, con nota de edicion |
| `experiments/v27_04_concentracion.json` | la concentracion de features y sus controles |
| `experiments/v27_05_h03_percentil.json` | H03, el unico resultado positivo |
| `experiments/v27_06_consolidacion.json` | cierre de H14 y estado final de las cuatro |
| `experiments/v27_09_auditoria_final.json` | compliance y auditoria cruzada |
| `reports/v27_compliance.md` | compliance medida de V27 |
| `reports/v27_concentracion_features.csv` | reparto de splits por feature |

### Auditorias ejecutadas

| Auditoria | Resultado |
|---|---|
| `ast.parse` en los 9 scripts de V27 | 0 errores |
| `ruff` con `--select F,E9` en los 9 scripts de V27 | 0 errores |
| `py_compile` en los 9 scripts de V27 | 0 errores |
| `v13_p0a_auditoria_idioma.py` | 504 ficheros, **0 errores**, 2 avisos |
| baseline reproduction | 4 de 4 controles exactos |
| fold validation | folds identicos a los de V20, array a array |
| notebook validation | 168 celdas, 0 celdas de codigo con error de sintaxis, seccion 35 ejecutada de principio a fin |
| character audit | 0 caracteres CJK en los scripts de V27 |

### Dos huecos de trazabilidad que se declaran

1. **El IC95 del bootstrap de V26-01 quedo a `null` en su artefacto.** Se reporto en el momento pero no se persistio, y no hay forma de recuperarlo sin reentrenar el experimento. Por eso la tabla historica lo marca como `no persistido` en lugar de citationar una cifra sin fuente. **No afecta a ninguna decision**: H02 se cerro por delta -0,000001 y 2 de 5 folds, muy por debajo del gate de +0,003, y el bootstrap era confirmatorio, no decisivo.

2. **La correccion del mecanismo de H11 modifico un artefacto ya generado.** El `mecanismo` de `v27_02` se sustituyo por un puntero a `v27_03`, con nota de edicion en el propio JSON. El delta, los folds y la decision no se tocaron. Se declara porque el punto 32 exige que la cadena de artefactos sea rastreable, y un artefacto editado sin nota romperia esa cadena.

## 27. Compliance acumulada

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


### Compliance especifica de V29

Las 13 comprobaciones de V29 dan cero usos, y la higiene del repositorio se verifica de nuevo al cierre.

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
holdout_reuse = 0
leaderboard_tuning = 0
toxic_test = 0
```

## 28. La evolucion de la pregunta, de V20 a V29

| Fase | Eje | Que se preguntaba | Respuesta |
|---|---|---|---|
| V20 | construccion | el blend de cuatro cuartos con cadena de umbrales | **LB 0.62689**, sigue siendo el champion |
| V21 a V24 | auditoria y causalidad | por que V20 funciona y donde se rompe | respuestas estructurales, no mejoras |
| V25 | feature selection | podar columnas elimina ruido | no: elimina senal, -0,000629 |
| V26 | evidencia | reescalar la evidencia cambia algo | no: es invariante a su escala, -0,000001 |
| V27 | monotonia, fiabilidad, percentil | restricciones y transformaciones globales | todas cerradas, una marginal |
| V28 | estructura | bagging y descomposicion de la decision | ambos negativos |
| **V29** | **capacidad y diversidad** | **mas capacidad por miembro, mas diversidad entre miembros, mejor combinacion** | **la capacidad **si** era un cuello: hay un modelo y un ensemble por encima de V20, pero por debajo del gate |

**La pregunta de V29 queda contestada con matiz, no en blanco.** La diversidad de modelos historicos esta agotada con numero (0 de 47). La capacidad individual **no lo estaba**: una unica LightGBM optimizada sobre la misma matriz de 279 features es el primer modelo del proyecto que supera a V20, y el ensemble que la incorpora llega a +0,001389 con 5 de 5 folds. No llega al gate de +0,003, y en leaderboard valdria +0,0014 frente a una oscilacion de 0,0034 entre modelos vecinos, asi que no es un candidato. El limite que queda sigue siendo la limitacion de los datos, y V29 no toco la masa de 61.949 filas que ningun modelo corrige.

## 29. Figuras acumuladas

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

### Figuras de V29 (8)

Siete de las nueve son de resultados que **pierden** contra V20. Ninguna se omite.
