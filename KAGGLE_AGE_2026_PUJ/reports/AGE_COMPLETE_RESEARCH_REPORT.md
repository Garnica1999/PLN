# AGE - Informe completo de la investigacion, de V20 al cierre actual

**Fecha:** 2026-09-28 · **Rama:** `proyecto-kaggle-age-2026-v26` · **Baseline congelado:** 0,6097782 · **Champion:** V20, LB 0.62689

Este informe reconstruye **toda** la trayectoria relevante de AGE desde V20 hasta el cierre actual. No se limita a V26 ni a V27, y no omite ninguna fase negativa: la mayor parte de lo que se midio en esta etapa fue negativo, y ese resultado es el producto.

**Ninguna cifra de este informe esta escrita a mano si existe un artefacto del que leerla.** El generador es `scripts/v27_10_informe_completo.py` y lee de `experiments.csv`, de `reports/v26_master_results.csv`, de los artefactos JSON y del notebook. Las unicas constantes fijadas en el generador son las que no viven en ningun artefacto: los scores ya obtenidos del leaderboard, los hashes y el historial, y cada una lleva su fuente declarada.

## 1. Resumen ejecutivo

| Concepto | Valor | Fuente |
|---|---|---|
| Baseline final | **0,6097782** corrected_cv | canonico del proyecto, reproducido por los 4 controles |
| Mejor corrected_cv | **0,610051** (H03, MARGINAL) | `experiments/v27_05_h03_percentil.json` |
| Mejor score verificado en leaderboard | **0.62689** (V20) | envio verificado de V20 |
| Mejor candidato validado internamente | H03, MARGINAL, no finalista | su IC95 incluye el cero y esta 11 veces bajo el gate |
| Numero de experimentos | **135** filas en `experiments.csv` | registro |
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

## 10. Reproducibilidad

### Identificadores y hashes

| Elemento | Valor |
|---|---|
| Baseline V20 | **0,6097782** corrected_cv |
| Hash de submission de V20 | `2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d` |
| Hash de datos | `32393abca28db4c9` |
| Ramas de trabajo | `proyecto-kaggle-age-2026-v26` |
| Folds | `StratifiedGroupKFold` 5 folds con `dup_clusters`, `[33868, 33868, 33867, 33866, 33867]` |
| Semilla | `cfg.SEED`, sin semilla nueva para seleccionar |
| Experimentos registrados | 135 filas, 0 duplicados |
| Notebook | 168 celdas, secciones 0 a 35, 4 aserciones obligatorias |
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

## 11. Compliance final

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

## 12. Conclusion cientifica: donde esta el cuello de botella de AGE

La pregunta es donde queda el limite de AGE despues de V20 a V27. Se responde por ejes, y se separa lo que tiene evidencia negativa de lo que no se ha medido.

| Eje | Estado | Evidencia |
|---|---|---|
| **Representacion de features** | agotado | H02 (escala) y H03 (percentil) medidos; H15 (seleccion) cerrado en V25 |
| **Combinacion de predicciones** | agotado | H04, H05, H06, H13 cerrados; H14 cerrado por equivalencia; V20 ya es el segundo nivel |
| **Calibracion** | agotado | H01 cerrado en D-029 |
| **Estructura ordinal** | agotado | H10 y CORAL (D-058) cerrados con numero |
| **Meta-modelo** | agotado | H07, H08, H09, H11, H12 cerrados |
| **Limitacion de los datos** | **lo que queda abierto** | el 38,3 % de filas que V19 y V20 fallan juntas |

### La respuesta, sin adornos

**El cuello de botella restante de AGE no esta en el modelo, esta en los datos.** Ninguna de las cinco propiedades del meta-modelo que se podian cambiar sin reentrenar todo el sistema queda abierta, y cada una tiene evidencia negativa reproducible. Y el mecanismo de por que fallaron es uno solo y esta medido: el meta-modelo de V20 abre **7 de sus 279 features**, todas del bloque de evidencia de los miembros.

Eso significa que **casi toda la ingenieria acumulada en las 279 features llega al resultado a traves de la prediccion de los miembros, no a traves del meta-modelo**. Añadir features, podarlas, reescalarlas, imponerles monotonia o darles fiabilidad son intervenciones sobre la parte del sistema que ya no se lee.

Lo que si queda abierto es la unica via que ninguna de las familias toco: **informacion nueva desde los datos**. Y dentro de eso, la masa de 38,3 % de filas que ambos modelos fallan juntas es la unica grande y medida, y esta agotada con la senal actual.

**No se afirma que el problema este resuelto.** Se afirma que el espacio de mejoras que era accesible sin senal nueva esta agotado, con evidencia.

## 13. Conclusion practica

| Pregunta | Respuesta |
|---|---|
| Baseline vigente | **V20**, 0,6097782 corrected_cv, LB 0.62689 |
| Mejor candidato interno | H03 (percentil), MARGINAL, no finalista |
| Existe candidato listo para submission | **No** |
| Debe cerrarse la investigacion de esta via | **Si**: no quedan lineas metodologicamente abiertas |
| Se genera submission | **No**, y no se genera automaticamente |
| Se modifica V20 | **No**, y no se modificara sin una hipotesis nueva con mecanismo distinto |

### Lineas definitivamente descartadas

1. **Seleccion supervisada de features.** Cerrada en V25, y no se reintenta con otro K ni con otro selector sin defecto metodologico demostrado.

2. **Reescalada de la evidencia** en cualquiera de sus formas: afin, percentil, temperatura o isotonica. Las cuatro quedan contestadas: H01 y H02 cerradas con numero, H03 medida y marginal, y un cambio de combinador ya cerrado como H04 y H06.

3. **Restricciones monotonas internas** del meta-modelo: no pueden actuar, porque las features con relacion defendible no son las que el modelo abre.

4. **Fiabilidad por modelo y clase** en formulacion agregada: es una constante por fila.

5. **Estructura ordinal y correccion de coste**: cerradas con numero en V22 y V23.

6. **Interacciones, meta-modelos no lineales distintos y meta-modelos por clase**: cerradas con numero, y ademas un arbol no puede aprovechar del todo una interaccion.

### Que queda por delante

No una hipotesis, sino una **decision de Carlos**. La unica via con recorrido sin explorar es senal nueva, y las restricciones de AGE (sin transformers, sin LLM, sin embeddings, sin etiquetas externas) cierran las fuentes de senal mas obvias. Cualquier linea nueva deberia empezar por demostrar que aporta senal que el bloque de evidencia de los miembros no aporta, y no por una transformacion de lo que ya existe.

## 14. Figuras

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

---

**Cierre de la fase.** V20 permanece como baseline absoluto en 0,6097782. El leaderboard no se utilizo para decidir entre hipotesis, no se toco el test ni el holdout, no se genero ninguna submission y no se modifico V20. La investigacion termina aqui porque las hipotesis defendibles restantes han sido contestadas o cerradas por evidencia, no porque se haya agotado el presupuesto.
