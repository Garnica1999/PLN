# V29 - Optuna y busqueda de ensemble diverso

**Fecha:** 2026-09-29 · **Rama:** `proyecto-kaggle-age-2026-v29` · **Baseline congelado:** 0,6097782 · **Champion:** V20, LB 0.62689

La fase completa de V29, generada desde los artefactos. **Ninguna cifra esta escrita a mano si existe un artefacto del que leerla.**

# V29 - Optuna y busqueda de ensemble diverso

**Fecha:** 2026-09-29 · **Rama:** `proyecto-kaggle-age-2026-v29` · **Baseline congelado:** 0,6097782 · **Champion:** V20, LB 0.62689

## 1. Motivacion y pregunta

V25 a V28 habian cerrado la representacion, la escala de la evidencia, la monotonia, los percentiles, la fiabilidad por modelo, la estructura ordinal, el bagging del meta-modelo, la descomposicion pairwise y la geometria del consenso. Lo que **no** se habia tocado es la capacidad de un miembro individual ni la composicion del ensemble.

La hipotesis central de V29 es una sola:

> V20 puede no estar limitado por la capacidad de sus miembros, sino por la **falta de diversidad util entre sus errores** y por una combinacion de pesos fija.

La palabra clave es **complementariedad**, y el punto 3 del brief es el que gobierna todo lo demas: *mejor modelo individual* no implica *mejor miembro del ensemble*.

## 2. Estado inicial

| Concepto | Valor |
|---|---|
| Baseline vigente | **0,6097782** corrected_cv, reproducido bit a bit en P0 |
| Leaderboard de V20 | **0.62689**, envio sin cambio |
|.Hash de la submission de V20 | `2c964af78c90bf08...`, intacto |
| Candidato para submission | ninguno |

## 3. Inventario historico: la surprise de V29

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

## 4. Diversidad: la medicion que decide la fase

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

## 5. Gap analysis: donde V29 casi reabria una familia cerrada

Dos de los tres ejes de V29 se parecen a familias ya cerradas, y decirlo es parte del trabajo:

- **Composicion del ensemble** es, en mecanismo, **H04** (ponderar modelos) y **H06** (apilar un segundo nivel), las dos cerradas con numero. El veredicto es **PARTIALLY_TESTED**, no GENUINELY_NEW, y la distincion que lo salva es concreta: H04 y H06 reevaluaron los miembros que V20 ya tenia; V29 amplía el conjunto de candidatos a modelos de 17 familias que nunca han entrado en un ensemble.

- **Capacidad individual** si es genuinamente nueva: ninguna familia cerrada de V25 a V28 modifico la capacidad de un miembro, todas transformaron la combinacion o la representacion.

- **Diversidad** es genuinamente nueva y ya medida, sin gastar un experimento.

Y queda un criterio de cierre declarado por si el mecanismo resultara ser el viejo: **si el optimum se apoya en una sola estructura, es H04 de nuevo y se cierra por equivalencia.**

Optuna queda ademas prohibido por decision propia sobre threshold, feature selection, calibracion, pairwise, ordinal, robust loss, grafo, ventana local: usar un metodo de busqueda nuevo sobre una familia cerrada no la abre, la reembala.

## 6. El pool: cuatro filtros declarados

De 52 candidatos de partida quedaron **16** en el pool, con los caidos por filtro visibles:

| Filtro | Caidos | Motivo |
|---|---|---|
| Util | 15 | CV por debajo de 0,50: ruido, no senal |
| Diverso | 9 | familia de V20 y correlacion de errores por encima del liston |
| No redundante | 0 | mismas decisiones que otro ya elegido |
| Tamano | 13 | tope de 16 miembros |

El criterio **no es el CV maximo**, que es lo que el brief prohibe. Es CV util, mas diversidad, mas rescate normalizado, con la tension entre rescate y dano **visible** en la tabla en vez de escondida en un score.

## 7. Optuna individual: estructura anidada

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

## 8. Composicion del ensemble: el resultado central

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

## 9. El control que decide: ¿es esto H04 otra vez?

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

## 10. Segunda particion: no ejecutada, y por que

ningun metodo supero el gate (delta >= +0.003 y >= 4/5 folds), luego hay 0 finalistas. El punto 31 del brief reserva la segunda particion para finalistas congelados, y aplicarla sin ninguno seria fabricar una validacion que nada sostiene.

Finalistas: **0**.

## 11. Resultados negativos, sin ocultar

| Resultado | Valor | Que cierra |
|---|---|---|
| Los 47 modelos historicos fuera del circulo de V20 | 0 de 47 con rescate neto positivo | que la hipotesis de diversidad de modelos **historicos** esta cerrada |
| Brazo de control: los mismos 3 metodos sin el modelo de Optuna | -0.000682, -0.000361, -0.000608, todas 2/5 folds | que repartir peso entre miembros diversos **sin uno capaz de destacar** no rinde nada |
| Promedio igual de todo el pool | -0,015377 | que la diversidad sin criterio es claramente peor: el punto de partida ingenuo |
| El mejor ensemble de V29 frente al gate | 0,002818 frente a 0,003 | que el margen es real pero **insuficiente** |
| Pista de Optuna sobre MLP | no ejecutada | el eje de capacidad queda medido solo a medias |

**El resultado negativo mas duro de la fase** es que 0 de 47 modelos historicos tienen rescate neto positivo frente a V20. Ninguno.

## 12. El limite empirico que se mantiene

La masa de filas que V19 y V20 fallan juntas seguia en **61.949 filas (36,6 %)** al cerrar V28. **V29 no la ha reducido.** El mejor ensemble de la fase cambia menos de 6.000 decisiones sobre 169.336 filas, y la masa dura queda intacta.

## 13. Decision

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

## 14. Reproducibilidad, compliance y git

| Comprobacion | Resultado |
|---|---|
| Control absoluto de V20 en P0 | 0,6097782 reproducido, folds identicos, OOF identico bit a bit |
| Folds | los mismos `StratifiedGroupKFold` de V20, sin cambiar grupos ni dataset |
| Estructura de Optuna | anidada, con asercion que aborta si el holdout interior toca la validacion exterior |
| Optuna | `TPESampler`, semilla 20260929 mas el indice de fold, segun artefacto trials por fold |
| Semilla de los folds | heredada de `cfg.SEED`, no se cambio |
| Compliance | **13 comprobaciones, todas en cero usos** |
| Higiene del repositorio | los 2 directorios ajenos a AGE siguen fuera de la rama: True |
| Test, holdout y leaderboard | **no utilizados en ningun punto de V29** |
| Artefactos de V20 tocados | **0** |
| Submissions generadas | **0** |

**No se declara R4.** No hubo reproduccion independiente por un tercero. La compliance busca firmas de las practicas prohibidas y no puede demostrar que no ocurran de otra forma.

### Una limitacion deOptuna que se declara

Optuna es un metodo de busqueda estocastica. Con `TPESampler` y semilla fija el resultado es reproducible **en la misma version de las librerias**, pero no se declara exactitud bit a bit entre reproducciones: una version distinta de LightGBM o de Optuna puede devolver otra solucion. Por eso el protocolo no depende de un unico trial, sino del gate de 5 folds y del bootstrap.

### La fuga indirecta, medida

Los OOF historicos de los miembros se produjeron con los folds de V20, asi que la prediccion de una fila del outer_train viene de un modelo que si vio parte del outer_val. Al ajustar pesos sobre outer_train, esa informacion llega de forma indirecta.

Se mide ajustando los pesos dos veces sobre mitades distintas del holdout interior: la distancia L1 entre los dos vectores es de 1,11 de media. **Los pesos son inherentemente inestables**, y eso no es culpa de la fuga: es una propiedad de estimar 16 pesos sobre un holdout de un solo fold. La fuga podria exacerbarlo, y un resultado **negativo** no se infla: aqui no hay nada que inflar, porque el metodo perderia por la misma razon con datos limpios.

## 15. Git

Rama `proyecto-kaggle-age-2026-v29`, creada desde `proyecto-kaggle-age-2026-v28`. **Sin push, sin merge, sin force-push y sin reset destructivo.** No se incluye `home/` ni `kaggle_toxic_2026_puj/`, ni ningun artefacto ajeno al proyecto.

### Figuras de V29 (8)

| Figura | Que muestra |
|---|---|
| `v29_cv_vs_diversidad.png` | CV individual contra diversidad frente a V20 |
| `v29_folds_ensemble.png` | consistencia fold a fold de cada metodo |
| `v29_maestra.png` | corrected_cv por metodo con V20 congelado, delta frente a V20 con el gate, y reparto de peso del ensemble |
| `v29_model_diversity_heatmap.png` | correlacion de errores entre los 28 mejores modelos y los 4 cuartos de V20 |
| `v29_optuna_lightgbm.png` | historia de optimizacion, distribucion de trials y outer_val por fold de LightGBM |
| `v29_pesos_ensemble.png` | F1 por clase de cada metodo y rescate frente a dano |
| `v29_pool_diversidad.png` | el pool en el plano CV contra correlacion de errores, con el liston de V20 dibujado |
| `v29_rescate_vs_dano.png` | rescate frente a dano de los 25 con mas rescate |

Ninguna oculta un resultado negativo: **siete de las nueve son de resultados que pierden contra V20.**

---

**Cierre de V29.** La hipotesis central queda contestada **con matiz**: la diversidad de modelos historicos no es el cuello de botella y queda cerrada con numero, pero la **capacidad individual si lo era parcialmente**, y una busqueda controlada de Optuna encuentra un modelo y un ensemble por encima de V20. No llegan al gate, asi que V20 permanece como baseline absoluto en 0,6097782. No se genera ninguna submission, no se toca el test, el holdout, el leaderboard ni ningun artefacto de V20.
