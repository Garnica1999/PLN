# V29 - Gap analysis de los tres ejes

Clasificacion previa a cualquier entrenamiento. La pregunta no es si V29 trae una idea nueva, sino **que parte de sus tres ejes lo es**, porque dos de ellos se parecen a familias que V25 a V28 cerraron con numero.

## Veredicto

| Eje | Veredicto | Experimentos | Razon |
|---|---|---|---|
| capacidad de modelo individual | **GENUINELY_NEW** | 2 | ninguna familia cerrada de V25 a V28 modifico la CAPACIDAD de un miembro individual; todas transformaron la combinacion, la escala o la representacion. Buscar mejores hiperparametros sobre l |
| diversidad de errores entre miembros | **GENUINELY_NEW** | 0 | el historial tiene 51 OOF en 17 familias y 47 modelos que NUNCA han formado parte de un ensemble. La pregunta de si alguno aporta senal que los cuatro cuartos de V20 no poseen no se ha formu |
| composicion del ensemble | **PARTIALLY_TESTED** | 1 | el MECANISMO de optimizar pesos es el de H04 y H06, y las dos estan cerradas. Lo que es genuinamente nuevo es el CONJUNTO de candidatos: H04 y H06 solo reevaluaron los miembros que V20 ya te |

Presupuesto previsto: **3 de 10** autorizados. Las familias historicas reached 17, con 51 OOF reutilizables, de los que **47 nunca han formado parte de un ensemble**.

## El problema serio: optimizar pesos roza H04 y H06

Optimizar los pesos de un ensemble es, en apariencia, lo mismo que **H04** (ponderar modelos) y **H06** (apilar un segundo nivel), y las dos estan cerradas con numero. Si V29 fuera solo eso, estaria reabriendo una familia cerrada.

La distincion que lo salva es concreta: **cuestion sobre quien es miembro**.

- **H04 y H06** respondieron: *mejora reponderar o re-apilar los miembros que V20 ya tenia?* Los candidatos eran los mismos cuatro. La respuesta fue no.

- **V29** pregunta: *aporta senal alguno de los otros modelos historicos, que nunca han entrado en un ensemble y tienen estructuras distintas?*

Por eso el veredicto de composicion es **PARTIALLY_TESTED** y no GENUINELY_NEW: el mecanismo es el mismo, el conjunto de candidatos es distinto.

**Y hay un criterio de cierre declarado por si el mecanismo resulta ser el viejo:** si el optimum de Optuna se apoya en un solo modelo, o devuelve los cuatro cuartos de V20 con otros pesos, entonces es H04 de nuevo y la composicion se cierra por equivalencia. Si reparte peso en familias estructuralmente distintas, entonces hay algo que decir.

## La referencia de diversidad: los miembros de V20

| Medida | Valor |
|---|---|
| Correlacion de errores media entre los 4 cuartos | **0.7802** |
| Discrepancia de decision media | 0.1644 |
| Discrepancia maxima entre dos cuartos | 0.1906 |
| Par de cuartos con mayor correlacion | 0,8675 (arbol contra cadena) |
| Par de cuartos con menor correlacion | 0,7461 (mlp contra cadena) |

Este es el liston que un candidato tiene que superar: **para ser mas diverso que los miembros actuales, su correlacion de errores con V20 tiene que bajar de 0.7802**.

## Los tres hallazgos de la medicion de diversidad

1. **28 modelos** tienen correlacion de errores por debajo del liston, es decir, mas diversos que cualquier par de los miembros de V20. De esos, **13** tienen ademas CV suficiente para no ser ruido (>= 0,50).

2. **0 modelos** tienen rescate unico mayor que dano unico frente a V20. Es el dato mas duro contra V29 y **no se oculta**: por si solo, ningun modelo historico es un complemento neto de V20.

3. El mayor rescate bruto lo tiene un modelo con CV 0,06, que *resucita* filas por azar. Es la trampa de leer el rescate sin skill, y por eso la metrica de decision es la fraccion de los errores de V20 que el modelo corrige, no el numero absolute.

Que el punto 2 sea desfavorable **no cierra la hipotesis de diversidad**, y esa es la distincion importante: un modelo puede tener rescate neto negativo frente a V20 y aun asi aportar a un blend ponderado, porque lo que un ensemble explota es la decorrelacion de los errores, no el acierto individual. Medir eso exige el ensemble, que es lo que hacen P9 y P10.

## Sobre que NO se permite usar Optuna

Optuna es un metodo de busqueda, no una tecnica. Aplicarlo sobre estas familias seria reabrir con mas trials lo que V25 a V28 ya cerraron:

| Familia | Decision que la cerro |
|---|---|
| feature selection supervisada | V25, D-060, -0,000629: podar columnas elimina senal, no ruido |
| escalado afil de la evidencia | V26-01, D-061, -0,000001: la evidencia es invariante a su escala |
| transformacion percentil | V27-05, MARGINAL +0,000272, IC95 incluye el cero |
| fiabilidad constante por modelo y clase | V27-01, H12, delta exactamente 0 |
| restricciones monotonas | V27-02, H11, -0,000355 |
| reponder blends equivalentes | V27, H14, cerrada por equivalencia funcional sin experimento |
| calibracion de probabilidades | H01, D-029 |
| ponderacion de modelos | H04, cerrada con numero |
| apilado de segundo nivel | H06, cerrada: V20 ya ES el segundo nivel |
| familias ordinales (H10, CORAL) | H10 y CORAL, D-058 |
| bagging del meta-modelo | V28-02, A, -0,000312 |
| descomposicion pairwise | V28-03, C, -0,067553 |
| geometria del consenso | V28, B, superconjunto ya implementado: +0,0001 |
| fusion de evidencia Dirichlet | V28, D, equivalente a H04 mas H06 |
| incertidumbre explicita | V28, E, duplicacion de B |
| subespacio aleatorio | V28, F, ya implementado en colsample_bytree |

Y queda prohibido por decision propia usar Optuna sobre: threshold, feature selection, calibracion, pairwise, ordinal, robust loss, grafo, ventana local. El punto 43 del brief lo pide y aqui se convierte en lista de verificacion.

Optuna se acota a tres ejes: **capacidad de modelo individual**, que nadie ha tocado en este proyecto; **diversidad**, que es medicion; y **composicion del ensemble**, con el conjunto de candidatos ampliado a modelos que nunca han entrado en uno.
