# P32+ - P00, procedencia del baseline

## Las tres cifras y que hacer con cada una

| Cifra | Que es | Que se puede hacer con ella |
|---|---|---|
| **0.6134471** | corrected_cv del baseline operativo: la mezcla con los **pesos medios**, que es la unica reproducible desde los artefactos y exactamente lo que se envio | la referencia de comparacion de toda la fase |
| **0.6143347** | la misma familia de modelos **reejecutada** sobre los mismos grupos con otra semilla, con reentrenamiento desde cero | **no** es una segunda medicion independiente. Se reporta como reejecucion |
| **0.63302** | leaderboard de la submission de V30 | evidencia de transferencia. No se usa para seleccionar nada |

## Por que las dos primeras NO son equivalentes

La diferencia es **+0.0008876**, y el brief tiene razon en exigir que no se trate como ruido menor. Se descompone en dos fuentes:

**1. Re-particion.** La segunda particion usa los **mismos grupos** `dup_clusters` con otra semilla. El artefacto de V30 ya lo registra: `particion_realmente_distinta = False`, con solapes por fold de 7155 a 7272 filas, alrededor del **21 %** de cada fold de validacion. El otro 79 % cambia, y con el la composicion de clases de cada fold. Como macro-F1 es no lineal, reagrupar las mismas predicciones ya mueve el numero.

**2. Reentrenamiento.** La segunda particion **si reentreno** V20 y los dos LightGBM desde cero. Y el proyecto ya midio que LightGBM no es reproducible bit a bit entre corridas con distinta carga de CPU (el fold 2 dio 608 frente a 644 iteraciones).

## La prueba de que la diferencia es ruido y no una propiedad del modelo

Si la diferencia entre 0,6134471 y 0,6143347 fuera una propiedad del candidato, el baseline V20 deberia quedarse quieto. No se queda:

| | primera particion | segunda particion | movimiento |
|---|---|---|---|
| V20 | 0.6097782 | 0.6104285 | **+0.0006503** |
| V30 | 0.6134471 | 0.6143347 | **+0.0008876** |
| delta | +0.0036689 | +0.0039062 | **+0.0006790** |

El baseline se movio casi tanto como el candidato y **en la misma direccion**. Y el delta candidato-minus-V20 **crecio**. Es decir: la segunda ejecucion no debilita la conclusion de V30, la refuerza. Lo que cambia no es el modelo, es el estimador.

## La descomposicion, y donde fallo la primera hipotesis

La primera version de este script le atribuyo la diferencia a **ruido de re-particion**, que era la explicacion obvia. **Era falsa.** Al medir la desviacion tipica de 200 re-particiones salio **0.0000161**, y la diferencia observada esta a **55 desviaciones**. Estar tan lejos no se atribuye al azar.

La comprobacion que faltaba es directa y concluyente: se toman **las mismas predicciones por fila** y se re-agrupan con la **estructura exacta** de la segunda particion de V30.

| Estructura de folds, mismas predicciones | media de folds |
|---|---|
| la original | 0.6134526 |
| la de la segunda particion | 0.613444 |

La reagrupacion aporta **-0.0000085**, o sea el **1.0%** de la diferencia total. Y las proporciones de clase de los folds son identicas entre particiones (desviacion maxima 0.000034).

**Conclusion: la diferencia es reentrenamiento.** La segunda particion de V30 reentreno los tres miembros desde cero sobre conjuntos de entrenamiento distintos, asi que **0,6134471 y 0,6143347 no son dos mediciones del mismo modelo: son dos modelos distintos de la misma familia.**

## La medicion que si importa: cuanto ruido tiene el estimador

La fuente 1 se puede medir **aislada y exactamente**. Se toma el OOF de V30 **congelado**, cuyas predicciones por fila son identicas, y se re-agrupa bajo 200 re-particiones distintas de las mismas filas. Sin reentrenar nada. Los grupos se reparten enteros, nunca filas sueltas, para que ningun near-duplicate pase de train a validacion y el ruido medido sea real.

| Magnitud | Valor |
|---|---|
| CV original | 0.6134471 |
| media de los 200 resplits | 0.6134188 |
| desviacion tipica | **1.61e-05** |
| minimo | 0.6133689 |
| maximo | 0.6134624 |
| amplitud total | 9.35e-05 |
| IC95 del estimador | [0.6133817, 0.6134481] |

El estimador tiene una desviacion de **0.0000161**, y el rango completo entre el minimo y el maximo es de 0.0000935. Reagrupar las mismas predicciones **no** es la fuente de incertidumbre relevante: la escala relevante es la del **reentrenamiento sobre otro reparto**, que el proyecto acaba de medir en ~0.0008876.

## Por que esto recalibra los gates del proyecto

El gate de PROMISING pide `delta >= +0,0015` con `4/5 folds` a favor. Los folds dan estabilidad **dentro** de una particion, pero no dicen nada de cuanto se mueve la cifra cuando cambia el reparto de entrenamiento. Y aqui hay que distinguir dos cosas muy distintas:

- **Ruido del estimador (reagrupar las mismas predicciones): 0.0000161.** Despreciable.
- **Variabilidad de entrenamiento (misma receta, otro reparto): ~0.0008876.** Cuarenta veces mayor, y es la que manda.

El gate de +0,0015 es **1.7 veces** esa segunda escala. Es exigente, pero no room-for-comfort: un delta que pase el gate por 1,6 veces sigue pudiendo ser entrenamiento.

Y los dos deltas que V31+ reporto, `+0,0000642` y `+0,0002948`, son **14 y 3 veces mas pequenos** que esa escala. Eso no los vuelve negativos, que es lo que dijo V31+. Los coloca donde de verdad estan: **muy por debajo de lo que el instrumento puede resolver**. Y por eso P32+ no va a perseguir el alpha: el problema no es afinar un parametro, es que la senal medida es mas pequena que la resolucion del metodo.

## La conclusion que este P00 obliga

El objetivo de la fase era superar **0.63302** en el leaderboard. Con una desviacion del estimador de 0.0000161 y una oscilacion entre modelos vecinos de 0,0034 que ya midio el proyecto, **superar ese numero exige un delta de CV de al menos 0.0034961** en el leaderboard, es decir unos **0.0033066** en CV para que sea atribuible al modelo y no al azar.

Ese es el liston real de esta fase, y ahora esta medido en vez de supuesto.

## Tabla de procedencia completa

| Metrica | Valor | Particion | Commit | Artefacto | Semilla | Condiciones | Reproducibilidad |
|---|---|---|---|---|---|---|---|
| V20 corrected_cv (primera particion) | 0.6097782 | StratifiedGroupKFold 5 sobre dup_clusters, semilla 42 (folds de V20) | 372127a | experiments/v20_* | 42 | 279 features, cadena de V20 con HIERBA sub-parametrizado | R3 bit a bit (dif 0,00e+00) |
| V29 mejor LightGBM individual | 0.6121698 | mismos folds de V20 | b044db8 | experiments/v29_04_optuna_lightgbm.json | Optuna por fold | 279 features, Optuna anidada, holdout interior del 20% | R2 (Optuna con semilla fija, determinista aqui) |
| V30 CV PUBLICADO (pesos por fold) | 0.6130051 | folds de V20 | 7bd3d42 | experiments/v30_02_ensemble.json (campo tabla) | 42 | mezcla de 3 miembros con los pesos POR FOLD que Optuna optimizo | R1 NO REPRODUCIBLE: esos pesos por fold no se persistieron |
| V30 CV OPERATIVO (pesos medios) | 0.6134471 | folds de V20 | 7bd3d42 / fc77f1c | experiments/v30_02_ensemble.json (manifest) + features/*.npy | 42 | la MISMA mezcla, con los pesos MEDIOS entre folds, que es lo que uso la submission | R3 reproduce exactamente; es el que se envio |
| V30 segunda particion (candidata) | 0.6143347 | mismos grupos dup_clusters, semilla 987654, solapes por fold 7155-7272 (~21% del fold de validacion) | 7bd3d42 | experiments/v30_05_segunda_particion.json | 987654 | configuracion CONGELADA, pero V20 y los dos LightGBM REENTRENADOS desde cero sobre los folds nuevos | R2: depende de la reentrenada, y LightGBM ya se midio no reproducible bit a bit entre ejecuciones con distinta carga de CPU |
| V20 segunda particion (recomputado) | 0.6104285 | la misma segunda particion | 7bd3d42 | experiments/v30_05_segunda_particion.json | 987654 | V20 reentrenado en los folds nuevos | R2 |
| leaderboard de V30 | 0.63302 | test oficial de AGE, 112.892 filas | posterior a 7bd3d42 | submissions/age_submission_v30_final_candidate.csv | n/a | pesos MEDIOS congelados; solo transform -> predict | fichero intacto, sha256 f543a715a1881c61 |
| leaderboard de V20 | 0.62689 | test oficial de AGE | 372127a | n/a | n/a | n/a | n/a |
| V30 CV recalculado en este script | 0.6134471 | folds de V20 | fc77f1c53dc591181375a3af70ac8ef40e2c10be | experiments/p32_00_provenance.json | 42 | reconstruccion desde el pool OOF y el manifest de pesos | coincide con el operativo: True |

## Lo que este P00 NO permitio encontrar

Una particion **verdaderamente independiente** no se puede construir aqui, y conviene decirlo: los grupos `dup_clusters` cubren el dataset entero, asi que cualquier particion agrupada nueva **reparte los mismos grupos** de otra manera. La unica forma de tener una particion independiente de verdad seria dejar clusters enteros fuera, lo que reduce el numero de filas disponibles para validar y cambia el estimador a otro. Es un experimento que el brief no pide y que costaria comparabilidad con todo el historico, asi que se documenta como limite y no se ejecuta.
