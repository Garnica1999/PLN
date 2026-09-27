# AGE · Informe de decision ejecutiva

- **Fecha**: 2026-09-26 · **Champion**: V12 · **Rama**: `kaggle-age-2026-puj-v18`
- **Experimentos registrados**: 109 · **Adopciones en las ultimas seis fases**: **0**
- **Este documento no propone ninguna fase.** Se genero desde los artefactos, para que sus cifras no puedan desincronizarse de los experimentos.

## 1. Estado actual

```
CLOSE AGE - V12 REMAINS CHAMPION
```

| Magnitud | Valor |
|---|---|
| macro-F1 pooled (CV LOFO de 5 folds, agrupado) | **0,6005017473** |
| macro-F1 media de folds | 0,6004103057 |
| Leaderboard del envio `age_submission_v12.csv` | **0.62182** |
| Leaderboard de la replica `age_submission_v12_reproducible.csv` | 0.62188 |
| Motivo de la diferencia | 93,88 % de acuerdo de fila, unos 6.900 test difieren: **0,00006** de diferencia en el leaderboard |
| Envios puntuados registrados | 11 |
| Clases | 6, balanceadas entre 16,04 % y 17,04 % de la masa |

V12 es `0,5 x MLP(64) + 0,5 x LogReg` sobre 279 variables: 78 columnas de 13 miembros OOF, features densas, features estructuradas, 73 columnas supervisadas cross-fitted y 6 de NB-SVM de palabra. **Permanece intacto.**

## 2. Que se intento, fase por fase

| Fase | Enfoque | Resultado | Decision |
|---|---|---|---|
| V13 | variables: evidencia por token, pronombres, modalidad, conjunciones; P3a especialista de frontera | 16 experimentos, **0 adopciones**; la fusion de variables quedo en -0,0002 | D-041 |
| V14 | relajar recursos: analizador linguistico espanol | **mejor bloque aislado del proyecto** (0,2409), fusion **-0,0015** con 0 de 4 folds en la segunda particion | D-044 |
| V15 | poda de redundancia (VIF, correlacion, clusters) | 197 variables conservadas, **-0,0040** con 0 de 5 folds; la redundancia es funcional, es el ensamble de estimadores debiles | D-045 |
| V16 | arquitectura del meta-modelo | A2 apilamiento **+0,0006** con 3 de 5 folds, el mejor delta del proyecto, con el IC95 contemplando el cero; fronteras +0,0102 y +0,0157 | D-046 |
| V17 | especialistas de frontera y decision local | cinco formulaciones: **+0,0002, +0,0000, -0,0001, -0,0002**; el techo con arbitro perfecto es +0,0107, pero ningun selector lo acerca | D-047 |
| V18 | funcion de entrenamiento: re-peso, margen, termino aditivo, destilacion | cuatro mecanismos, **+0,0008 como maximo** y 0 de 5 folds; la destilacion recupera el 0,8 % de lo que el especialista tiene | D-049 |

Lineas adicionales cerradas con numero: D-037 (33 capas de decision ordinal), D-039 (0 de 72 columnas lexicograficas no redundantes), especialistas por longitud (+0,0000), BM25 (-0,0002), kNN sobre SVD (0,3088 sin aporte), Ridge y SGD como meta (-0,0028), y la recuperacion train-test descartada.

## 3. Lo que se audito en esta sesion, y que se encontro

### 3.a La hipotesis de compresion de outputs: muerta por construccion

Los 62 artefactos de train de `features/` se auditaron uno a uno. **Todos estan consumidos o cerrados con numero.** Y el punto decisivo no es el consumo sino la forma: **todos** los OOF de miembros estan ya comprimidos a seis columnas o menos, porque cada miembro es un modelo de seis clases y no puede emitir mas. La compresion **no la introduce V12**, se introdujo al construir cada miembro. La hipotesis no falla por evidencia: **no se puede formular**.

### 3.b D-049 estaba mal razonado, y ahora esta medido

D-049 cerro V18 diciendo que la frontera es una **minoria de la masa**. Leida al pie de la letra es **falsa**: `18-24` son el 16,93 % y `25-34` el 16,74 %, luego juntas son **33.67 %**, un tercio. Y la lever **no es pequena**:

| Correcciones | `18-24\|25-34` | `25-34\|35-49` |
|---|---|---|
| 250 | +0.0015 | +0.0016 |
| 500 | +0.0031 | +0.0031 |
| 1,000 | +0.0061 | +0.0063 |
| 2,000 | +0.0122 | +0.0125 |
| 5,000 | +0.0303 | +0.0308 |

Medido por contrafactual sobre el OOF existente, sin reentrenar nada. El gate de adopcion de +0,003 **se cruza con unas 478 correcciones netas**, de las 28.721 y 28.438 que V12 comete en cada par. Es decir, **la magnitud de la oportunidad es de sobra** y el cierre no puede sostenerse en ella.

### 3.c El gap CV -> LB no da puntos gratis

Regresando el leaderboard sobre el CV de los cuatro envios cuyos dos extremos son verificables en disco (v01, v07, v08, v12): **LB = +0.00223 + 1.02825 x CV**, **R2 = 0.9968**, desviacion del residuo 0.00147.

- Un gate de **+0,003** en CV vale **+0.00308** en LB y llevaria a **0.62278**, sobre el 0.62182 del envio real de V12 y el 0.62188 de su replica.
- El gap es un **offset aditivo estable**, de +0.01716 a +0.02132, con desviacion 0.00164.
- La pendiente es **1,03**, luego los gates locales estan calibrados: ni mas cerrados ni mas abiertos de lo que parecen en terminos de leaderboard. La preocupacion correcta es la contraria a la habitual, y es la que sostiene el gate: filtra candidatos de +0,001 que no mejorarian nada.

## 4. Que queda realmente abierto

**Nada con diferencia informacional demostrable.**

| Mecanismo | Estado |
|---|---|
| Desplazamiento de prior por clase | sondeado por el especialista, +0,0010, **sin artefacto en disco** y por tanto no verificable; y tres veces por debajo del gate |
| Sustituto *soft-F1* de la entropia cruzada | el unico item tecnicamente no ejecutado; con clases al 16-17 % de masa, macro-F1 y macro-recall son la misma magnitud, luego el optimo del *surrogate* y el de la entropia cruzada casi coinciden |
| Optimizar o seleccionar sobre las fronteras | **no puede alcanzar el leaderboard**: la metrica oficial es macro-F1 de seis clases y P1 midio que re-pesar la frontera es un intercambio de suma cero |
| V13 y V18 P2/P6/P7/P9 | pre-registrados y **bloqueados** por la condicion de sus briefs, que no se cumplio |
| Identificador de autor para agrupar el LOFO | **no es un mecanismo: es un dato que no existe** en el dataset (D-003) |

## 5. Opinion del especialista

La auditoria independiente del agente `kaggle specialist` concluyo, en respuesta directa a la pregunta de si existe una hipotesis con probabilidad razonable de superar V12:
>
> **No.** Ninguna de las candidatas que sobreviven a la auditoria tiene mas de un 15 % de > probabilidad de superar 0,62188, y la que mas tiene esta prevista por debajo del gate por un > factor de tres a diez.

| Campo | Contenido |
|---|---|
| Hipotesis recomendada | **ninguna**. La senal de frontera existe y es grande, y es cobrable en principio; el bloqueo es la identificabilidad de las filas corregibles, no la informacion, ni la magnitud, ni la funcion objetivo |
| Evidencia a favor | especialista sobre las 279 variables saca 0,7136 y 0,7493 contra 0,4963 y 0,4974 de V12 (D-048); techo con enrutamiento +0,0107 global; y la elasticidad medida en esta sesion, +0.0061 y +0.0063 por 1.000 correcciones |
| Evidencia en contra | cinco selectores fallidos en V17; cuatro mecanismos de objetivo fallidos en V18; errores de V12 en frontera **confiados** (margen 0,809 cuando el especialista acierta contra 0,539 cuando V12 acierta); solo el 12,7 % y el 8,0 % de las victorias del especialista caen en la region de frontera; y la pagina de confianza de V16, que con 90 puertas no desvia ni una fila |
| Riesgo | bajo de contaminacion; el riesgo real es gastar tiempo y anadir una entrada de +0,001 al registro |
| Coste | irrelevante: la reconstruccion son 2 min y el calculo segundos, sin GPU. El problema es que **el techo ya esta medido** y queda por debajo del gate |
| Decision | **cerrar**. Un experimento cuyo techo medido es +0,001 no se justifica ni aunque cueste cero |

**Una salvedad de trazabilidad**: la sonda de prior-shift que el especialista menciona **no dejo artefacto** en el proyecto, luego su +0,0010 no puede usarse como evidencia. El resto de sus cifras se han verificado una a una contra disco y son correctas, incluida la elasticidad, que aqui se ha recalado de forma independiente.

## 6. Las ocho condiciones para ejecutar un experimento, evaluadas una a una

El brief permite **un** experimento adicional solo si se cumplen **todas** las condiciones. Ninguna se cumple entera:

| # | Condicion | Estado |
|---|---|---|
| 1 | el especialista lo considera hipotesis competitiva real | **FALLA**: su veredicto es cerrar |
| 2 | la informacion es genuinamente nueva | **FALLA**: la compresion no existe como hipotesis y los 62 artefactos estan cerrados |
| 3 | no fue probada historicamente | **FALLA**: los cinco selectores de V17 y los cuatro mecanismos de V18 son exactamente esta pregunta |
| 4 | los artefactos necesarios ya existen | se cumple, y es irrelevante: el techo esta medido |
| 5 | existe senal OOF clara | **parcial**: la senal existe, el selector no |
| 6 | afecta directamente a una frontera | se cumple |
| 7 | el coste experimental es bajo | se cumple, y es irrelevante |
| 8 | hay estrategia clara para convertirlo en mejora de LB | se cumple: una submission |

Fallan 1, 2 y 3, que son las tres no negociables. **El brief dice que si cualquiera falla, no se experimenta.**

## 7. Decision

```
CLOSE AGE - V12 REMAINS CHAMPION
```

**V12 permanece champion.** No se ejecuta ningun experimento adicional, no se genera ninguna submission y AGE deja de ser una linea de trabajo experimental.

**AGE no se reabre** salvo que aparezca una fuente de informacion nueva, una regla nueva del concurso o evidencia externa que cambie sustancialmente el espacio de busqueda. En particular, y de forma permanente: no mas fases, ni busqueda de hiperparametros, ni variantes de perdidas, ni de features, ni de ensembles, ni de destilacion, ni de especialistas.

**Lo que este trabajo deja recomendado no es un experimento**: optimizar o seleccionar **sobre las fronteras directamente**, lo que exige cambiar la metrica efectiva del entrenamiento. Es un problema distinto del que se cerro, porque la lever es grande y lo que falta es un selector, no informacion. Queda para quien retome el problema con un dataset que permita agrupar por autor o con etiquetas mas informativas.
