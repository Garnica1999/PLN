# Informe completo del proyecto — Predicción de rango de edad a partir de tweets

- **Competición**: `tweet-age-2026-puj`
- **Ámbito**: `KAGGLE_AGE_2026_PUJ/`
- **Rama de las últimas fases**: `kaggle-age-2026-puj-v17`
- **Fecha**: 2026-09-26
- **Estado**: V12 congelado y entregado. V13 a V17 cerradas **sin ninguna adopción**.
- **102 experimentos** registrados en `experiments.csv` · **47 decisiones** en `DECISIONS.md`
  (D-001 a D-047) · **0 errores** de auditoría de idioma sobre 371 ficheros

Este documento es la cuenta completa y legible del proyecto: qué se entregó, cómo se decidió, qué se
buscó después y por qué no se encontró. Los resultados que aquí se citan salen de los artefactos en
`experiments/`, y cada afirmación va acompañada del fichero donde vive.

---

## 1. Resumen ejecutivo

El proyecto empezó como un problema de clasificación de texto con seis clases de edad a partir
de un tweet, y terminó siendo un estudio sobre **dónde está el límite de un conjunto de datos**. Se
entregó un modelo con **macro-F1 0,6005** en validación cruzada agrupada y **0,62188** en el
leaderboard, y después se invirtieron cinco fases buscando una mejora que no apareció. La razón por la
que no apareció es la parte valioso del trabajo, y está medida:

1. **Las variables no eran el cuello de botella.** Se construyeron bloques nuevos cinco veces
   (evidencia por token, pronombres, conjunciones, modalidad, variables lingüísticas de un analizador
   morfosintáctico externo) y **ninguno mejoró el modelo**, incluidos dos que en aislamiento eran los
   mejores del proyecto.
2. **La redundancia no era el problema.** Se identificaron 58 clusters de colinealidad que cubren 171
   de 310 variables y se podó dejando una por cluster: **empeoró en 0,0040**, de forma reproducible en
   dos particiones. Las variables casi idénticas estaban cargando con algo.
3. **La arquitectura sí tenía margen, pero pequeño.** El combinador actual es un promedio fijo
   0,5 + 0,5 que nunca se optimizó. Un combinador aprendido da **+0,0006**, que es el primer delta
   positivo del proyecto y no supera el gate.
4. **Ese pequeño margen estaba en las fronteras, y es real.** El combinador aprendido mejora las dos
   fronteras centrales en **+0,0102** y **+0,0157**, tres a cinco veces su efecto global.
5. **Pero no se puede cobrar.** Un clasificador binario de frontera ya existente acierta **0,6864** y
   **0,7218** en las filas de cada frontera, frente a **0,4963** y **0,4974** del modelo, y resuelve
   entre 10.837 y 12.699 filas de forma neta. El techo con un enrutamiento perfecto sería **+0,0107**,
   que supera el gate de adopción. Y sin embargo, **ninguna señal disponible encuentra dónde cobrarlo**,
   porque los errores del modelo en la frontera son **errores confiados** y las victorias del
   especialista caen fuera de la región de frontera.

La conclusión que la evidencia sostiene es una sola, y es incómoda: **el cuello de botella es la
función objetivo**. El modelo se optimiza con macro-F1 sobre seis clases, y el problema está en las
fronteras. Mientras no se cambie eso, añadir variables, podarlas, reentrenar el meta-modelo o
enrutar a un especialista son movimientos que no pueden mejorar el modelo. Y eso no es una conjetura:
es lo que Hanmidido 21 experimentos intending exactamente lo contrario.

---

## 2. El problema y los datos

| Magnitud | Valor |
|---|---|
| Train | 169.336 filas |
| Test | 112.892 filas |
| Clases | 6, en orden de edad: `13-17`, `18-24`, `25-34`, `35-49`, `50-64`, `65-xx` |
| Distribución | casi uniforme: de 28.863 a 27.160 filas por clase |
| Variable objetivo | el propio autor del tweet, en su rango de edad declarado |
| Restricción de la competición | **sin embeddings, sin vectorizadores aprendidos, sin modelos externos** |

**El problema del leakage por duplicados** (`reports/`, sección 2 del notebook). El dataset está
poblado de **casi duplicados**: tweets que coinciden en su texto con pequeñas variaciones. Con una
partición aleatoria, un casi duplicado del test aparece en el train y el modelo "recuerda" en vez de
"generaliza". Por eso **todas** las fases usan particiones **agrupadas** por clúster de duplicados
(`features/dup_clusters.parquet`), y por eso las cifras de este informe no son comparables con las de
ninguna distribución publicada sin agrupar.

---

## 3. Qué se entregó: V12

| Componente | Detalle |
|---|---|
| Caras | dos: `LogReg` sobre 279 variables y un `MLP` codificado de 64 unidades |
| Combinación | **promedio fijo 0,5 + 0,5** de las probabilidades de ambas |
| Variables | 279 = 78 scores de 14 miembros + 35 densas + 87 estructurales + 73 supervisadas + 6 NB-SVM |
| Referencia CV | **0,6005** (pooled, `metricas_globales`) |
| Referencia LB | **0,62188**, reproducible |
| Submission | `age_submission_v12.csv`, reconstruida al 100,0000 % desde la matriz original (D-034) |

### 3.1 La discrepancia 0,6004 contra 0,6005, resuelta

El proyecto usaba dos números indistintamente. **No son el mismo número redondeado**: son dos
estimadores del mismo macro-F1.

- **Pooled** (la confusión de todas las filas a la vez, que es lo que devuelve `metricas_globales`):
  `0,6005017473`, que redondea a `0,6005`.
- **Media de los cinco folds**: `0,6004103057`, que redondea a `0,6004`.
- **Diferencia**: `0,0000914416`, es decir 1,8 veces la unidad de redondeo del último dígito. No se
  arregla redondeando.

No coinciden porque el macro-F1 de una confusión agregada no es la media de los macro-F1 de sus
partes: es la desigualdad de Jensen aplicada a la media por clases.

**Canónico: 0,6005, en escala pooled.** El motivo no es estético. El umbral de efecto de D-042 se
calibró contra controles permutados medidos en pooled (0,6000 a 0,6003); el intervalo de bootstrap
del gate vive en pooled porque remuestrea la confusión conjunta; y las cinco fases modernas
(V13, V14, V15, V16, V17) miden su referencia en pooled. Usar 0,6004 regalaría **+0,0001** de delta y
rompería el emparejamiento con todo lo anterior.

*Fuente: `experiments/v17_p0_reproduccion.json`, `reports/v17_p0_reproduccion.md`.*

---

## 4. El protocolo con el que se decidió

Un proyecto que cierra 21 experiments en negativo necesita un protocolo que no se pueda mover
después de ver el resultado. Este es el que se usó, y se mantuvo constante de V13 en adelante.

| Elemento | Definición |
|---|---|
| Partición | 5 folds **agrupados** por clúster de duplicados, `make_folds(y, groups=..., scheme="group")` |
| Segunda partición | 4 folds agrupados distintos, usada como **control de selección** en los candidatos que pasan el filtro |
| Significancia | **bootstrap pareado** sobre la tabla de confusión conjunta, IC95. Remuestrear las dos tablas por separado daba intervalos que no contenían el cero en diferencias nulas; el método pareado se validó contra el clásico (difieren 5e-5) |
| Doble métrica | macro-F1 global **y** accuracy en las dos fronteras centrales. Ninguna adopción puede mejorar una frontera empeorando la otra |
| Controles de permutado | 3 semillas que destruyen la relación señal-etiqueta. Una candidata que no supera sus propios controles no es candidata |
| Gate de D-042 | Δ ≥ +0,001, ≥3/5 folds positivos, IC95 pareado que excluya el cero, segunda partición ≥3/4, y ninguna frontera central empeore |
| Gate de V17 | Exploratorio Δ ≥ +0,0015 con ≥4/5 folds; de adopción Δ ≥ +0,003 con el resto de condiciones |
| Huella numerica | `autotest_numerico()` como primera línea de todo script, por un bug de `numpy 1.26.4` con `Python 3.14.7` que calcula mal `X @ X.T` (D-040) |
| Restricción de decisiones | Ninguna variable, umbral, hiperparámetro o calibración puede usar la validación exterior |

El gate relajado de D-042 no es unaXF concessions: se rebajó el efecto de +0,003 a +0,001 **antes**
de correr V14, precisamente para darle a esa fase la mejor oportunidad posible. Falló igualmente.

---

## 5. V13 — Auditoría de brechas y diagnóstico causal (15 experimentos, 0 adopciones)

V13 no buscó una mejora: construyó el inventario de lo que se había hecho, lo que estaba cerrado, lo
que estaba a medias y lo que era genuinamente nuevo, y luego cerró cada línea con un mecanismo en vez
de una conjetura.

### 5.1 Lo que encontró la auditoría

**P0a, auditoría de idioma**: el proyecto arrastraba **6 ideogramas CJK y 260 emojis o flechas** en
sus propios textos escritos. Corregidos: 0 errores.

**P0b, colinealidad** (sobre la matriz completa, sin reentrenar): **401 pares** con |ρ| ≥ 0,80, **229**
redundantes, **15 duplicados exactos**, **54 clusters**, una columna constante, rango efectivo estricto
**263 de 278**, 87 direcciones para el 95 % de la varianza, y **170 variables con VIF ≥ 10** (mediana
18,87). La matriz estaba altamente colineal y, aun así, nadie había comprobado si eso afectaba a la
predicción. V15 lo comprobó.

### 5.2 Las cuatro familias de información nueva, y por qué se cerraron

| Familia | Control aislado | Fusión con V12 | Veredicto |
|---|---|---|---|
| **Evidencia por token** `P(clase\|token)` | **0,3997** frente a un azar de 0,1667 | **−0,0002**, 3/5 folds, IC95 [−0,0015; +0,0010] | cerrada |
| **Pronombres de persona** (26 variables) | 0,1918 | neto **−64.075** filas | cerrada |
| **Árboles de conjicciones** (36 variables) | 0,1539 | por debajo del azar | cerrada |
| **Modalidad e incertidumbre** (23 variables) | 0,1270 | por debajo del azar | cerrada |

**El caso de la evidencia por token merece una línea propia**, porque es el más instructivo de todo el
proyecto. En aislamiento alcanza 0,3997, casi el doble del azar: la información de edad **está** en los
tokens, y es buena de verdad. Pero al fusionarla con V12 el resultado es −0,0002, y sus controles
permutados (0,6000, 0,6003, 0,6000) son **indistinguibles** de la candidata. La señal es real y a la
vez inaccesible al combinador.

**Diagnóstico causal de por qué** (P16a a P16d, el trabajo más fino de V13):

- **P16c, reconstrucción**: el mejor R² mediano reconstruido es **+0,522** para el bloque de tokens
  (máximo +0,959, con 21 de 36 columnas por encima de 0,5), frente a **+0,204** para pronombres y
  **+0,035** para modalidad, y **0,0485** en el control. El bloque de tokens contiene estructura relevante
  recuperable de forma verificable.
- **P16b, activación**: 122.870 filas (72,6 %) tienen al menos un token conocido, y **1,65 tokens
  conocidos de media**.
- **P16d, puerta de confianza**: el AUC de la confianza del bloque es **0,7173**, y el techo con una
  puerta perfecta sería 0,6770. **Pero** el máximo de las **90 combinaciones de umbral** da 0,6005 con
  **0 filas desviadas**: la puerta nunca cambia nada.

Los tres mecanismos conviven: saturación del espacio, información inaccesible al combinador, y modelo
en el límite. De ahí la conclusión de D-041, que es el resultado más importante de V13: **para seguir
había que cambiar el alcance, no hacer más experimentos**.

*Fuentes: `reports/v13_final_research.md`, `reports/v13_hypothesis_registry.csv` (8 hipótesis
pre-registradas, 8 falsadas), `reports/v13_master_results.csv`, `experiments/v13_p15_*.json`.*

---

## 6. V14 — Relajar la restricción de recursos (2 experimentos, 0 adopciones)

V13 dejó una sola vía abierta, y era de alcance: permitir un recurso lingüístico **externo**. D-043 lo
autorizó de forma acotada (un analizador morfosintáctico español, `es_core_news_md` de spaCy), con la
procedencia documentada y sin cruzar la línea de D-026.

**V14a construyó 37 variables nuevas**: morfología verbal y nominal, lemas, entidades nombradas y
sintaxis de dependencias. En aislamiento, un `LogReg` sobre ellas da **0,2409** y un `LinearSVC`
**0,2270**, frente a un azar de 0,1667. **Es el mejor bloque aislado de todo el proyecto**, por encima
de los lexicones (0,2033), los pronombres (0,1918) y la modalidad (0,1270). Solo 4 de 37 resultaron
redundantes, y eran las de longitud y diversidad, exactamente lo que P0b había predicho para la
textometría.

**V14b lo fusionó con V12, con el criterio más favorable posible**: Δ **−0,0015**, **1 de 5 folds**
positivos. Falla cinco de los seis criterios del gate. Y los controles de permutado (0,5962, 0,5975,
0,5986) son **peores** que la candidata, lo que significa que la señal es real, y aun así la segunda
partición da **0 de 4** folds positivos con media −0,0031, y el IC95 pareado es [−0,0027; −0,0002] con
solo un 0,9 % de resmuestras positivas. La frontera `18-24|25-34` empeora de 0,4963 a 0,4919.

La conclusión de D-044 generaliza la de V13: **la restricción de recursos no era el cuello de
botella**. El meta-modelo no puede explotar señal nueva dentro de este espacio de variables, y se
confirmó con un tratamiento experimental más fuerte, no se refutó.

*Fuentes: `reports/v14_linguistico.md`, `reports/v14_fusion_linguistico.md`.*

---

## 7. V15 — Correlación de entrada y eliminación de variables redundantes (2 experimentos)

V13 había medido colinealidad sin evaluar sus consecuencias. V15 cerró ese círculo, porque era la
pregunta pendientey era la más fácil de responder mal.

**V15a**, sobre la matriz completa de **312 variables** (279 de V12 más 33 de V14) y **dentro de cada
fold** — porque elegir variables por su correlación con la etiqueta sobre todo el train sería una fuga
de información:

- **2 columnas constantes**, que se podan por definición.
- **402 pares** con |ρ| ≥ 0,80, **245** con |ρ| ≥ 0,90, **15 duplicados exactos**.
- **58 clusters** que cubren **171 de 310** variables, dejando **197** tras podar.
- Asociación con la salida por **eta cuadrado** (que es la medida adecuada cuando la salida es
  categórica): mediana **0,0306**, máximo **0,5599**, y **110 de las 310** variables analizables por
  debajo de 0,01.
- El **Jaccard de la estructura de pares entre folds es 1,0000**: la redundancia es **perfectamente
  reproducible**, así que podar no es una decisión que dependa de la partición.

**V15b** evaluó la poda con el gate completo: un representante por cluster, el de mayor eta² dentro del
fold. Resultado: **0,5965** frente a 0,6005, Δ **−0,0040** con **0 de 5** folds positivos; segunda
partición **0 de 4** con media −0,0065; IC95 pareado [−0,0054; −0,0027], enteramente negativo.
**NO ADOPTAR**.

**Lo que el resultado afirma, y es contraintuitivo**: la redundancia no es cosmética. Las variables
con ρ ≈ 0,95 no son la misma variable, son lecturas ligeramente distintas del mismo fenómeno, y
promediarlas reduce la varianza igual que reduce el ruido cualquier promedio. La colinealidad afecta a
la **interpretación** de los coeficientes, no a la **predicción**: es exactamente lo que P0b insinuaba,
confirmado ahora con una evaluación completa en lugar de una inferencia.

*Fuentes: `reports/v15_correlacion.md`, `reports/v15_poda.md`, `reports/v15_clusters.csv`.*

> **Una corrección sobre esa cifra.** El campo `variables_tras_poda` del JSON de V15a vale 81 y está mal
> calculado: resta el número de clusters una vez de más, porque un cluster de tamaño *k* elimina *k−1*
> variables, no *k*. El número correcto es 310 − (171 − 58) = **197**, y es el que se usó de verdad, como
> demuestra el índice `features/v15_conservadas_fold{0..4}.npy` con 197 columnas en los cinco folds. El
> script no se editó porque ya se ejecutó; el campo se declara erróneo para que nadie lo cite. La
> conclusión de V15 no cambia, porque la poda se evaluó sobre las 197 columnas reales.

---

## 8. V16 — Arquitectura del meta-modelo (1 experimento, 0 adopciones)

Todos los diagnósticos de V13, V14 y V15 apuntaban al mismo sitio: **el cuello de botella no eran los
datos, era la arquitectura**. V12 arrastra desde su primera versión un promedio fijo 0,5 + 0,5 que
nadie optimizó nunca. V16 midió tres formas de combinar, sobre las mismas 279 variables y sin variables
nuevas.

| Arquitectura | macro-F1 | Δ | Folds positivos | Decisión |
|---|---|---|---|---|
| **A0** promedio fijo (producción) | 0,6005 | — | — | referencia |
| **A1** peso aprendido por fold | 0,6002 | −0,0003 | 0/5 | descartada |
| **A2** combinador logístico (apilamiento) | **0,6011** | **+0,0006** | 3/5 | **NO ADOPTAR** |

**A1 queda descartada con un argumento limpio**: los pesos que elige por fold son 0,1 · 0,5 · 0,5 ·
0,4 · 0,5. El 0,5 de producción se elige solo en tres de los cinco folds, así que el promedio fijo no
era una decisión desafortunada, era la correcta.

**A2 es el primer delta positivo del proyecto**, y aparece en la partición de 5 folds (3/5) y en la
segunda (3/4), pero el IC95 pareado **[−0,0003; +0,0015] contiene el cero** y el efecto queda bajo el
mínimo. Falla dos de los cinco criterios.

**Y aquí está el hallazgo que destapó V17**: las dos fronteras centrales mejoran de forma notable.

| Frontera | A0 | A2 | Ganancia |
|---|---|---|---|
| `18-24\|25-34` | 0,4963 | **0,5065** | **+0,0102** |
| `25-34\|35-49` | 0,4974 | **0,5131** | **+0,0157** |

Son ganancias de uno a dos puntos, **tres a cinco veces el efecto global**, y en el mismo sentido. La
macro-F1 global apenas se mueve porque la media ponderada por clases la dominan las clases fáciles, y
ahí el combinador no tiene nada que ganar.

**La lección operativa, y es accionable**: una mejora de **+0,0157** en la frontera `25-34|35-49` es
**invisible** en la macro-F1 global. Cualquier trabajo futuro que se dirija a las fronteras tiene que
**medir sobre ellas desde el principio**, no como métrica secundaria. El gate D-042 quedó validado por
un segundo caso difícil, y este es el instructivo: la intuición (las fronteras mejoran, luego mejora el
modelo) y el gate discrepan, y el gate gana porque el intervalo de confianza no excluye el cero.

*Fuentes: `reports/v16_arquitectura.md`, `experiments/v16_arquitectura.json`, D-046.*

---

## 9. V17 — Especialistas de frontera y decisión local (8 de 16 experimentos, 0 adopciones)

V17 se abrió con una pregunta directa: si la señal que V16 encontró en las fronteras puede convertirse
en ganancia global con modelos entrenados específicamente para esas fronteras y activados cuando V12 es
ambiguo. Es la única fase del proyecto que empieza por **auditar** en vez de por entrenar.

### 9.1 La referencia, reconciliada antes de nada

Antes de cualquier experimento se resolvió la discrepancia 0,6004 contra 0,6005 que se arrastraba desde
hacía meses (sección 3.1 de este informe) y se verificó que A2 de V16 se reproduce desde sus artefactos.
**No hubo STOP**: las tres arquitecturas coinciden con lo publicado y las fronteras quedan confirmadas.

### 9.2 El especialista de frontera ya existía, y es muchísimo mejor

El hallazgo de P1 y P1.1 es que **no hacía falta entrenar un especialista**: ya había uno, el
`E028a`, un `LinearSVC` sobre el TF-IDF disperso entrenado **solo con las dos clases de la frontera**,
con los mismos folds agrupados del proyecto.

| Par | V12 | especialista | aporta | rompe | techo con enrutamiento |
|---|---|---|---|---|---|
| `18-24\|25-34` (n=57.023) | 0,4963 | **0,6864** | 14.356 | 3.519 | **0,7481** |
| `25-34\|35-49` (n=56.577) | 0,4974 | **0,7218** | 15.571 | 2.872 | **0,7726** |

**+0,19 y +0,22 puntos** en las filas de cada frontera. Y no es una versión mejorada de V12: es una
decisión distinta sobre las mismas filas. El techo con enrutamiento perfecto es **0,7481** y **0,7726**.

Dos matices que importan y que se publicaron junto al número:

- En un **23 a un 25 %** de las filas de la frontera **ambos fallan**. Ese techo nadie lo puede cobrar.
- Estas accuracies están **restringidas al par** y **no** son macro-F1 de seis clases: no se comparan
  con el 0,6005 de referencia.

### 9.3 Las dos señales que sí separan, y no están donde se buscaban

El brief proponía compuerta sobre las probabilidades de V12. La medición de separabilidad (área bajo
la curva, que es una medida de orden y no puede sobreajustar) dice que **no sirven**:

| Señal | AUC 18-24\|25-34 | AUC 25-34\|35-49 |
|---|---|---|
| **acuerdo entre miembros** | **0,7070** | **0,7300** |
| **margen del especialista** | **0,7294** | **0,7539** |
| separación de V12 | 0,5780 | 0,5268 |
| entropia de V12 | 0,4323 *(invertida)* | 0,4230 *(invertida)* |

Las dos que sirven **no forman parte de las probabilidades de V12**, y apuntan en **sentidos
opuestos**: el margen alto indica escucharle, y el acuerdo bajo también.

### 9.4 El mecanismo, medido, que explica por qué nada funciona

Esta es la parte más útil de la fase, y se obtuvo mirando **dónde** están las victorias del especialista
y **cuán segura** está V12 en cada grupo de filas.

| Grupo de filas | % con el par arriba en V12 | separación de V12 |
|---|---|---|
| lo que el especialista **rescata** | **12,7 %** y **8,0 %** | **0,809** y **0,759** |
| lo que el especialista **rompe** | **70,6 %** y **67,8 %** | 0,539 y 0,682 |

**V12 está más confiado cuando se equivoca en la frontera.** Sus errores ahí no son de baja confianza,
son errores **confiados**. Y las victorias del especialista caen **fuera** de la región de frontera, que
es la única donde escucharle no destruye aciertos ajenos. De ahí se siguen las tres cosas:

- Escuchar al especialista en la región de frontera **sin más** vale **−0,0080** de macro-F1 global.
- La compuerta por ambiguedad con la señal que propone el brief da **+0,0002** como máximo y se vuelve
  **negativa** a partir de un margen de 0,20.
- La regla conjunta de las dos señales da **+0,0000**.

### 9.5 Las dos formulaciones razonables, medidas y fallidas

| Candidato | Qué pregunta | macro-F1 | Δ | Folds | Decisión |
|---|---|---|---|---|---|
| compuerta de ambiguedad (P1b) | ¿cuándo duda V12? | 0,6007 | +0,0002 | — | cerrada |
| regla conjunta (P1c) | ¿cuándo dudan las dos señales? | 0,6005 | +0,0000 | — | cerrada |
| **P9** arbitro aprendido | ¿a quién escucho? | 0,6004 | **−0,0001** | **0/5** | **NO ADOPTAR** |
| **P5** combinador local | ¿qué clase es? | 0,6003 | **−0,0002** | 1/5 | **NO ADOPTAR** |
| *techo con árbitro perfecto* | — | *0,6112* | *+0,0107* | — | *cota, no resultado* |

**P9** es el experimento más fuerte de la fase: un `LogReg` binario por frontera sobre **47 columnas
derivadas que ya existían** (los votos y márgenes de los 13 miembros, los 5 márgenes de pares, los 3 de
grupo grueso, las 6 probabilidades de V12, las 6 del combinador A2 y el acuerdo entre miembros), con
**protocolo anidado en tres niveles**: el árbitro se ajusta en un bloque interior, el **umbral se elige
en otro bloque que el árbitro no ve**, y la validación exterior no participa en nada. **Ninguna
representación nueva**, que es lo que el brief exige.

Su comportamiento es la información: eligió umbrales de 0,60 a 0,90, activó entre 327 y 1.256 filas, y
**cuando activó perdió: 3 rescates contra 22 daños**. No es un árbitro mal entrenado: es que **la
frontera que habría que aprender no existe con estas señales**.

**P5** es la hipótesis restante y era genuinamente distinta, porque pregunta qué clase es en lugar de a
quién escuchar. Su diseño es elegante por una razón concreta: **redistribuye la masa que V12 ya pone
en las dos clases del par conservando el total**, de modo que no puede mejorar las clases fáciles ni
dañar a las vecinas, y actúa en todas las filas. También falló, y además **empeoró las fronteras** que
pretendía mejorar (0,4953 y 0,4968 frente a 0,4963 y 0,4974).

### 9.6 La respuesta a la pregunta científica de la fase

| Opción | Veredicto |
|---|---|
| A. V12 no tiene información | **Falso** |
| B. Un miembro de V12 sí la tiene | Parcial: la expone, no alcanza |
| C. Un especialista aprende información nueva | **Cierto y grande**, pero ya existía |
| D. El problema es solo de decisión | Parcial: el techo es +0,0107 |
| E. Se resuelve con gating | **Falso**, con tres evidencias independientes |
| F. La mejora local no tiene impacto global | **Cierto en la práctica**: +0,0000 alcanzable |

La formulación más precisa es: **la señal existe, el techo la supera, y la identificación es lo que no
existe**. No es un problema de datos, no es un problema de umbral, y no es un problema de métrica.

### 9.7 La asimetría que queda abierta, y que no es un experimento

**Los márgenes de pares ya son entradas de V12** —lo midió V15a al listar la matriz de 312
variables— y V12 no los explota, cuando por sí solos valen 0,6864 y 0,7218 en las filas de su frontera.

Hay información accesible en el espacio de entradas que el combinador global no usa. Y eso **no se
resuelve con otra compuerta ni con otro meta-modelo**: exige **cambiar la función objetivo**, que hoy
es macro-F1 sobre seis clases y no sobre las fronteras. Es la única acción de este informe que no es
"probar otra vez".

*Fuentes: `reports/v17_final_research.md`, `reports/v17_candidate_comparison.csv`,
`reports/v17_master_results.csv`, `reports/v17_hypothesis_registry.csv`, D-047.*

---

## 10. Tabla maestra de las cinco fases

| Fase | Objeto | Experimentos | Mejor resultado | Adopciones |
|---|---|---|---|---|
| **V13** | auditoría de brechas y diagnóstico causal | 15 | 0,4028 de control reconstruido; diagnóstico causal en 3 mecanismos | 0 |
| **V14** | relajar la restricción de recursos | 2 | control aislado 0,2409, el mejor del proyecto; fusión −0,0015 | 0 |
| **V15** | correlación de entrada y poda | 2 | poda −0,0040, 0/5 y 0/4 folds | 0 |
| **V16** | arquitectura del meta-modelo | 1 | A2 +0,0006; fronteras +0,0102 y +0,0157 | 0 |
| **V17** | especialistas de frontera y decisión local | 8 | techo +0,0107; alcanzable +0,0000 | 0 |
| **Total** | | **28** | | **0** |

Con las 74 etapas anteriores de construcción del modelo, el proyecto acumula **102 experimentos**
registrados.

**Costo**: las cinco fases de búsqueda suman unos 40 minutos de CPU en su mayor parte, porque se
aplicó una puerta de rescate antes de cada evaluación cara: medir el intercambio de rescate contra daño
con artefactos inertes antes de entrenar nada.

---

## 11. Lo que se aprendió

Cinco lecciones, y todas tienen el mismo pedigree: se aprendieron midiendo, no meditando.

1. **Una señal puede ser fuerte en aislamiento e inaccesible al combinador.** El bloque de evidencia por
   token saca 0,3997 contra un azar de 0,1667, y aun así su fusión da −0,0002 con controles permutados
   indistinguibles. El mismo patrón se repitió con el bloque lingüístico de V14. Medir el bloque por
   separado es necesario y no es suficiente: hay que medir **si el combinador puede usarla**, y esa
   pregunta tiene su propio coste.
2. **La redundancia de entradas puede ser funcional.** Podar 115 variables, entre ellas 15 duplicados
   exactos, empeoró de forma reproducible. Las variables casi idénticas son lecturas ligeramente
   distintas del mismo fenómeno, y promediarlas reduce la varianza. La colinealidad afecta a la
   interpretación de los coeficientes, no a la predicción.
3. **Una decisión de diseño que nunca se midió puede estar equivocada, y también puede estar
   correcta.** El promedio fijo 0,5 + 0,5 resultó ser el óptimo en 3 de 5 folds, y medirlo costó
   media hora y evitó semanas de búsqueda. Merece la pena dudar de todo lo que se heredó sin justificación.
4. **Medir en la métrica que importa, no en la que se informes por costumbre.** Una mejora de +0,0157
   en una frontera es invisible en la macro-F1 global. Ese error de encuadre costó cuatro
   fases, y V16 lo pagó.
5. **Un error confiado no se puede enrutar.** Es la lección de V17 y la más transferible: una compuerta
   de ambigüedad solo funciona si los errores del modelo son errores de baja confianza. Aquí eran lo
   contrario, y por eso tres compuertas independientes fallaron con mecanismos distintos.

---

## 12. Estado final

### 12.1 Entregado

- **Modelo**: V12, combinador con promedio fijo 0,5 LogReg + 0,5 MLP sobre 279 variables.
- **Referencia**: 0,6005 pooled en CV agrupada, **0,62188** en el leaderboard.
- **Submission**: `age_submission_v12.csv`, reproducible al 100,0000 % desde la matriz original.

### 12.2 No entregado, y deliberadamente

- **Ninguna submission nueva en V13 a V17.** Ningún candidato superó el gate de adopción.
- **Ninguna modificación de V12, V15 ni V16**, ni de las submissions históricas.
- **Ninguna interacción con el leaderboard** para seleccionar nada.

### 12.3 Reproducibilidad

| Elemento | Estado |
|---|---|
| Semilla | `cfg.SEED`, fijada en todos los scripts |
| Intérprete | `/home/carlos/venv/bin/python` (Python 3.14.7) |
| Autocomprobación | `autotest_numerico()` como primera línea de todo script |
| Git | ramas `kaggle-age-2026-puj-v13` y `kaggle-age-2026-puj-v17`, **sin push**, sin PR, sin merge, sin reescritura de historia, sin modificar remotes |
| Memoria | `PROJECT_STATE.md`, `RUNLOG.md`, `DECISIONS.md` (47 decisiones), `experiments.csv` (102 experimentos) |
| Trazabilidad | 28 informes de fase, 2 CSV maestros, 1 registro de hipótesis, 18 figuras, notebook consolidado de 78 celdas con la PARTE III de V16 y V17 ejecutada |
| Auditoría | **0 errores** sobre 371 ficheros, con verificación independiente de caracteres |

### 12.4 Una advertencia sobre cómo leer las cifras

Dos trampas documentadas que conviene tener presentes al leer cualquier informe del proyecto:

- **El estimador.** Un 0,6004 en un informe antiguo es la media por fold, no la pooled. Los deltas solo
  son comparables si se sabe qué estimador se usó en cada lado.
- **Las accuracies de frontera.** Las de V17 están restringidas al par y no son macro-F1 de seis
  clases. Un 0,72 en la tabla de fronteras no está en escala con un 0,60 en la tabla global.

### 12.5 Qué queda abierto

Una sola cosa, y es la única acción que este informe recomienda:

> **Cambiar la función objetivo.** El problema está en las fronteras de edad, y el modelo se optimiza
> con una media que las diluye. Los márgenes de pares ya están entre las entradas de V12 y no se
> explotan. Cualquier esquema de entrenamiento o evaluación que mida y optimice sobre las fronteras
> directamente es la vía que la evidencia deja abierta; el resto de vías está cerrada con medición.

---

## 13. Índice de artefactos

| Documento | Contenido |
|---|---|
| `reports/v13_final_research.md` | cierre de V13, diagnóstico causal, 8 hipótesis falsadas |
| `reports/v14_linguistico.md`, `reports/v14_fusion_linguistico.md` | V14 completa |
| `reports/v15_correlacion.md`, `reports/v15_poda.md` | V15 completa |
| `reports/v16_arquitectura.md` | V16 completa |
| `reports/v17_final_research.md` | cierre de V17, respuesta a la pregunta científica |
| `reports/v17_gap_audit.md` | auditoría de brechas, 42 familias |
| `reports/v17_candidate_comparison.csv` | todos los candidatos de V17 con su decisión |
| `reports/v17_master_results.csv`, `reports/v17_hypothesis_registry.csv` | tablas maestras de la fase |
| `reports/v17_reproducibility.md`, `reports/v17_compliance.md` | reproducibilidad y cumplimiento del brief |
| `reports/figures/` | 18 figuras, 10 de ellas de V17 |
| `notebooks/99_entrega_consolidado.ipynb` | 78 celdas, con la PARTE III de V16 y V17 ejecutada |
| `experiments.csv` | 102 experimentos con pregunta, mecanismo, resultado y decisión |
| `DECISIONS.md` | 47 decisiones con contexto, evidencia y consecuencias |
| `PROJECT_STATE.md`, `RUNLOG.md` | estado del proyecto y bitácora completa |

---

*Documento generado el 2026-09-26. Todas las cifras citadas salen de los artefactos en
`experiments/` y `reports/`, y son verificables celda a celda contra ellos.*
