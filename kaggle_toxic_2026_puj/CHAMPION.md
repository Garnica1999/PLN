# CHAMPION

Proyecto: KAGGLE_TOXIC_2026_PUJ (https://www.kaggle.com/competitions/tweet-toxic-2026-puj)
Metrica oficial: F1 Score, media column-wise de la F1 binaria de la clase positiva
sobre HS, TR y AG (`f1_media_columnas`, D-009).
Ultima actualizacion de este fichero: 2026-09-28, fase P46 (auditoria del campeon).

**Estado verificado del campeon (P46, 2026-09-28).** `leaderboard_max_verified = 0.78306`,
que corresponde a `v03_lr_union_thr` (V03), sha256
`7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`. Las 22 submissions
registradas en `submissions/leaderboard_log.csv` tienen artefacto en disco, sin duplicados ni
valores nulos, y **ninguna supera 0.78306**. Los valores 0.78535 y 0.78650 que aparecen mas
abajo en este fichero son predicciones, no mediciones; ver el aviso de la seccion 4.

---

## 1. Champion actual

```
=== Version v03 - LogisticRegression balanced C=2.0 x3 sobre la union TF-IDF, con umbral por etiqueta ===
Fecha: 2026-09-22
Features: 27.431 (word (1,2) min_df=2 sublinear + char_wb (3,5) min_df=2 sublinear)
Modelo: LogisticRegression C=2.0, class_weight=balanced, solver=liblinear,
        max_iter=1000, random_state=42, una cabeza por etiqueta (HS, TR, AG)
Umbrales: HS 0.44 | TR 0.51 | AG 0.46
CV local: 0.7772114525165721 (OOF canonico, 1.483 filas de desarrollo,
        StratifiedGroupKFold de 5 pliegues con near-duplicados fusionados)
LB: 0.78306
Cambios vs version anterior: ninguno. Artefacto congelado, no se modifica.
```

Ficheros del champion:

| Elemento | Ruta | sha256 |
|---|---|---|
| Submission | `experiments/baseline_v03_frozen/submission_v03.csv` | `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e` |
| Submission (copia en submissions) | `submissions/toxic_submission_v03_lr_union_thr.csv` | `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e` |
| Modelo | `experiments/baseline_v03_frozen/model_v03.joblib` | `263ed5bd0fc0de7cabfbaa85e88fca6fa65cf0a1b4e60a6e796b5120e26d13ca` |
| OOF canonico | `features/oof_v03_final.npz` (1.483 x 3) | ver `experiments/baseline_v03_frozen/hashes.txt` |
| Traza | `experiments/baseline_v03_frozen/pipeline_v03.json` | - |

El champion esta congelado. Ninguna fase lo regenera, lo reentrena ni lo sobrescribe.

## 2. Formato de submission que exige la competencia

- Fichero CSV, codificacion UTF-8, separador **coma**.
- Exactamente cuatro columnas y en este orden: `id,HS,TR,AG`.
- 1.154 filas, una por cada tweet de `data/raw/toxic_test.csv`.
- La columna `id` debe conservar el **orden y los valores** de `data/raw/toxic_test.csv`.
- Las tres etiquetas son enteros binarios 0 o 1. Sin nulos, sin indice, sin columna extra.
- Comprobacion automatizada en `experiments/v16_01_generar_submissions.py`: el
  generador relee cada CSV que escribe y aborta si las columnas, los nulos o el
  orden de identificadores no cuadran.

## 3. Medidas de la fase v16 que gobiernan cualquier candidato

### 3.1 Prevalencia real del test

`k = 0.935177`, es decir el test tiene un **6.48 por ciento menos de positivos** que
train. Obtenido de forma analitica, sin etiquetas de test: la plantilla `v00_plantilla`
predice 1 en las tres columnas y su LB fue 0.51494; con prediccion todo-uno la F1
binaria de la clase positiva es exactamente `2p/(1+p)`, de modo que la unica incognita
del LB es la prevalencia de cada columna del test.

| Etiqueta | Prevalencia train | Prevalencia test estimada | Positivos test |
|---|---|---|---|
| HS | 0.436742 | 0.408431 | 471 |
| TR | 0.299827 | 0.280391 | 324 |
| AG | 0.382438 | 0.357647 | 413 |
| Media | 0.373002 | 0.348823 | - |

Incertidumbre: +-0.0025 de `k` por cada +-0.001 del LB de la sonda.

Validacion cruzada independiente: el EM de Saerens sobre las probabilidades de test,
sin usar etiquetas, estima por etiqueta HS 0.936, TR 0.955, AG 0.938
(`reports/v9_02_prior_shift_em.md`). Coincide con el estimador analitico.

### 3.2 Pendiente empirica dLB/d(tasa de positivos)

- **Pareja controlada V02 -> V03** (mismo modelo, mismas features, unica diferencia
  la regla de decision): tasa 0.3385 -> 0.3645, LB 0.77993 -> 0.78306.
  **Pendiente causal: +0.12011 por unidad de tasa.**
- Ajuste descriptivo de los 14 modelos reales con LB conocido: **+0.18416**, R2 0.22.
- Ajuste dentro de la familia de V03 (13 modelos, menos de 250 celdas de cambio):
  **+0.17616**, R2 0.22.

**El signo es positivo en las tres: subir la tasa de positivos predicha sube el LB.**
La hipotesis de bajar la tasa para igualar la prevalencia del test queda refutada por
dos vias independientes, y el registro ya lo habia notado para v09_p5.

Advertencia metodologica que hay que tener presente al usar la pendiente: la regresion
sobre las 14 submissions es **descriptiva, no causal**, porque las submissions difieren
a la vez en modelo, features y umbrales. La pendiente de la pareja V02 -> V03 es la
unica estimacion cuasi experimental del proyecto y es la que se usa para predecir.
La plantilla todo-uno se excluye del ajuste por sistema: es un punto degenerado
(tasa 1.0, LB 0.51494) que por si solo sube el R2 de 0.22 a 0.97.

Contrafactual causal adicional, sobre el OOF ponderado a la prevalencia del test: el
desplazamiento cuesta -0.006071 de F1 con los umbrales de V03 congelados, y el argmax
de F1 se mueve a umbrales **mas bajos** (0.410, 0.475, 0.450), no mas altos. Es decir
la respuesta correcta a menos prevalencia es predecir **mas** positivos. La direccion
opuesta a la del encargo.

### 3.3 Techo de lo que puede mover el LB

107 cambios netos de acierto mueven el LB 0.003. El error estandar teorico del
leaderboard con 1.154 filas es 0.01219. Por tanto **una candidata que cambie menos de
unas 107 celdas no puede mejorar el LB de forma detectable**, y ninguna perturbacion
de baja capacidad de V03 puede esperar mas de +0.003. El OOF no sirve para elegir: las
cuatro sondas con corrected_cv por encima de V03 transfirieron a la baja, y el gap
local-LB crece con la capacidad.

## 4. Candidatas de la fase v16, pendientes de enviar

Generadas por `experiments/v16_01_generar_submissions.py`, con control de identidad del
champion (hash, reproduccion celda a celda desde el artefacto congelado, reentreno
independiente y OOF canonico) y hash sha256 de cada salida. Medidas en
`resultados/v16_01_tabla_de_challengers.csv`.

> **AVISO ANADIDO EN P46 (2026-09-28). La columna `LB combinado` de la tabla siguiente NO
> contiene puntuaciones del leaderboard: contiene PREDICCIONES.** Se titula asi y el propio
> fichero declara la formula que las produce, unas lineas mas abajo. `ch1` y `ch5` si se
> enviaron, y sus puntuaciones reales son **0.77828** y **0.77495**, ambas REJECTED. Esos
> valores medidos estan en la seccion *Estado al cierre de V16*, mas abajo en este mismo
> fichero, y son los que valen. La tabla se conserva intacta como registro de lo que se
> predijo en su momento, no como registro de lo que se midio. Verificacion y evidencia:
> `reports/p46_champion_audit.{json,md,csv}`.

| # | Clave | Fichero | Tasa | Celdas | LB combinado (PREDICTO, no medido) | sha256 |
|---|---|---|---|---|---|---|
| 1 | ch1 | `submissions/toxic_submission_v16_ch1_tasa_optima_transportada.csv` | 0.3836 | 66 | 0.78535 | `a1b91806f7201aa14a16d4b6d95f8032f04bb36812e417cd9e1c481a5a1ac032` |
| 2 | ch5 | `submissions/toxic_submission_v16_ch5_tasa_optima_absoluta.csv` | 0.3980 | 116 | 0.78650 | `dac0d815465ada1df4b1c554deeb1b4a39d863226e0d8154dd3349d5cd0492d9` |
| 3 | ch2 | `submissions/toxic_submission_v16_ch2_prior_shift_em.csv` | 0.3752 | 37 | 0.78418 | `118817f26703fcb8385241a39ac885eae74a37643789f770871732f178a25efe` |
| 4 | ch4 | `submissions/toxic_submission_v16_ch4_control_tasa_baja.csv` | 0.3576 | 24 | 0.78242 | `e0d6bb1cd71e0b49e8e14beb585779b8dc3eb8300892899db4ec21a4f8739d9c` |
| 5 | ch3 | `submissions/toxic_submission_v16_ch3_peso_char_070.csv` | 0.3873 | 183 | 0.77431 | `d0d5e42e2b3ec5125721e801deb52206d904bb5bd41d506a547928893dec20a8` |

`LB combinado` (PREDICCION, no medicion) = 0.78306 + 0.12011 x delta_tasa + aciertos_netos_oof x 2.804e-05, y para
ch3, que cambia el modelo y no solo la regla, la base de perdidas por cambio de modelo
del registro (-0.00875, media de v04, v05, v06, v07 y v09_p5).

Que pregunta responde cada una, y por que deberia transferir mejor:

- **ch1** (66 celdas, LB 0.78535). Tasa por cuantil al optimo de F1 del OOF
  transportado por cociente, es decir un paso de +0.019 de tasa, la mitad del paso
  controlado V02 -> V03. Es la unica candidata cuya tasa esta **interpolada** entre los
  dos unicos puntos con LB conocido de la familia, de modo que es donde la pendiente
  causal esta mejor identificada. Acierto entre las modificadas 0.5000, aciertos netos
  0: no aporta evidencia nueva pero tampoco resta.
- **ch5** (116 celdas, LB 0.78650). El mismo mecanismo carried al optimo de F1 absoluto
  (+0.0335 de tasa). Es la de mayor techo y **supera las 107 celdas**, o sea la unica
  que puede mover el LB de forma detectable. En contra: el OOF la penaliza, con 0.3793 de
  acierto entre las modificadas y -21 aciertos netos, y v04 perdio 0.008 en una tasa
  casi identica.
- **ch2** (37 celdas, LB 0.78418). Correccion de prior en el logit con los factores por
  etiqueta del EM, umbrales de V03 intactos. Es la unica candidata que usa informacion de
  test medida fuera de modelo. La validacion anidada de la fase v9 le.media solo
  +0.000574, asi que el techo es bajo, pero es la de menor riesgo: 37 celdas.
- **ch4** (24 celdas, LB 0.78242). Control negativo: la hipotesis del encargo sometida a
  control, mismo modelo y misma regla por cuantil, solo baja la tasa. Acierto entre las
  modificadas 0.5376 y +7 aciertos netos, el unico positivo de las cinco, pero con 24
  celdas es indetectable. No se envia esperando ganar, sino para cerrar la pregunta de la
  direccion con un contraste limpio.
- **ch3** (183 celdas, LB 0.77431). Peso de mezcla 0.3 word / 0.7 char sin tocar ni el
  numero de features ni el C, para no repetir los modos de fallo ya pagados en v05, v06 y
  v07. Es la de mayor movimiento y la de peor soporte: corrected_cv 0.754593, es decir
  **-0.0226** respecto al champion, con 0.3299 de acierto entre las modificadas y -100
  aciertos netos. Se la envia solo por su tamano de movimiento, con la prediccion de
  perdida.

**Aviso de honestidad sobre el ranking**: el orden por valor esperado puro es
ch5 > ch1 > ch2 > ch4 > ch3, y ese es el orden de la tabla. Pero ch5 y ch3 son las unicas
que superan las 107 celdas, y las dos tienen el soporte local mas debil. Por probabilidad
de transferencia el orden es **ch1 > ch5 > ch2 > ch4 > ch3**. Si solo se puede enviar
una, enviar ch1; si hay presupuesto para varias, enviar ch1 y ch5 en ese orden.

## 5. Lo que queda descartado y por que

| Linea | Motivo |
|---|---|
| Segundo nivel (meta, stacking, Ridge/LR sobre scores OOF) | Ninguno ha transferido en 14 sondas. v08_meta_v2, v09_candidate_a, v10_candidate_b y v11_candidate_c quedaron por debajo de V03, y v12_meta36 fue rechazado por el holdout (-0.0122). |
| Promediar probabilidades | Falla de forma consistente: v04 -0.0234, v08_p1 -0.0084, v12_p7 -0.0118. La causa esta medida: el promedio contrae las probabilidades y con umbrales fijos eso mueve celdas en la direccion equivocada. Por eso las candidatas de tasa se fijan por **cuantil**, que es invariante a la escala. |
| Subir umbrales / igualar la tasa a la prevalencia del test | Refutado por la pendiente (positiva en las tres estimaciones) y por el contrafactual causal. Ademas la tasa optima de F1 se situa **por encima** de la prevalencia, y la proporcion se ensancha cuando la prevalencia baja, asi que igualar tasa y prevalencia es el objetivo equivocado para esta metrica. |
| Post-proceso jerarquico sobre V03 | Medido y refutado antes de gastar una submission: baja el corrected_cv de 0.777211 a 0.773024 y el acierto entre las 55 celdas modificadas es 0.4909, por debajo del azar. V03 no lo lleva y no deberia llevarlo. |
| Mas capacidad o menos capacidad en la union | v05 (32k, C=3.26) -0.02788, v07 (min_df=5) -0.02896, v06 (C=1) -0.02063. Todo el espectro de capacidad cae por debajo del champion. |

## 6. Nota sobre el presupuesto de submissions

El leaderboard conserva la mejor puntuacion, asi que enviar candidatas **no puede
degradar el 0.78306 ya registrado**. El coste de enviar es de presupuesto, no de score.
El valor de las candidatas de la v16 no es esperar una mejora, sino reducir la
incertidumbre sobre el signo y la magnitud de la respuesta del LB a la tasa de
positivos, que es el unico eje del proyecto con un contraste cuasi experimental
(V02 -> V03). Conviene recordar que de 15 submissions historicas solo una supero a la
anterior, y que V03 es el maximo observado, de modo que su 0.78306 esta sesgado hacia
arriba por seleccion y el nivel esperado de la familia es 0.776 a 0.780.

## 7. Reglas heredadas que siguen vigentes

- El test nunca se usa para entrenar, ajustar vocabularios, umbrales ni seleccionar
  features. Solo para generar predicciones y para el diagnostico no supervisado del
  EM de prevalencia, que no consume etiquetas.
- El holdout2 esta agotado (5 de 5) y no se usa desde la fase v3.
- Sin datos externos etiquetados, sin transduccion, sin reconstruir etiquetas del test.
- Toda correccion propia se documenta antes de interpretar resultados. Ningun resultado
  se edita a posteriori.

## 8. Correcciones propias registradas en la fase v16

Ninguna se aplico despues de mirar el resultado; se documentan aqui para que el
lector pueda auditar las cifras.

1. `resolver_k` biseccionaba en el sentido equivocado (la funcion objetivo es creciente
   en `k`, no decreciente) y devolvia el borde `k = 2.0`. Corregido antes de leer
   ninguna prevalencia.
2. `f1_ponderada` multiplicaba `pesos` de forma `(n,)` por `y` de forma `(n, 1)`, es
   decir un producto exterior que devolvia una matriz `(n, n)` en vez de `n` sumandos.
   El sintoma fue un contrafactual completamente insensible a `k`: los seis escenarios
   devolvian umbrales y F1 identicos. Corregido aplanando y anadiendo una comprobacion
   de longitudes que hace el fallo imposible de repetir en silencio.
3. Los dos primeros accumulator de `contrafactual_prevalencia` estaban intercambiados
   respecto a su docstring, de modo que la linea de informe atribuia al "coste" un numero
   que era el signo del gasto. Se renombraron a `f1_umbral_v03_prevalencia_train` y
   `f1_umbral_v03_prevalencia_test`.
4. El marginal `dF1/d(umbral)` se media con paso 0.002 y devolvia +0.14, +0.13 y -0.44
   para las tres etiquetas, una dispersion de 3.5 veces que es ruido de
   discretizacion y no senal. Se paso a paso 0.01.
5. `json.dumps` reventaba con `TypeError` al volcar los `ndarray` de umbrales, despues de
   haber ejecutado todos los experimentos. Se anadio `sanear`.
6. El bloque de informe de `v16_00` quedo huerfano dentro de `contrafactual_prevalencia`
   tras un `return`, de modo que el script no imprimia nada y salia con codigo 0. Las
   funciones se movieron a nivel de modulo.
7. `Path.write_text(..., encoding='ascii')` trunco `CHAMPION.md` a 0 bytes al fallar la
   codificacion a mitad de escritura. Se reescribio el fichero en ASCII puro.

## Estado al cierre de V16

**Ninguna de las 5 submissions enviadas en V16 supero al champion.** V03 sigue como champion.

| challenger | CV | LB | dLB | estado |
| --- | ---: | ---: | --: | --- |
| **V03 (champion)** | 0.777211 | **0.78306** | +0.00000 | CHAMPION |
| v16_ch1, tasa predicha 0.3836 | n/d | 0.77828 | -0.00478 | REJECTED |
| v16_ch5, tasa predicha 0.3980 | n/d | 0.77495 | -0.00811 | REJECTED |
| v16_ch7, C=1.5 | 0.776653 | 0.77296 | -0.01010 | REJECTED |
| v16_ch8, C=2.5 | 0.776752 | 0.77707 | -0.00599 | REJECTED |
| v16_ch9, min_df=1 | 0.765532 | 0.76978 | -0.01328 | REJECTED |
| v16_ch6, promedio 5 semillas | n/d | no enviada | n/d | NOT SUBMITTED |

**ch6 no se envio, y la razon es el hallazgo mas util de la fase**: las cinco semillas distintas
dan modelos **bit a bit identicos**, porque liblinear es determinista aqui. No hay varianza que
reducir, luego el promedio no es un ensemble. Ademas su tasa predicha era 0.3833, casi la de ch1
que ya habia perdido 0.00478, asi que enviarla habria sido gastar una submission en una
configuracion ya medida como perdedora.

**Las tres dimensiones de un solo parametro tienen su optimo en V03**:

| dimension | por debajo | V03 | por encima |
| --- | --- | --- | --- |
| C | 1.0 -> 0.77227, 1.5 -> 0.77296 | **2.0 -> 0.78306** | 2.5 -> 0.77707, 3.26 -> 0.77602 |
| min_df | 1 -> 0.76978 | **2 -> 0.78306** | 5 -> 0.77034 |
| tasa predicha | 0.3385 -> 0.77993 | **0.3645 -> 0.78306** | 0.3836 -> 0.77828, 0.3980 -> 0.77495 |

**Y una distincion que no teniamos antes**: la anticorrelacion entre CV y LB **no es una propiedad
del proyecto, es una propiedad de la familia de segundo nivel**. Las cuatro sondas con CV mejor
que V03 que son meta o stacking transferieron a la baja, y en cambio las tres sondas de vecinos
inmediatos de esta fase, con CV peor, tambien transferieron a la baja **pero en el mismo orden que
el CV**. En la familia de vecinos el CV local si es util.

**Conclusiones de la fase, con la formulacion que la instruccion prescribe**:

> Los vecinos inmediatos de regularizacion/capacidad alrededor de V03 tampoco producen una mejora
> de leaderboard. Junto con la curva de tasa positiva, la ausencia de diversidad entre seeds y el
> historial de transferencias, esto deja a V03 como el mejor score observado.

> V03 continua siendo la mejor solucion reproducible bajo las restricciones actuales. Bajo las
> pruebas competitivas realizadas y las reglas disponibles, este es el mejor score obtenido.

**Correccion al encargo de las tres sondas**: se tomaba V03 con min_df=1 y se proponia min_df=2
como tercera sonda. V03 tiene **min_df=2**, verificado en las constantes PARAMS_WORD y PARAMS_CHAR
del arnes validado, en el manifiesto de la submission campeona y en el pipeline congelado. Ademas
el proyecto ya habia enviado V03 con ref 56571480 y obtenido 0.78306. La tercera sonda habria sido
V03 misma. Se sustituyo por min_df=1, que es el vecino inmediato por debajo del valor real.

**Champion conservado**
- Modelo: LogisticRegression balanced C=2.0 x3, liblinear, max_iter 2000, semilla 42
- Representacion: TF-IDF word(1,2) + char_wb(3,5), min_df=2, sublinear_tf
- Thresholds: HS 0.44, TR 0.51, AG 0.46
- corrected_cv: 0.7772114525165721
- LB: 0.78306
- submission ref: 56571480
- sha256: 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e

<!-- P63 2026-09-29 -->
## P63 (2026-09-29) - Representacion a nivel de registro, estructura del dataset y composicion

Seis experimentos, dos auditorias y un dataset auxiliar. Ninguno supera a V03, y lo
interesante no son los negativos sino que cada uno lleva un mecanismo MEDIDO.

```text
P63-00  ID forensics      D. STRUCTURED_IDENTIFIER, confianza alta, 0 experimentos
P63-02  id predictivo     0,419136 contra 0,424038 de un control de igual dimension
P63-03  relacional        el bloque esta 92,92 por ciento vacio
P63-04  composicion       brecha especifica +0,0083528268, y +0,1055 en solitario
P63-05  texto largo       la brecha en MUY_LARGA es -0,009437
P63-06  integracion       brecha +0,0085891400, MARGINAL
P63-07  residuo           los bloques restan al margen, luego son redundantes con el
                         margen y no con los n-gramas
```

**El id no es un contador.** V15-00 lo afirmo sin medirlo. Train y test son dos muestras
disjuntas del mismo bloque contiguo, con interseccion 0 y union
2885 sobre 4999 posiciones. Es un indice de
coleccion, y su contenido es la posicion en un ensamblado, que no correlaciona con la
etiqueta.

**La composicion en ratios porta senal y no cabe.** Es un resultado distinto del de P62, no
uno contrario: un canal disperso de varios cientos de columnas compite por el presupuesto
L2 de un espacio de 21000 dimensiones, y 42 proporciones acotadas no compiten. Y aun asi
no mejora a V03, porque lo que aportan ya esta implicito en la confianza de V03.

**La hipotesis mas antigua del repositorio queda refutada.** V30-01 escribio que el efecto de
la URL y la mencion cambia de signo con la longitud, y nunca gasto un experimento. El bloque
de posiciones es PEOR que un bloque aleatorio del mismo ancho precisamente en la region
muy larga, con -0.009437. Refutada en su prediccion, no solo
no confirmada.

**El limite que queda es de orden, no de decision.** El margen hasta el techo moviendo solo
el corte va de +0.0000 en la region corta a +0.0890 en la
muy larga, y en la region CORTA es exactamente +0.0000: el umbral de V03 ya
es el optimo ahi sobre las tres etiquetas.

**Dos correcciones cambiaron la direccion de una conclusion** y tienen fila propia en el
registro. El confound de escala de P63-04 paso el delta de -0,1899987838 a
-0,0102687019 y cambio el signo de la brecha. El techo de P63-06 paso el margen de MUY_LARGA
de +0,1589 a +0,0890.

```text
no evidence of exploitable remaining model capacity in the record-level and
aggregate-composition dimensions of this representation

V03 remains the best verified leaderboard result.
```
<!-- /P63 2026-09-29 -->

<!-- P64 2026-09-29 -->
## P64 (2026-09-29) - Representacion del tweet largo y distribucion de la evidencia interna

Cuatro experimentos, dos auditorias, trece figuras. Ninguno supera a V03, y la respuesta
tiene dos mitades que no se pueden separar.

**La estructura interna es REAL.** P59-07 habia explicado su fracaso por cobertura, no por
concepto: con ventana de 32 tokens solo el
9.6 por ciento de las filas tiene mas
de una unidad, mientras que la segmentacion por puntuacion llega al
50.6 por ciento. Un factor de
5.29 en las celdas alcanzables, que no es una variacion
de parametro.

**Y el leave-one-out, que P59 nunca computo, muestra el mecanismo.** El 80,88 por ciento de
los documentos multi segmento de la region MUY_LARGA tienen segmentos que empujan la decision
en signos opuestos, la entropia interna sube de 0,1048 a 0,4565 con la longitud, y la
concentracion cae de 0,7554 a 0,6170. En los tweets largos la decision no descansa en un
segmento dominante sino en un reparto contradictorio. Eso es exactamente lo que una bolsa de
n-gramas aplana.

**Y no aporta.** El bloque de evidencia queda -0.0140959650 frente al
margen de V03, y el leave-one-out -0.0004650203, que es casi nula.
Tampoco gana en la region larga, que es donde P63 midio el margen, y el margen tampoco mejora
ahi, luego el problema de MUY_LARGA no es de representacion ni de confianza.

**El perfil segmental es longitud disfrazada.** El mejor rasgo da
0.589847 de AUC frente a
0.600506 de contar tokens, luego queda
cerrado como rasgo.

**La razon de fondo.** El margen de V03 solo ya explica sus propios errores con AUC 0,770875
contra el acierto, y el mejor rasgo de evidencia interna llega a 0,605302. La estructura no
falta: esta dentro de la confianza de V03.

```text
the internal evidence distribution of long tweets is real, measured and reproducible,
and V03 already carries it inside its own confidence

V03 remains the best verified leaderboard result.
```
<!-- /P64 2026-09-29 -->

<!-- P65 2026-09-29 -->
## P65 (2026-09-29) - Representacion de largo alcance, alcance de operadores y jerarquia

Cuatro experimentos y una auditoria, trece figuras. Ninguno supera a V03, y la respuesta tiene
una forma que no estaba prevista.

**LA RESPUESTA: la senal existe y la adicion la rechaza.** Cinco candidatos de este proyecto
baten a los dos controles, el de capacidad y el de la confianza de V03, y ninguno mejora:

| candidato | contra el control de anchura | contra el margen de V03 | delta vs V03 |
|---|---:|---:|---:|
| P64-07 alcance de negacion | +0,0201566146 | +0,0088975231 | -0,0021826774 |
| P65-01 word (1,3) | +0.0596671770 | +0.0071281317 | -0,0039520688 |
| P65-03 cohesion documental | +0.0090835614 | +0.0083711943 | -0,0027090062 |

Lo que falta no es informacion: es sitio. Unas cientos de columnas compitiendo por el
presupuesto L2 de un espacio de veinte mil, y lo que la columna nueva necesita decir es mas
pequeno que lo que cuesta decirlo. Eso es una afirmacion sobre la ARQUITECTURA de la adicion,
y es falsable: predice que la misma evidencia fundida en el espacio de probabilidad, y no
concatenada en el de caracteristicas, sobrevive.

**EL GAP ANALYSIS CERRO MAS DE LO QUE ABRIO.** 6 abiertas y
10 cerradas, leidas en el codigo y no en el registro. Los skip-grams ya los
habia medido V6-05, con un vocabulario un 83 por ciento mayor y un resultado neutro. La
negacion y la cita las habian medido E2-07 y E2-08, y despues P64-06 y P64-07.

**EL CONTROL DE IMPLEMENTACION REPRODUCE A V6-05.** El rango de hueco 1 a 2, que es
exactamente el suyo, da
-0.0010515908 frente al -0,0001 que el
registro. Luego la implementacion de los pares con hueco es la misma, y los rangos que V6-05 no
alcanzo, de tres a cinco y de seis a diez tokens, tampoco aportan.

**Y LA LONGITUD DEL TWEET NO ES EL MECANISMO.** Un especialista ajustado en LARGA y MUY_LARGA
es BIT A BIT el mismo numero que un reentrenado sobre las mismas
714 filas sin ningun criterio de longitud:
+0.0000000000. Un negativo asi no es *faltan datos*,
que seria una excusa valida: dice que la especializacion no es el mecanismo, punto. Y el
especialista gana
+0.0212 en MUY_LARGA y
pierde -0.0257 en LARGA, que tiene
el triple de filas, luego hay una mejora local real que la metrica primaria rechaza.

```text
the information about long-range relations is present, measurable and not what V03 already
knows, and none of it fits through concatenation

the length of a tweet is not the mechanism either: a model retrained on the long rows with no
length criterion is bit for bit the specialist

V03 remains the best verified leaderboard result.
```
<!-- /P65 2026-09-29 -->

<!-- P66 2026-09-29 -->
## P66 (2026-09-29) - Late fusion: la evidencia de largo alcance en el espacio de scores

Dos experimentos y dos auditorias, trece figuras. Y es la primera vez que la senal de largo
alcance mueve el numero.

**LA RESPUESTA A LA PREGUNTA DE LA FASE: SI, Y ES ATRIBUIBLE AL CANDIDATO.**

| fusion | delta vs V03 | pliegues | brecha de atribucion |
|---|---:|---:|---:|
| MI_arbol + residual anidada | +0.0026743167 | 4/5 | +0.0041133588 |
| MI_arbol + residual con score barajado | -0.0014390421 | | |

La brecha de atribucion es la cifra que manda: el mismo mecanismo con el score del candidato
**barajado** pierde casi todo.

**Y MI_arbol es el unico que funciona porque es NO LINEAL.** Las correlaciones entre z_V03 y
z_C de los siete candidatos van de +0,9294 a +0,9988: son casi la misma puntuacion. El AUC del
residuo deja a cinco de siete por debajo de 0,5, luego su parte unica destruye informacion en
vez de anadirla. El de MI_arbol es el unico por encima del azar, y su modelo es el unico no
lineal de la lista. Si el error de V03 fuera una funcion lineal del error de MI_arbol, la
regresion residual no dejaria nada.

**EL MATIZ, Y VA PEGADO AL RESULTADO.** De las
53 celdas que la fusion rescata, solo
6 son errores de
alta confianza y las demas estan pegadas al umbral, luego es sobre todo **recalibracion**. Y el
bootstrap por 1415 grupos da IC95
[-0.0044311634, +0.0102108745], que
**incluye el cero**. Veredicto PROMISING, no FINALIST.

**EL INVENTARIO FUE EL PROBLEMA REAL.** Ningun script de P64 y P65 escribio sus
probabilidades a disco, y `features/` esta en `.gitignore`. Los scores se regeneraron y se
verificaron contra el numero de cada fase: 5 de 7 reproducen con diferencia de 1e-11, uno es
no verificable y uno tiene una diferencia declarada. El regenerador **se paro** cuando dos no
cuadraron.

**Y UNA CORRECCION QUE CAMBIO EL VEREDICTO DEL MEJOR RESULTADO.** `blend_prob` usaba siempre
la probabilidad REAL del candidato y nunca la barajada, luego su control de barajado salia
identico al real POR CONSTRUCCION, la brecha daba cero falso, y el mejor resultado de la fase
queda INVALIDATED por un motivo que no era el suyo. Corregido, ese mismo resultado pasa a
+0,0033 con brecha +0,0093.

```text
adding columns to a twenty thousand dimension space does not fit, and correcting a three
column score does, for the SAME evidence and with the SAME base model

the only measured exception is MI_arbol, and its distinguishing feature is that the model
producing it is NON LINEAR

and the caveat: it is mostly recalibration, because 6 of 53 rescues are high confidence and
the IC95 includes zero

V03 remains the best verified leaderboard result.
```
<!-- /P66 2026-09-29 -->

<!-- P67 2026-09-29 -->
## P67 (2026-09-29): el residuo de MI_arbol es MIXED, no una senal transferible

- Categoria del mecanismo: **MIXED_CON_ATRIBUCION_PARCIAL**.
- Reentreno anidado de un escalar por etiqueta: `-0.0029731513`, es decir la calibracion **no** explica el efecto.
- AUC del residuo de MI_arbol contra la etiqueta **fuera** de la banda: `0.579597`, sobre 0,5. Los otros seis candidatos caen por debajo.
- Sustitucion cruzada: `4 de 7` candidatos dan positivo con la misma operacion.
- Enriquecimiento de los rescates: `1.338302`.
- Efecto en celdas: `4.01` de 1483 filas.
- 1 de 12 tracks principales ejecutados, 0 de 4 reservas.
- 0 nuevas representaciones, 0 submissions, 0 Kaggle, 0 holdout, 0 test.
- Figuras: 6 en `figuras/p67/`.

> V03 remains the best verified leaderboard result.
<!-- /P67 2026-09-29 -->
\n<!-- POST-P67 2026-09-29 -->\n## Post-P67 (2026-09-29): el gap analysis y una linea nueva cerrada

- Poblacion **B = 215 celdas** donde V03 falla y alguien acierta. El 28,4 por ciento de sus
  errores. **122 exclusivas** de un solo candidato.
- Once familias clasificadas: 6 `CLOSED`, 2 `INVALIDATED`, 1 `DUPLICATE`, 1 `COMPLEMENTARY`,
  1 `PARTIALLY_OPEN` (MI_arbol).
- `P68-H1` (enrutado por estructura del ejemplo): **INVALIDATED**, delta
  -0.0018992302, 1 de 5 pliegues.
- El techo era `+0,0390871336`. El enrutador no lo captura porque la estructura predice
  dificultad (AUC 0,6080), no ganador (AUC 0,5491).
- 1 experimento principal, 2 controles, 1 ablacion. Sin grid.

> V03 remains the best verified leaderboard result.
<!-- /POST-P67 2026-09-29 -->\n
<!-- CIERRE 2026-09-29 -->
### P68-H2 y el cierre (2026-09-29)

- `P68-H2 (rescue re-weighting)`: **DUPLICATE**. El mecanismo (sample weighting) esta medido
  nueve veces; el indicador propuesto es un subconjunto estricto del de V4-B03, y en la
  direccion que salio mal. **Coste: 0 experimentos.**
- Los ocho ejes del encargo tienen cita historica. **0 sin recorrer.**
- **Fase de investigacion cerrada**, con la condicion del encargo cumplida.
- 122 rescates exclusivos, 77 de MI_arbol (46 en LARGA). Techo oracle de las celdas
  delegables: +0,0391, no alcanzado por cuatro mecanismos distintos.
<!-- /CIERRE 2026-09-29 -->

<!-- DOWNWEIGHT-AUDIT 2026-09-29 -->
### Auditoria de A_cost_sensitive_downweight (2026-09-29)

- **REPRODUCIBLE, nivel R3.** `0,7854160387` reproducido exacto a `3,2e-11`; delta
  `+0,0082045862`; cinco deltas por pliegue identicos; 99 rescates y 74 danos identicos.
- **Criterio 6 = `A_falta_de_artefacto`**: faltaba el CSV de submission, y la directiva 12
  prohibia crearlo. Fallo procedimental, no experimental.
- **Segunda particion CONFIRMED**: delta `+0,0105253130` con 5 de 5 pliegues.
- **ESTADO FINAL: MARGINAL.** Bootstrap por grupos `[-0,000691, +0,011818]` incluye el cero.
  Con 1415 grupos sobre 1483 filas, el remuestreo por filas subestimaba la incertidumbre.
- Correccion documental: el mecanismo **no es un meta-modelo**; P58 lo etiquetaba mal.
<!-- /DOWNWEIGHT-AUDIT 2026-09-29 -->

<!-- DOWNWEIGHT-CONFIRM 2026-09-29 -->
### Confirmacion de A_cost_sensitive_downweight (2026-09-29)

- **Tres de tres particiones positivas**: canonica `+0,0082045862`, segunda `+0,0105253130`,
  tercera `+0,0048825807`. Media `+0,0078708266`.
- **Test de permutacion**: `p = 0,000999`. **0 de 1000** asignaciones equivalentes alcanzan el
  delta observado; el maximo de la distribucion nula es `+0,0045971670`, la mitad del real.
  La regla funciona por **que** ejemplos reciben menos peso.
- **Defecto corregido**: `ajustar_predecir` dejaba `class_weight='balanced'` con multiplicador
  1.0, y sklearn recalcula 'balanced' con los `sample_weight`. La canonica no se afecto
  (carga el OOF); las particiones segunda y tercera si, y se repitieron.
- **ESTADO FINAL: PROMISING_UNCONFIRMED** (5 de 6 componentes). El bootstrap por grupos sigue
  incluyendo el cero, luego **no se adopta** y **no se envia submission**.
<!-- /DOWNWEIGHT-CONFIRM 2026-09-29 -->

<!-- DOWNWEIGHT-CLOSE 2026-09-29 -->
### Cierre de A_cost_sensitive_downweight (2026-09-29)

**Conclusion congelada**: mejora reproducible, consistente entre tres particiones y altamente
improbable bajo la permutacion de la asignacion de pesos; pero el bootstrap por grupos continua
incluyendo cero, luego la magnitud no queda resuelta con precision suficiente para ADOPTED.

```text
ORIGINAL                     +0,0082045862   nunca enviado, nunca auditado
REPRODUCED                   +0,0082045862   R3, diferencia 3,2e-11
SECOND_PARTITION             +0,0105253130   5 de 5
THIRD_PARTITION_CORRECTED    +0,0048825807   4 de 5
PERMUTATION                  p = 0,000999    0 de 1000 superan al observado
FINAL_STATUS                 PROMISING_UNCONFIRMED

research_status              CANDIDATE_PENDING_GOVERNANCE
SUBMISSION  NO     TEST  NO     LEADERBOARD  NO
```

**Limitacion de precision estadistica**: el IC95 por grupos `[-0,0006906975, +0,0118175641]`
incluye el cero. El limite inferior esta a seis diezmilesimas. No se resuelve con mas
particiones: seria seleccion, no evidencia.

Las tres pruebas responden a preguntas distintas: particiones = estabilidad del reparto;
permutacion = especificidad de la correspondencia peso-ejemplo; bootstrap = magnitud.

Nota de gobernanza en `reports/a_cost_sensitive_downweight_governance_note.md`.
<!-- /DOWNWEIGHT-CLOSE 2026-09-29 -->
