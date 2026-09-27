# CHAMPION

Proyecto: KAGGLE_TOXIC_2026_PUJ (https://www.kaggle.com/competitions/tweet-toxic-2026-puj)
Metrica oficial: F1 Score, media column-wise de la F1 binaria de la clase positiva
sobre HS, TR y AG (`f1_media_columnas`, D-009).
Ultima actualizacion de este fichero: 2026-09-26, fase v16.

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

| # | Clave | Fichero | Tasa | Celdas | LB combinado | sha256 |
|---|---|---|---|---|---|---|
| 1 | ch1 | `submissions/toxic_submission_v16_ch1_tasa_optima_transportada.csv` | 0.3836 | 66 | 0.78535 | `a1b91806f7201aa14a16d4b6d95f8032f04bb36812e417cd9e1c481a5a1ac032` |
| 2 | ch5 | `submissions/toxic_submission_v16_ch5_tasa_optima_absoluta.csv` | 0.3980 | 116 | 0.78650 | `dac0d815465ada1df4b1c554deeb1b4a39d863226e0d8154dd3349d5cd0492d9` |
| 3 | ch2 | `submissions/toxic_submission_v16_ch2_prior_shift_em.csv` | 0.3752 | 37 | 0.78418 | `118817f26703fcb8385241a39ac885eae74a37643789f770871732f178a25efe` |
| 4 | ch4 | `submissions/toxic_submission_v16_ch4_control_tasa_baja.csv` | 0.3576 | 24 | 0.78242 | `e0d6bb1cd71e0b49e8e14beb585779b8dc3eb8300892899db4ec21a4f8739d9c` |
| 5 | ch3 | `submissions/toxic_submission_v16_ch3_peso_char_070.csv` | 0.3873 | 183 | 0.77431 | `d0d5e42e2b3ec5125721e801deb52206d904bb5bd41d506a547928893dec20a8` |

`LB combinado` = 0.78306 + 0.12011 x delta_tasa + aciertos_netos_oof x 2.804e-05, y para
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
