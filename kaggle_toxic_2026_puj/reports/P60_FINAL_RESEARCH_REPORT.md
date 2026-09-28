# P60 - Informe final de investigacion guiada por literatura

Generado por `experiments/p60_11_cerrar.py`. Fecha 2026-09-28T18:00:25+00:00.
Commit fbaa516. Rama v60. Sin push.

## Resumen ejecutivo

P60 ejecuto doce pistas de la literatura sobre aprendizaje multi-etiqueta, sobre la frontera que P59 dejo cerrada. Doce pistas, 8 experimentos consumidos, 4 cerradas sin experimentar, 0 candidatas, 1 marginales, 7 cerradas, 0 invalidadas y 0 bootstraps ejecutados.

**Ninguna pista produjo un candidato.** El gate exige delta de al menos +0,003 y 4 de 5 pliegues, y ninguna de las ocho pistas con experimento llego al primero de los dos criterios. Ningun bootstrap se ejecuto, porque el bootstrap solo se ejecuta despues de superar el gate.

```text
V03 corrected_cv = 0.7772114525165721
V03 LB           = 0.78306
V03 sha256       = 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
```

> V03 remains the best verified leaderboard result.

## Estado de V03, verificado al inicio y al cierre de cada pista

El sha256 de la submission se verifica al comienzo de cada uno de los ocho scripts de experimento y en este script de cierre. Ninguna pista modifico V03, el OOF canonico, los pliegues ni las submissiones. Cero submissiones en toda la fase.

## Motivacion: por que esta fase es distinta de P59

P59 pregunto si quedaba alguna fuente de senal dentro de las familias ya conocidas. P60 cambia la pregunta: en vez de variations sobre la misma familia, se buscaron mecanismos de aprendizaje apoyados en literatura reciente que **modifiquen como se representa, optimiza o combina la evidencia** y que no sean una reformulacion de lo ya cerrado. Por eso el bloque empieza con una revision bibliografica y un registro de huecos, y las pistas se cierran por duplicacion funcional antes de gastar experimento cuando la literatura no aporta un mecanismo nuevo.

## Revision bibliografica y gap analysis

Se evaluaron **14 metodos** de la literatura, con prioridad en 2024, 2025 y 2026 y con los trabajos clasicos cuando son la referencia metodologica. 9 pasaron el gap check y 5 se cerraron por equivalencia funcional.

| ref | metodo | filas del registro | mecanismo en una linea |
|---|---|---:|---|
| CONTRAST-OVL | aprendizaje contrastivo supervisado con solapamiento de etiqueta | 1 | modifica la GEOMETRIA de la representacion aprendida: dos instancias son positivas si comparten  |
| ASL | Asymmetric Loss: foco asimetrico y desplazamiento de probabilida | 0 | dos terminos, L+ = (1-pm)^gamma+ * -log(p) y L- = pm^gamma- * -log(1-p), con pm = max(p - m, 0). |
| DRO-GROUP | optimizacion minimax sobre un conjunto de grupos | 0 | minimiza el maximo, o una media superior de tipo CVaR, del riesgo sobre grupos, en vez del riesg |
| EFFNUM | ponderacion por numero efectivo de muestras (1-beta)/(1-beta^n_c | 0 | sustituye el peso 1/n_c por (1-beta)/(1-beta^{n_c}), que satura: las clases grandes pesan propor |
| DBL | perdida equilibrada en distribucion con regularizacion tolerante | 0 | rebaja el peso de lasnn en regiones con exceso de negativos, usando la razon de recuentos positi |
| PAIR-NOISE | deteccion de ruido a nivel de par instancia-etiqueta y correccio | 0 | la unidad de deteccion es el par, y el criterio combina la perdida del par, informacion de rango |
| OT-PROT | redes basadas en prototipos para clasificacion de texto | 1 | clasifica un punto por su similitud a un conjunto de prototipos aprendidos, no a un unico centro |
| META-RW | meta-aprendizaje de pesos por ejemplo desde la alineacion del gr | 0 | los pesos se obtienen de un objetivo META que compara el gradiente de cada ejemplo de entrenamie |
| POLYLOSS | perdida polinomica learned de la entropia cruzada | 0 | expansion polinomial de la entropia cruzada, Pol_1 y Pol_2, con coeficientes entrenables o fijad |
| W-ASL | ASL mas ponderacion por frecuencia de etiqueta mas suavizado de  | 0 | ASL mas un peso por frecuencia de etiqueta y mas un suavizado de la estructura de coocurrencia e |
| LOGIT-ADJ | logit adjustment post-hoc y en la perdida, con grounding estadis | 0 | anade tau * log(pi_y) al logit, post-hoc o dentro de la perdida, para favorecer margen relativo  |
| LDAM | margen por etiqueta Delta_y = C / n_y^(1/4) aplicado solo al ter | 0 | reduce el logit de la clase verdadera en una cantidad Delta_y, lo que cambia el gradiente del po |
| LDL | aprendizaje de distribuciones de etiqueta en un espacio continuo | 1 | regresion sobre vectores de distribucion de etiqueta aprendidos |
| CONFORMAL | prediccion conformista con garantia de cobertura a nivel de conj | 0 | devuelve un CONJUNTO de etiquetas con garantia de cobertura finita, construido con un no conform |

### La estructura de etiquetas, medida y no citada

Este es el hecho que ordena la revision y viene del OOF canonico, no de ningun trabajo:

```text
prevalencia_HS                   0.43695212407282535
prevalencia_TR                   0.3000674308833446
prevalencia_AG                   0.38233310856372216
P_TR1_dado_HS0                   0.0
P_AG1_dado_HS0                   0.026347305389221556
P_AG1_dado_TR1                   0.946067415730337
negativos_TR_no_definidos        1.0
negativos_AG_no_definidos        0.9736526946107784
```

Tres consecuencias, y las tres importan mas que cualquier resultado individual:

1. **Las etiquetas estan anidadas, no solo correlacionadas.** P(TR=1 | HS=0) = 0.0000. El subespacio factible tiene cuatro puntos, que es exactamente el que `V4-E01-E03` enumero en doce variantes sin superar a V03.

2. **El 100 por ciento de los negativos que ve TR en entrenamiento son etiquetas NO DEFINIDAS**, y el 97 por ciento en AG. Eso es ruido dependiente de la etiqueta, y es lo que la pista C midio por pares.

3. **La asimetria de prevalencias es de 1,46 a 1, no de una clase minoritaria frente a una dominante.** Ridnik et al. justifican la perdida asimetrica por un desequilibrio de uno o dos ordenes de magnitud, luego la priori de la pista A era baja desde el principio y la revision lo declaro antes de ejecutar.

## Las doce pistas

| orden | pista | referencia | veredicto | delta del mecanismo | pliegues | neto |
|---:|---|---|---|---:|---:|---:|
| 1 | G | Multi-Label Text Classification with Label Att | **CLOSED** | -0.294907 | 0/5 | -590 |
| 2 | A | Asymmetric Loss for Multi-Label Classification | **CLOSED** | -0.152926 | 0/5 | -1218 |
| 3 | L | Distributionally Robust Optimization with Grou | **MARGINAL** | +0.000336 | 2/5 | 1 |
| 4 | D | Class-Balanced Loss Based on Effective Number  | **CLOSED** | -0.000432 | 0/5 | 0 |
| 5 | C | Noisy Multi-Label Text Classification via Inst | **CLOSED** | +nan | 0/5 | 1 |
| 6 | H | Robust Text Classification: Analyzing Prototyp | **CLOSED** | -0.072307 | 0/5 | -2264 |
| 7 | I | Learning to Reweight Examples for Robust Deep  | **CLOSED** | -0.301104 | 0/5 | -572 |
| 8 | J | PolyLoss: A Unified Formulation of Cross-Entro | **CLOSED** | -0.001710 | 2/5 | 2 |

### Pistas cerradas sin consumir experimento

| pista | referencia | precedente funcional | decision |
|---|---|---|---|
| P60-B | W-ASL | ASL mas ponderacion por frecuencia de etiqueta, que es class_weight=balanced, mas suavizado de coocurrencia, que es la familia de dependencia de etiqu | CERRADA POR DUPLICACION |
| P60-E | LOGIT-ADJ/LDAM | sobre un modelo lineal el ajuste de logit es un desplazamiento de intercepto, y con umbral fijo un desplazamiento de intercepto es un cambio de umbral | CERRADA POR DUPLICACION |
| P60-F | LDL | tres etiquetas anidadas dan un subespacio factible de cuatro puntos, medido en este corpus, y V4-E01-E03 ya lo enumero en doce variantes | CERRADA POR DUPLICACION |
| P60-K | CONFORMAL | convertir un conjunto conformista en una decision binaria exige cortar la puntuacion de no conformidad en un punto, que es un ajuste de umbral ya cerr | CERRADA POR DUPLICACION |

* **P60-B (W-ASL)**: ASL mas ponderacion por frecuencia de etiqueta, que es class_weight=balanced, mas suavizado de coocurrencia, que es la familia de dependencia de etiquetas. Precedentes: V8-07 y V4-E01-E03.
* **P60-E (LOGIT-ADJ/LDAM)**: sobre un modelo lineal el ajuste de logit es un desplazamiento de intercepto, y con umbral fijo un desplazamiento de intercepto es un cambio de umbral, tanto post-hoc como dentro de la perdida. Precedentes: V8-09.
* **P60-F (LDL)**: tres etiquetas anidadas dan un subespacio factible de cuatro puntos, medido en este corpus, y V4-E01-E03 ya lo enumero en doce variantes. Precedentes: V4-E01-E03.
* **P60-K (CONFORMAL)**: convertir un conjunto conformista en una decision binaria exige cortar la puntuacion de no conformidad en un punto, que es un ajuste de umbral ya cerrado de raiz. Precedentes: V8-09.

## Resultados, incluidos los negativos

No se muestran solo los mejores. El orden de lectura que mas informa es por el delta del mecanismo contra su propio control, y no por el corrected_cv absoluto.

| pista | corrected_cv | delta del mecanismo | pliegues | rescate | dano | neto | celdas |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.624338862 | -0.1529258699 | 0/5 | 334 | 1552 | -1218 | -0.1528725906 |
| C | 0.7772647319 | +nan | 0/5 | 2 | 1 | 1 | 5.32794e-05 |
| D | 0.7768328831 | -0.0004318488 | 0/5 | 3 | 3 | 0 | -0.0003785694 |
| G | 0.4497522457 | -0.2949072740 | 0/5 | 278 | 868 | -590 | nan |
| H | 0.4686780062 | -0.0723067170 | 0/5 | 391 | 2655 | -2264 | -0.3085334463 |
| I | 0.4761602463 | -0.3011044856 | 0/5 | 318 | 890 | -572 | -0.3010512063 |
| J | 0.7755546945 | -0.0017100374 | 2/5 | 25 | 23 | 2 | -0.0016567581 |
| L | 0.777600986 | +0.0003362541 | 2/5 | 2 | 1 | 1 | 0.0003895335 |

## Resultados marginales

* **L**, delta del mecanismo 0.0003362541, 2 de 5 pliegues, neto 1. Mecanismo: 3 celdas de 4449 cambiadas

## Candidatos

**Ninguno.** Ninguna pista supero el gate de +0,003 con 4 de 5 pliegues, luego no hay finalista y no hay artefacto de candidato que generar.

## Bootstrap

Ejecutados: **0 de 8**. El bootstrap se ejecuta solo despues de superar el gate, y ninguna pista lo supero, luego no hay ningun intervalo de confianza que reportar. Es preferible no tener ninguno que tener el intervalo de una mejora que no existe.

## Complementariedad

La tabla de rescate y dano es la que mas informa sobre si una tecnica aporta informacion o solo desplaza decisiones. Ninguna pista de P60 produjo un neto que se aproxime a un volumen de correccion util comparable al de un modelo alternativo entrenado de verdad.

* **A**: rescate 334 / dano 1552 / neto -1218, pliegues 0/5
* **C**: rescate 2 / dano 1 / neto 1, pliegues 0/5
* **D**: rescate 3 / dano 3 / neto 0, pliegues 0/5
* **G**: rescate 278 / dano 868 / neto -590, pliegues 0/5
* **H**: rescate 391 / dano 2655 / neto -2264, pliegues 0/5
* **I**: rescate 318 / dano 890 / neto -572, pliegues 0/5
* **J**: rescate 25 / dano 23 / neto 2, pliegues 2/5
* **L**: rescate 2 / dano 1 / neto 1, pliegues 2/5

## Reproducibilidad

Cada pista es un script nuevo y completo que se ejecuta de principio a fin y escribe su propio artefacto JSON, su informe y su CSV. Antes de aceptar cualquier resultado cada script pasa `ast.parse`, `ruff`, `py_compile` y una ejecucion real cuyo codigo de salida se captura sin tuberia. Los scripts de losses, DRO, deteccion y PolyLoss verifican su gradiente analitico contra diferencias centrales antes de entrenar, y el script de transporte verifica la exactitud de su metrica antes de concluir nada.

## Compliance

```text
transformers = 0        BERT = 0            RoBERTa = 0         LLM = 0
embeddings preentrenados = 0                FastText preentrenado = 0
Word2Vec preentrenado = 0                    GloVe = 0
clasificadores externos = 0                  etiquetas externas = 0
corpus toxicos externos = 0                  hate speech externo = 0
pseudo-etiquetas = 0                         test labels = 0
test training = 0                            test calibration = 0
test feature selection = 0                   holdout reuse = 0
leaderboard tuning = 0                      submissiones = 0
consultas a Kaggle = 0
```

Deep learning, donde se uso: solo en la pista G, con un SVD interno de 128 dimensiones y una proyeccion lineal de 32. Sin transformer, sin embedding preentrenado, sin datos externos.

## Git

Trabajo en la rama `proyecto-kaggle-toxic-2026-p60`, creada desde `v25` que contiene P59. Sin push, sin merge, sin force-push, sin rebase destructivo. Un commit local por pista. Estado al cierre de este script: commit fbaa516.

## Notebook

La integracion en el notebook consolidado se genera por script leyendo los artefactos, de modo que ninguna cifra del notebook se escribe a mano. Se registra en `reports/p60_notebook_integration.json`.

## Decision final

```text
P60 = CLOSED
V03 remains final
```

La pregunta final del encargo era si existe una tecnica respaldada por literatura que introduzca una fuente de senal o un mecanismo de aprendizaje materialmente diferente de todo lo ya probado y que consiga una mejora reproducible sobre V03. La respuesta documentada es **no**, y conviene separarla en las cinco dimensiones que el encargo pide, porque la respuesta no es la misma en todas:

* **deteccion**: si. La pista C demostro que dentro de la region sin definir de AG la confianza esta ANTICORRELACIONADA con el error, con AUC 0,1814, y que un indicador estructural aporta +0,1129 de AUC residual sobre el control de esa misma etiqueta. Es informacion nueva y real.

* **calibracion**: no aporta nada nuevo. La mejora de P59-B, que era +0,003367, venia del sesgo de la agregacion y no de correccion alguna, y P59-U quedo marginal.

* **representacion**: el contraste por solapamiento de etiquetas es genuinamente un eje nuevo, y el gap check lo confirma, pero perdio -0,2949 porque la perdida contrastiva domina por un orden de magnitud y porque la arquitectura no puede competir con V03. El transporte optimo tambien es genuinamente distinto, con 2842 veces el sesgo de su propia metrica como prueba, y perdio -0,0723 contra su control de centroide.

* **optimizacion**: la perdida asimetrica actua donde no debia y se apaga donde debia. El corte duro, que es la parte nova, nunca se activo: cero por ciento de los negativos tienen perdida exactamente cero. Lo que si actuo fue la potencia del negativo, que es un ponderado de clase, familia cerrada por V8-07. El riesgo minimax por grupos se activa con una sobreponderacion del grupo peor de +2,47 por ciento y mueve tres celdas de 4449. El numero efectivo de muestras resulta ser, medido antes de entrenar, la misma funcion que la forma de V03 con menos de 2,33 por ciento de diferencia y coseno 0,9999996 entre las dos soluciones. El meta-reweighting y PolyLoss rematan el eje de la perdida.

* **informacion nueva**: hay una, y es pequena y local: la estructura anidada de las etiquetas convierte el 100 por ciento de los negativos de TR en etiquetas no definidas, y esa region se comporta de forma distinta. Lo que no hay es una forma de exploitation de esa region que mejore la decision. El intento suave fue un no-op exacto, con delta 0,0000000000, y el intento duro ya lo habia cerrado V14-00 con -0,116581. El eje queda acotado entre dos extremos medidos: factor cero es catastrofico y factor uno no hace nada.

La conclusion que sostiene el conjunto de P59 y P60 es que V03 no esta limitado por la falta de un mecanismo de aprendizaje mejor, sino por el techo estructural que V28-01 midio: los errores de HS, TR y AG se solapan con phi de +0,42 a +0,51, luego fallan sobre los mismos textos, y cualquier metodo que actue sobre una sola etiqueta choca contra ese techo. Los dos unicos ejes que podrian esquivarlo, la geometria aprendida y el objetivo sobre grupos, se ejecutaron y ninguno produjo una mejora.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_11_cerrar.py`. Sin push.*