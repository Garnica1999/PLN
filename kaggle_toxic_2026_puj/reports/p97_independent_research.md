# P97 - Investigacion independiente: mecanismos nuevos hacia 0.80

Fecha: 2026-10-13. Objetivo: encontrar una mejora real y transferible sobre V03
(corrected_cv 0.7772114525165721; LB 0.78306; faltan +0.01694 para 0.80000).
V03 permanece congelado. Restricciones academicas vigentes (sin transformers,
sin preentrenados, sin datos externos etiquetados, sin pseudo-labeling, sin
test, sin holdouts, sin leaderboard para elegir). Presupuesto: 3 hipotesis
principales, 6 experimentos principales como maximo; los controles se registran
pero no consumen presupuesto.

## 1. Metodologia de la busqueda externa

- Fuentes primarias: arXiv (paginas de articulo y formulas), ACL Anthology
  (Findings incluidos), PMLR (ICML/ICLR), proceedings de NeurIPS, JMLR,
  JASA, Pattern Recognition, y documentacion tecnica de implementaciones
  (scikit-learn) cuando el algoritmo lo exige.
- 16 busquedas independientes el 2026-10-13, organizadas por las ocho areas
  del encargo; se extrajeron mecanismos (formulas y algoritmos) y no solo
  resumenes. Cada referencia quedo en `reports/p97_method_registry.csv` con
  DOI/URL primaria, mecanismo, datos, compatibilidad, limitaciones y
  diferencia exacta frente al historial.
- Regla de aceptacion: una tecnica no es nueva por su nombre; la comparacion
  con el proyecto se hizo contra el codigo y los resultados registrados
  (P93-P96, `experiments.csv`, DECISIONS). Si no hay evidencia de que algo se
  probo, se registra como no probado.

## 2. Cobertura por area (que se examino y que se descarto)

**A1. Aprendizaje discriminativo desde cero / objetivos.** Se reviso la
linea de sustitutos de F1 (sigmoidF1, 2021; cost-sensitive F1, NeurIPS 2014;
plug-in para F-macro, ICML 2013) y de seleccion/correccion por instancia
(DISC CVPR 2023; pares (i,l) NAACL 2024; co-teaching NeurIPS 2018; BAN
ICML 2018). Veredicto: los sustitutos son la familia cerrada V7/P60 (A);
la seleccion por memorizacion requiere modelos profundos que el proyecto ya
midio por debajo de V03 (C); BAN y suavizado de objetivos quedan en la
familia de pesos/objetivos (C). No se encontro un objetivo nuevo compatible
con LR sparse fuera de esas clases.

**A2. Optimizacion y decision multietiqueta.** Se reviso Lipton/Elkan 2014
(umbral optimo depende de la densidad local), Dembczynski 2013 (plug-in
F-macro: umbrales por etiqueta), Narasimhan 2014 (reduccion a costes),
BOVA (blog tecnico, sin peer review; no se acepta como evidencia) y la linea
de multicalibracion/omniprediccion (2018-2025). Hallazgo central: **ninguna
variante historica condiciono el umbral a los atributos del ejemplo**. Las
probaditas fueron globales por etiqueta (TH-01 adoptado), anidadas globales
(E2-16, V7-03) y anidadas completas con regla de la mediana (P55/V56-01,
+0.000925). La regla condicional por segmento es un objeto distinto y falsable
por separado (candidato C97-A, clase D).

**A3. Ruido y calidad de etiquetas por instancia y etiqueta.** Se reviso
nnPU (Kiryo 2017; formula en el registro W01), PNU (Sakai 2017), DC-PU
(NeurIPS 2025), matrices de transicion dependientes de la instancia (2020,
2022, 2024), Confident Learning (2021) y el analisis de suavizado de etiquetas
(2019-2024). Contra el proyecto: el detector propio de ruido quedo MARGINAL
(V28-03); el reetiquetado esta fuera del marco; la limpieza CL solo valdria
como pesos (familia cerrada). **Pero** la formulacion de riesgo PU/PNU - usada
aqui para tratar las filas HS=0 como mezcla no supervisada de TR/AG - no
existe en el historial y V14 dejo esa puerta explicitamente abierta
(candidato C97-C, clase D; ver desafio CH3).

**A4. Modelos locales, metricas y grafos.** NCA (metrica aprendida para kNN),
Laplacian SVM (Belkin 2006), LabelSpreading. El kNN puro nunca se ejecuto
(P94 H-P94-03) pero su sustrato (SVD) y su geometria ya tienen evidencia en
contra; la regularizacion por variedad queda en territorio de vecindad y de
near-duplicados cuarentenados (D-288). Descartados como C (C97-D, C97-F).

**A5. No linealidad e interacciones.** No aparecio ningun mecanismo nuevo
compatible fuera de kernels, n-gramas, features densas y MLP (familias
cerradas con medicion); los hallazgos 2023-2026 en texto sin transformers
siguen siendo BoW/TF-IDF + lineal (revision comparativa 2024: "Are We Really
Making Much Progress in Text Classification?"), lo que consolida a V03 como
referencia fuerte, no como un error a corregir.

**A6. Generalizacion y transferencia.** Bates/Hastie/Tibshirani (NCV, JASA
2024): IC clasicos de CV subcubren porque cada dato se usa en train y test;
la CV estima el error de modelos de otros trains, no del modelo final. Esto
respalda la banda de particion de D-070 y el diseno de replicas de P95
(W23, clase B: criterio de evaluacion, no modelo). Group DRO (Sagawa 2020)
queda como el unico mecanismo de modelado orientado a robustez/transferencia
no ejecutado de verdad (ver desafio CH1; candidato C97-B, clase B).

**A7. Uso distinto de los datos etiquetados.** La particion natural del
corpus (TR/AG definidas solo si HS=1) permite formulaciones P/N/U (Sakai
2017) y PUbN (negativos sesgados) sin supervision externa. Es la fuente de
informacion estructural mas infrautilizada del proyecto.

**A8. Metodos de otras areas.** Auto-destilacion (BAN), suavizado aprendido,
stability selection, CL: todos desembocan en familias ya medidas (objetivos,
pesos, seleccion de features) o requieren modelos no permitidos. Descartados
con registro.

## 3. Desafio explicito a P94/P95/P96 (falsos negativos buscados)

**CH1 - M-P94-12 (DRO) se cerro por "equivalencia de re-ponderacion".**
Que se probo realmente: V6-03 (peso estatico por tamano de grupo, 1/sqrt(g) y
1/g) y V4-B (downweighting por coste/cercania): reglas fijas de peso.
Que no se probo: el objetivo minimax iterativo (pesos adversarios por peor
perdida de grupo), que no equivale a ninguna regla fija. Por que la evidencia
anterior no resuelve la pregunta nueva: cambia la hipotesis causal - de
"corregir una fuente de ponderacion" a "comprar robustez/transferencia por
grupo" - y esa es exactamente la pregunta abierta por el freno de
transferencia. Falsificador: el peor grupo no mejora y la media cae o no hay
estabilizacion de pliegues.

**CH2 - M-P94-04 (reglas de decision) cubre solo umbrales globales.**
Que se probo: umbral global por etiqueta (adoptado: +0.0031 en LB pareado),
re-optimizaciones globales anidadas (sin mejora: 0.776613 en E2-16),
anidamiento completo con mediana (+0.000925), calibracion (P76). Que no se
probo: el umbral como funcion del ejemplo (segmentos pre-registrados), que la
teoria de Lipton/Elkan (umbral optimo depende de la densidad local) y la linea
de multicalibracion respaldan. Falsificador: la regla condicional no supera a
la regla global anidada en las mismas condiciones.

**CH3 - V14 cerro la perdida enmascarada, no la formulacion no supervisada.**
La identidad de V14 (enmascarar = H2-M) es algebraica para perdidas con peso
cero; la propia auditoria V14-00 escribio: "Una formulacion que de verdad
tratara HS=0 como no supervisado tendria que aportar una senal distinta de
cero". Esa formulacion existe, tiene 60 anos de teoria (PU/PNU) y nunca se
implemento ni se registro en P94. Falsificador: nnPU no supera al control
torch BCE identico en pooled o pierde en >=4/5 pliegues.

No se reabre nada mas: los demas descartes de P94 se re-verificaron y se
mantienen (kNN simple, sustitutos de F1, features, ensemble, cost-sensitive).

## 4. Candidatos y estimaciones (separa CV / complementariedad / robustez / transferencia)

| Id | Mecanismo | Clase | Coste | Riesgo | Que se afirmaria si gana |
|---|---|---|---|---|---|
| C97-A | Umbral condicionado por segmento (nested) | D | CPU-min | medio | CV + complementariedad por region; transferencia esperable si el efecto es de region |
| C97-B | Group DRO minimax | B | CPU-min | medio | robustez entre pliegues + transferencia por grupos |
| C97-C | nnPU/PNU en TR (HS=0 como U) | D | CPU-min | alto | CV en TR/AG con informacion estructural nueva |
| C97-D | kNN metrica aprendida | C | CPU-min | alto | solo diagnostico (no seleccionado) |
| C97-E | Suavizado de objetivos por ruido | C | CPU-min | alto | no seleccionado (familia de pesos) |
| C97-F | Laplaciano kNN | C | CPU-min-h | alto | no seleccionado |
| C97-G | Auto-destilacion BAN | C | CPU-min | medio | no seleccionado |
| C97-H | Stability selection | C | CPU-min | medio | no seleccionado |

Sin probabilidades numericas inventadas: los juicios son cualitativos y estan
anclados a las magnitudes historicas (mejora anidada maxima +0.0009; banda de
particion 0.017; freno de transferencia para mejoras locales <= +0.024).

## 5. Pre-registro de las tres hipotesis seleccionadas

Protocolo comun a las tres: pliegues canonicos (StratifiedGroupKFold(5, seed
42) con grupos = near-duplicado UNION pseudo-autor); ajuste solo con el train
de cada pliegue; sin test, sin holdouts, sin leaderboard; metrica primaria
`corrected_cv`; evaluacion adicional por pliegue, salvage/damage vs V03 y
bootstrap por clusters (unidad = grupo, 2000 replicas, semilla 20260925);
artefactos ASCII con hashes; fila nueva en experiments.csv por unidad.

### H97-A - Reglas de decision condicionadas por segmento
- PREDICTION: una regla nested con umbral por (etiqueta, cuartil de longitud,
  fronteras calculadas en el train del pliegue) obtiene un `corrected_cv`
  pooled superior al de la regla global nested en las mismas condiciones
  (delta >= +0.000) y >= 3/5 pliegues con mejora; complementariedad
  salvage > damage.
- NULL HYPOTHESIS: la regla condicional no supera a la global nested
  (delta <= 0 o < 3/5 pliegues).
- EXPECTED MECHANISM: la densidad local p(s|t=1) varia por region (V28-01:
  el error se concentra en textos largos); un umbral unico es un compromiso.
- PRIMARY METRIC: corrected_cv pooled de la regla (frozen V03 = referencia;
  global-nested = control del protocolo).
- FALSIFICATION CRITERION: delta(condicional - global-nested) <= 0 con
  bootstrap por clusters que no excluya el cero.
- STOP CONDITION: un unico pre-registro (cuartiles de longitud primario;
  mencion y hashtag como secundarios descriptivos, sin seleccion entre
  ellos). Sin grids adicionales.
- Coste: CPU-min (usa el OOF guardado; no entrena).

### H97-B - Group DRO minimax (peor grupo)
- PREDICTION: con grupos = fusionados del proyecto, el minimax con T = 15
  rondas y eta = 0.5 mejora la robustez (menor dispersion entre pliegues y
  peor-grupo mejor) pero la media pooled puede subir o bajar; se considera
  senal si pooled >= V03 + 0.003 con >= 4/5 pliegues, o si estabiliza la
  dispersion sin perdida material (> -0.001).
- NULL HYPOTHESIS: el minimax no mejora pooled ni el peor grupo.
- EXPECTED MECHANISM: reponderacion adversaria de grupos heterogeneos
  transfiere mejor que la media ERM (Sagawa 2020; analogia con el freno de
  transferencia).
- PRIMARY METRIC: corrected_cv pooled con umbrales V03 congelados (misma
  regla de decision que la referencia); secundario: cuantiles de perdida por
  grupo y dispersion entre pliegues.
- FALSIFICATION CRITERION: pooled < control A0 (mismo arnes, pesos
  uniformes) en >= 4/5 pliegues y sin mejora de peor-grupo.
- STOP CONDITION: una sola configuracion pre-registrada (T=15, eta=0.5,
  pesos normalizados, sin grid); si se descarta no se exploran variantes.
- Coste: CPU-min (ajustes LR liblinear sobre sparse).

### H97-C - nnPU/PNU para la cabeza TR (y AG si TR rescata)
- PREDICTION: el estimador de riesgo PU/PNU (positives = TR=1; unlabeled =
  filas HS=0; negativos definidos = TR=0 con HS=1; pi = proporcion observada
  en train) entrenado con optimizacion directa produce un score que, con
  umbrales propios nested, supera al control torch BCE identico (mismo
  arnes, mismos datos) en pooled o mejora TR en >= 3/5 pliegues.
- NULL HYPOTHESIS: el estimador no supera al control (diferencia <= 0 en
  pooled o <= 2/5 pliegues).
- EXPECTED MECHANISM: 80.44 por ciento de los negativos observados de TR no
  estan definidos (V12-02); ignorarlos (H2-M) o etiquetarlos 0 (V03) sesga
  el concepto; el riesgo no negativo usa la mezcla como informacion.
- PRIMARY METRIC: corrected_cv pooled con umbrales propios nested
  (mecanismo con regla de decision propia, P97 seccion 7); secundario:
  corrected_cv con umbrales V03 congelados (comparabilidad directa).
- FALSIFICATION CRITERION: el control torch BCE iguala o supera al
  estimador en pooled y pliegues (el mecanismo no aporta).
- STOP CONDITION: una configuracion (pasos=400, Adam lr=0.05, wd=0, pi
  unico); sin grid. Si TR no rescata, no se extiende a AG.
- Coste: CPU-min (modelo lineal denso-sparse en torch).

Presupuesto restante: 3 principales planeadas; margen para 3 experimentos
principales mas solo si un falsificador pide una comprobacion distinta y
pre-registrada; los controles (global-nested en A, A0 en B, torch-BCE en C)
no consumen presupuesto y se registran como unidades.

## 6. Entregables de la fase

`p97_independent_research.md` (este informe), `p97_method_registry.csv`,
`p97_novelty_matrix.csv`, resultados por experimento en `reports/p97_*.md/json`,
OOF en `features/p97_*.npz`, figuras en `reports/figuras/p97_*.png`, filas
nuevas en `experiments.csv`, seccion 157 del notebook y verificacion final
con `tools/verificar_caracteres.py`.

## 7. Resultados de la ejecucion (addendum, escrito DESPUES de ejecutar)

| Hipotesis | Experimento | Veredicto pre-registrado | Cifras clave |
|---|---|---|---|
| H97-A decision condicional | P97-01 | FALSIFICADA | segmento 0.7702914 vs global-nested 0.7781368 (delta -0.007845, 1/5); el control global-nested reproduce el +0.000925 historico (P55/V56) |
| H97-B DRO minimax | P97-02 (+2 invalidas) | FALSIFICADA | DRO 0.7720416 vs A0 0.7772115 (delta -0.005170, 3/5); p90/p99 de error por grupo iguales; mas dispersion |
| H97-C nnPU/PNU TR | P97-03 (+2 invalidas) | SENAL (MARGINAL por gate) | nnpu 0.7770004 vs control 0.7748502 (delta +0.002150, 5/5 pliegues; TR +0.006450, 5/5); IC95 clusters [-0.000461, +0.004896]; vs V03 en crudo -0.0002 |
| H97-C replicacion | P97-04 | FALSIFICADA | delta -0.001738, 1/5 pliegues en particion agrupada nueva (seed 43) |

Clasificacion final de los mecanismos ejecutados:

- **H97-A (decision condicional por segmento): INVALIDATED.** Fragmentar la
  seleccion de umbrales sobreajusta el lado de decision; el propio control
  nested confirma que la re-optimizacion global conocida (+0.000925) es el
  limite practico de este eje.
- **H97-B (Group DRO): INVALIDATED.** El minimax no compro robustez medible
  (cola de error por grupo identica) y costo media y dispersion. Nota de
  implementacion: la primera instanciacion (pesos sin normalizar frente a la
  escala de C de liblinear) colapso por defecto de escala y quedo preservada;
  la ejecucion valida (pesos a media 1) es la que sostiene el veredicto.
- **H97-C (nnPU/PNU): MARGINAL.** En el arnes identico y con el mecanismo
  validamente instanciado (dos instanciaciones invalidas previas, ambas
  preservadas: escala L2 de Adam y codificacion P/N/U) mejora al control en
  5/5 pliegues (+0.002150 pooled, TR +0.006450) pero queda por debajo del gate
  (+0.003), su IC de clusters incluye el cero (93.7 por ciento de remuestreos
  positivos) y NO se replica en la segunda particion agrupada (-0.001738,
  1/5). No es una via hacia 0.80 con la evidencia actual; queda documentado
  como la primera formulacion realmente no supervisada de la semantica TR/AG
  que se ejecuta, con senal direccional en una particion y sin robustez.

Disciplina de ejecucion: 3 hipotesis principales, 4 experimentos principales
(P97-01..04), 3 controles registrados, 2 instanciaciones invalidas preservadas
y documentadas, 0 grids, 0 submissions, 0 usos de test/holdout/leaderboard.
Las correcciones de implementacion se detectaron con la disciplina de P96
(los controles del arnes deben reproducir V03; cuando no lo hicieron, se
aislo el defecto antes de leer el resultado del mecanismo).

Que cambio de evidencia ampliaria la investigacion (sin inventar vias):
autorizacion de una particion externa nueva, una fuente de etiquetas
independiente (P95-N1/N3) o un cambio de reglas que permita otra familia de
representacion; nada de eso esta disponible hoy.
