# P94 - Revision actualizada del estado del arte (SOTA) bajo las restricciones de la competencia

Fecha: 2026-10-13. Fase de investigacion y diseno: sin entrenamiento, sin submissions,
sin test, sin holdout, sin holdout2, sin leaderboard, sin rankings de Kaggle.

Objetivo: identificar mecanismos de modelado GENUINAMENTE nuevos (en el sentido del
protocolo P2: mecanismo distinto, no otra implementacion del mismo mecanismo) que
puedan mejorar el leaderboard frente a V03 (corrected_cv 0.7772114525165721, LB 0.78306),
en el espacio permitido por las reglas de la competencia y del proyecto.

Espacio permitido (filtro aplicado a toda referencia):
texto corto en espanol; clasificacion multietiqueta HS/TR/AG; dataset supervisado
limitado (1.731 filas de desarrollo); representacion word + char; sin transformers,
BERT, RoBERTa, LLM ni modelos preentrenados; sin datasets externos etiquetados, sin
clasificadores externos, sin embeddings de terceros, sin pseudo-labeling, sin corpus
externos nuevos; sin rankings ni scores de test para seleccionar tecnicas.

Preguntas de decision (Q1-Q5) que la revision debe responder:
- Q1: existe un objetivo/optimizacion genuinamente nuevo para macro-F1 o desbalanceo
  que no equivalga a las perdidas ya medidas?
- Q2: existe una representacion sparse que preserve negacion/contexto de forma no
  equivalente a las features historicas?
- Q3: existe un mecanismo de estructura de salida multietiqueta no equivalente a
  cadenas, meta-dependencia, especializacion o inferencia conjunta ya cerrada?
- Q4: existe reduccion de varianza o transferencia entre etiquetas que no replique
  ensembles, bagging, stacking ni promediado ya cerrado?
- Q5: queda algun clasificador discriminativo sobre representacion sparse sin medir
  con evidencia a favor?

## Fichas de referencias (fuentes primarias verificables)

Nombres transliterados a ASCII por la politica del proyecto. Cada referencia debe
justificar una decision concreta; las que no cambian el conjunto de hipotesis viables
se declaran y no generan linea nueva.

### R01 - Joachims, T. (1998). Text Categorization with Support Vector Machines: Learning with Many Relevant Features. ECML 1998.
- DOI: 10.1007/BFb0026683. Revisado por pares (conferencia).
- Metodo: SVM lineal sobre representaciones sparse de alta dimension; pliegues de
  regularizacion para espacios con mas rasgos que ejemplos.
- Supuestos: la frontera lineal sobre n-gramas es competitiva cuando d >> n.
- Limitaciones: no aborda multi-etiqueta ni metrica macro directa.
- Relevancia TOXIC: es la base teorica de V03 (ya citada en el informe final).
- Solapamiento historico: F01 (familia campeona).
- Decision: ALREADY_TESTED. No cambia hipotesis; consolida la eleccion del campeon.

### R02 - Wang, S.; Manning, C. D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. ACL 2012 (short).
- URL: https://aclanthology.org/P12-2016. Revisado por pares (ACL).
- Metodo: NB-SVM (interpolacion del log-count ratio de Naive Bayes con pesos SVM).
- Supuestos: el conteo de bigramas con NB-SVM compite con modelos mas complejos.
- Limitaciones: binario, no multi-etiqueta; sin mecanismo de umbrales.
- Relevancia TOXIC: clasico fuerte de texto corto sobre sparse.
- Solapamiento historico: E2-15 (NB-WORD -0.0234; NBSVM -0.0160; NBSVM-SVC -0.0343).
- Decision: ALREADY_TESTED (2/5 pliegues, sin gate). Cierra Q5 para la variante principal.

### R03 - Lan, M.; Tan, C. L.; Su, J.; Lu, Y. (2009). Supervised and Traditional Term Weighting Methods for Automatic Text Categorization. IEEE TPAMI 31(4):721-735.
- DOI: 10.1109/TPAMI.2008.110. Revisado por pares (revista).
- Metodo: comparativa de esquemas de ponderacion de terminos supervisados (chi2, IG)
  frente a tf-idf.
- Supuestos: la ponderacion supervisada mejora categorizacion clasica en algunos
  corpus.
- Limitaciones: categorizacion de temas, no odio; sin metrica macro multi-etiqueta.
- Relevancia TOXIC: cubre la familia "ponderacion alternativa" sobre sparse.
- Solapamiento historico: v4-C (log-odds/chi2/MI/BM25/binario, degradan);
  V37-01 (BM25 cerrado; BM25-char declarado no ejecutado).
- Decision: ALREADY_TESTED. Cierra Q5 para ponderaciones supervisadas.

### R04 - Chen, K.; Zhang, Z.; Long, J.; Zhang, H. (2016). Turning from TF-IDF to TF-IGM for term weighting in text classification. Expert Systems with Applications 66:245-260.
- DOI: 10.1016/j.eswa.2016.09.009. Revisado por pares (revista).
- Metodo: TF-IGM (information gain moment) como ponderacion alternativa a tf-idf.
- Supuestos: mejora sobre tf-idf en corpus desbalanceados.
- Limitaciones: mismo mecanismo de ponderacion supervisada de R03.
- Relevancia TOXIC: parte de la misma familia de ponderacion.
- Solapamiento historico: v4-C; V37-01. La ponderacion supervisada degrada en TOXIC.
- Decision: DUPLICATE de R03 (mismo mecanismo, otra formula). No abre linea.

### R05 - Lipton, Z. C.; Elkan, C.; Naryanaswamy, B. (2014). Thresholding Classifiers to Maximize F1 Score.
- URL: https://arxiv.org/abs/1402.1892. Preprint (arXiv; sin revision de pares verificada).
- Metodo: regla de umbral optima para maximizar F1 bajo probabilidades calibradas;
  extension a macro-F1 multi-etiqueta.
- Supuestos: probabilidades calibradas; F1 depende de la matriz de confusion completa.
- Limitaciones: lado de decision, no de entrenamiento; sensible a la calibracion.
- Relevancia TOXIC: es el fundamento de la capa de umbrales adoptada (TH-01, parte de
  V03) y de las familias de re-optimizacion (ya cerradas).
- Solapamiento historico: TH-01; S03; V7-P0 (D-055: umbrales debilmente
  identificados); P55 (D-126/D-127); P76 (D-259); la EDA de probabilidades demostro
  que los errores de V03 NO son de decision (D-142).
- Decision: ALREADY_TESTED. Cierra Q1 para el lado de decision.

### R06 - Dembczynski, K.; Jachnik, A.; Kotlowski, W.; Waegeman, W.; Hullermeier, E. (2013). Optimizing the F-measure in multi-label classification: Plug-in rule approach versus structured loss minimization. ICML 2013.
- URL: https://proceedings.mlr.press/v28/dembczynski13.html. Revisado por pares (ICML).
- Metodo: compara la regla plug-in (estimacion de parametros y decision) con
  minimizacion de perdida estructurada para F-measure multi-etiqueta.
- Supuestos: distribucion subyacente estimable; perdida estructurada aproximable.
- Limitaciones: teoricas y sinteticas; en la practica converge a umbrales por etiqueta
  para F-measure binaria.
- Relevancia TOXIC: cubre ambos lados (decision y estructura) de la F multi-etiqueta.
- Solapamiento historico: lado decision = TH-01/V7-P0/D-142; lado estructura =
  E2-01/V45-01/D-116 (no explotable) con techo oraculo +0.046271.
- Decision: ALREADY_TESTED en sus dos lados. Cierra Q1 (estructura) y Q3.

### R07 - Parambath, S. P.; Usunier, N.; Grandvalet, Y. (2014). Optimizing F-measures by cost-sensitive classification. NeurIPS 2014.
- URL: https://papers.nips.cc/paper/2014 (NIPS 27, pp. 2757-2765). Revisado por pares.
- Metodo: reduce la optimizacion de F-measure a clasificacion sensible a costes con
  una eleccion de costes derivada.
- Supuestos: acceso a estimaciones de la matriz de confusion por instancia.
- Limitaciones: el mecanismo subyacente es re-ponderacion por costes.
- Relevancia TOXIC: la re-ponderacion por costes es la familia mas transitada del
  proyecto.
- Solapamiento historico: class_weight (V03); V4-B (downweighting); V30 (ruido);
  P84-03 (ROS cerrado por equivalencia con reweighting); D-191..D-217
  (A_cost_sensitive_downweight: PROMISING_UNCONFIRMED, LB 0.77522, nunca adoptado).
- Decision: ALREADY_TESTED. Cierra Q1 (lado costes) y Q4.

### R08 - Eban, E.; Schain, M.; Mackey, A.; Gordon, A.; Rifkin, R.; Elidan, G. (2017). Scalable Learning of Non-Decomposable Objectives. AISTATS 2017.
- URL: https://proceedings.mlr.press/v54/eban17a.html. Revisado por pares (AISTATS).
- Metodo: formulacion con restricciones (TFCO) para optimizar metricas no
  descomponibles (F-measure, cobertura) durante el entrenamiento.
- Supuestos: modelos con salida continua entrenada por gradiente (suelen asumir redes);
  hiperparametros de restriccion sintonizables.
- Limitaciones: la implementacion de referencia es para frameworks neuronales; para
  un modelo lineal disperso el mecanismo se reduce a una perdida sustituta de la
  metrica mas una restriccion (hiperparametro nuevo).
- Relevancia TOXIC: es la version "moderna" del entrenamiento orientado a metrica.
- Solapamiento historico: V7-01 probo la F1-suave (el sustituto literal del objetivo:
  -0.0033, la PEOR de las dos perdidas orientadas, con control BCE a -0.000046 de
  V03); P60-02 (perdida asimetrica cerrada); V12-H2 (perdida propia MARGINAL +0.0016,
  derivada verificada, D-092/D-096); la leccion medida es "aproximar mejor el objetivo
  no es mejor" (D-053); ademas hay freno de transferencia: mejoras locales <= +0.02 no
  se trasladaron al LB (R1-C1, R2-MINDF5, OPT-01).
- Decision: ALREADY_TESTED. La formulacion con restricciones cambia el punto del
  espacio de soluciones (otro hiperparametro de la misma familia), no el mecanismo;
  no supera el criterio de novedad P2. Cierra Q1.

### R09 - Lee, N.; Yang, H.; Yoo, H. (2021). A surrogate loss function for optimization of F-beta score in binary classification with imbalanced data.
- URL: https://arxiv.org/abs/2104.01459. DOI: 10.48550/arXiv.2104.01459. Preprint
  (arXiv; sin revision de pares verificada).
- Metodo: perdida sustituta diferenciable para F-beta; los caminos de gradiente
  aproximan los de F-beta en el limite de muestra grande.
- Supuestos: entrenamiento por gradiente (redes en sus experimentos).
- Limitaciones: presentada y validada con ResNets sobre imagenes; sin multi-etiqueta.
- Relevancia TOXIC: misma familia exacta que V7-01 (sustituto literal del objetivo).
- Solapamiento historico: V7-01 (F1-suave -0.0033 vs control, 3/5 pliegues).
- Decision: ALREADY_TESTED. Cierra Q1 (beta es un parametro, no un mecanismo nuevo).

### R10 - Berger, A. (2020). Threshold optimization for F measure of macro-averaged performance. Pattern Recognition 102:107175.
- DOI: 10.1016/j.patcog.2020.107175 (S003132032030056X). Revisado por pares (revista).
- Metodo: optimizacion de umbrales para F macro-promediada en multi-clase.
- Supuestos: probabilidades del clasificador disponibles; se optimiza solo la decision.
- Limitaciones: lado de decision; sensible al estimador de probabilidades.
- Relevancia TOXIC: refuerza el cierre de la capa de decision.
- Solapamiento historico: TH-01 adoptado; re-optimizaciones cerradas (F05);
  el lado de decision tiene techo medido (D-142).
- Decision: ALREADY_TESTED. Cierra Q1 (decision).

### R11 - Opitz, J.; Burst, S. (2019). Macro F1 and Macro F1.
- URL: https://arxiv.org/abs/1911.03347. Preprint (arXiv).
- Metodo: analisis de la ambiguedad de la definicion de macro-F1 (media de F1 por
  etiqueta frente a F1 de macro-P y macro-R) y su sensibilidad a la prevalencia.
- Supuestos: ninguno de modelado; es un analisis de metricas.
- Limitaciones: no propone mecanismo de modelado.
- Relevancia TOXIC: la metrica oficial de la competencia es la media column-wise de
  la F1 binaria (identificada empiricamente en E00); esta referencia explica por que
  variantes de definicion cambian el ranking.
- Solapamiento historico: E00/D-009/D-115 (definicion congelada por escrito).
- Decision: INFORMATIONAL. No genera hipotesis de modelado; confirma la definicion
  ya congelada del proyecto.

### R12 - Abderrouaf, C.; Oussalah, M. (2019). On Online Hate Speech Detection. Effects of Negated Data Construction. IEEE International Conference on Big Data (Big Data) 2019, pp. 5595-5602.
- URL: https://ieeexplore.ieee.org (DOI de conferencia IEEE). Revisado por pares.
- Metodo: ingenieria de features y esquema de transferencia para odio online; estudia
  los efectos de la construccion de datos negados.
- Supuestos: la negacion cambia la etiqueta ("no es X" vs "es X") y debe representarse.
- Limitaciones: no multi-etiqueta; corpus propio en ingles.
- Relevancia TOXIC: es la referencia natural de la familia "negacion" que P94 nombra.
- Solapamiento historico: E2-08-NEG y E2-08-NEG-TXT (patrones NEG_* de spaCy
  simbolico: 0.5267 sola; 0.7657 con TXT: -0.0115 vs V03, 0/5 pliegues); E2-07
  (cita); la linea sintaxis/cita/negacion quedo cerrada sin mejora.
- Decision: ALREADY_TESTED. Una formulacion alternativa (marcado de alcance por
  token) no cambia el mecanismo (features de evidencia negada sobre LR sparse).
  Cierra Q2.

### R13 - Khanday, A. M. U. D.; Rabani, S. T.; Khan, Q. R.; Rouf, N.; Mohiuddin, M. (2022). Detecting twitter hate speech in COVID-19 era using machine learning and ensemble learning techniques.
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC9481233. Revisado por pares (revista).
- Metodo: TF/IDF y Bag of Words con clasicos (LR, SVM, NB) y ensembles sobre odio en
  Twitter.
- Supuestos: representacion sparse suficiente para el dominio.
- Limitaciones: binario; el ensemble no se valida contra un protocolo agrupado.
- Relevancia TOXIC: confirma que el estandar clasico es exactamente V03 (TF-IDF +
  lineales) y que los ensembles son la extension tipica.
- Solapamiento historico: F03 (ensembles cerrados con medicion propia y prohibicion
  directa en P94 de replicarlos).
- Decision: ALREADY_TESTED para mecanica; context only. Cierra Q4.

### R14 - Bel-Enguix, G.; Gomez-Adorno, H.; Sierra, G.; Vasquez, J.; Andersen, S.-T.; Ojeda-Trueba, S. (2023). Overview of HOMO-MEX at IberLEF 2023: Hate speech detection in Online Messages directed towards the MEXican spanish speaking LGBTQ+ population. Procesamiento del Lenguaje Natural 71.
- URL: https://ceur-ws.org/Vol-3496/homomex-paper2.pdf (paper de sistema) + PLN 71 (overview). Revisado por pares (revista PLN / taller IberLEF).
- Metodo (panorama): sistemas clasicos y neuronales; los clasicos usan TF-IDF y SVM.
- Limitaciones: dominios y etiquetas distintos a TOXIC.
- Relevancia TOXIC: confirma el papel de los clasicos en espanol y el salto que dan
  los transformers (prohibidos aqui).
- Solapamiento historico: F01; F12 (DL desde cero cerrado).
- Decision: INFORMATIONAL. No abre hipotesis viable bajo las restricciones.

### R15 - Wiegand, M.; Ruppenhofer, J.; Kleinbauer, T. (2019). Detection of Abusive Language: the Problem of Biased Datasets. NAACL 2019.
- URL: https://aclanthology.org/N19-1202. Revisado por pares (NAACL).
- Metodo: analisis de sesgos de anotacion/dominio y su efecto en el rendimiento.
- Limitaciones: no propone modelo.
- Relevancia TOXIC: explica el techo de anotacion (frontera no observable, 40.49%
  de la region de insulto interpersonal etiquetada (0,0,0)) y el limite de datos.
- Solapamiento historico: FINAL_REPORT secciones 20 y 23; P90 casebook de errores.
- Decision: INFORMATIONAL. Sustenta el techo medido; no abre linea.

### R16 - Wei, J.; Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. EMNLP-IJCNLP 2019.
- URL: https://aclanthology.org/D19-1670. Revisado por pares (EMNLP).
- Metodo: sustitucion/insercion/borrado/intercambio de palabras como aumento.
- Supuestos: corpus de entrenamiento pequeno; requiere sinonimos externos para SR.
- Limitaciones: ganancias modestas y dependientes del dominio; las operaciones que
  preservan etiqueta ya estaban representadas aqui (espaciado, elongacion,
  puntuacion) y empeoraron.
- Relevancia TOXIC: la familia augmentation.
- Solapamiento historico: V6-08 (aumento peor); V47-01 (TRAIN_AUGMENTATION =
  CLOSED); D-118 (no existe expansion legal).
- Decision: ALREADY_TESTED. Cierra Q4 (varianza por datos).

### R17 - Bogatinovski, J.; Todorovski, L.; Dzeroski, S.; Kocev, D. (2022). Comprehensive comparative study of multi-label classification methods. Expert Systems with Applications 203:117215.
- DOI: 10.1016/j.eswa.2022.117215. Revisado por pares (revista).
- Metodo: comparativa a gran escala de metodos multi-etiqueta (binary relevance,
  chains, label powerset/RAkEL, adaptaciones y ensembles).
- Supuestos: los metodos de estructura explotan correlaciones de etiquetas.
- Limitaciones: sin estudio especifico de F1 macro con umbrales por etiqueta en
  corpus pequenos de odio.
- Relevancia TOXIC: cubre la estructura de salida (Q3).
- Solapamiento historico: E2-01 (cadenas, +0.002 = ruido); V45-01 (estructura
  conjunta CLOSED); V12-02 (par MARGINAL); el techo con oraculo es +0.046271 y no
  esta en la estimacion (D-103); especialistas = reentrenado bit a bit (D-165).
- Decision: ALREADY_TESTED. Cierra Q3 (LP/RAkEL son ensembles de powersets:
  ensemble cerrado + estructura cerrada).

## Sintesis Q1-Q5

- Q1 (objetivo macro-F1/desbalanceo): NO. Las tres rutas formales (perdida sustituta
  R08/R09, regla plug-in R05/R06, costes R07) estan medidas o equivalen a mecanismos
  medidos y cerrados (V7-01, V12-H2, P60, TH-01, F14). Ademas, la metrica local no
  transfiere por si misma (freno de transferencia) y los errores de V03 no son de
  decision (D-142).
- Q2 (negacion/contexto en sparse): NO. R12 mapea a E2-08/E2-07, cerrados por
  medicion; las reformulaciones son el mismo mecanismo (P2 prohibe tratarlas como
  nuevas).
- Q3 (estructura de salida): NO. R06/R17 mapean a cadenas/estructura/inferencia
  conjunta, cerradas con techo medido.
- Q4 (varianza/transferencia): NO. R13/R16 mapean a ensembles/augmentation,
  cerrados; P94 excluye replicar esas familias; el candidato cost-sensitive
  existente no es nuevo (F14) y su sonda externa fallo.
- Q5 (lineal alternativo sin medir): NO con evidencia a favor. Todos los clasicos
  razonables ya se midieron (F07/F08/F09/F21) con resultado <= V03. La unica celda
  sin medicion directa (instance-based/kNN) tiene evidencia adyacente en contra
  (memorizacion del RBF, regularizacion por vecindad marginal, prototipos cerrados)
  y ninguna literatura que la situe por encima de un lineal regularizado ajustado
  en texto corto sparse.

Conclusion de la busqueda: NINGUNA referencia de las auditadas cambia el conjunto de
hipotesis viables. La busqueda bibliografica se declara CERRADA conforme a la regla
de P1 ("si no cambia el conjunto de hipotesis viables, cierra la busqueda").

## Politica de citacion

- Solo se citan fuentes primarias con URL/DOI verificables; los preprints van
  marcados como tales (R05, R09, R11).
- Prohibido citar rankings de Kaggle o scores de test como criterio de seleccion.
- Los scores de leaderboard mencionados (0.78306 etc.) son del propio proyecto y
  provienen de su registro interno de sondas; no se usaron para seleccionar tecnicas.
