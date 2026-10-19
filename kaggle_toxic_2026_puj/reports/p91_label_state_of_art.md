# P91 - Estado del arte metodologico (verificado)

Fecha de acceso: 2026-10-10. Busqueda dirigida a UNA decision: como auditar la calidad de
etiquetas multietiqueta con procedimiento ciego y reproducible. No es una busqueda de
arquitecturas. Cada referencia se verifico contra su fuente primaria (ACL Anthology,
JAIR/MIT Press, arXiv API); no se cita ningun trabajo no verificado.

## 1. Confident learning (estimacion de ruido con probabilidades fuera de muestra)

- Northcutt, C. G.; Jiang, L.; Chuang, I. L. (2021). "Confident Learning: Estimating
  Uncertainty in Dataset Labels". Journal of Artificial Intelligence Research (JAIR),
  70:1373-1411. DOI 10.1613/jair.1.12125 (verificado). Revisado por pares.
- Metodo: estima la distribucion conjunta etiqueta dada x etiqueta verdadera (matriz
  confident joint) usando solo probabilidades fuera de muestra y las etiquetas dadas;
  identifica y ordena posibles errores.
- Supuestos: proceso de ruido condicional a la clase; requiere probabilidades OOF bien
  calibradas en el sentido de rango; no exige etiquetas limpias de referencia.
- Multietiqueta: la adaptacion natural es por par instancia-etiqueta (extension por
  etiqueta binaria); la documentacion de cleanlab (herramienta, no paper) ofrece
  `multilabel_classification` con esa lectura y advierte que NO modela etiquetas
  verdaderas omitidas (falsos negativos de anotacion). URL:
  https://docs.cleanlab.ai (acceso 2026-10-10; documentacion de herramienta).
- Limites relevantes aqui: la confianza del modelo no es certeza humana; con 4 modelos
  muy correlacionados el desacuerdo no es independiente.
- Cambio de decision que aporta: fija el criterio de ordenacion de candidatos
  (self-confidence por celda) y su limite. **Precedente interno: P59-C ya implemento
  confident learning desde cero (sin cleanlab) y el resultado fue MARGINAL** (el estimador
  de ruido no supero al control de dificultad: AUC 0.973019 vs 0.974143). No se instala ni
  ejecuta cleanlab.

## 2. Errores de etiqueta a escala (motivacion, no prueba local)

- Northcutt, C. G.; Athalye, A.; Mueller, J. (2021). "Pervasive Label Errors in Test Sets
  Destabilize Machine Learning Benchmarks". NeurIPS 2021 Track on Datasets and Benchmarks
  (arXiv 2103.14749; estado verificado). Revisado por pares.
- Aporte: los errores de etiqueta son frecuentes en datasets populares y afectan rankings.
- Uso en TOXIC: justifica auditar, NO demuestra que este dataset tenga el mismo problema.

## 3. Deteccion de errores de anotacion en NLP (metodos y evaluacion)

- Klie, J.-C.; Webber, B.; Gurevych, I. (2023). "Annotation Error Detection: Analyzing the
  Past and Present for a More Coherent Future". Computational Linguistics, 49(1):157-198.
  DOI 10.1162/coli_a_00464 (verificado en ACL Anthology, 2023.cl-1.4).
- Aporte: reimplementa 18 metodos, evaluacion uniforme y mejores practicas; ninguno
  domina; recomienda protocolos ciegos y analisis por pares.
- Cambio de decision: fija que la cola use MULTIPLES criterios (no un solo detector) y que
  la evaluacion de la revision se haga por pares con tasas, no con una metrica unica.

## 4. Consenso con multiples anotadores

- Goh, H. W.; Tkachenko, U.; Mueller, J. (2022). "CROWDLAB: Supervised learning to infer
  consensus labels and quality scores for data with multiple annotators". NeurIPS 2022
  Human-in-the-Loop Learning Workshop (arXiv 2210.06812; estado verificado: workshop).
- Aporte: combina predicciones del modelo y anotaciones para consenso y calidad por
  ejemplo; util DESPUES de la revision, cuando existan 2+ anotaciones.
- Cambio de decision: la agregacion de la revision no sera mayoria simple a ciegas; se
  reporta el acuerdo completo y el consenso ponderado como analisis secundario.

## 5. Sesgo de anotacion en discurso de odio (diseno ciego)

- Sap, M.; Card, D.; Gabriel, S.; Choi, Y.; Smith, N. A. (2019). "The Risk of Racial Bias
  in Hate Speech Detection". ACL 2019, pp. 1668-1678 (P19-1163, verificado). Revisado por
  pares.
- Aporte: el contexto del anotador (priming de dialecto/raza) cambia las etiquetas de
  toxicidad; los desacuerdos reflejan a veces sesgo del anotador, no error del texto.
- Cambio de decision: cegamiento estricto, sin priming, y prohibicion de discutir casos
  antes de cerrar la primera ronda; el desacuerdo se interpreta con cautela.

## 6. Sintesis y decision

La literatura no cambia la metodologia que este proyecto recomendaria: la revision debe
ser ciega, con 2+ anotadores, adjudicacion separada, acuerdo por etiqueta con metricas
robustas a prevalencia, y deteccion de candidatos multicriterio con probabilidades OOF.
Los aportes concretos son: (i) la formalizacion de la tarea y mejores practicas de Klie
et al. 2023; (ii) el limite de confident learning (no modela etiquetas omitidas) y su
precedente interno MARGINAL (P59-C); (iii) el diseno ciego y sin priming (Sap et al.
2019); (iv) CROWDLAB solo aplica DESPUES de obtener multiples anotaciones. No se busca mas
literatura: ninguna decision pendiente depende de ello.

## 7. Nota de consistencia P91-R (aditiva, 2026-10-09)

Correccion documental aditiva; el texto anterior se conserva integro (registro original:
sha256 776f00a7a2a3a9a69bc2a4b9cf3cbc7e5e0a825246befcff1cac2238c235992c). La fecha de
acceso "2026-10-10" registrada en la cabecera y en la referencia de cleanlab **no pudo
verificarse** contra un registro independiente: la fecha del proyecto es 2026-10-09 y la
consulta se realizo durante la sesion de trabajo de P91 (evidencia: los resultados de
busqueda de esa sesion). Se registra la consulta como **fecha no verificada**: la
documentacion de cleanlab fue consultada durante la fase P91, pero la fecha exacta no es
confirmable; no se realizo ninguna consulta nueva en P91-R ni se repite la busqueda.
