# V25: revision de literatura con verificacion en linea de nueve referencias

Proyecto `kaggle_toxic_2026_puj`, rama `v25`.
Fichero de literatura. NO es un informe de experimento: aqui no se entrena nada,
no se ejecuta nada y no hay ninguna medicion propia.
Todas las cifras que aparecen son o bien de la fuente bibliografica citada, o
bien de informes ya existentes del repositorio, con el fichero citado al lado.

Formato: ASCII puro estricto. La comprobacion es
`sum(1 for c in open(ruta,'rb').read() if c > 127) == 0`.

Reglas del concurso que gobiernan cada seccion de compatibilidad:
prohibidos transformers, prohibidos LLM, prohibidos embeddings preentrenados,
prohibidos datasets externos etiquetados. Se permite entrenar desde cero
sobre texto no etiquetado y usar las 1.483 filas de desarrollo.

---

## 1. Steffen Rendle (2010). "Factorization Machines." ICDM 2010

### 1.1 Verificacion

- `https://api.crossref.org/works/10.1109/ICDM.2010.127` devuelve **200**.
  `title` = "Factorization Machines". `author` = ["Steffen Rendle"] (autor unico).
  `container-title` = "2010 IEEE International Conference on Data Mining".
  `published` = 2010-12. `page` = "995-1000". `publisher` = IEEE.
  `type` = proceedings-article. **Coincide con lo declarado.**
- Corroboracion independiente: Semantic Scholar para el mismo DOI devuelve
  `externalIds.DBLP = conf/icdm/Rendle10`, `year = 2010`,
  `venue = 2010 IEEE International Conference on Data Mining`, y el resumen
  integral. El DOI tambien resuelve en `https://doi.org/10.1109/ICDM.2010.127`.
- **FALLO EN UN CAMPO, y es importante.** El identificador alternativo que
  ofrecio el encargo, arXiv `2010.10593`, **no es este trabajo**. La API de arXiv
  (`https://export.arxiv.org/api/query?id_list=2010.10593`, estado 200) resuelve a:
  - `title` = "Cross-Modal Information Maximization for Medical Imaging: CMIM"
  - `author` = Tristan Sylvain, Francis Dutil, Tess Berthier, Lisa Di Jorio,
    Margaux Luck, Devon Hjelm, Yoshua Bengio
  - `published` = 2020-10-20, `comment` = "ICASSP 2021",
    `journal_ref` = ICASSP 2021, pp. 1190-1194
  Es decir, el arXiv `2010.10593` es un trabajo de imagen medica de 2021, no
  existe ninguna relacion con Factorization Machines.
- No hay preprint de repuesto. La busqueda de arXiv
  `ti:"Factorization Machines" AND au:Rendle` devuelve **cero entradas**. Por
  tanto **no existe ninguna version en arXiv de este trabajo**, y el fallback
  propuesto hay que descartarlo, no solo corregirlo.

### 1.2 Referencia bibliografica

Rendle, S. (2010). Factorization Machines. En *2010 IEEE International Conference
on Data Mining (ICDM 2010)*, pp. 995-1000. IEEE.

### 1.3 Enlace

`https://doi.org/10.1109/ICDM.2010.127`

### 1.4 Resumen de la idea

Introduce las Factorization Machines como una clase de modelo que combina las
ventajas de una SVM (un predictor general que trabaja con cualquier vector de
rasgos de valor real) con las de un modelo de factorizacion. Frente a la SVM, la
FM modela **todas** las interacciones entre variables usando parametros
factorizados en vez de un peso explicito por par, y por eso puede estimarlas
incluso con una esparsidad muy alta, donde la SVM falla. La ecuacion del modelo
se puede calcular en tiempo lineal, de modo que la FM se optimiza directamente
sin necesidad de una transformacion a la forma dual.

### 1.5 Mecanismo relevante

La forma es `y = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j`, con un
vector latente `v_i` en R^k por rasgo activo. La identidad que hace el coste
lineal es `(sum_i v_i x_i)^2 = sum_{i<j} <v_i, v_j> x_i x_j + sum_i ||v_i||^2 x_i^2`:
el sumatorio sobre pares se recupera de un cuadrado y de una resta. Es
exactamente la propiedad por la que `docs/v25_mechanism_audit.md` (seccion 4,
punto 1) obliga a verificar la implementacion numericamente contra una version
explicita de pares antes de fiarse del resultado grande. Numero de parametros:
`n*k + n + 1`.

### 1.6 Que parte es compatible con TOXIC

La FM entera. Es un modelo lineal en los coeficientes, entrenado desde cero,
que solo consume la matriz dispersa de 27.431 rasgos y las 1.483 etiquetas de
desarrollo. No usa transformers, ni LLM, ni embeddings preentrenados, ni
ningun corpus externo etiquetado. Se implementa en numpy sobre la matriz CSR
sin ninguna dependencia nueva.

### 1.7 Que parte esta prohibida o no es aplicable

- Las implementaciones de biblioteca (libFM, xLearn, fastFM) no estan
  instaladas en este entorno. No es una prohibicion del concurso, es una
  dependencia; la algebra se puede escribir a mano sobre la matriz dispersa.
- La nocion de "campo" (`F = M` con Metadata) que el trabajo usa para sistemas
  de recomendacion **no tiene equivalente natural** en TF-IDF de n-gramas: habria
  que inventar una particion en campos (por ejemplo bloque word(1,2) frente a
  bloque char_wb(3,5)) que el trabajo no define. La FM de campo es opcional y
  no aporta a este corpus.

### 1.8 Relacion con V03

V03 es exactamente el caso `k = 0` de la FM: el termino lineal con 27.431
coeficientes y ningun termino de interaccion. Los numeros de V03 que estan
congelados son `LogisticRegression C=2.0, class_weight=balanced,
solver=liblinear, max_iter=1000, random_state=42`, umbrales congelados
HS 0.44 / TR 0.51 / AG 0.46, `corrected_cv` canonico 0.7772114525165721 y
LB 0.78306 (`CHAMPION.md` seccion 1). El margen que anade V25 lo dice
`docs/v25_mechanism_audit.md` seccion 4 punto 2: una FM con `k = 4` son
27.431 coeficientes lineales mas 27.431 x 4 factores latentes, del orden de
137.000 parametros frente a los 27.431 de V03, un factor de 5 (no de 70 como
los vectores de soporte del kernel). Y el punto de partida es que V03 ya tiene
capacidad de sobra: F1 en train 0,9614 y gap train-OOF de +0,1842, con C=1 en
0,77227 y C=1,5 en 0,77296, o sea que el optimo de capacidad esta en el interior
de la curva (`reports/v24_01_transferencia_complejidad.md`, citado en
`reports/v24_02_dictamen_gate.md` seccion 2.3).

### 1.9 Relacion con V25

Es **el mecanismo que V25 propone**. Se implementa. Lo que queda por hacer fuera
de este fichero: (a) verificar la identidad de tiempo lineal contra una
implementacion explicita de pares en una matriz sintetica pequena antes de
aceptar el resultado grande, y (b) reportar la complejidad efectiva junto al CV,
no solo el CV. La unica correccion que sale de esta revision es bibliografica:
el identificador de arXiv que traia el encargo hay que borrarlo, porque senala
a otro trabajo.

---

## 2. Shu Wu, Zekun Li, Yunyue Su, Zeyu Cui, Xiaoyu Zhang, Liang Wang et al. (2025). "GraphFM: Graph Factorization Machines for Feature Interaction Modelling." Machine Intelligence Research 22, 239-253

### 2.1 Verificacion

- `https://api.crossref.org/works/10.1007/s11633-024-1505-5` devuelve **200**.
  `title` = "GraphFM: Graph Factorization Machines for Feature Interaction Modelling".
  `author` = ["Shu Wu", "Zekun Li", "Yunyue Su", "Zeyu Cui", "Xiaoyu Zhang",
  "Liang Wang"].
  `container-title` = "Machine Intelligence Research". `volume` = 22,
  `issue` = 2, `page` = "239-253", `published` = 2025-01-07, `publisher` = Springer.
  **Coincide con lo declarado en todo menos en un detalle menor**, ver abajo.
- Preprint en arXiv: `https://export.arxiv.org/api/query?id_list=2105.11866`
  devuelve 200, `title` = "GraphFM: Graph Factorization Machines for Feature
  Interaction Modeling" (una sola "l" en Modeling, frente a las dos de
  "Modelling" en la version de revista), mismos 6 autores, `published` =
  2021-05-25. Es decir el trabajo es de 2021 en preprint y de 2025 en revista;
  citar 2025 con el DOI de revista es correcto, pero conviene saber que el
  trabajo tiene cuatro anos.
- **Detalle menor a registrar.** El encargo escribe "Liang Wang et al." El
  registro tiene **exactamente 6 autores** y no mas. El "et al." anade un
  septimo autor fantasma. No es un error de substance pero se corrige.
- El resumen tambien se leyo de la fuente (Crossref, campo `abstract`), para no
  describir el mecanismo de memoria.

### 2.2 Referencia bibliografica

Wu, S., Li, Z., Su, Y., Cui, Z., Zhang, X., Wang, L. (2025). GraphFM: Graph
Factorization Machines for Feature Interaction Modelling. *Machine Intelligence
Research*, 22(2), 239-253. Springer. DOI 10.1007/s11633-024-1505-5.
Preprint: arXiv:2105.11866.

### 2.3 Enlace

`https://doi.org/10.1007/s11633-024-1505-5`

### 2.4 Resumen de la idea

La FM es el enfoque dominante para interacciones de features de segundo orden
con datos dispersos de alta dimension, pero segun los autores falla por dos
motivos a la vez: no captura interacciones de orden superior por la expansion
combinatoria, y considerar el producto de **todas** las parejas de features
introduce ruido y degrada la precision. GraphFM representa las features de
forma natural como nodos de un grafo, disena un mecanismo para seleccionar las
interacciones beneficiosas y las formula como aristas entre features, e
integra la funcion de interaccion de la FM dentro de la estrategia de agregacion
de una red neuronal de grafos, de modo que apilar capas da interacciones de
orden arbitrario.

### 2.5 Mecanismo relevante

Dos piezas, y las dos son el problema, no la solucion:
1. un selector de que pareja de features merece interaccion, que se convierte en
   el conjunto de aristas del grafo;
2. message passing de GNN en ese grafo de features, con la funcion de la FM como
   funcion de agregacion, y apilado de capas para subir de orden.
La evaluacion del trabajo es sobre recomendacion / CTR (el codigo que enlazan es
`github.com/CRIPAC-DIG/GraphCTR`).

### 2.6 Que parte es compatible con TOXIC

Ninguna pieza es ilegal por si misma: un grafo y un selector de pares se pueden
construir con las 27.431 features y las 1.483 etiquetas de desarrollo, sin
transformers, sin LLM, sin embeddings preentrenados y sin corpus externo
etiquetado. Es decir, la referencia **no se descarta por las reglas del
concurso**, se descarta por la aritmetica del corpus.

### 2.7 Que parte esta prohibida o no es aplicable

- El selector de pares tiene que elegir sobre el producto cartesiano de las
  27.431 features. `docs/v25_mechanism_audit.md` seccion 4 punto 1 calcula que
  los pares son **376 millones**. Un grafo sobre features con 27.431 nodos y
  una fraccion no despreciable de esas aristas, aprendido con 1.483 etiquetas, es
  una estructura con mas parametros libres que filas de entrenamiento, sin que
  exista ninguna regularizacion que la haga caber en el presupuesto.
- El campo de la publicacion es recomendacion / CTR, no clasificacion de
  toxicidad en espanol. El texto de los experimentos no se ha leido (paywall de
  Springer), asi que no se cita ninguna cifra de este trabajo. La comparacion
  de referencia es modelo de interacciones contra modelo de interacciones, no
  contra TF-IDF.
- "Orden arbitrario" por apilado de capas de GNN choca de frente con el
  veredicto de `reports/v24_02_dictamen_gate.md` seccion 2.3: la restriccion de
  V03 no es la expresividad, son las 1.483 etiquetas de desarrollo. Anadir capas
  sobre el eje equivocado no ataca esa restriccion.

### 2.8 Relacion con V03

Ninguno. V03 es un modelo plano sobre la matriz dispersa, sin grafo de
features, sin aristas, sin GNN. Este es justamente el hueco que
`docs/v25_mechanism_audit.md` seccion 3 describe: el proyecto ha explorado
interacciones en el eje de las etiquetas (v6-B, v12-03) y en el espacio de
modelos (ensembles, stacking), nunca en el eje de las features. GraphFM es la
primera referencia de esta lista que vive en ese eje.

### 2.9 Relacion con V25

Es la referencia de **que viene despues** de FM, y por eso se cita como el
limite superior al que V25 **no** sube. No se implementa nada de ella. Su
utilidad en esta fase es negativa: justifica quedarse en el orden 2 de bajo
rango y no escalar a un grafo de 376 millones de aristas aprendidas con 1.483
filas.

---

## 3. Junbo Huang, Ricardo Usbeck (2024). "Revisiting Supervised Contrastive Learning for Microblog Classification." EMNLP 2024

### 3.1 Verificacion

- `https://aclanthology.org/2024.emnlp-main.876/` devuelve **200**.
  `https://aclanthology.org/2024.emnlp-main.876.bib` devuelve **200** con
  `title` = "Revisiting Supervised Contrastive Learning for Microblog
  Classification", `author` = "Huang, Junbo" and "Usbeck, Ricardo",
  `year` = 2014... no: `year` = 2014 en el month, el campo `year` = **2024**,
  `booktitle` = "Proceedings of the 2024 Conference on Empirical Methods in
  Natural Language Processing", `pages` = "15644--15653",
  `doi` = "10.18653/v1/2024.emnlp-main.876", `address` = Miami, Florida, USA,
  `month` = nov.
- `https://api.crossref.org/works/10.18653/v1/2024.emnlp-main.876` devuelve
  **200** con el mismo titulo, los mismos dos autores, `published` = 2024 y
  `page` = "15644-15653". **Coincide con lo declarado** (las paginas no venian
  declaradas y no hay conflicto).
- Resumen leido de la pagina ACL: se examina la eficacia de afinar modelos de
  lenguaje basados en transformer, regularizados con una perdida de aprendizaje
  contrastivo supervisado, para clasificacion de microblogs en ingles. La
  evaluacion en TweetEval y Tweet Topic Classification muestra mejora sobre los
  modelos base, con una ganancia de hasta **11,9 puntos porcentuales** en todas
  las subtareas.

### 3.2 Referencia bibliografica

Huang, J., Usbeck, R. (2024). Revisiting Supervised Contrastive Learning for
Microblog Classification. En *Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing (EMNLP 2024)*, pp. 15644-15653.
Association for Computational Linguistics.

### 3.3 Enlace

`https://aclanthology.org/2024.emnlp-main.876/`

### 3.4 Resumen de la idea

El contenido de microblogs (tweets) es ruidoso por su registro informal y por la
falta de contexto dentro de cada publicacion, y por eso los modelos de estado
del arte se apoyan en modelos de lenguaje preentrenados. El trabajo reexamina si
el aprendizaje contrastivo supervisado (SCL) basta por si solo con recursos
pequenos, y encuentra que afinar un encoder de transformer con una perdida SCL
mejora la clasificacion de microblogs en ingles.

### 3.5 Mecanismo relevante

Una perdida de aprendizaje contrastivo supervisado: en el espacio de
representaciones, attraction entre ejemplos de la misma etiqueta y repulsion
entre ejemplos de etiquetas distintas, usada como termino regularizador sobre un
encoder de transformer afinado.

### 3.6 Que parte es compatible con TOXIC

Ninguna parte del mecanismo. La pieza de attraction/repulsion en un espacio
denso se podria escribir sin transformer, pero entonces seria otra cosa (un
metodo de metric learning sobre las salidas de V03) y no este trabajo.

### 3.7 Que parte esta prohibida o no es aplicable

- **Transformers preentrenados: prohibido de forma expresa por las reglas del
  concurso.** El objeto del trabajo es afinar un encoder de transformer, asi que
  el nucleo del metodo cae entero dentro de la prohibicion.
- Los dos benchmarks son en ingles (TweetEval y Tweet Topic Classification) y la
  tarea es clasificacion de microblogs en general, no toxicidad en espanol. No
  hay lectura directa de un numero para esta tarea y este idioma.
- La **motivacion** del trabajo es que preentrenar un LM propio es
  resource-intensive y no apto para laboratorios pequenos. Aqui la restriccion
  de recursos es una prohibicion dura, no una limitacion practica.
- El unico dato numerico del trabajo (hasta 11,9 puntos) es sobre microblogs en
  ingles con transformer, asi que **no es transferible** a 1.483 filas de
  toxicidad en espanol con matriz dispersa. Se cita como lo que es, no como
  una expectativa.

### 3.8 Relacion con V03

No se ejecuto nunca un objetivo contrastivo. La sonda estructural de
`experiments/v12_00_auditoria_brecha.py` (registrada en
`reports/v12_auditoria_brecha.json`, familia "Entrenamiento contrastivo o por
pares sobre TF-IDF disperso") encontro **cero ocurrencias** de `contrastive`,
`pairwise`, `margin_loss` o `hinge` **como objetivo propio** en el arbol
sintactico de los 127 modulos del proyecto, y la familia queda como
`GENUINELY_UNTESTED` en `reports/v12_gap_audit.csv` (fila 24). El propio
informe avisa de que la perdida hinge si se probo, pero como clasificador
completo (LinearSVC), no como termino auxiliar.

Lo que **si** se ejecuto y es lo mas proximo es otra cosa:
`experiments/v12_03_entrenamiento_por_pares.py`
acopla **pares de etiquetas** dentro de una misma fila mediante un termino de
margen fijo 1,0 en unidades de logit, no pares de ejemplos ni pares de features.
Resultado en `reports/v12_03_entrenamiento_por_pares.md`: `lambda=0.0` da
0,777265 (+0,000053), `lambda=0.25` da 0,774742 (**-0,002469**), `lambda=1.0` da
0,769274 (-0,007938), curva monotonamente decreciente con maximo en `lambda=0`,
y H3 queda **cerrada**.

### 3.9 Relacion con V25

**No se implementa y no se va a implementar.** No es candidata de V25: es una
referencia del eje del objetivo de entrenamiento, ortogonal al eje de las
interacciones de features que V25 explora. Se cita por dos razones operativas:
para dejar constancia de que el eje del objetivo ya esta sondeado y cerrado, y
para dejar constancia de que la unica pieza que de aqui seria legal
(contraste en un espacio denso) no es este trabajo.

---

## 4. Horacio Jarquin Vazquez, Hugo Jair Escalante, Manuel Montes, Mario Ezra Aragon (2025). "GAttention: Gated Attention for the Detection of Abusive Language." Findings of EMNLP 2025, 20314-20329

### 4.1 Verificacion

- `https://aclanthology.org/2025.findings-emnlp.1105/` devuelve **200**.
  `https://aclanthology.org/2025.findings-emnlp.1105.bib` devuelve **200** con
  `title` = "{GA}ttention: Gated Attention for the Detection of Abusive Language",
  `booktitle` = "Findings of the Association for Computational Linguistics: EMNLP 2025",
  `pages` = "20314--20329" (coincide exactamente con lo declarado),
  `month` = nov, `year` = **2025**, `address` = Suzhou, China,
  `publisher` = Association for Computational Linguistics,
  `doi` = "10.18653/v1/2025.findings-emnlp.1105", `ISBN` = 979-8-89176-335-7.
- `https://api.crossref.org/works/10.18653/v1/2025.findings-emnlp.1105` devuelve
  **200**, mismo titulo, mismos cuatro autores, `published` = 2025,
  `page` = "20314-20329".
- **DISCREPANCIA EN EL CAMPO DE AUTORIA.** Crossref devuelve el autor como
  `given = "Horacio Jarquin Vasquez"` y `family = "Vazquez"`, es decir su
  apellido es **Vazquez** y el nombre de pila lleva "Jarquin" como segundo
  nombre. La ficha de ACL BibTeX da `Vasquez, Horacio Jarquin`. Lo declarado
  "Horacio Jarquin Vazquez" es la misma persona, pero funde el nombre de pila
  con el apellido. La forma correcta, en ASCII, es **Vazquez, Horacio Jarquin**.
  Se deja constancia porque en una cita bibliografica el apellido es el campo
  por el que se cita.
- Los otros tres autores si coinciden exactamente: Hugo Jair Escalante,
  Manuel Montes, Mario Ezra Aragon. El tercero y el cuarto aparecen ademas en
  `references.bib` con los mismos roles en MEX-A3T (`aragon2018mexa3t`,
  `alvarezcarmona2018mexa3t`).
- Resumen leido de la pagina ACL: GAttention combina un mecanismo de
  atencion contextual con uno de auto-atencion, integrando las relaciones
  internas de una secuencia y las relaciones globales entre secuencias
  distintas, y se aplica sobre modelos de encoder basados en transformer.

### 4.2 Referencia bibliografica

Vazquez, H. J., Escalante, H. J., Montes, M., Aragon, M. E. (2025). GAttention:
Gated Attention for the Detection of Abusive Language. En *Findings of the
Association for Computational Linguistics: EMNLP 2025*, pp. 20314-20329.
Association for Computational Linguistics.

### 4.3 Enlace

`https://aclanthology.org/2025.findings-emnlp.1105/`

### 4.4 Resumen de la idea

El lenguaje abusivo en linea crea entornos toxicos y agrava las tensiones
sociales, y eso hace falta modelos que interpreten senales linguisticas sutiles.
El trabajo introduce una atencion con puerta (Gated Attention) que combina las
ventajas de la atencion contextual y de la auto-atencion para sortear las
limitaciones de los modelos de atencion existentes en clasificacion de texto.
La combinacion permite entender y procesar mejor los elementos de la secuencia,
que es especialmente util en tareas sensibles al contexto como la deteccion de
lenguaje abusivo.

### 4.5 Mecanismo relevante

Una capa de atencion con puerta que fusiona un vector de consulta **contextual**
(relaciones globales entre secuencias distintas) con un vector de consulta de
**auto-atencion** (relaciones internas dentro de una misma secuencia), insertada
dentro de un encoder basado en transformer. Es una intervencion sobre la
arquitectura de un encoder, no sobre la representacion ni sobre el objetivo.

### 4.6 Que parte es compatible con TOXIC

Nada del mecanismo. La tarea (deteccion de lenguaje abusivo) es la nuestra, pero
la tarea no es el mecanismo.

### 4.7 Que parte esta prohibida o no es aplicable

- **Transformers: prohibido de forma expresa.** El mecanismo es una capa de
  atencion para encoder de transformer, asi que el objeto del trabajo esta
  entero dentro de la prohibicion.
- La "atencion contextual entre secuencias distintas" necesita un conjunto de
  secuencias relacionadas entre si (documentos del mismo tema, por ejemplo).
  Este corpus son 1.483 tweets sueltos sin ninguna estructura de documento que
  los relacione, asi que la mitad del mecanismo no tiene insumo.
- Los numeros del trabajo **no se citan aqui**: no se han leido las tablas. El
  resumen disponible en la pagina ACL no contiene cifras y no se inventa ninguna.

### 4.8 Relacion con V03

Ninguno. V03 no tiene atencion, ni encoder, ni ninguna capa recurrente. Lo
mas proximo que se ejecuto en el proyecto es una arquitectura recurrente en
lugar de atencion: `experiments/v10_11_bigru_finetuning.py`, informe
`reports/v10_11_bigru_finetuning.md`, con los tres brazos
`congelado` 0,693994, `afinado` 0,711639, `afinando_todo` 0,699086, todos por
debajo de V03 en 0,777211 (`afinado menos V03` = **-0,065572**,
`gana a V03: False`). El veredicto del informe es "H1 QUEDA CERRADA con dos
variantes" y la consecuencia escrita es "**la via de representacion secuencial
queda cerrada**". Ademas `docs/v25_mechanism_audit.md` seccion 1 registra la CNN
de caracteres (DL-02, `11_dl_desde_cero.py`) en 0,7410, tambien cerrada.

### 4.9 Relacion con V25

**No se implementa y no se va a implementar.** La unica palabra que podria
parecer transferible es "puerta", y no lo es: una puerta es un producto de
parametros aprendidos dentro de una arquitectura neuronal, y V25 trabaja sobre
una matriz dispersa con un clasificador lineal. Se cita como precedent de que la
literatura reciente de deteccion de lenguaje abusivo sigue siendo casi toda
arquitecturas de transformer, y por tanto **fuera de las reglas del concurso**:
eso es informacion util para no perder tiempo buscandolas.

---

## 5. Naganand Yadati (2024). "GAINER: Graph Machine Learning with Node-specific Radius for Classification of Short Texts and Documents." EACL 2024, 609-626

### 5.1 Verificacion

- `https://aclanthology.org/2024.eacl-long.36/` devuelve **200**.
  `https://aclanthology.org/2024.eacl-long.36.bib` devuelve **200** con
  `title` = "{GAINER}: Graph Machine Learning with Node-specific Radius for
  Classification of Short Texts and Documents",
  `author` = "Yadati, Naganand" (autor unico),
  `booktitle` = "Proceedings of the 18th Conference of the European Chapter of
  the Association for Computational Linguistics (Volume 1: Long Papers)",
  `pages` = "609--626" (**coincide exactamente con lo declarado**),
  `address` = "St. Julian's, Malta", `month` = mar, `year` = **2024**,
  `doi` = "10.18653/v1/2024.eacl-long.36".
- `https://api.crossref.org/works/10.18653/v1/2024.eacl-long.36` devuelve
  **200** con el mismo titulo, el mismo autor unico, `published` = 2024,
  `page` = "609-626". **Coincide con lo declarado en todos los campos.**
- Resumen leido de la pagina ACL: el message passing es esencial para el
  rendimiento empirico en los modelos de grafos para NLP, pero el numero de
  pasos de message passing (el "radio") es fijo para todos los nodos en los
  modelos existentes. Fijar el radio es una restriccion fundamental porque los
  nodos tienen propiedades y cantidades distintas de estructura local
  informativa. GAINER propone un radio especifico por nodo, con enfoques no
  neuronales y neuronales nuevos.

### 5.2 Referencia bibliografica

Yadati, N. (2024). GAINER: Graph Machine Learning with Node-specific Radius for
Classification of Short Texts and Documents. En *Proceedings of the 18th
Conference of the European Chapter of the Association for Computational Linguistics
(Volume 1: Long Papers) (EACL 2024)*, pp. 609-626. Association for
Computational Linguistics.

### 5.3 Enlace

`https://aclanthology.org/2024.eacl-long.36/`

### 5.4 Resumen de la idea

En los modelos de grafos para NLP el radio del message passing esta fijado por
igual para todos los nodos, y eso restringe el modelo de forma fundamental
porque cada nodo tiene su propia cantidad de estructura local informativa. GAINER
("Graph mAchine learnIng with Node-spEcific Radius") sustituye el radio global
por un radio decidido nodo a nodo, y valida la idea con una variante no neuronal
y otra neuronal.

### 5.5 Mecanismo relevante

Convertir el numero de pasos de message passing, que es un hiperparametro global
fijo, en una decision por nodo. Operativamente: en lugar de `k` iteraciones de
agregacion para todos los nodos, cada nodo tiene su `k` y se detiene cuando su
propia estructura local deja de aportar.

### 5.6 Que parte es compatible con TOXIC

Ninguna pieza concreta. La idea abstracta de "campo receptor adaptativo por
unidad" no es ilegal, pero no hay ninguna unidad en este proyecto a la que
asignarle un radio.

### 5.7 Que parte esta prohibida o no es aplicable

- Requiere un grafo de entrada con nodos y aristas. Este corpus no tiene grafo:
  ni de documentos, ni de features, ni de autores. La busqueda de `gnn`,
  `graph neural`, `red de grafos`, `message passing`, `node embedding` y `grafo`
  sobre `reports/`, `docs/`, `DECISIONS.md` y `PROJECT_STATE.md` devuelve **cero
  coincidencias**. Sin grafo, el radio por nodo no tiene objeto.
- Los dos experimentos citados en el resumen son sobre tareas de NLP con modelos de
  grafos, no sobre toxicidad en espanol.
- El presupuesto: aprender un radio por nodo es un hiperparametro por unidad de
  entrada. Con 1.483 filas de desarrollo y `reports/v24_02_dictamen_gate.md`
  seccion 2.1 explicando que el umbral de gate esta dentro del ruido del
  instrumento, anadir un hiperparametro por fila es exactamente el tipo de
  palanca que ese dictamen declara no rentable.

### 5.8 Relacion con V03

Ninguno. V03 no tiene grafo, ni message passing, ni radio. Es la segunda
referencia de esta lista (con GraphFM) que vive enteramente fuera de lo que
V03 es y de lo que el repositorio ha ejecutado.

### 5.9 Relacion con V25

**No se implementa y no se va a implementar.** Se cita como contrapeso de
GraphFM: si alguien propusiera algun dia un grafo sobre features, GAINER es la
referencia que dice que el radio es lo que hay que vigilar, y que lo que hay que
vigilar es un hiperparametro y no un mecanismo nuevo. No aporta ninguna eje a
V25.

---

## 6. Jiaqi Mu, Pramod Viswanath (2018). "All-But-The-Top: Simple and Effective Postprocessing for Word Representations." ICLR 2018

### 6.1 Verificacion

- `https://arxiv.org/abs/1702.01417` devuelve **200**.
  `title` = "All-but-the-Top: Simple and Effective Postprocessing for Word
  Representations".
  `Authors` = **"Jiaqi Mu, Suma Bhat, Pramod Viswanath"**.
  `Submitted on 5 Feb 2017 (v1), last revised 19 Mar 2018 (this version, v2)`.
  `Subjects` = Computation and Language (cs.CL); Machine Learning (stat.ML).
  DOI de arXiv: `10.48550/arXiv.1702.01417`.
  Tambien verificado por la API de arXiv
  (`https://export.arxiv.org/api/query?id_list=1702.01417`, estado 200), que
  devuelve los mismos tres autores.
- **FALLO EN EL CAMPO DE AUTORIA.** Lo declarado lista **dos** autores,
  "Jiaqi Mu, Pramod Viswanath". El registro tiene **tres**: falta
  **Suma Bhat**, en posicion intermedia. Esto no es una variante de citacion
  admisible, es un autor omitido.
- **DISCREPANCIA MENOR EN EL TITULO.** El titulo publicado es "All-but-the-Top",
  con la segunda palabra en minuscula. Lo declarado escribe "All-But-The-Top",
  con B y T mayusculas. Mismo trabajo, grafia distinta.
- **DISCREPANCIA MENOR EN EL ANIO.** La version de arXiv es de febrero de 2017
  (v1) y la revisada de marzo de 2018 (v2). La cita a ICLR 2018 es correcta para
  la version de conferencia, que es la que se cita. La entrada `arXiv:1702.01417`
  es de 2017.
- No se pudo verificar la ficha de OpenReview: la API de `openreview.net`
  devuelve 400 con un aviso de migracion a `api.openreview.net`, y el intento
  por DBLP devuelve una pagina de verificacion anti-bot. Se hace constar como
  limitacion, no como resultado. La version de arXiv esta verificada y es la
  misma obra.
- **El mismo error esta dentro del repositorio.** En
  `reports/v21_p1_01_embeddings_desde_cero.md`, referencias numeradas, entrada
  **9**, se escribe "Mu, J., Viswanath, P. (2018). All-but-the-Top: ...". El
  mismo error de autoria esta propagado en el informe interno.

### 6.2 Referencia bibliografica

Mu, J., Bhat, S., Viswanath, P. (2018). All-but-the-Top: Simple and Effective
Postprocessing for Word Representations. *ICLR 2018*. Preprint arXiv:1702.01417
(v1 2017-02-05, v2 2018-03-19). DOI 10.48550/arXiv.1702.01417.

### 6.3 Enlace

`https://arxiv.org/abs/1702.01417`

### 6.4 Resumen de la idea

Las representaciones vectoriales reales de palabras han transformado las
aplicaciones de NLP, y ejemplos populares son word2vec y GloVe por su capacidad
de capturar regularidades linguisticas. El trabajo demuestra un
postprocesamiento **muy simple y contra-intuitivo** que hace que las
representaciones ya disponibles sean aun mas fuertes: eliminar el vector medio
comun y unas pocas direcciones dominantes de los vectores de palabra. El
procedimiento se valida en tareas intrinsecas de nivel lexical y en tareas de
nivel de oracion, en varios idiomas, y en cada caso las representaciones
procesadas son consistentemente mejores que las originales.

### 6.5 Mecanismo relevante

Centrar la matriz de vectores de palabra restando el vector medio comun, y a
continuacion retirar las `D` primeras direcciones principales (localizadas por
SVD), sin renormalizar despues. Es una funcion determinista de una matriz ya
entrenada: no entrena nada, no necesita etiquetas, no necesita corpus nuevo.
Cuesta segundos.

### 6.6 Que parte es compatible con TOXIC

Compatible en principio y compatible en la practica, porque ya se ejecuto: es
una funcion determinista de los vectores que el proyecto ya tenia, no usa
supervision, ni datos externos, ni nada preentrenado. Es de las pocas ideas de
esta lista que el reglamento no toca.

### 6.7 Que parte esta prohibida o no es aplicable

- No es aplicable a V03 directamente: el mecanismo necesita una matriz densa de
  vectores y V03 no tiene ninguna, es un clasificador lineal sobre una matriz
  dispersa de TF-IDF. En este repositorio se aplico al bloque denso de FastText
  de la fase V5, que es donde hay una matriz densa.
- La ponderacion SIF asociada (pesos `a/(a+p(w))`) se descarto por evidencia
  propia, segun `reports/v21_p1_01_embeddings_desde_cero.md` seccion 2.2: el
  equipo midio pooling ponderado por TF-IDF en 0,6348 frente a media en 0,6530, y
  la critica publicada a SIF encuentra que la ventaja desaparece cuando ambos
  metodos usan eliminacion de componente. Se aplica `R` sin `W`.

### 6.8 Relacion con V03

No toca V03. Modifica el bloque denso **antes** de unirlo, es decir actua sobre
la unica via por la que el embedding puede sumar algo a los 27.431 rasgos
dispersos. Y esa via ya esta acotada: `reports/v24_02_dictamen_gate.md` seccion
2.4 fija el techo de fusion de FastText en **+0,0064** de OOF con **0
submissions**, y lo llama "la mejor mejora OOF disponible en todo el
repositorio, en cualquier rama". Cualquier reformulacion del bloque denso tiene
que pasar por ese techo.

### 6.9 Relacion con V25

**YA EJECUTADO Y CERRADO. No queda nada que hacer.**
Script: `experiments/v22_03_all_but_the_top.py`.
Informes: `reports/v22_03_all_but_the_top.md`,
`reports/v22_03_all_but_the_top.json`, `reports/v22_03_cierre.md`.
Control de cordura con `D = 0`: 0,781766 reproducido con desviacion 0,000000,
dentro de tolerancia. Aplicado sobre la matriz de los 6.059 vectores de tipo que
el pipeline consulta, antes del pooling, sin renormalizacion posterior.
Resultados frente al control: `D=1` **-0,001831**, `D=2` **-0,006121**,
`D=5` **-0,003622**. Veredicto escrito: `ALL_BUT_THE_TOP_NO_APORTA`. Ninguna
eliminacion de componentes mejora la fusion por encima del control, que ya es
el embedding sin postprocesar.
`docs/v25_mechanism_audit.md` seccion 1 lo registra como **cerrado**, con la
nota explicita de que no es una interaccion sino un postprocesado de un espacio
vectorial.

---

## 7. Duyu Tang, Furu Wei, Nan Yang, Ming Zhou, Ting Liu, Bing Qin (2014). "Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification." ACL 2014, 1555-1565

### 7.1 Verificacion

- `https://aclanthology.org/P14-1146/` devuelve **200**.
  `https://aclanthology.org/P14-1146.bib` devuelve **200** con
  `title` = "Learning Sentiment-Specific Word Embedding for {T}witter Sentiment
  Classification", `author` = "Tang, Duyu" and "Wei, Furu" and "Yang, Nan" and
  "Zhou, Ming" and "Liu, Ting" and "Qin, Bing" (**los 6, en el mismo orden**),
  `booktitle` = "Proceedings of the 52nd Annual Meeting of the Association for
  Computational Linguistics (Volume 1: Long Papers)", `month` = jun,
  `year` = **2014**, `address` = Baltimore, Maryland,
  `pages` = "1555--1565" (**coincide exactamente**),
  `doi` = "10.3115/v1/P14-1146".
- `https://api.crossref.org/works/10.3115/v1/P14-1146` devuelve **200** con el
  mismo titulo, los mismos 6 autores en el mismo orden, `published` = 2014,
  `page` = "1555-1565". **Coincide con lo declarado en todos los campos.**
- Resumen leido via Semantic Scholar para el mismo DOI (la ficha ACL no expone
  bloque de resumen en el HTML que se pudo leer): el metodo aprende embeddings
  de palabra para clasificacion de sentimiento en Twitter; los algoritmos
  habituales solo modelan el contexto sintactico e ignoran el sentimiento, lo
  que es un problema porque palabras con contexto similar y polaridad opuesta
  quedan cerca; SSWE codifica informacion de sentimiento en la representacion
  continua; se desarrollan **tres redes neuronales** que incorporan la
  supervision de polaridad en sus funciones de perdida; para obtener corpus a
  gran escala se aprende de **tweets con supervision distante recogidos por
  emoticonos positivos y negativos**; en SemEval 2013 (1) la feature SSWE
  rinde de forma comparable a las features hechos a mano en el sistema con
  mejor resultado, y (2) el rendimiento mejora mas al concatenar SSWE con el
  conjunto de features existente.

### 7.2 Referencia bibliografica

Tang, D., Wei, F., Yang, N., Zhou, M., Liu, T., Qin, B. (2014). Learning
Sentiment-Specific Word Embedding for Twitter Sentiment Classification. En
*Proceedings of the 52nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) (ACL 2014)*, pp. 1555-1565. Association for
Computational Linguistics.

### 7.3 Enlace

`https://aclanthology.org/P14-1146/`

### 7.4 Resumen de la idea

Las representaciones continuas de palabras suelen modelar solo el contexto
sintactico e ignorar el sentimiento, de modo que palabras con contexto similar y
polaridad opuesta acaban con vectores vecinos. SSWE resuelve eso metiendo la
informacion de sentimiento dentro de la representacion continua de la palabra,
con tres redes neuronales distintas que incorporan la supervision de polaridad
en la funcion de perdida, y se entrena sobre un corpus massive de tweets con
supervision distante por emoticonos.

### 7.5 Mecanismo relevante

Entrenar los vectores de palabra **no** con un objetivo de lenguaje sino con un
objetivo que ve la etiqueta de sentimiento de la frase o del tweet, para que el
subespacio relevante para la clase quede alineado con el subespacio que el
clasificador necesita. El corpus de entrada son tweets etiquetados
automaticamente por emoticonos.

### 7.6 Que parte es compatible con TOXIC

La **variante con las 1.483 etiquetas propias**. El proyecto ya lo decidio por
escrito: `reports/v21_p1_01_embeddings_desde_cero.md` seccion 3, candidata
numero 2 ("Embeddings con objetivo dirigido a la tarea"), declara
"**Es legal en la variante Liu et al. (2018) / Tang et al. (2014) / Zheng et al.
(2015) / Kenter et al. (2016)**: usa las etiquetas del propio train, no un
dataset externo". Es legal porque no introduce ninguna fuente de informacion
nueva: reordena la misma supervision que V03 ya consume.

### 7.7 Que parte esta prohibida o no es aplicable

- **El corpus real del trabajo esta prohibido.** Es un corpus masivo de tweets
  externos con supervision distante por emoticonos, es decir un dataset externo
  etiquetado (por heuristica), y las reglas del concurso lo prohiben de forma
  expresa. La variante que se podria ejecutar **no es la del trabajo**, es una
  adaptacion.
- Las tres redes son neuronales, asi que la variante ejecutable exige
  entrenamiento real de una red, no una reparametrizacion del pipeline.
- El modo de fallo que el dictamen identifica es el mismo que el del NB-SVM: al
  colapsar cada tipo a su afinidad con la clase, el objetivo conserva **si** el
  tipo es toxico y descarta **cuanto aparece**, y la frecuencia es senal en
  tweets toxicos.

### 7.8 Relacion con V03

No se ejecuto nunca contra V03. Lo que el dictamen ofrece es la
analogia mas cercana ya resuelta en este proyecto: el NB-SVM, que es la manera
mas directa de reescalar la representacion por clase con senal supervisada, se
implemento bien y **perdio**, 0,7638 contra 0,7772. Y el techo empirico es el
mismo de la seccion 6.8: +0,0064 de fusion, 0 submissions.

### 7.9 Relacion con V25

**NO EJECUTADO, y cerrado por dictamen antes de ejecutar.**
`reports/v22_02_dictamen_especialista.md`, Pregunta 2, respuesta directa:
"**NO merece ejecutarse.**" Los cuatro hechos que lo cierran, tal como estan
escritos alli:
1. **Redundancia con el clasificador lineal.** El gradiente de SSWE y el del
   clasificador lineal apuntan al mismo sitio, y el bloque resultante es el mas
   redundante con el que ya existe. Hay una tension logica entre "ser util" y
   "ser distinto" que el objetivo supervisado resuelve siempre en contra de
   "ser distinto".
2. **El precedente NB-SVM**, que perdio 0,7638 contra 0,7772 porque binarizar y
   descartar el term frequency pierde senal.
3. **La escala de etiquetas y el circulo vicioso de la regularizacion.** Se
   dispone de 1.483 etiquetas binarias frente a un bloque de 6.059 vectores de
   tipo por 200 dimensiones, del orden de 1,2 millones de parametros. La
   conclusion escrita es literally: "**el unico regimen de hiperparametros en el
   que SSWE es seguro es el regimen en el que SSWE es un no-op**".
4. **El techo empirico**, que SSWE no cambia porque no cambia la densidad.
Ademas, el riesgo se declara **nuevo y no cubierto**: es riesgo de estimacion
(ajustar ruido de etiqueta con 1.483 ejemplos) y no el riesgo de seleccion
sobre ruido de las sondas que ya se descartaron. El coste estimado era el mas
alto de V22 por un orden de magnitud (3 a 5 horas de calculo).
Otros ficheros: `reports/v22_03_cierre.md` ("SSWE: NO merece ejecutarse"),
`reports/v21_03_cierre_de_fase.md` (tabla: SSWE, objetivo dirigido, "no
medida", estado "nueva", riesgo "alto", autorizada "**no**"),
`reports/v21_02_dictamen_competitivo.md` seccion B ("no aporta informacion
nueva"). No hay nada que implementar.

---

## 8. Carl Saroufim, Akram Almatarky, Mohammad Abdel Hady (2018). "Language Independent Sentiment Analysis with Sentiment-Specific Word Embeddings." WASSA 2018

### 8.1 Verificacion

- `https://aclanthology.org/W18-6204/` devuelve **200**.
  `https://aclanthology.org/W18-6204.bib` devuelve **200** con
  `title` = "Language Independent Sentiment Analysis with Sentiment-Specific
  Word Embeddings", `author` = "Saroufim, Carl" and "Almatarky, Akram" and
  "Abdel Hady, Mohammad" (**los 3, en el mismo orden**),
  `booktitle` = "Proceedings of the 9th Workshop on Computational Approaches to
  Subjectivity, Sentiment and Social Media Analysis", `month` = oct,
  `year` = **2018**, `address` = Brussels, Belgium,
  `pages` = "14--23" (no declarado en el encargo, sin conflicto),
  `doi` = "10.18653/v1/W18-6204".
- `https://api.crossref.org/works/10.18653/v1/W18-6204` devuelve **200** con el
  mismo titulo, los mismos 3 autores, `published` = 2018, `page` = "14-23".
  **Coincide con lo declarado en todos los campos.**
- Resumen leido de la pagina ACL: la anotacion de datos es un paso critico pero
  tedioso, caro y lento; se presenta un metodo independiente del idioma para
  entrenar un modelo de polaridad de sentimiento con poca cantidad de datos
  etiquetados a mano; word embeddings como Word2Vec son eficientes pero pueden
  mapear palabras de polaridad opuesta a vectores cercanos; se entrenan SSWE
  sobre un modelo Word2Vec no supervisado, usando RNN o CNN, sobre datos
  autoetiquetados como "Positive" o "Negative"; para ello se apoya en la
  **universalidad de emojis y emoticonos** para autoetiquetar un numero grande de
  **tweets en frances** con un conjunto pequeno de emojis y emoticonos
  positivos y negativos; despues se aplica **transfer learning** para refinar
  los pesos de la red con un conjunto pequeno de datos etiquetados a mano; se
  evalua en clasificacion de sentimiento en frances sobre los datos de
  SemEval 2016; se logra mejora usando SSWE sobre Word2Vec; y se usa ademas un
  **enfoque basado en grafos para propagacion de etiquetas** para autogenerar un
  lexico de sentimiento.

### 8.2 Referencia bibliografica

Saroufim, C., Almatarky, A., Abdel Hady, M. (2018). Language Independent Sentiment
Analysis with Sentiment-Specific Word Embeddings. En *Proceedings of the 9th
Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media
Analysis (WASSA 2018)*, pp. 14-23. Association for Computational Linguistics.

### 8.3 Enlace

`https://aclanthology.org/W18-6204/`

### 8.4 Resumen de la idea

Entrenar un modelo de polaridad de sentimiento con muy pocos datos etiquetados a
mano, aprovechando que los emojis y emoticonos son universales: con un conjunto
pequeno de emojis y emoticonos positivos y negativos se autoetiqueta un corpus
grande de tweets en frances, se entrena un embedding especifico de sentimiento
sobre Word2Vec no supervisado con una RNN o una CNN, y despues se refinan los
pesos con el conjunto pequeno de etiquetas manuales. El resultado es una mejora
sobre Word2Vec, ademas de un lexico de sentimiento autogenerado por propagacion
de etiquetas sobre un grafo.

### 8.5 Mecanismo relevante

Autoetiquetado a escala con una semilla pequena y barata (emojis y emoticonos)
como fuente de Supervision distante, entrenamiento de un embedding de sentimiento
sobre el bloque no supervisado, y afinamiento con un conjunto manual pequeno. La
pieza de propagacion de etiquetas sobre un grafo genera el lexico de sentimiento
por consenso de vecinos.

### 8.6 Que parte es compatible con TOXIC

Solo el paso final de **transfer learning** sobre un conjunto manual pequeno,
que aqui serian las 1.483 filas de desarrollo. Eso es legal por definicion: es
la supervision que ya existe en el proyecto.

### 8.7 Que parte esta prohibida o no es aplicable

- **El autoetiquetado por emoji esta prohibido.** Es exactamente un dataset
  externo etiquetado, construido con una heuristica. El concurso prohibe
  datasets externos etiquetados, y esta es la forma mas economica y por tanto
  la mas tentadora de conseguir uno, asi que conviene nombrarla
  explicitamente para que nadie la intente por el camino corto.
- **El lexico de sentimiento autogenerado por propagacion de etiquetas sobre un
  grafo esta prohibido** por la misma razon: es un recurso etiquetado externo
  derivado.
- El nombre "language independent" es enganoso para este caso: el metodo es
  independiente del idioma en el sentido de que funciona en cualquier idioma,
  pero suautoetiquetado se apoya en la universalidad de los emojis y su
  evaluacion es en **frances**. La competencia es en espanol y no hay ninguna
  lectura de resultado de espanol en este trabajo.
- La arquitectura es una RNN o una CNN entrenada, no una reparametrizacion del
  pipeline disperso.

### 8.8 Relacion con V03

Ninguna, y no se ejecuto nada. Lo mas cercano en el repositorio son dos lineas
previas: el NB-SVM, que perdio 0,7638 contra 0,7772 segun
`reports/v22_02_dictamen_especialista.md`, y la familia de reglas y lexico, con
`reports/rule_audit_v2.md` y `reports/rule_audit_v4.md`. Ninguna de las dos es
este trabajo.

### 8.9 Relacion con V25

**NO EJECUTADO, y ni siquiera mencionado en el repositorio.** La busqueda de
`saroufim`, `WASSA`, `language independent` e `independiente del idioma` sobre
`reports/`, `docs/`, `DECISIONS.md`, `PROJECT_STATE.md` y `references.bib`
devuelve **cero coincidencias** de esta referencia (la unica coincidencia de
"independiente del idioma" en `reports/v21_p1_02_toxicidad_nlp_clasico.md` linea
466 es un metodo distinto, basado en emoji, con arXiv:2201.06723, y no este
trabajo). Tampoco figura en `references.bib`, que tiene 553 lineas y no contiene
ninguna de las nueve referencias de este informe. Su papel en V25 es
**negativo y documental**: es el ejemplo mas claro de como se construiria un
dataset etiquetado externo barato, y por tanto la referencia que hay que citar
para dejar escrito que ese camino esta cerrado por regla y no por falta de
evidencia. No hay nada que implementar.

---

## 9. Albert Gu, Tri Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."

### 9.1 Verificacion

- `https://export.arxiv.org/api/query?id_list=2312.00752` devuelve **200** con
  `title` = "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
  `author` = ["Albert Gu", "Tri Dao"], `published` = 2023-12-01T18:01:34Z,
  `updated` = 2024-05-31T17:55:27Z.
- Resumen leido de la misma fuente: los modelos fundacionales que alimentan
  casi todas las aplicaciones =(interesantes) de deep learning se basan casi de
  forma universal en la arquitectura Transformer y en su modulo de atencion;
  muchas arquitecturas subcuadraticas (atencion lineal, convolucion con
  puerta, modelos recurrentes y modelos estructurados de espacio de estados) se
  han desarrollado para tratar la ineficiencia computacional del Transformer en
  secuencias largas, pero no han rendido tan bien como la atencion en
  modalidades importantes como el lenguaje; los autores identifican una
  debilidad clave de esos modelos, su incapacidad para hacer razonamiento
  basado en contenido, y hacen varias mejoras, la primera de las cuales es dejar
  que los parametros del SSM sean funciones de la entrada.
- `https://arxiv.org/abs/2312.00752` responde y el identificador es valido.
  **Coincide con lo declarado** en titulo, autores y anio.
- **LIMITACION A DECLARAR.** El registro de arXiv **no tiene `journal_ref` ni
  comentario de publicacion**: es un preprint. Semantic Scholar respondio 429
  (limite de peticiones) cuando se le pidio, asi que no se pudo confirmar una
  version de revista o de conferencia. Solo queda verificada la version de
  preprint. El encargo tampoco declaraba version de publicacion, asi que la
  referencia es correcta tal como esta escrita, pero conviene saber que es un
  preprint.

### 9.2 Referencia bibliografica

Gu, A., Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective
State Spaces. Preprint arXiv:2312.00752. DOI 10.48550/arXiv.2312.00752.

### 9.3 Enlace

`https://arxiv.org/abs/2312.00752`

### 9.4 Resumen de la idea

Casi todos los modelos fundacionales se basan en el Transformer, y hay
arquitecturas subcuadraticas alternativas que no han igualado a la atencion en
lenguaje. El trabajo identifica la debilidad compartida de esas alternativas, que
no pueden hacer razonamiento basado en contenido, y la corrige haciendo que los
parametros del modelo de espacio de estados sean funciones de la entrada, lo que
introduce la selectividad dentro de la propia recurrence.

### 9.5 Mecanismo relevante

Un modelo de espacio de estados selectivo: los parametros de discretizacion
dependen de la entrada, de modo que la actualizacion del estado es selectiva y
guiada por el contenido, y el coste total sigue siendo lineal en la longitud de
la secuencia. El logro computacional es convertir una interaccion de coste
aparentemente cuadratico en algo lineal, **sin(kernel)**: por identidad
algebraica, no por aproximacion.

### 9.6 Que parte es compatible con TOXIC

Ninguna pieza del modelo. La unica propiedad conceptual transferible es la de
V25, y conviene decirlo con cuidado para no sobrevalorarla: **la idea de bajar a
coste lineal un objeto que parece cuadratico por identidad algebraica y no por
kernel**. Eso es exactamente lo que hace la identidad de la FM en la seccion 1.5
con el sumatorio sobre pares. Es una analogia de propiedad computacional, no de
modelo.

### 9.7 Que parte esta prohibida o no es aplicable

- Es una arquitectura neuronal profunda entrenada desde cero sobre secuencias, con
  barrido secuencial lineal. Nada de eso cabe en 1.483 filas.
- La secuencia de este corpus es un tweet de unos 15 a 25 tokens. La problema que
  Mamba resuelve (secuencias largas, coste cuadratico) **no existe aqui**: a
  esa longitud, la atencion es trivialmente barata y la ventaja de Mamba no tiene
  donde aparecer.
- El proyecto ya midio la ruta secuencial y la cerro. Ver seccion 4.8.

### 9.8 Relacion con V03

Ninguno. V03 es un clasificador lineal sobre una bolsa de n-gramas, que ya es
permutacion invariante y no necesita modelar el orden. Mamba es un modelo
**secuencial**; aqui la secuencia es irrelevante por construccion del modelo
campeon.

### 9.9 Relacion con V25

**NO EJECUTADO.** La busqueda de `mamba`, `state space` y `espacio de estados`
sobre `reports/`, `docs/`, `DECISIONS.md` y `references.bib` devuelve **cero
coincidencias**. Los unicos intentos secuenciales del repositorio son
`reports/v9_04_dl_secuencial_fasttext.md` (embeddings congelados, 0,6943 frente
a 0,6636 del control de media mas regresion lineal) y
`reports/v10_11_bigru_finetuning.md` (0,711639 en el mejor brazo), y ambos
quedan por debajo de V03. No hay nada que implementar, y por el tamano de la
secuencia no hay nada que implementar siquiera utilmente.

---

# Estado de la literatura dentro del proyecto

Esta seccion es la que de verdad importa. Para cada una de las nueve se indica si
el mecanismo **ya fue ejecutado en este repositorio** y con que resultado.
"Ejecutado" significa que hay un script en `experiments/` que lo produce y un
informe en `reports/` con su numero. "No ejecutado" significa que no hay tal
script, y se dice donde se comprobo.

| # | referencia | mecanismo | ejecutado en TOXIC | resultado |
|---|---|---|---|---|
| 1 | Rendle 2010 | FM, interacciones de segundo orden de bajo rango sobre la matriz de features | **NO EJECUTADO** | -- |
| 2 | GraphFM 2025 | grafo de features + GNN sobre pares seleccionados | **NO EJECUTADO** | -- |
| 3 | Huang y Usbeck 2024 | perdida de aprendizaje contrastivo supervisado | **NO EJECUTADO** como objetivo contrastivo | -- |
| 4 | Jarquin Vazquez et al. 2025 | atencion con puerta sobre encoder de transformer | **NO EJECUTADO** | -- |
| 5 | Yadati 2024 | radio de message passing por nodo | **NO EJECUTADO** | -- |
| 6 | Mu, Bhat y Viswanath 2018 | eliminar media y direcciones dominantes | **EJECUTADO Y CERRADO** | `ALL_BUT_THE_TOP_NO_APORTA` |
| 7 | Tang et al. 2014 | embeddings con objetivo dirigido a la tarea | **NO EJECUTADO** | -- |
| 8 | Saroufim et al. 2018 | SSWE con autoetiquetado por emoji | **NO EJECUTADO** | -- |
| 9 | Gu y Dao 2023 | espacio de estados selectivo | **NO EJECUTADO** | -- |

Detalle de cada fila, con los ficheros donde se comprobo.

### 1. Rendle 2010, Factorization Machines: NO EJECUTADO

Es el mecanismo de V25 y es genuinamente nuevo aqui.
`docs/v25_mechanism_audit.md` seccion 1, primera fila de la tabla: "Factorization
Machine | **ninguno** | 0 ocurrencias de `FactorizationMachine`, `xlearn`,
`libffm`, `fastFM` | -- | **si, es V25** | **nuevo**". Las cuatro filas
siguientes confirman que tampoco existe, en ninguna parte del codigo, la
multiplicacion elemento a elemento de la matriz consigo misma, un polinomio de
grado 2, una factorizacion de bajo rango aplicada a la matriz de features, ni
field-aware factorization.
Seccion 3, conclusion de P0, textual: "**Factorization Machines son
genuinamente nuevas en este repositorio.** No existe ninguna implementacion
previa de interacciones de features de segundo orden, ni de factorizacion de bajo
rango sobre la matriz de features, ni de variantes field-aware, ni de las
bibliotecas que las implementan."
Lo que si existe y no es esto, y esta enumerado en la misma seccion 1 para evitar
confusiones: bajo rango **entre etiquetas** (`v8_00_gap_audit.py`,
`v12_01_regularizacion_grupo.py`), ponderacion por columnas
(`30_supervised_weighting_v4.py` linea 300, `matriz.multiply(vector[None, :])`),
norma al cuadrado por columna (`v10_09_correlacion_eliminacion.py` linea 211,
`matriz.multiply(matriz).sum(axis=0)`), pares de **etiquetas**
(`v12_03_entrenamiento_por_pares.py`) y factorizacion condicional de etiquetas
(`v6_02_label_factorization.py`, B1 **-0,0107** con 1 de 5 pliegues).
Contexto adicional del eje equivocado: `reports/v20_00_auditoria_interacciones.md`
documenta que la unica aparicion de productos cruzados por pares del proyecto
esta en `experiments/v9_05_proyecciones_supervisadas.py` lineas 100-101, y son
productos de **proyecciones de texto**, no de features, con veredicto -0,070578
frente a V03.

### 2. GraphFM 2025: NO EJECUTADO

No hay grafo de ningun tipo en el proyecto. La busqueda de `gnn`, `graph
neural`, `red de grafos`, `message passing`, `node embedding` y `grafo` sobre
`reports/`, `docs/`, `DECISIONS.md` y `PROJECT_STATE.md` devuelve **cero
coincidencias**. Por tanto ni la estructura de grafo ni el selector de pares ni
la agregacion tipo GNN existen. El unico trabajo de V25 que toca el eje de las
features sigue siendo la FM simple.

### 3. Huang y Usbeck 2024, SupCon: NO EJECUTADO como objetivo contrastivo

Aqui hay que ser preciso, porque el repositorio si ejecuto algo con nombre
parecido. Lo que **no** existe: un objetivo contrastivo. La sonda estructural de
`experiments/v12_00_auditoria_brecha.py` (artefactos
`reports/v12_auditoria_brecha.json` y `reports/v12_gap_audit.csv`, fila 24)
registra la familia "Entrenamiento contrastivo o por pares sobre TF-IDF disperso"
con estado **GENUINELY_UNTESTED** y el veredicto textual: "**Cero ocurrencias de
`contrastive`, `pairwise`, `margin_loss` o `hinge` como objetivo propio en todo el
codigo.** La sonda AST no encuentra ninguna clase ni ninguna llamada", sobre los
**127 modulos** analizados. El propio informe anade que la perdida hinge si se
probo, pero como clasificador completo (LinearSVC), no como termino auxiliar, y
que el escaner se autovalida con un positivo conocido: la unica aparicion de la
palabra `contrastive` en el proyecto es el propio script de la auditoria
(vease `RUNLOG.md` linea 1427 y `reports/v12_final_research.md` linea 125).
Lo que **si** se ejecuto, y es otra cosa: `experiments/v12_03_entrenamiento_por_pares.py`
con informe `reports/v12_03_entrenamiento_por_pares.md`, un termino de margen
fijo 1,0 sobre **pares de etiquetas** (HS contra TR y AG) dentro de una misma
fila, que acopla los tres clasificadores en la funcion objetivo sin tocar la
prediccion. Resultados: `lambda=0.0` 0,777265 (+0,000053), `lambda=0.10` 0,777187
(-0,000024), `lambda=0.25` 0,774742 (-0,002469), `lambda=0.50` 0,771042
(-0,006169), `lambda=1.00` 0,769274 (-0,007938), `lambda=2.00` 0,766954
(-0,010258): maximo en `lambda=0` y curva monotonamente decreciente. H3 **cerrada**.
Con la aclaracion que hace el propio informe y que `docs/v25_mechanism_audit.md`
seccion 2 repite como la confusion mas probable de esta fase: eso acopla
**etiquetas**, no **features**. Un eje distinto del producto cartesiano.

### 4. Jarquin Vazquez et al. 2025, GAttention: NO EJECUTADO

Ninguna atencion, ningun transformer, ninguna capa con puerta. La busqueda de
`gated attention`, `gattention` y `jarquin` en todo el proyecto devuelve **cero
coincidencias**. Lo mas proximo ejecutado es una arquitectura recurrente en
lugar de atencion: `experiments/v10_11_bigru_finetuning.py` y
`reports/v10_11_bigru_finetuning.md`, con `congelado` 0,693994, `afinado`
0,711639, `afinando_todo` 0,699086, frente a V03 en 0,777211; `afinado menos V03`
= -0,065572; `gana a V03: False`; veredicto "H1 QUEDA CERRADA con dos variantes,
congelada y afinada" y la frase "**la via de representacion secuencial queda
cerrada**". Ademas `docs/v25_mechanism_audit.md` seccion 1 registra la CNN de
caracteres (DL-02, `11_dl_desde_cero.py`) en **0,7410**, tambien cerrada. Y el
precedente de la CNN de caracteres importa para V25: 0,7410 es la medida de lo
que rinde un modelo no lineal con capacidad de sobra frente a un modelo lineal
sobre esta matriz.

### 5. Yadati 2024, GAINER: NO EJECUTADO

Misma comprobacion que la fila 2: cero ocurrencias de grafo, message passing o
radio en `reports/`, `docs/`, `DECISIONS.md` y `PROJECT_STATE.md`. No hay nodos a
los que asignar un radio.

### 6. Mu, Bhat y Viswanath 2018, All-But-The-Top: EJECUTADO Y CERRADO

Es el unico de los nueve que se ha ejecutado de verdad, y con veredicto negativo.
- Script: `experiments/v22_03_all_but_the_top.py`.
- Informes: `reports/v22_03_all_but_the_top.md`,
  `reports/v22_03_all_but_the_top.json`, `reports/v22_03_cierre.md`.
- Control de cordura con `D = 0`: **0,781766** reproducido con desviacion
  **0,000000**, dentro de tolerancia, lo que valida a la vez la carga de
  vectores, el centrado, la SVD y la reconstruccion.
- Donde se aplica: sobre la matriz de los **6.059 vectores de tipo** que el
  pipeline consulta, antes del pooling, que es donde los autores colocan el
  procedimiento. Sin renormalizacion posterior, siguiendo a los autores.
- Resultados frente al control (no frente a V03, que es la comparacion
  correcta aqui):

| D | varianza eliminada | delta frente al control | neto de celdas |
|---:|---:|---:|---:|
| 1 | 0,0531 | **-0,001831** | +1 |
| 2 | 0,0863 | **-0,006121** | -12 |
| 5 | 0,1684 | **-0,003622** | -2 |

- Veredicto escrito: `ALL_BUT_THE_TOP_NO_APORTA`. La hipotesis de que los
  componentes dominantes ocultan senal recuperable no se sostiene en este
  pipeline.
- Contexto de techo: la via del bloque denso esta acotada por
  `reports/v24_02_dictamen_gate.md` seccion 2.4, que llama a FastText "la mejor
  mejora OOF disponible en todo el repositorio, en cualquier rama" con un techo
  de fusion de **+0,0064** y **0 submissions**, y lo convierte con la pendiente
  del modelo de transferencia en **+0,0016 de dLB, o 0,13 errores estandar**.
- En la matriz de `docs/v25_mechanism_audit.md` seccion 1, ultima fila, figura
  como All-But-The-Top con `v22-03`, resultados "D=1 -0.0018, D=2 -0.0061,
  D=5 -0.0036", y la anotacion "**no**: post-procesado de un espacio vectorial,
  no interacciones", estado **cerrado**.
Los antecedentes del bloque denso estan repartidos en `reports/embedding_*.md`
(por ejemplo `reports/embedding_complementarity.md`, que da la correccion neta de
+1 a +6 filas sobre 1.483 en cuatro configuraciones, y
`reports/embedding_phase_closeout_v5.md`).

### 7. Tang et al. 2014, SSWE: NO EJECUTADO, cerrado por dictamen

Nunca se midio. Se esta informado en cuatro sitios, todos con el mismo veredicto:
- `reports/v21_p1_01_embeddings_desde_cero.md` (lineas 171-172 la referencia;
  lineas 196-201 la arquitectura; lineas 927 la legalidad; lineas 1083-1084 la
  lista de referencias). Ficha 2, candidata numero 2 del ranking de la seccion
  3, "Embeddings con objetivo dirigido a la tarea", declarada legal en la
  variante Liu/Tang/Zheng/Kenter.
- `reports/v22_02_dictamen_especialista.md`, Pregunta 2, seccion completa desde
  la linea 155: respuesta directa "**NO merece ejecutarse**", con los cuatro
  hechos, el coste estimado mas alto de V22 (3 a 5 horas), y el riesgo
  declarado como **nuevo y no cubierto** (riesgo de estimacion, no de
  seleccion). Cierre en la linea 337.
- `reports/v22_03_cierre.md` linea 90: "**SSWE: NO merece ejecutarse.** El
  especialista le concedes que puede producir una" [mejora, texto truncado en el
  original], y lineas 95-97 con el argumento de la regularizacion.
- `reports/v21_03_cierre_de_fase.md` linea 13, tabla de la fase: SSWE (objetivo
  dirigido) | **no medida** | nueva | riesgo alto | autorizada **no**.
- `reports/v21_02_dictamen_competitivo.md` lineas 356-360, seccion B "SSWE /
  embeddings con objetivo dirigido a la tarea", primer motivo: "no aporta
  informacion nueva".
Y el precedente que el dictamen usa como analogia mas cercana, y que si se
ejecuto: el NB-SVM, en `reports/v21_p1_03_nbsvm_y_hibridos.md`, con 0,7638
contra 0,7772.

### 8. Saroufim et al. 2018: NO EJECUTADO y nunca mencionado

Cero coincidencias de `saroufim`, `WASSA`, `language independent` e
`independiente del idioma` en `reports/`, `docs/`, `DECISIONS.md`,
`PROJECT_STATE.md` y `references.bib`. No hay script, no hay informe, no hay
cita. Es la referencia de las nueve con menos presencia en el repositorio, y en
concreto no aparece ni en `references.bib`, que tiene 553 lineas y **no contiene
ninguna** de las nueve referencias de este informe. Su unico papel es
documental: es el ejemplo canonico de como se construiria un dataset etiquetado
externo barato, y deja por escrito que ese camino esta cerrado por regla.

### 9. Gu y Dao 2023, Mamba: NO EJECUTADO

Cero coincidencias de `mamba`, `state space` y `espacio de estados` en
`reports/`, `docs/`, `DECISIONS.md` y `references.bib`. Los unicos intentos
neuronales secuenciales del repositorio son `reports/v9_04_dl_secuencial_fasttext.md`
y `reports/v10_11_bigru_finetuning.md`, ambos cerrados y por debajo de V03, y la
CNN de caracteres en 0,7410 segun `docs/v25_mechanism_audit.md`.

### Lectura de conjunto

De las nueve referencias, **siete no fueron ejecutadas nunca** (1, 2, 3, 4, 5, 8,
9) y **una se ejecuto y se cerro en contra** (6). La octava (7) fue analizada y
cerrada por dictamen antes de gastar una hora de GPU en ella. Y la novena
referencia, que es la propia FM, es **el unico mecanismo de esta lista que V25 va
a implementar**, precisamente porque es el unico que la auditoria historica
declara ausente del repositorio.

De las siete no ejecutadas, cuatro (2, 3, 4, 5) viven en territorio prohibido o
estructuralmente ausente: dos requieren grafos que el corpus no tiene (2, 5), dos
requieren transformers que el concurso prohibe (3, 4). Las tres restantes (1, 8,
9) son legales en su principio pero o no aplican a una matriz dispersa (8, 9) o
son justamente la que se va a hacer (1).

El precedente de V22-03 es la leccion operativa de todo esto: el mecanismo mas
barato de los nueve (una funcion determinista de una matriz ya entrenada, sin
supervision, sin datos nuevos, coste de segundos) se ejecuto y dio **-0,0018**
contra su propio control. Si lo mas barato no aporta, el filtro que
`reports/v24_02_dictamen_gate.md` seccion 5 fija para cualquier candidata (perfil
de celdas superior a 200 y `dOOF` estimado superior a +0,0376) es el que hay que
aplicar al resultado de la FM antes de plantearse una submission.

---

## Resumen de la verificacion

| # | referencia | identificador consultado | estado | campos que NO coinciden |
|---|---|---|---|---|
| 1 | Rendle 2010 | DOI 10.1109/ICDM.2010.127 via Crossref | 200, correcto | ninguno del DOI. **arXiv 2010.10593 es otro trabajo** (CMIM, Sylvain et al., ICASSP 2021) y no hay preprint de Rendle en arXiv |
| 2 | Wu et al. 2025, GraphFM | DOI 10.1007/s11633-024-1505-5 via Crossref | 200, correcto | ninguno sustantivo; el "et al." sobra (son 6) y el preprint arXiv:2105.11866 es de 2021 con "Modeling" de una l |
| 3 | Huang y Usbeck 2024 | ACL 2024.emnlp-main.876 y su DOI | 200, correcto | ninguno |
| 4 | Vazquez et al. 2025, GAttention | ACL 2025.findings-emnlp.1105 y su DOI | 200, correcto | paginas 20314-20329 exactas. **Campo autoria**: el apellido es Vazquez y el nombre de pila Horacio Jarquin; lo declarado funde ambos |
| 5 | Yadati 2024, GAINER | ACL 2024.eacl-long.36 y su DOI | 200, correcto | ninguno |
| 6 | Mu et al. 2018, All-But-The-Top | arXiv 1702.01417 | 200 | **Falta un autor: Suma Bhat**. Titulo con otra caja ("All-but-the-Top"). Anio de arXiv 2017 (v2 en 2018) |
| 7 | Tang et al. 2014, SSWE | ACL P14-1146 y su DOI | 200, correcto | ninguno |
| 8 | Saroufim et al. 2018, WASSA | ACL W18-6204 y su DOI | 200, correcto | ninguno (paginas 14-23, no declaradas) |
| 9 | Gu y Dao 2023, Mamba | arXiv 2312.00752 | 200, correcto | ninguno; es preprint, sin `journal_ref` (Semantic Scholar dio 429) |

Resultado agregado: **siete de las nueve coinciden en todos los campos**. Dos
tienen discrepancias. La referencia 1 tiene el identificador de reserva
**invalidado** (apunta a un trabajo de imagen medica de 2021) y no existe
preprint sustituto. La referencia 6 tiene **un autor omitido**, y el mismo error
esta propagado en `reports/v21_p1_01_embeddings_desde_cero.md`, entrada 9. La
referencia 4 tiene una fusion de nombre y apellido en el primer autor. Ninguno
de los tres casos cambia la naturaleza del trabajo, y los tres estan corregidos
en las secciones correspondientes.
