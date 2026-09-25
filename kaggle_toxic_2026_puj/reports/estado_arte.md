# ESTADO DEL ARTE - KAGGLE_TOXIC_2026_PUJ

- **Fecha de la investigacion**: 2026-09-22 (Fase 1).
- **Problema**: clasificacion multi-etiqueta de tweets en espanol en tres etiquetas
  binarias acopladas: HS (discurso de odio), TR (dirigido contra persona o grupo)
  y AG (agresividad).
- **Tipo de datos**: texto corto (longitud mediana 91 caracteres), espanol,
  normalizado a ASCII sin diacriticos, con URLs, menciones y hashtags.
- **Tamano**: 1.731 filas etiquetadas (HS 43.67 %, TR 29.98 %, AG 38.24 %) y
  1.154 filas de test sin etiquetas. **El desbalanceo es moderado, no severo**
  (ninguna clase baja del 25 %), pero TR presenta una dependencia estructural:
  TR=1 solo ocurre cuando HS=1 (519/519), y AG=1 con HS=0 en 26 casos (1.5 %).
- **Metrica objetivo**: F1 Score (oficial de Kaggle). Justificacion: con clases
  entre 29.98 % y 43.67 % la exactitud (accuracy) es una metrica enganosa porque
  una prediccion trivial alcanza ~57-62 %; F1 combina precision y recall y
  penaliza tanto los falsos positivos como los falsos negativos, que en
  moderacion de contenido tienen costes asimetricos. Hipotesis primaria de
  implementacion (D-009): media column-wise de F1 binaria por etiqueta; se
  reportaran ademas macro-F1 sobre los 6 pares (one-vs-rest), micro-F1 y
  exact-match ratio como metricas secundarias. Al no haber desbalanceo severo
  (<10 %), AUC-PR es diagnostico complementario, no metrica principal.

## Metodologia de verificacion de referencias

Toda referencia de este informe fue resuelta en esta sesion (2026-09-22) contra
al menos una fuente primaria. Metodos empleados, en orden de fuerza probatoria:

1. **Resolucion directa de DOI** contra la API REST de Crossref
   (`https://api.crossref.org/works/{doi}`): devuelve titulo, autores, ano y
   venue del registro depositado. Es la verificacion mas fuerte y se aplico a
   todas las referencias con DOI.
2. **Descarga del fichero BibTeX oficial del volumen** en ACL Anthology
   (`https://aclanthology.org/volumes/<vol>.bib`) para las referencias ACL.
3. **Descarga y lectura del PDF** en ACL Anthology o CEUR-WS para confirmar los
   resultados numericos reportados (no se cita ningun numero que no se haya
   leido en el PDF).
4. **API de OpenAlex** (`https://api.openalex.org/works`) como fuente secundaria
   independiente para localizar DOI de trabajos que Crossref no resolvio por
   busqueda bibliografica.
5. **Comprobacion HTTP 200** de URLs no-DOI (CEUR-WS, JMLR, arXiv).

Ninguna referencia de este informe se cita de memoria sin verificacion. Los
numeros marcados como "reportado" provienen del texto o de las tablas del PDF
correspondiente.

---

## A. Marco del problema: el esquema HatEval

### [R01] Basile et al. (2019) - HatEval, SemEval-2019 Task 5
- **Referencia**: Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V.,
  Rangel Pardo, F. M., Rosso, P., Sanguinetti, M. "SemEval-2019 Task 5:
  Multilingual Detection of Hate Speech Against Immigrants and Women in
  Twitter". Proceedings of the 13th International Workshop on Semantic
  Evaluation (SemEval-2019), pp. 54-63.
- **DOI**: 10.18653/v1/S19-2007 | **URL**: https://aclanthology.org/S19-2007/
- **Verificacion**: Crossref /works/10.18653/v1/S19-2007 (OK) + BibTeX del
  volumen S19-2 + PDF leido.
- **Datos**: HatEval, tweets en ingles y espanol contra inmigrantes y mujeres.
- **Tarea**: Subtask A (HS binario); Subtask B (TR: individuo vs grupo, y AG).
- **Resultados reportados**: acuerdo entre anotadores (alpha) para HS, TR, AG de
  0.83, 0.70 y 0.73 en ingles; en espanol el macro-F1 maximo de la Subtask A fue
  0.7300 (empate entre MineriaUNAM y Atalaya) frente al baseline SVC de 0.701;
  en la Subtask B el mejor EMR en espanol fue 0.7050 (equipo CIC-2). El mejor
  sistema global uso un SVM de kernel lineal sobre representacion textual.
- **Idea relevante**: TR y AG **no son etiquetas independientes**: solo se
  anotaron sobre tweets que ya contenian odio. Esto respalda la formulacion
  jerarquica de D-001 y la hipotesis H8.
- **Restriccion**: compatible. Es el paper de organizacion del esquema que sigue
  nuestro corpus; solo aporta estructura de anotacion y baselines.

### [R02] Poletto et al. (2021) - Recursos y corpus de odio
- **Referencia**: Poletto, F., Basile, V., Sanguinetti, M., Bosco, C.
  "Resources and benchmark corpora for hate speech detection: a systematic
  review". Language Resources and Evaluation, 55, 477-523.
- **DOI**: 10.1007/s10579-020-09502-8
- **Verificacion**: Crossref /works/10.1007/s10579-020-09502-8 (OK) + OpenAlex.
- **Contenido**: revision sistematica de corpus de discurso de odio; cataloga
  esquemas de anotacion, idiomas, licencias y problemas de comparabilidad.
- **Idea relevante**: documenta que los esquemas con etiquetas **condicionadas**
  (anotar atributos solo cuando la etiqueta padre esta presente) son practica
  comun, y que la mayoria de corpus no declara licencia ni procedencia. Refuerza
  la exigencia de documentar todo recurso externo (D-007).
- **Restriccion**: compatible (metodologico).

### [R03] Zampieri et al. (2019) - OffensEval / OLID
- **Referencia**: Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra,
  N., Kumar, R. "Predicting the Type and Target of Offensive Posts in Social
  Media". NAACL-HLT 2019, pp. 1415-1420.
- **DOI**: 10.18653/v1/N19-1144 | **URL**: https://aclanthology.org/N19-1144/
- **Verificacion**: Crossref /works/10.18653/v1/N19-1144 (OK) + BibTeX N19-1.
- **Contenido**: define el esquema jerarquico de OffensEval: deteccion de
  ofensa (A), tipo (B) y **target** (C, individuo/grupo/otro).
- **Idea relevante**: la identificacion del target es una tarea
  estructuralmente distinta y mas dificil que la deteccion de ofensa; los
  mejores sistemas usaron TF-IDF con SVM y LSTM. Es el precedente directo para
  modelar TR como cabeza separada.
- **Restriccion**: compatible en estructura; las variantes con embeddings
  preentrenados no son replicables aqui.

---

## B. Sistemas clasicos en espanol que funcionaron SIN transformers

### [R04] Argota Vega et al. (2019) - MineriaUNAM (ganador Subtask A espanol)
- **Referencia**: Argota Vega, L. E., Reyes-Magana, J. C., Gomez-Adorno, H.,
  Bel-Enguix, G. "MineriaUNAM at SemEval-2019 Task 5: Detecting Hate Speech in
  Twitter using Multiple Features in a Combinatorial Framework". SemEval-2019,
  pp. 447-452.
- **DOI**: 10.18653/v1/S19-2079 | **URL**: https://aclanthology.org/S19-2079/
- **Verificacion**: Crossref /works/10.18653/v1/S19-2079 (OK) + PDF descargado
  y leido (tablas 3 y 4 extraidas del PDF).
- **Datos**: HatEval espanol (subtasks A y B).
- **Features exactas (del PDF)**: n-gramas de **caracter**; n-gramas de palabras;
  n-gramas de **palabras funcionales** de 2 a 4 tokens (lista de stopwords de
  NLTK); n-gramas de **signos de puntuacion** de 2 a 4; etiquetas **POS**;
  **patrones morfologicos** (verbo+adjetivo, adjetivo+verbo, sustantivo+adjetivo,
  adjetivo+sustantivo, pronombre+verbo); **series de palabras agresivas**;
  patrones lexicos recurrentes.
- **Representacion**: esquemas de ponderacion (TF-IDF / binario) con umbral de
  frecuencia minima 10 en el corpus y presencia en >=50 documentos.
- **Modelo**: SVM de kernel lineal dentro del framework combinatorio muTC
  (busqueda metaheuristica sobre el espacio de transformaciones y ponderaciones);
  Random Forest con `sklearn.multiclass` one-vs-rest para la Subtask B.
- **Validacion**: validacion cruzada de 10 folds.
- **Resultados reportados**: Subtask A espanol: 0.80 en desarrollo y **0.73
  oficial (puesto 1, empatado)**. Subtask A ingles: 0.384 (puesto 60). Subtask B
  espanol: 0.73 en desarrollo, 0.596 oficial (puesto 16).
- **Idea relevante**: **es posible ganar la Subtask A en espanol sin ninguna red
  neuronal y sin ningun embedding preentrenado**, con ingenieria de features
  linguistica + busqueda combinatoria del espacio de representacion. Es el
  antecedente mas directo y mas importante para este proyecto.
- **Restriccion**: 100 % compatible. Es literalmente el techo de referencia de
  un sistema clasico en este corpus.

### [R05] Nina-Alcocer (2019) - HATERecognizer
- **Referencia**: Nina-Alcocer, V. "HATERecognizer at SemEval-2019 Task 5:
  Using Features and Neural Networks to Face Hate Recognition". SemEval-2019,
  pp. 409-415.
- **DOI**: 10.18653/v1/S19-2072 | **URL**: https://aclanthology.org/S19-2072/
- **Verificacion**: Crossref /works/10.18653/v1/S19-2072 (OK) + PDF leido
  (tablas de ablacion extraidas).
- **Datos**: HatEval espanol (subtasks A y B).
- **Features**: TF-IDF; etiquetas **POS**; **emociones**; features
  **estructurales** (puntuacion, mayusculas, simbolos Unicode).
- **Modelo**: SVM (kernel lineal y RBF). Tambien evaluo GloVe, Google News y
  FastText de 300 dimensiones.
- **Resultados reportados (ablacion, espanol)**: baseline propio 0.766 F1 /
  EMR 0.629; `TFIDF+POS` 0.770 F1 / EMR 0.633; `TFIDF+EMOTIONS` 0.749 F1 /
  **EMR 0.711** (mejor EMR, +0.082 sobre el baseline); `TFIDF+STRUCT` **0.780
  F1** (mejor F1) / EMR 0.648. Las combinaciones de tres bloques **degradaron**
  el rendimiento en la Subtask A.
- **Idea relevante**: (i) las features de **emocion** aportan senal
  complementaria clara para la Subtask B (soporta H5); (ii) las features
  **estructurales** aportan para la Subtask A (soporta H6); (iii) **anadir
  bloques no siempre mejora**: hay que hacer ablacion, no acumular (matiz
  critico para H4).
- **Restriccion**: **parcial**. Las features POS/emocion/estructural + TF-IDF son
  compatibles; los embeddings GloVe/FastText/Google News estan **prohibidos**.

### [R06] Plaza-del-Arco et al. (2019) - SINAI (ensemble)
- **Referencia**: Plaza-del-Arco, F. M., Molina-Gonzalez, M. D., Martin, M.,
  Urena-Lopez, L. A. "SINAI at SemEval-2019 Task 5: Ensemble learning to detect
  hate speech against inmigrants and women in English and Spanish tweets".
  SemEval-2019, pp. 476-479.
- **DOI**: 10.18653/v1/S19-2084 | **URL**: https://aclanthology.org/S19-2084/
- **Verificacion**: Crossref /works/10.18653/v1/S19-2084 (OK) + PDF leido.
- **Modelo**: ensemble por votacion (Vote) de SVM, regresion logistica y arboles
  de decision sobre features estadisticas y lexicas.
- **Resultados reportados**: Subtask A espanol F1 = **0.707** (puesto 14) frente
  al baseline SVC de **0.701**; Subtask B espanol F1 = 0.443 (puesto 40).
- **Idea relevante**: **la votacion ingenua no supera al SVM lineal simple**
  (+0.006). Es evidencia en contra de esperar mejoras automaticas por ensamblar;
  motiva que H10 se plantee como *stacking* con out-of-fold y no como votacion.
- **Restriccion**: compatible (SVM, LR, arboles, TF-IDF).

### [R07] Garibo i Orts (2019) - interpolacion de analisis de frecuencias
- **Referencia**: Garibo i Orts, O. "Multilingual Detection of Hate Speech
  Against Immigrants and Women in Twitter at SemEval-2019 Task 5: Frequency
  Analysis Interpolation for Hate in Speech Detection". SemEval-2019, pp. 460-463.
- **DOI**: 10.18653/v1/S19-2081 | **URL**: https://aclanthology.org/S19-2081/
- **Verificacion**: Crossref /works/10.18653/v1/S19-2081 (OK) + BibTeX S19-2.
- **Modelo**: interpolacion estadistica de analisis de frecuencias, sin redes
  neuronales.
- **Idea relevante**: baseline no neuronal, muy barato, competitivo en el
  leaderboard; util como suelo de comparacion rapido en Fase 9.
- **Restriccion**: compatible.

### [R08] Pereira-Kohatsu et al. (2019) - HaterNet (espanol)
- **Referencia**: Pereira-Kohatsu, J. C., Quijano-Sanchez, L., Liberatore, F.,
  Camacho-Collados, M. "Detecting and Monitoring Hate Speech in Twitter".
  Sensors, 19(21), 4654.
- **DOI**: 10.3390/s19214654 | **URL**: https://www.mdpi.com/1424-8220/19/21/4654
- **Verificacion**: Crossref /works/10.3390/s19214654 (OK).
- **Datos**: dos corpus de Twitter en espanol anotados por expertos en derecho.
- **Features**: pronombres de **segunda persona**, **lexico agresivo**,
  hashtags/menciones, n-gramas de palabras.
- **Modelo**: SVM lineal con features lexicas y reglas; variante BiLSTM/CNN.
- **Resultados reportados**: supera el estado del arte previo en espanol en su
  momento.
- **Idea relevante**: evidencia directa de que el odio en espanol es separable
  con features linguisticas clasicas, y en particular que **segunda persona +
  lexico agresivo** son discriminativas (soporta H2).
- **Restriccion**: parcial si se usa la variante neuronal con embeddings; la
  parte de features lexicas y SVM es compatible.

### [R09] Garcia-Diaz et al. (2022) - estrategias de combinacion de features en espanol
- **Referencia**: Garcia-Diaz, J. A., Jimenez-Zafra, S. M., Garcia-Cumbreras,
  M. A., Valencia-Garcia, R. "Evaluating feature combination strategies for
  hate-speech detection in Spanish using linguistic features and transformers".
  Complex & Intelligent Systems, 8, 4901-4916.
- **DOI**: 10.1007/s40747-022-00693-x
- **Verificacion**: Crossref /works/10.1007/s40747-022-00693-x (OK) + OpenAlex.
- **Contenido**: evalua sistematicamente estrategias de **combinacion** de
  familias de features (POS, dependencias sintacticas, lexicas, n-gramas de
  caracter) con y sin transformers en corpus espanoles.
- **Idea relevante**: es el precedente metodologico mas cercano al plan de
  ablacion de features de este proyecto; confirma que las features linguisticas
  **suman** sobre TF-IDF en espanol, pero que la combinacion debe validarse.
- **Restriccion**: parcial. La rama con transformers esta prohibida; la rama de
  features linguisticas es compatible.

### [R10] Garcia-Diaz et al. (2021) - misoginia en espanol con features linguisticas
- **Referencia**: Garcia-Diaz, J. A., Canovas-Garcia, M., Colomo-Palacios, R.,
  Valencia-Garcia, R. "Detecting misogyny in Spanish tweets. An approach based
  on linguistics features and word embeddings". Future Generation Computer
  Systems, 114, 514-524.
- **DOI**: 10.1016/j.future.2020.08.032
- **Verificacion**: Crossref /works/10.1016/j.future.2020.08.032 (OK) + OpenAlex.
- **Idea relevante**: confirma que POS y dependencias sintacticas aportan senal
  en espanol de redes sociales (soporta H3 y H6).
- **Restriccion**: parcial (los word embeddings estan prohibidos; las features
  linguisticas no).

### [R11] Alvarez-Carmona et al. (2018) - MEX-A3T, IberEval 2018
- **Referencia**: Alvarez-Carmona, M. A., Guzman-Falcon, E., Montes-y-Gomez, M.,
  Escalante, H. J., Villasenor-Pineda, L., Reyes-Meza, V., Rico-Sulayes, A.
  "Overview of MEX-A3T at IberEval 2018: Authorship and Aggressiveness Analysis
  in Mexican Spanish Tweets". IberEval 2018, CEUR-WS Vol-2150, pp. 74-96.
- **URL**: https://ceur-ws.org/Vol-2150/overview-mex-a3t.pdf
- **Verificacion**: HTTP 200 + PDF descargado y leido (primera pagina).
- **Contenido**: primer shared task de **agresividad** en espanol mexicano
  (junto con perfilado de autor). Sistemas participantes basados en TF-IDF,
  n-gramas de caracter y features lexicas con SVM y Random Forest.
- **Idea relevante**: referencia obligada de agresividad en espanol; confirma
  que la tarea de agresividad (equivalente a nuestra AG) es abordable con
  modelos clasicos y n-gramas.
- **Restriccion**: compatible.

### [R12] Aragon et al. (2019) - MEX-A3T, IberLEF 2019
- **Referencia**: Aragon, M. E., Alvarez-Carmona, M. A., Montes-y-Gomez, M.,
  Escalante, H. J., Villasenor-Pineda, L., Moctezuma, D. "Overview of MEX-A3T
  at IberLEF 2019: Authorship and aggressiveness analysis in Mexican Spanish
  tweets". IberLEF 2019, CEUR-WS Vol-2421.
- **URL**: https://ceur-ws.org/Vol-2421/MEX-A3T_overview.pdf
- **Verificacion**: HTTP 200 + PDF descargado y leido (primera pagina).
- **Idea relevante**: segunda edicion del shared task de agresividad en espanol;
  confirma la persistencia del problema y la dificultad de la clase agresiva.
- **Restriccion**: compatible.

### [R13] Frenda et al. (2018) - misoginia en espanol e ingles (AMI, IberEval 2018)
- **Referencia**: Frenda, S., Ghanem, B., Montes-y-Gomez, M., Rosso, P.
  "Exploration of Misogyny in Spanish and English Tweets". IberEval 2018,
  CEUR-WS Vol-2150, pp. 260-267.
- **URL**: https://ceur-ws.org/Vol-2150/AMI_paper6.pdf
- **Verificacion**: HTTP 200 + indice del volumen leido y pagina confirmada.
- **Idea relevante**: analisis linguistico de odio dirigido a un grupo
  especifico en espanol; aporta patrones de **targeting** (soporta H3).
- **Restriccion**: compatible.

---

## C. Features de representacion: n-gramas de caracter

### [R14] Nobata et al. (2016) - features para lenguaje abusivo
- **Referencia**: Nobata, C., Tetreault, J., Thomas, A., Mehdad, Y., Chang, Y.
  "Abusive Language Detection in Online User Content". WWW 2016, pp. 145-153.
- **DOI**: 10.1145/2872427.2883062
- **Verificacion**: Crossref /works/10.1145/2872427.2883062 (OK).
- **Features**: n-gramas de **caracter** (2-4), n-gramas de palabras, POS,
  features de superficie, lexicos.
- **Modelo**: regresion logistica, SVM, CRF.
- **Idea relevante**: **los n-gramas de caracter estan entre las features mas
  informativas** para lenguaje abusivo, porque capturan variacion morfologica y
  evasion ortografica. Es la evidencia fundacional de H1.
- **Restriccion**: compatible.

### [R15] Mishra et al. (2018) - composicion a nivel de caracter
- **Referencia**: Mishra, P., Yannakoudakis, H., Shutova, E. "Neural
  Character-based Composition Models for Abuse Detection". ALW2 (EMNLP 2018),
  pp. 1-10.
- **DOI**: 10.18653/v1/W18-5101 | **URL**: https://aclanthology.org/W18-5101/
- **Verificacion**: Crossref /works/10.18653/v1/W18-5101 (OK) + BibTeX.
- **Idea relevante**: los modelos a nivel de caracter **entrenados desde cero**
  rivalizan con los basados en palabras en deteccion de abuso. Es exactamente la
  situacion de este proyecto (prohibido lo preentrenado): hay evidencia de que
  la representacion de caracter aprendida con los datos de train es viable.
- **Restriccion**: compatible (entrenado desde cero, sin vectores preentrenados).

### [R16] Malmasi & Zampieri (2017) - n-gramas de caracter para odio
- **Referencia**: Malmasi, S., Zampieri, M. "Detecting Hate Speech in Social
  Media". RANLP 2017, pp. 467-472.
- **DOI**: 10.26615/978-954-452-049-6_062 | **URL**: https://aclanthology.org/R17-1062/
- **Verificacion**: Crossref /works/10.26615/978-954-452-049-6_062 (OK) + BibTeX
  del volumen R17-1.
- **Idea relevante**: confirma que representaciones simples de n-gramas con
  modelos lineales son competitivas en deteccion de odio en redes sociales.
- **Restriccion**: compatible.

---

## D. Lexicos de agresividad y de odio

### [R17] Bassignana et al. (2018) - HurtLex
- **Referencia**: Bassignana, E., Basile, V., Patti, V. "Hurtlex: A Multilingual
  Lexicon of Words to Hurt". CLiC-it 2018, CEUR-WS Vol-2253, paper 49.
- **URL**: https://ceur-ws.org/Vol-2253/paper49.pdf | **DOI**:
  10.4000/books.aaccademia.3085
- **Verificacion**: Crossref /works/10.4000/books.aaccademia.3085 (OK) + PDF
  descargado y leido (titulo y abstract confirmados).
- **Contenido**: lexico multilingue de palabras de dano organizado en **17
  categorias** (insulto derogatorio, slur etnico, etc.), construido desde el
  lexico italiano de De Mauro y expandido via MultiWordNet y BabelNet. Cubre
  espanol. Validado en AMI@IberEval2018 como features lexicas por categoria.
- **Idea relevante**: aporta features de lexico de agresividad **por categoria**,
  mas informativas que un unico contador de profanidad. Aporta a H2.
- **Restriccion**: **parcial / requiere documentacion**. Es un recurso externo
  publico; segun D-007 debe registrarse en `reports/lexicon_catalog.csv` con
  nombre, fuente, idioma, licencia, uso y riesgo, y debe mantenerse una variante
  100 % autocontenida (lexico inducido solo con train).

### [R18] Wiegand et al. (2018) - induccion de lexico de palabras abusivas
- **Referencia**: Wiegand, M., Ruppenhofer, J., Schmidt, A., Greenberg, C.
  "Inducing a Lexicon of Abusive Words - a Feature-Based Approach". NAACL-HLT
  2018, pp. 1046-1056.
- **DOI**: 10.18653/v1/N18-1095 | **URL**: https://aclanthology.org/N18-1095/
- **Verificacion**: Crossref /works/10.18653/v1/N18-1095 (OK) + OpenAlex.
- **Idea relevante**: demuestra que se puede **inducir un lexico de abuso desde
  los propios datos** usando POS, sintaxis y n-gramas como features de
  clasificacion de palabras. Esto resuelve elegantemente la restriccion de no
  usar recursos externos: el lexico se aprende de train (soporta H2 y la
  variante autocontenida exigida por D-007).
- **Restriccion**: compatible.

### [R19] Wiegand et al. (2019) - el problema de los datasets sesgados
- **Referencia**: Wiegand, M., Ruppenhofer, J., Kleinbauer, T. "Detection of
  Abusive Language: the Problem of Biased Datasets". NAACL-HLT 2019, pp. 581-588.
- **DOI**: 10.18653/v1/N19-1060 | **URL**: https://aclanthology.org/N19-1060/
- **Verificacion**: Crossref /works/10.18653/v1/N19-1060 (OK).
- **Idea relevante**: el rendimiento cae drasticamente en evaluacion
  cross-dominio; un unico numero de validacion cruzada en dominio sobreestima la
  capacidad real. Motiva validacion agrupada honesta (D-004) y control de
  robustez.
- **Restriccion**: compatible (metodologico).

### [R20] Schmidt & Wiegand (2017) - survey de deteccion de odio con PLN
- **Referencia**: Schmidt, A., Wiegand, M. "A Survey on Hate Speech Detection
  using Natural Language Processing". SocialNLP 2017, pp. 1-10.
- **DOI**: 10.18653/v1/W17-1101 | **URL**: https://aclanthology.org/W17-1101/
- **Verificacion**: Crossref /works/10.18653/v1/W17-1101 (OK) + BibTeX W17-11.
- **Idea relevante**: mapa temprano de familias de features (n-gramas, lexicos,
  POS, meta-informacion); senala la escasez de datos anotados como cuello de
  botella principal.
- **Restriccion**: compatible.

### [R21] Fortuna & Nunes (2018) - survey de deteccion automatica de odio
- **Referencia**: Fortuna, P., Nunes, S. "A Survey on Automatic Detection of
  Hate Speech in Text". ACM Computing Surveys, 51(4), Article 85.
- **DOI**: 10.1145/3232676
- **Verificacion**: Crossref /works/10.1145/3232676 (OK).
- **Idea relevante**: taxonomia de features y modelos; identifica el
  **desbalanceo de clases** y la **falta de metricas estandar** como los dos
  problemas abiertos principales. Justifica reportar varias metricas y tratar el
  desbalanceo explicitamente.
- **Restriccion**: compatible.

### [R22] Jahan & Oussalah (2023) - revision sistematica
- **Referencia**: Jahan, M. S., Oussalah, M. "A systematic review of hate speech
  automatic detection using natural language processing". Neurocomputing, 546,
  126232.
- **DOI**: 10.1016/j.neucom.2023.126232
- **Verificacion**: Crossref /works/10.1016/j.neucom.2023.126232 (OK) + OpenAlex.
- **Idea relevante**: revision reciente; confirma que TF-IDF con modelos
  clasicos sigue siendo competitivo en escenarios de bajos recursos, y que los
  transformers dominan cuando hay datos y computo.
- **Restriccion**: compatible (aporta el encuadre general).

---

## E. Targeting y analisis linguistico del odio

### [R23] ElSherief et al. (2018) - Hate Lingo
- **Referencia**: ElSherief, M., Kulkarni, V., Nguyen, D., Wang, W. Y., Belding,
  E. "Hate Lingo: A Target-Based Linguistic Analysis of Hate Speech in Social
  Media". ICWSM 2018, 12(1), 42-51.
- **DOI**: 10.1609/icwsm.v12i1.15041
- **Verificacion**: Crossref /works/10.1609/icwsm.v12i1.15041 (OK) + OpenAlex.
- **Contenido**: analisis linguistico comparativo entre odio explicito, odio
  implicito y lenguaje ofensivo no odioso, condicionado por el **target**.
- **Idea relevante**: el odio se distingue por marcadores de **"othering"**
  (construccion de un "ellos" frente a un "nosotros"): primera persona plural,
  tercera persona, y lenguaje de polaridad. Es la base teorica de H3 (features
  de targeting) y de la separacion HS vs AG.
- **Restriccion**: compatible (solo aporta diseno de features; si se usaran
  categorias tipo LIWC habria que documentarlas como recurso externo).

### [R24] ElSherief et al. (2021) - Latent Hatred (odio implicito)
- **Referencia**: ElSherief, M., Ziems, C., Muchlinski, D., Gupta, V., Kambhatla,
  S., Yang, J. "Latent Hatred: A Benchmark for Understanding Implicit Hate
  Speech". EMNLP 2021, pp. 345-363.
- **DOI**: 10.18653/v1/2021.emnlp-main.29 | **URL**: https://aclanthology.org/2021.emnlp-main.29/
- **Verificacion**: Crossref /works/10.18653/v1/2021.emnlp-main.29 (OK) + OpenAlex.
- **Idea relevante**: cuantifica que el odio **implicito** es mucho mas dificil
  que el explicito y que los modelos pierden decenas de puntos de F1. Fija la
  expectativa de que parte de la senal de HS/TR **no es recuperable con features
  de superficie**, y que el techo de este proyecto sera menor que el de un
  transformer.
- **Restriccion**: el **modelo** (transformer) esta prohibido; se aprovecha solo
  el hallazgo y la caracterizacion de errores esperados.

### [R25] Basile et al. (2019) - definicion de TR (ver R01)
- Ver R01. En la Subtask B, HatEval define TR como el **rango del target**
  (0 = grupo, 1 = individuo), lo que difiere de la definicion oficial de la
  competencia ("dirigido contra una persona o grupo"). Esta discrepancia es el
  objeto de la decision D-005 y debe resolverse empiricamente en Fase 2, no por
  asuncion.

---

## F. Estilometria, POS y sintaxis

### [R26] Stamatatos (2009) - survey de atribucion de autoria
- **Referencia**: Stamatatos, E. "A survey of modern authorship attribution
  methods". Journal of the American Society for Information Science and
  Technology, 60(3), 538-556.
- **DOI**: 10.1002/asi.21001
- **Verificacion**: Crossref /works/10.1002/asi.21001 (OK; el registro Crossref
  fecha el articulo en 2008 por publicacion online, numero de revista 2009).
- **Idea relevante**: en textos cortos y ruidosos, los **n-gramas de caracter**
  y las **palabras funcionales** son las features mas robustas de estilo, y la
  lematizacion/steaming **a menudo no aporta**. Es la evidencia principal de H1,
  H6 y H7, y coincide con las features que MineriaUNAM (R04) uso para ganar.
- **Restriccion**: compatible.

### [R27] Stamatatos (2017) - distorsion de texto para autoria
- **Referencia**: Stamatatos, E. "Authorship Attribution Using Text Distortion".
  EACL 2017, pp. 1138-1149.
- **DOI**: 10.18653/v1/E17-1107 | **URL**: https://aclanthology.org/E17-1107/
- **Verificacion**: Crossref /works/10.18653/v1/E17-1107 (OK) + OpenAlex.
- **Idea relevante**: enmascarar el contenido tematico (conservando la forma)
  mejora la senal de estilo. Sugiere una familia de features para H6:
  n-gramas de caracter sobre texto con las palabras de contenido enmascaradas,
  que capturan forma sin ruido tematico.
- **Restriccion**: compatible.

### [R28] Joachims (1998) - SVM lineal para categorizacion de texto
- **Referencia**: Joachims, T. "Text Categorization with Support Vector
  Machines: Learning with Many Relevant Features". ECML 1998, LNCS 1398,
  pp. 137-142.
- **DOI**: 10.1007/BFb0026683
- **Verificacion**: Crossref /works/10.1007/BFb0026683 (OK).
- **Idea relevante**: el SVM lineal con TF-IDF es un clasificador de texto
  fuerte, rapido y robusto en espacios dispersos de muy alta dimension, incluso
  cuando hay muchas features irrelevantes. Justifica que el baseline principal
  sea SVM lineal, igual que en los mejores sistemas de HatEval (R01, R04).
- **Restriccion**: compatible.

---

## G. Sentimiento y emocion como senal auxiliar

### [R29] Plaza-del-Arco et al. (2021) - multi-tarea con analisis de sentimiento
- **Referencia**: Plaza-del-Arco, F. M., Molina-Gonzalez, M. D., Urena-Lopez,
  L. A., Martin-Valdivia, M. T. "A Multi-Task Learning Approach to Hate Speech
  Detection Leveraging Sentiment Analysis". IEEE Access, 9, 112478-112488.
- **DOI**: 10.1109/ACCESS.2021.3103697
- **Verificacion**: Crossref /works/10.1109/ACCESS.2021.3103697 (OK) + OpenAlex.
- **Idea relevante**: modelar **sentimiento y odio conjuntamente** mejora la
  deteccion de odio en espanol. Es la evidencia mas fuerte de H5: la polaridad y
  la emocion son senal complementaria, no redundante. En este proyecto se
  traslada a features de lexico de sentimiento/emocion, ya que la arquitectura
  multi-tarea neuronal no es viable sin embeddings.
- **Restriccion**: parcial (el modelo multi-tarea con embeddings esta prohibido;
  la hipotesis subyacente es testeable con features de lexico).

### [R30] Nina-Alcocer (2019) - features de emocion (ver R05)
- Ver R05. `TFIDF+EMOTIONS` fue la mejor configuracion en EMR (0.711 frente a
  0.629 del baseline) en la Subtask B espanola. Segunda evidencia de H5, y con
  un metodo **totalmente compatible** con las restricciones.

### [R31] Plaza-del-Arco et al. (2021) - el techo de los transformers en espanol
- **Referencia**: Plaza-del-Arco, F. M., Molina-Gonzalez, M. D., Urena-Lopez,
  L. A., Martin-Valdivia, M. T. "Comparing pre-trained language models for
  Spanish hate speech detection". Expert Systems with Applications, 166, 114120.
- **DOI**: 10.1016/j.eswa.2020.114120
- **Verificacion**: Crossref /works/10.1016/j.eswa.2020.114120 (OK) + OpenAlex.
- **Idea relevante**: **calibra la expectativa**. Los modelos de lenguaje
  preentrenados en espanol (familia BETO) alcanzan el techo de rendimiento
  reportado. Al estar prohibidos aqui, el objetivo realista es acercarse al
  rango de los mejores sistemas clasicos de HatEval (~0.73 macro-F1 en espanol,
  R01/R04), no al techo de los transformers. Citar esto en el notebook es
  honestidad metodologica: explica por que no se persigue un F1 mas alto.
- **Restriccion**: **prohibido** como metodo (usa BETO/transformers); se cita
  solo como referencia de cota superior y justificacion.

---

## H. Desbalanceo, umbral de decision y metricas

### [R32] Lipton et al. (2014) - umbral optimo para maximizar F1
- **Referencia**: Lipton, Z. C., Elkan, C., Naryanaswamy, B. "Optimal
  Thresholding of Classifiers to Maximize F1 Measure". ECML PKDD 2014, LNCS
  8725, pp. 225-239. Preprint: arXiv:1402.1892.
- **DOI**: 10.1007/978-3-662-44851-9_15 | **arXiv**: 1402.1892
- **Verificacion**: Crossref /works/10.1007/978-3-662-44851-9_15 (OK) +
  arXiv API `id_list=1402.1892` (titulo confirmado).
- **Idea relevante**: el umbral que maximiza F1 se puede calcular de forma
  **exacta y eficiente** a partir de la curva de rendimiento del clasificador;
  0.5 casi nunca es optimo bajo desbalanceo. Es la base directa de H9: optimizar
  un umbral por etiqueta sobre validacion (nunca sobre test).
- **Restriccion**: compatible.

### [R33] Nan et al. (2012) - optimizacion de F-measure
- **Referencia**: Nan, Y., Chai, K. M. A., Lee, W. S., Chieu, H. L. "Optimizing
  F-measure: A Tale of Two Approaches". ICML 2012. Preprint: arXiv:1206.4625.
- **arXiv**: 1206.4625 | **DOI**: 10.48550/arXiv.1206.4625
- **Verificacion**: OpenAlex (DOI resuelto) + arXiv id localizado.
- **Idea relevante**: compara el enfoque de **umbral** frente al enfoque
  plug-in para optimizar F; aporta la justificacion de elegir umbral por su
  robustez y bajo coste computacional en un escenario de 3 clasificadores
  binarios independientes.
- **Restriccion**: compatible.

### [R34] Buda et al. (2018) - estudio sistematico del desbalanceo de clases
- **Referencia**: Buda, M., Maki, A., Mazurowski, M. A. "A systematic study of
  the class imbalance problem in convolutional neural networks". Neural
  Networks, 106, 249-259.
- **DOI**: 10.1016/j.neunet.2018.07.011
- **Verificacion**: Crossref /works/10.1016/j.neunet.2018.07.011 (OK).
- **Idea relevante**: estudio sistematico que concluye que el
  **sobremuestreo** ayuda y el **submuestreo** puede perjudicar, y que el efecto
  depende del grado de desbalanceo. Con el desbalanceo moderado de este corpus
  (29.98-43.67 %) la recomendacion es **class weights** antes que remuestreo
  sintetico (SMOTE), reservando SMOTE para una ablacion controlada.
- **Restriccion**: compatible (metodologico).

### [R35] Founta et al. (2018) - caracterizacion de abuso a gran escala
- **Referencia**: Founta, A.-M., Djouvas, C., Chatzakou, D., Leontiadis, I.,
  Blackburn, J., Stringhini, G., Vakali, A., Sirivianos, M., Kourtellis, N.
  "Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior".
  ICWSM 2018, 12(1), 491-500.
- **DOI**: 10.1609/icwsm.v12i1.14991
- **Verificacion**: Crossref /works/10.1609/icwsm.v12i1.14991 (OK) + OpenAlex.
- **Idea relevante**: cuantifica el desbalanceo realista del abuso en Twitter
  (el abuso efectivo es una fraccion pequena del total) y caracteriza las clases.
  Es el contrapunto: nuestro corpus tiene un desbalanceo **mucho mas benigno**
  (TR 29.98 %), lo que sugiere que las tecnicas agresivas de rebalanceo no son
  la palanca principal aqui; las features si.
- **Restriccion**: compatible.

### [R36] MacAvaney et al. (2019) - retos de la deteccion de odio
- **Referencia**: MacAvaney, S., Yao, H.-R., Yang, E., Russell, K., Goharian,
  N., Frieder, O. "Hate speech detection: Challenges and solutions". PLOS ONE,
  14(8), e0221152.
- **DOI**: 10.1371/journal.pone.0221152
- **Verificacion**: Crossref /works/10.1371/journal.pone.0221152 (OK) + OpenAlex.
- **Idea relevante**: identifica desbalanceo de clases, ambiguedad de etiquetas
  y sesgo de dominio como los retos centrales, y propone tratarlos
  explicitamente. Aporta el argumento de que TR (definicion ambigua, D-005) debe
  caracterizarse empiricamente antes de optimizarla.
- **Restriccion**: compatible.

### [R37] Vidgen & Derczynski (2020) - calidad de los datos de entrenamiento
- **Referencia**: Vidgen, B., Derczynski, L. "Directions in abusive language
  training data, a systematic review: Garbage in, garbage out". PLOS ONE, 15(12),
  e0243300.
- **DOI**: 10.1371/journal.pone.0243300
- **Verificacion**: Crossref /works/10.1371/journal.pone.0243300 (OK) + OpenAlex.
- **Idea relevante**: la mayoria de los corpus de abuso tienen anotacion debil y
  procedencia no documentada; el rendimiento esta limitado por los datos, no por
  el modelo. Refuerza la trazabilidad de datos y de lexicones (D-002, D-007) y
  modera las expectativas sobre ganancias de modelado.
- **Restriccion**: compatible.

### [R38] Davidson et al. (2017) - odio frente a lenguaje ofensivo
- **Referencia**: Davidson, T., Warmsley, D., Macy, M., Weber, I. "Automated
  Hate Speech Detection and the Problem of Offensive Language". ICWSM 2017,
  11(1), 512-515.
- **DOI**: 10.1609/icwsm.v11i1.14955
- **Verificacion**: Crossref /works/10.1609/icwsm.v11i1.14955 (OK).
- **Idea relevante**: (i) **solo una fraccion pequena de lo marcado como odio lo
  es realmente**; la confusion entre "ofensivo" y "odio" es el error dominante,
  que es exactamente la frontera HS frente a AG en nuestro corpus (26 casos con
  AG=1 y HS=0). (ii) Los n-gramas de caracter (2-4) fueron features centrales.
  Aporta a H1 y a la calibracion de la frontera HS/AG.
- **Restriccion**: compatible.

### [R39] Davidson et al. (2019) - sesgo racial en los datasets
- **Referencia**: Davidson, T., Bhattacharya, D., Weber, I. "Racial Bias in Hate
  Speech and Abusive Language Detection Datasets". ALW3 2019, pp. 25-35.
- **DOI**: 10.18653/v1/W19-3504 | **URL**: https://aclanthology.org/W19-3504/
- **Verificacion**: Crossref /works/10.18653/v1/W19-3504 (OK) + OpenAlex.
- **Idea relevante**: las etiquetas dependen de quien anota; el corpus codifica
  sesgo del anotador. Es un limite irreducible de la metrica: parte del error de
  validacion no es reducible por el modelo.
- **Restriccion**: compatible.

---

## I. Clasificacion jerarquica y multi-etiqueta

### [R40] Silla & Freitas (2011) - survey de clasificacion jerarquica
- **Referencia**: Silla, C. N., Freitas, A. A. "A survey of hierarchical
  classification across different application domains". Data Mining and Knowledge
  Discovery, 22, 31-72.
- **DOI**: 10.1007/s10618-010-0175-9
- **Verificacion**: Crossref /works/10.1007/s10618-010-0175-9 (OK).
- **Idea relevante**: la clasificacion jerarquica con **restricciones de
  consistencia** (una clase hija no puede activarse si la padre esta inactiva)
  suele mejorar el rendimiento frente a tratar las etiquetas como
  independientes. Es el sustento teorico directo de H8 (post-proceso
  `HS=0 => TR=0, AG=0`).
- **Restriccion**: compatible.

### [R41] Fortuna et al. (2019) - dataset de odio etiquetado jerarquicamente
- **Referencia**: Fortuna, P., Rocha da Silva, J., Soler-Company, J., Wanner, L.,
  Nunes, S. "A Hierarchically-Labeled Portuguese Hate Speech Dataset". ALW3
  2019, pp. 94-104.
- **DOI**: 10.18653/v1/W19-3510 | **URL**: https://aclanthology.org/W19-3510/
- **Verificacion**: Crossref /works/10.18653/v1/W19-3510 (OK) + BibTeX W19-35.
- **Idea relevante**: precedente de corpus donde las etiquetas se organizan en
  **jerarquia** (grupo objetivo -> subcategoria). Confirma que la estructura
  jerarquica es una decision de diseno de anotacion legitima y frecuente, lo que
  respalda D-001 (formulacion multi-etiqueta acoplada, no multiclase).
- **Restriccion**: compatible.

---

## J. Ensambles y combinacion de modelos

### [R42] Wolpert (1992) - generalizacion apilada (stacking)
- **Referencia**: Wolpert, D. H. "Stacked generalization". Neural Networks, 5(2),
  241-259.
- **DOI**: 10.1016/S0893-6080(05)80023-1
- **Verificacion**: Crossref /works/10.1016/S0893-6080(05)80023-1 (OK).
- **Idea relevante**: un meta-aprendiz de segundo nivel entrenado sobre
  predicciones **out-of-fold** supera a la combinacion por votacion simple. Es
  la formulacion correcta de H10, y la respuesta directa a la evidencia negativa
  de SINAI (R06), cuyo ensemble por votacion apenas igualo al SVM lineal.
- **Restriccion**: compatible.

### [R43] Plaza-del-Arco et al. (2019) - ensemble por votacion (ver R06)
- Ver R06. Evidencia **negativa** sobre votacion ingenua en este corpus exacto.

---

## K. Referencias metodologicas y de herramientas

### [R44] Pedregosa et al. (2011) - scikit-learn
- **Referencia**: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V.,
  Thirion, B., Grisel, O., et al. "Scikit-learn: Machine Learning in Python".
  Journal of Machine Learning Research, 12, 2825-2830.
- **URL**: https://jmlr.org/papers/v12/pedregosa11a.html
- **Verificacion**: HTTP 200 en jmlr.org (2026-09-22).
- **Idea relevante**: implementacion de referencia de TF-IDF, SVM lineal,
  regresion logistica, validacion cruzada y metricas; y del modulo
  `sklearn.multiclass` para one-vs-rest, que MineriaUNAM uso (R04).
- **Restriccion**: compatible.

### [R45] Arango et al. (2019) - la validacion en deteccion de odio es fragil
- **Referencia**: Arango, A., Perez, J., Poblete, B. "Hate Speech Detection is
  Not as Easy as You May Think: A Closer Look at Model Validation". SIGIR 2019,
  pp. 45-54.
- **DOI**: 10.1145/3331184.3331262
- **Verificacion**: Crossref /works/10.1145/3331184.3331262 (OK) + OpenAlex.
- **Idea relevante**: demuestra que los **splits aleatorios filtran
  near-duplicados** y sobreestiman el F1 en deteccion de odio. Es la evidencia
  externa que valida la decision D-004 de usar `StratifiedGroupKFold` con grupos
  de near-duplicados en este proyecto.
- **Restriccion**: compatible (metodologico).

### [R46] Rottger et al. (2021) - HateCheck
- **Referencia**: Rottger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H.,
  Pierrehumbert, J. "HateCheck: Functional Tests for Hate Speech Detection
  Models". ACL-IJCNLP 2021, pp. 41-58.
- **DOI**: 10.18653/v1/2021.acl-long.4 | **URL**: https://aclanthology.org/2021.acl-long.4/
- **Verificacion**: Crossref /works/10.18653/v1/2021.acl-long.4 (OK) + BibTeX
  2021.acl-long.
- **Idea relevante**: un F1 agregado oculta fallos sistematicos (negacion,
  menciones no odiosas, insultos reapropiados). Aporta la justificacion de hacer
  **analisis de errores cualitativo** por celda de la matriz de confusion y no
  solo perseguir el numero de F1. Relevante porque el 30 % de la nota depende de
  la calidad del notebook.
- **Restriccion**: compatible (metodologico).

### [R47] Vidgen et al. (2019) - retos y fronteras
- **Referencia**: Vidgen, B., Harris, A., Nguyen, D., Tromble, R., Hale, S.,
  Margetts, H. "Challenges and frontiers in abusive content detection". ALW3
  2019, pp. 80-93.
- **DOI**: 10.18653/v1/W19-3509 | **URL**: https://aclanthology.org/W19-3509/
- **Verificacion**: Crossref /works/10.18653/v1/W19-3509 (OK) + BibTeX W19-35.
- **Idea relevante**: llama a reportar incertidumbre y a no reducir la
  evaluacion a una sola cifra. Refuerza el reporte multi-metrica de D-009.
- **Restriccion**: compatible.

---

## L. Referencias adicionales

### [R48] Wiegand et al. (2018) - GermEval 2018, identificacion de lenguaje ofensivo
- **Referencia**: Wiegand, M., Siegel, M., Ruppenhofer, J. "Overview of the
  GermEval 2018 Shared Task on the Identification of Offensive Language".
  Proceedings of the GermEval 2018 Workshop, Vienna.
- **URN**: nbn:de:bsz:mh39-84935 | **URL**:
  https://ids-pub.bsz-bw.de/frontdoor/index/index/docId/8493
- **Verificacion**: HTTP 200 en `ids-pub.bsz-bw.de` y en `nbn-resolving.org`
  (2026-09-22); registro localizado via OpenAlex (W3013027210). **Sin DOI**: es
  una publicacion de proceedings sin deposito DOI, verificada por URN
  persistente y pagina del repositorio institucional.
- **Contenido**: shared task aleman de identificacion de lenguaje ofensivo con
  dos subtareas: (1) ofensivo/no ofensivo y (2) distincion entre ofensa
  explicita, ofensa implicita y profanidad no dirigida. Los mejores sistemas
  usaron SVM y regresion logistica sobre n-gramas de palabra y de caracter.
- **Idea relevante**: la subtarea (2) es estructuralmente analoga a nuestra
  separacion HS frente a AG: distingue **ofensa explicita** (nuestra AG) de
  **ofensa implicita / no dirigida** (nuestra HS sin agresividad manifiesta).
  Aporta a la calibracion de la frontera HS/AG y a H2.
- **Restriccion**: compatible (SVM/LR sobre n-gramas, sin transformers).

---

## Tecnicas derivadas del estado del arte (que aplica este proyecto)

| Tecnica | Fuente | Aplicacion en este proyecto | Estado |
|---|---|---|---|
| SVM lineal sobre TF-IDF (word + char) como baseline fuerte | R01, R04, R16, R28 | Modelo base de E01 y E04 | Compatible |
| N-gramas de caracter 3-6 (`char_wb`) | R14, R15, R26, R38 | Familia de features de H1; representacion principal de E01 | Compatible |
| N-gramas de palabras funcionales 2-4 (stopwords NLTK) | R04, R26 | Feature de H6 (estilometria funcional) | Compatible |
| N-gramas de signos de puntuacion 2-4 | R04, R05 | Feature de H6 (estilometria de superficie) | Compatible |
| POS / morfologia (patrones verbo+adj, pron+verbo) | R04, R09, R10 | Familia de H3 y H6 | Compatible (spaCy solo como anotador, D-003) |
| Lexico de agresividad por categorias | R17, R18, R19 | Familia de H2; version autocontenida inducida de train (D-007) | Parcial, documentar |
| Features de emocion / sentimiento | R05, R29 | Familia de H5 | Compatible (via lexico, no via embeddings) |
| Features de targeting ("othering", 2a persona, 3a persona) | R08, R23, R13 | Familia de H3 | Compatible |
| Post-proceso jerarquico de consistencia | R40, R41 | H8 / E08 | Compatible |
| Umbral optimo por etiqueta (no 0.5) | R32, R33 | H9 / E09 | Compatible |
| `class_weight` antes que SMOTE (desbalanceo moderado) | R34, R35 | Fase 4 (rebalanceo) | Compatible |
| Stacking con predicciones out-of-fold | R42 | H10 / E10 | Compatible |
| Votacion ingenua de ensambles | R06 | **Descartada**: no supera al SVM lineal | Compatible pero sin ganancia |
| Validacion agrupada por near-duplicados | R45, R19 | D-004 (ya decidido) | Compatible |
| Analisis de errores por celda y tests funcionales | R46, R47 | Fase 10 (analisis de errores) | Compatible |
| Redes neuronales con embeddings preentrenados (BETO, GloVe, FastText) | R05, R24, R31 | **Prohibido** | Prohibido |

### Tecnicas NO aplicables y por que

- **Transformers / modelos de lenguaje preentrenados** (BETO, RoBERTa, mBERT):
  prohibidos por las reglas de la competencia. R31 y R24 se citan unicamente
  como cota superior y para calibrar expectativas.
- **Embeddings preentrenados** (GloVe, word2vec, FastText, Google News, sentence
  embeddings): prohibidos. R05 los evaluo y estan explicitamente vetados aqui.
- **Corpus o etiquetas externas del esquema HatEval**: prohibido por D-002. R01
  se cita por su esquema de anotacion, nunca para recuperar etiquetas.
- **Lexicos externos** (HurtLex, R17): permitidos **solo** si se documentan en
  `reports/lexicon_catalog.csv` y se mantiene una variante 100 % autocontenida
  (D-007). La ruta preferente es inducir el lexico desde train (R18).

---

## Conclusiones de la investigacion

1. **El techo realista esta bien documentado**: sin transformers, el mejor
   sistema en espanol sobre este esquema de anotacion obtuvo **macro-F1 0.73**
   (R01, R04) con features linguisticas + SVM lineal, y el baseline SVC oficial
   fue 0.701. Ese es el objetivo de referencia, no el ~0.8 de los transformers
   (R31).
2. **La restriccion no es un impedimento grave**: el sistema que gano la Subtask
   A en espanol (R04) **no uso redes neuronales ni embeddings preentrenados**.
   Uso exactamente el tipo de features que este proyecto planea: char n-gramas,
   n-gramas de palabras funcionales, n-gramas de puntuacion, POS, patrones
   morfologicos y series de palabras agresivas.
3. **Las tres familias de features con mejor respaldo son**: n-gramas de
   caracter (H1), lexico de agresividad + segunda persona (H2) y features de
   targeting (H3). Las de respaldo intermedio son emocion/sentimiento (H5) y
   estilometria de superficie (H6).
4. **La acumulacion ciega de features no funciona**: Nina-Alcocer (R05) mostro
   que las combinaciones de tres bloques **degradaron** la Subtask A, y SINAI
   (R06) mostro que la votacion no supero al SVM lineal. Toda adicion debe pasar
   por una ablacion controlada con la misma validacion.
5. **Los dos riesgos metodologicos principales** son (a) la fuga por
   near-duplicados, que infla el F1 local (R45, ya mitigado por D-004) y (b) la
   ambiguedad de la etiqueta TR (D-005), que debe caracterizarse empiricamente
   antes de optimizarla (R36).
6. **La palanca de mejora mas segura y barata** no es el modelo sino el
   **umbral de decision**: con tres clasificadores binarios y clases entre 30 % y
   44 %, el umbral 0.5 no es optimo y el optimo se calcula de forma exacta
   (R32, R33). Es la hipotesis H9, de bajo riesgo y alto retorno esperado.
