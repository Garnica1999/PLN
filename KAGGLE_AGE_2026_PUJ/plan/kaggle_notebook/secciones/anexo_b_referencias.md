[DOC]

## Anexo B. Referencias del estado del arte

**Criterio de inclusión.** Solo se listan referencias que aparecen en el repositorio: la sección 54 del notebook `99_entrega_consolidado.ipynb` (bibliografía SOTA formal del proyecto), la bibliografía verificada `references.bib` (52 fichas: la corriente de dominio 2006-2021 y las 22 fichas de métodos SOTA incorporadas en el cierre de 2026, volcadas en `reports/state_of_art.csv`), las citas inline de `DECISIONS.md`, `RUNLOG.md`, `FINAL_REPORT.md`, `CHAMPION.md` y docstrings de `scripts/`, y los informes de `reports/` (incluidas la adenda de cierre SOTA de octubre de 2026 en `age_literature_review_2026.md` y el registro de métodos `age_literature_methods_registry.csv`). El DOI/URL se incluye solo cuando está en el repositorio; si no está, se marca `[VERIFICAR]`.

### B.1 Perfilado de autor y edad en texto (corriente de dominio)

Referencias que guiaron el diseño (según `FINAL_REPORT.md` §4 y `references.bib`), con su aporte concreto:

1. **Schler, J., Koppel, M., Argamon, S. y Pennebaker, J. W. (2006).** Effects of Age and Gender on Blogging. Proceedings of the AAAI Spring Symposium on Computational Approaches to Analyzing Weblogs (SS-06-03). URL: http://u.cs.biu.ac.il/~koppel/papers/springsymp-blogs-07.10.05-final.pdf — Uso: esquema de 6 rangos idéntico al de la competencia y bloque de features baratas (pronombres, palabras funcionales).
2. **Argamon, S., Koppel, M., Pennebaker, J. W. y Schler, J. (2007).** Mining the Blogosphere: Age, gender and the varieties of self-expression. First Monday, 12(9). DOI: 10.5210/fm.v12i9.2003 — Uso: discurso "outer-directed" vs "inner-directed" como marco del bloque de estilo.
3. **Mukherjee, A. y Liu, B. (2010).** Improving Gender Classification of Blog Authors. EMNLP 2010. URL: https://aclanthology.org/D10-1021/ — Uso: normalización weighted log-count ratio, base del NB-SVM del proyecto.
4. **Nguyen, D., Smith, N. A. y Rosé, C. P. (2011).** Author Age Prediction from Text using Linear Regression. LaTeCH@ACL-HLT 2011. URL: https://dongnguyen.nl/publications/nguyen-latech2011.pdf — Uso: regresión ordinal auxiliar sobre el punto medio del rango (E-015).
5. **Nguyen, D., Gravel, R., Trieschnigg, D. y Meder, T. (2013).** How Old Do You Think I Am? A Study of Language and Age in Twitter. ICWSM 2013. URL: https://dongnguyen.nl/publications/nguyen-icwsm2013.pdf — Uso: catálogo de features de Twitter (RT, URL, hashtag, alargamiento) y prioridad de los n-gramas de carácter.
6. **Nguyen, D., Trieschnigg, D., Doğruöz, A. S., Gravel, R., Theune, M., Meder, T. y de Jong, F. (2014).** Why Gender and Age Prediction from Tweets is Hard: Lessons from a Crowdsourcing Experiment. COLING 2014. URL: https://dongnguyen.nl/publications/nguyen-coling2014.pdf — Uso: cota de realismo del problema (los anotadores humanos no superan al clasificador).
7. **Sap, M., Park, G., Eichstaedt, J., Kern, M., Stillwell, D., Kosinski, M., Ungar, L. y Schwartz, H. A. (2014).** Developing Age and Gender Predictive Lexica over Social Media. EMNLP 2014. DOI: 10.3115/v1/D14-1121 — Uso: inspiración de los lexicones internos por regresión L2 sobre n-gramas.
8. **Flekova, L. y Gurevych, I. (2013).** Can We Hide in the Web? Large Scale Simultaneous Age and Gender Author Profiling in Social Media. PAN@CLEF 2013, CEUR-WS Vol-1179. URL: https://ceur-ws.org/Vol-1179/ — Uso: bloque explícito de superficie/legibilidad combinado con n-gramas.
9. **Hovy, D. y Fornaciari, T. (2018).** Increasing In-Class Similarity by Retrofitting Embeddings with Demographic Information. EMNLP 2018. DOI: 10.18653/v1/D18-1070 — Uso: idea de aumentar la similitud intra-clase con SVD/LSA sin embeddings (representación original prohibida por D-001).
10. **Morgan-Lopez, A. A., Kim, A. E., Chew, R. F. y Ruddle, P. (2017).** Predicting age groups of Twitter users based on language and metadata features. PLOS ONE, 12(8), e0183537. DOI: 10.1371/journal.pone.0183537 — Uso: fija el baseline solo-texto como cota inferior al no haber metadatos.
11. **Liu, Y., Singh, L. y Mneimneh, Z. (2021).** A Comparative Analysis of Classic and Deep Learning Models for Inferring Gender and Age of Twitter Users. DeLTA 2021. DOI: 10.5220/0010559500480058 — Uso: justifica la apuesta clásica (TF-IDF + modelos lineales) frente a deep learning.
12. **Rangel, F., Rosso, P., Koppel, M., Stamatatos, E. e Inches, G. (2013).** Overview of the Author Profiling Task at PAN 2013. CEUR-WS Vol-1179. URL: https://ceur-ws.org/Vol-1179/CLEF2013wn-PAN-RangelEt2013.pdf — Uso: calibración de techos con 3 clases de edad.
13. **Rangel, F., Rosso, P., Chugur, I., Potthast, M., Trenkmann, M., Stein, B., Verhoeven, B. y Daelemans, W. (2014).** Overview of the 2nd Author Profiling Task at PAN 2014. CEUR-WS Vol-1180. URL: https://ceur-ws.org/Vol-1180/CLEF2014wn-Pan-RangelEt2014.pdf — Uso: calibración con 5 clases de edad.
14. **Rangel, F., Celli, F., Rosso, P., Potthast, M., Stein, B. y Daelemans, W. (2015).** Overview of the 3rd Author Profiling Task at PAN 2015. CEUR-WS Vol-1391. URL: https://ceur-ws.org/Vol-1391/inv-pap12-CR.pdf — Uso: referencia de techos según el ancho de las clases.
15. **Rangel, F., Rosso, P., Verhoeven, B., Daelemans, W., Potthast, M. y Stein, B. (2016).** Overview of the 4th Author Profiling Task at PAN 2016: Cross-Genre Evaluations. CEUR-WS Vol-1609. URL: https://ceur-ws.org/Vol-1609/16090750.pdf — Uso: anticipa el riesgo de sobreajuste al género textual (aquí, retuits).
16. **Rangel, F., Rosso, P., Potthast, M. y Stein, B. (2017).** Overview of the 5th Author Profiling Task at PAN 2017: Gender and Language Variety Identification in Twitter. CEUR-WS Vol-1866. URL: https://ceur-ws.org/Vol-1866/invited_paper_11.pdf — Uso: rasgos dialectales para el análisis lingüístico.
17. **Rangel, F., Rosso, P., Montes-y-Gómez, M., Potthast, M. y Stein, B. (2018).** Overview of the 6th Author Profiling Task at PAN 2018: Multimodal Gender Identification in Twitter. CEUR-WS Vol-2125. URL: https://ceur-ws.org/Vol-2125/invited_paper_15.pdf — Uso: confirma que la imagen aporta poco frente al texto.
18. **Rangel, F. y Rosso, P. (2019).** Overview of the 7th Author Profiling Task at PAN 2019: Bots and Gender Profiling. CEUR-WS Vol-2380. URL: https://ceur-ws.org/Vol-2380/ — Uso: en PAN 2019 (ES) los char n-grams superaron a los embeddings y los RT no deben limpiarse.
19. **Sapkota, U., Bethard, S., Montes, M. y Solorio, T. (2015).** Not All Character N-grams Are Created Equal: A Study in Authorship Attribution. NAACL-HLT 2015. DOI: 10.3115/v1/N15-1010. URL: https://aclanthology.org/N15-1010/ — Uso: justifica char n-grams 2-5 con marcado de espacios (char_wb).
20. **Sapkota, U., Solorio, T., Montes-y-Gómez, M. y Ramírez-de-la-Rosa, G. (2013).** Author Profiling for English and Spanish Text. PAN@CLEF 2013, CEUR-WS Vol-1179. URL: https://ceur-ws.org/Vol-1179/ — Uso: n-gramas de carácter y palabra + SVM en inglés y español.
21. **López-Monroy, A. P., Montes-y-Gómez, M., Escalante, H. J., Villaseñor-Pineda, L. y Villatoro-Tello, E. (2013).** INAOE's Participation at PAN'13: Author Profiling Task. CEUR-WS Vol-1179. URL: https://ceur-ws.org/Vol-1179/ — Uso: atributos de segundo orden como antecedente.
22. **Verhoeven, B., Daelemans, W. y Plank, B. (2016).** TwiSty: A Multilingual Twitter Stylometry Corpus for Gender and Personality Profiling. LREC 2016. DOI: 10.63317/3yne5b7ve4p2 — Uso: referencia metodológica (no entrenable por D-001: sin datos externos).
23. **Verhoeven, B. y Daelemans, W. (2014).** CLiPS Stylometry Investigation (CSI) corpus: A Dutch corpus for the detection of age, gender, personality, sentiment and deception in text. LREC 2014. DOI: 10.63317/5ihmqb5uqzzs — Uso: referencia metodológica de etiquetas de edad.

Nota: la ficha de Rosenthal y McKeown (2011) aparece en la revisión, pero `references.bib` la marca "NO VERIFICADO... No citar en la entrega final sin confirmar" (ver B.9).

### B.2 Aprendizaje estadístico y arquitecturas neuronales

Núcleo metodológico de la sección 54 del notebook, complementado con citas inline de `DECISIONS.md` y `RUNLOG.md`:

1. **Ke, G. et al. (2017).** LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 30, 3146-3154. URL: https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html — la ficha bibliográfica está en `references.bib`; el uso de la librería sí es central (V19-B1, V29, V30 y el árbol `HIERBA` de V20).
2. **Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. y Gulin, A. (2018).** CatBoost: unbiased boosting with categorical features. NeurIPS 31, 6638-6648. URL: https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html — Uso: challenger P2 (V32-E2) y falsación de árboles obliviosos (V49).
3. **Gorishniy, Y., Rubachev, I., Khrulkov, V. y Babenko, A. (2021).** Revisiting Deep Learning Models for Tabular Data. NeurIPS 34, 18932-18943. URL: https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html — Uso: fundamento del TabularResNet de los componentes neuronales V38-V47.
4. **Hu, J., Shen, L. y Sun, G. (2018).** Squeeze-and-Excitation Networks. CVPR 2018, 7132-7141. DOI: 10.1109/CVPR.2018.00745. URL: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html — Uso: compuertas SE en los bloques residuales (V40-V47).
5. **Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. y Wilson, A. G. (2018).** Averaging Weights Leads to Wider Optima and Better Generalization. UAI 2018, 876-885. URL: https://arxiv.org/abs/1803.05407 — Uso: SWA en V41-V47 (D-094).
6. **Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. y Wojna, Z. (2016).** Rethinking the Inception Architecture for Computer Vision. CVPR 2016, 2818-2826. DOI: 10.1109/CVPR.2016.308 — Uso: label smoothing 0.10 (V42-V47).
7. **Müller, R., Kornblith, S. y Hinton, G. E. (2019).** When does label smoothing help? NeurIPS 32, 4694-4703. URL: https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html — Uso: cita complementaria del suavizado de etiquetas.
8. **Loshchilov, I. y Hutter, F. (2017).** SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR 2017. URL: https://openreview.net/forum?id=Skq89Scxx — Uso: enfriamiento cosenoidal `CosineAnnealingLR` en V38-V47.
9. **Klambauer, G., Unterthiner, T., Mayr, A. y Hochreiter, S. (2017).** Self-Normalizing Neural Networks. NeurIPS 30, 971-980. URL: https://proceedings.neurips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html — Uso: falsación SNN con SELU y AlphaDropout (V52-B, D-105).
10. **Hinton, G. E. (2002).** Training Products of Experts by Minimizing Contrastive Divergence. Neural Computation, 14(8), 1771-1800. DOI: 10.1162/089976602760128018 — Uso: consenso Product-of-Experts del campeón V48 (D-101).
11. **Heskes, T. (1998).** Selecting weighting factors in logarithmic opinion pools. NeurIPS 11, 266-272. URL: https://proceedings.neurips.cc/paper/1998/hash/9a96876e2f8f3dc4f3cf45f02c61c0c1-Abstract.html — Uso: antecedente del pooling logarítmico ponderado.
12. **Huang, G., Liu, Z., van der Maaten, L. y Weinberger, K. Q. (2017).** Densely Connected Convolutional Networks. CVPR 2017, 4700-4708. DOI: 10.1109/CVPR.2017.243 — Uso: red densa multi-escala de V53 (Dense-SE-TabularNet) y componente del consenso no lineal V56.

### B.3 Ordinalidad y transporte óptimo

1. **Frank, E. y Hall, M. (2001).** A simple approach to ordinal classification. ECML 2001, 145-156. Springer. DOI: 10.1007/3-540-44795-4_13 — Uso: miembro E-021a del ensamble (5 binarios `P(y>k)`; 0.5527) y descomposición pairwise (V28-03).
2. **Cao, W., Mirjalili, V. y Raschka, S. (2020).** Rank consistent ordinal regression for neural networks with application to age estimation. Pattern Recognition Letters, 140, 325-331. DOI: 10.1016/j.patrec.2020.11.008 — Uso: CORAL en V22/V23 (cerrado en V23-01).
3. **McCullagh, P. (1980).** Regression Models for Ordinal Data. Journal of the Royal Statistical Society: Series B (Methodological), 42(2), 109-127. DOI: 10.1111/j.2517-6161.1980.tb01109.x — citado como "McCullagh 1980" en `DECISIONS.md` (D-099) y `RUNLOG.md` (V47); la hipótesis fue falsada (0.5810757).
4. **Cramér, H. (1928).** On the composition of elementary errors. Scandinavian Actuarial Journal, 1928(1), 13-74. DOI: 10.1080/03461238.1928.10416862 — Uso: base de la distancia de Cramér-von Mises empleada como pérdida en V43/V46/V47/V51.
5. **von Mises, R. (1931).** Wahrscheinlichkeitsrechnung und ihre Anwendung in der Statistik und theoretischen Physik. F. Deuticke, Leipzig. — Uso: segunda mitad del nombre de la distancia.
6. **Rizzo, M. L. y Székely, G. J. (2016).** Energy distance. Wiley Interdisciplinary Reviews: Computational Statistics, 8(1), 27-38. DOI: 10.1002/wics.1375; y Székely & Rizzo (2013), JSPI 143(8), 1249-1272, DOI: 10.1016/j.jspi.2013.03.018 — Uso: conexión transporte óptimo / estadísticos de energía citada junto a la pérdida de Cramér.
7. **Frogner, C., Zhang, C., Mobahi, H., Araya, M. y Poggio, T. A. (2015).** Learning with a Wasserstein Loss. NeurIPS 28, 2053-2061. URL: https://proceedings.neurips.cc/paper/2015/hash/70efdf2ec9b086079795c442636b55fb-Abstract.html — Uso: marco general del transporte óptimo ordinal (pérdidas V43-V47, V52-A).

### B.4 Selección de términos, NB-SVM y optimización de F-medida

1. **Wang, S. y Manning, C. D. (2012).** Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. ACL 2012, 90-94. URL: https://aclanthology.org/P12-2018/ — Uso: NB-SVM (log-count ratio + LinearSVC por clase) implementado como miembros del ensamble ADV05 (char+menciones, CV 0.5641) y ADV07 (palabra, CV 0.5254); ver nota de verificación en B.9.
2. **Monroe, B. L., Colaresi, M. P. y Quinn, K. M. (2008).** Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict. Political Analysis, 16(4), 372-403. DOI: 10.1093/pan/mpn018 — Uso: log-odds con prior de Dirichlet para términos distintivos por clase (`class_lexicon_scores`, análisis contrastivo de V45).
3. **Lipton, Z. C., Elkan, C. y Narayanaswamy, B. (2014).** Thresholding Classifiers to Maximize F1 Score. arXiv:1402.1892. URL: https://arxiv.org/abs/1402.1892 — Uso: base teórica del ajuste de umbral de la cadena plug-in.
4. **Puthiya Parambath, S. A., Usunier, N. y Grandvalet, Y. (2014).** Optimizing F-Measures by Cost-Sensitive Classification. NIPS 2014. URL: https://arxiv.org/abs/1505.00199 — Uso: reducción de F-medida a clasificación cost-sensitive (evaluada y descartada como clasificador; ver `FINAL_REPORT.md` §4).
5. **Waegeman, W., Dembczyński, K., Jachnik, A., Cheng, W. y Hüllermeier, E. (2014).** On the Bayes-optimality of F-measure maximizers. Journal of Machine Learning Research, 15, 3333-3388. URL: https://arxiv.org/abs/1310.4849 — Uso: análisis del optimizador de F1.
6. **Narasimhan, H., Kar, P. y Jain, P. (2015).** Optimizing Non-decomposable Performance Measures: A Tale of Two Classes. ICML 2015. URL: https://arxiv.org/abs/1505.06812 — Uso: esquemas estocásticos para F-medida.
7. **Eban, E., Schain, M., Mackey, A., Gordon, A., Saurous, R. A. y Elidan, G. (2017).** Scalable Learning of Non-Decomposable Objectives. AISTATS 2017. URL: https://arxiv.org/abs/1608.04802 — Uso: cotas sobre la matriz de confusión para optimizar F.

### B.5 Integridad experimental y fenómenos estadísticos

1. **Kaufman, S., Rosset, S., Perlich, C. y Stitelman, O. (2012).** Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data (TKDD), 6(4), 1-21. DOI: 10.1145/2382577.2382579 — Uso: marco de la auditoría de fuga por near-duplicados y de los folds agrupados (D-003).
2. **Simpson, E. H. (1951).** The Interpretation of Interaction in Contingency Tables. Journal of the Royal Statistical Society: Series B (Methodological), 13(2), 238-241. DOI: 10.1111/j.2517-6161.1951.tb00088.x — Uso: fundamento de la paradoja de Simpson demostrada en V52 con métricas armónicas (D-105).

### B.6 Aprendizaje online: la línea FTRL

1. **McMahan, H. B., Holt, G., Sculley, D. et al. (2013).** Ad Click Prediction: a View from the Trenches. Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '13), 1222-1230. DOI: 10.1145/2487575.2488215 — citado como "McMahan 2013" en `DECISIONS.md` (D-083), `PROJECT_STATE.md` y `FINAL_REPORT.md`; implementado como candidato P3 (V32-E3, V33-V35) y cerrado por mecanismo (degeneró a 0.040169, bajo el suelo trivial 0.048542).

### B.7 Comparación con el techo publicado (acta D-087)

Referencias usadas para fijar el techo alcanzable y comparar con lo publicado (según `DECISIONS.md` D-087):

1. **Sadeghi, Akbari y Jaziriyan (2024).** ExaAUAC. Journal of Big Data. DOI: 10.1007/s44163-024-00145-0 — Twitter árabe, 181k perfiles, los mismos 6 bins: macro-F1 0.44 con BERT fine-tuned + metadatos (0.58 colapsando a 3 grupos). El campeón (0.616) queda +0.176 por encima.
2. **Cimino, A., Dell'Orletta, F. y Nissim, M. (2020).** Overview of the EVALITA 2020 Topic, Age, and Gender Attribute Identification from Tweets (TAG-it) Task. CEUR-WS Vol-2765, paper165. URL: http://ceur-ws.org/Vol-2765/paper165.pdf — edad macro-F1 0.416 (UmBERT) frente a 0.382 (SVM).
3. **Lupo, Bose, Habibi, Hovy y Schwarz (2024).** DADIT. LREC-COLING 2024. arXiv: 2403.05700 — hasta +53% de F1 en edad con XLM.
4. **O'Connor et al. (2024).** Revisión de métodos de edad y género desde Twitter. JMIR. DOI: 10.2196/72521 — rango publicado de F1 0.31-0.94; el extremo alto usa edad autodeclarada verificada, protocolo incompatible.
5. **Morgan-Lopez, A. A., Kim, A. E., Bibeau, C. y Dubin, M. (2017).** Predicting age groups of Twitter users: Comparison of survey vs Twitter metadata. American Journal of Public Health, 107(8). DOI: 10.2105/AJPH.2017.303831 — 74% de F1 con lenguaje + metadatos frente a 57% solo metadatos.
6. **Álvarez-Carmona, M. Á. et al. (2015).** INAOE's participation at PAN'15: Author Profiling task. CEUR-WS Vol-1391, paper 116. URL: https://ceur-ws.org/Vol-1391/116-CR.pdf — publica accuracy media combinada, no macro-F1 desglosado por edad.
7. **Rangel, F. et al. (2020).** Overview de PAN 2020. PMC7148014 — mismo problema de métrica agregada; la cifra desglosada de edad de PAN 2015 no pudo verificarse.

### B.8 Mecanismos revisados en 2026 y no adoptados (evidencia de cierre)

Citados en `reports/age_literature_review_2026.md`, `reports/age_literature_methods_registry.csv` y `reports/age_literature_gap_registry.csv` como evidencia para cerrar o mantener cerradas las familias C1-C5. Ninguno fue adoptado.

| Referencia (como aparece en el repo) | Papel en la decisión |
|---|---|
| Díaz y Marathe (CVPR 2019), SORD | Revisado en profundidad en C1 (2026): inspiró la combinación de etiquetas suaves con distancia ordinal adoptada en V42/V43; el candidato puro se cerró por desajuste de métrica nominal |
| Vargas et al. (Pattern Recognition 2022; Information Fusion 2023; Information Sciences 2023; ESWA 2024) | Familias de kernel ordinales; la CE categorica rinde mejor en métricas nominales |
| Castagnos et al. (COLING 2022), OLL | Pérdida ordinal de la familia C1 |
| Kasai et al. (arXiv 2405.11775) | Comparativa explícita: trade-off ordinal vs nominal en texto |
| Garg y Manwani (ACML 2020), Ord-NLL | Ruido ordinal explícito (C2); colapsa a target suave en expectativa |
| Yuan et al. (PRICAI 2023) y Takezaki et al. (IEEE Access 2026) | Evidencia adicional de Ord-NLL (C2) |
| ORCU (arXiv 2410.15658v4); stick-breaking (Neurocomputing 2020); Beckham y Pal (ICML 2017) | Regularización de forma unimodal (C3); ganancia de calibración, no de F1 |
| Dou et al. (COLING 2022), DePro; Ravfogel et al. (ACL 2020), INLP; Chew et al. (EACL Findings 2024), NFL; Seroussi et al. (CoNLL 2011) | Separación estilo/contenido (C4); la ganancia es estrictamente OOD |
| Wang et al. (AAAI 2023), CPL | Geometría ordinal de clases (C5); dependiente de un extractor prohibido |
| Natarajan et al. (NeurIPS 2013) | Desplazamiento de umbral por ruido; el intercepto óptimo ya estaba medido en 0 (P33) |

Detalles bibliográficos completos (título, venue, DOI/URL) de estas referencias: `[VERIFICAR]` en su mayoría; el repositorio las cita por autor, año y venue abreviado en los informes indicados.

### B.9 Referencias descartadas o marcadas por verificación

1. **Rosenthal, S. y McKeown, K. (2011).** Age Prediction in Blogs: A Study of Online Social Networks, Style and Personality. Informe técnico, Columbia University. DOI/URL: `[VERIFICAR]` — `references.bib` la marca "NO VERIFICADO: no se localizó DOI ni URL confirmada... No citar en la entrega final sin confirmar".
2. **Wang y Manning (2012)** (ficha completa en B.4): `reports/tables/dataset_provenance.md` registra que su ficha no se localizó en Crossref ni en el índice de ACL Anthology con las consultas realizadas; DOI/URL: https://aclanthology.org/P12-2018/.
3. **DOI 10.1007/s44163-026-01695-1**: citado como no verificable en D-087 (acceso bloqueado); no se incluye como referencia utilizable.

### B.10 Herramientas estándar citadas inline

Nota: fichas de la publicación original; no provienen del `references.bib` del repositorio. Se incorporaron en la segunda pasada de citas inline (2026-10-17) a las secciones del notebook.

1. **Salton, G. y Buckley, C. (1988).** «Term-Weighting Approaches in Automatic Text Retrieval». Information Processing & Management, 24(5), 513-523. — Uso: TF-IDF (00_portada, 01_fase1, fase3_5_residual_v13v17, fase4_1_v18_v29, fase4_3_fracasos, anexo_c_glosario).
2. **Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R. y Lin, C.-J. (2008).** «LIBLINEAR: A Library for Large Linear Classification». Journal of Machine Learning Research, 9, 1871-1874. — Uso: LinearSVC (03_fase3, fase3_5_residual_v13v17, fase4_3_fracasos, anexo_c_glosario).
3. **Bottou, L. (2010).** «Large-Scale Machine Learning with Stochastic Gradient Descent». COMPSTAT 2010, 177-186. — Uso: SGD (fase4_3_fracasos, anexo_c_glosario).
4. **Rennie, J. D. M., Shih, L., Teevan, J. y Karger, D. R. (2003).** «Tackling the Poor Assumptions of Naive Bayes Text Classifiers». ICML 2003, 616-623. — Uso: ComplementNB (anexo_c_glosario).
5. **Hoerl, A. E. y Kennard, R. W. (1970).** «Ridge Regression: Biased Estimation for Nonorthogonal Problems». Technometrics, 12(1), 55-67. — Uso: Ridge (anexo_c_glosario).
6. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. y Duchesnay, É. (2011).** «Scikit-learn: Machine Learning in Python». Journal of Machine Learning Research, 12, 2825-2830. — Uso: scikit-learn (06_fase6).
7. **Akiba, T., Sano, S., Yanase, T., Ohta, T. y Koyama, M. (2019).** «Optuna: A Next-generation Hyperparameter Optimization Framework». KDD 2019, 2623-2631. — Uso: Optuna (04_fase4, fase4_1_v18_v29, fase2_5_auditoria_historial, fase5_4_matematicos, anexo_c_glosario).
8. **Efron, B. (1979).** «Bootstrap Methods: Another Look at the Jackknife». The Annals of Statistics, 7(1), 1-26. — Uso: bootstrap (01_fase1, 05_fase5, fase2_5_auditoria_historial, fase4_1_v18_v29, fase4_3_fracasos, fase4_5_v44_v52, anexo_c_glosario).
9. **Chen, T. y Guestrin, C. (2016).** «XGBoost: A Scalable Tree Boosting System». KDD 2016, 785-794. — Uso: XGBoost (ablación de V44, fase4_5_v44_v52).
10. **Zhang, H., Cissé, M., Dauphin, Y. N. y Lopez-Paz, D. (2018).** «mixup: Beyond Empirical Risk Minimization». ICLR 2018. — Uso: mixup (ablación de V44, fase4_5_v44_v52).
11. **Inoue, H. (2019).** «Multi-Sample Dropout for Accelerated Training and Better Generalization». arXiv:1905.09788. — Uso: Multi-Sample Dropout (ablación de V44, fase4_5_v44_v52).
12. **Arik, S. Ö. y Pfister, T. (2021).** «TabNet: Attentive Interpretable Tabular Learning». AAAI 2021, 6679-6687. — Uso: TabNet (mención «SE-TabNet» en fase4_5_v44_v52).
13. **Huang, X., Khetan, A., Cvitkovic, M. y Karnin, Z. (2020).** «TabTransformer: Tabular Data Modeling Using Contextual Embeddings». arXiv:2012.06678. — Uso: TabTransformer (V50, fase4_5_v44_v52).
14. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. y Polosukhin, I. (2017).** «Attention Is All You Need». Advances in Neural Information Processing Systems 30 (NIPS 2017), 5998-6008. — Uso: self-attention/MHSA (V50, fase4_5_v44_v52).
15. **Ioffe, S. y Szegedy, C. (2015).** «Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift». ICML 2015, 448-456. — Uso: BatchNorm (04b_fase4_nn, fase4_5_v44_v52, anexo_c_glosario).
16. **Jaccard, P. (1912).** «The Distribution of the Flora in the Alpine Zone». New Phytologist, 11(2), 37-50. — Uso: índice de Jaccard (02_fase2, fase2_5_auditoria_historial, fase5_4_matematicos, fase5_5_calibracion, anexo_c_glosario).
17. **Breiman, L. (1996).** «Bagging Predictors». Machine Learning, 24(2), 123-140. — Uso: bagging (V28-A, fase4_1_v18_v29).
18. **Hinton, G., Vinyals, O. y Dean, J. (2015).** «Distilling the Knowledge in a Neural Network». NIPS 2014 Deep Learning Workshop (arXiv:1503.02531). — Uso: destilación (V18-P5, fase4_1_v18_v29).
19. **Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S. y Müller, K.-R. (2021).** «Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology». Machine Learning and Knowledge Extraction, 3(2), 392-413. — Uso: metodología CRISP-ML(Q) (00_portada).
20. **Rocchio, J. J. (1971).** «Relevance Feedback in Information Retrieval». The SMART Retrieval System: Experiments in Automatic Document Processing, Prentice-Hall, 313-323. — Uso: centroides Rocchio (anexo_c_glosario).
