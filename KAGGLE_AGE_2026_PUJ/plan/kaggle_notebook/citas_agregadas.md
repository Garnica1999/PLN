# Citas inline agregadas a las secciones (2026-10-17)

Regla: primera mención por archivo con (Autores, «Título», Venue, Año) según `references.bib` (≤4 autores completos; >4 "et al."); repeticiones en el mismo archivo con (Autores, Año). No se tocaron fences ```python, tablas, cifras ni encabezados. McMahan 2013 y McCullagh 1980 van sin título/venue: el repo no los da (regla 5).

## 00_portada.md (2)
- Product-of-Experts → Hinton 2002 (Neural Computation). plug-in → Lipton, Elkan y Narayanaswamy 2014 (arXiv).

## 01_fase1.md (4)
- n-gramas de carácter → Sapkota, Bethard, Montes y Solorio 2015 (NAACL-HLT). canales léxicos → Sap et al. 2014 (EMNLP). apuesta clásica vs DL → Liu, Singh y Mneimneh 2021 (DeLTA). near-duplicados → Kaufman, Rosset, Perlich y Stitelman 2012 (ACM TKDD).

## 02_fase2.md (4 eventos / 6 obras)
- forma del texto → Flekova y Gurevych 2013 (PAN@CLEF). marcadores léxicos → Nguyen, Gravel, Trieschnigg y Meder 2013 (ICWSM). ordinales (puntos medios/acumulativos/transporte) → Nguyen, Smith y Rosé 2011 + Frank y Hall 2001 + Frogner et al. 2015. near-duplicados → Kaufman et al. 2012.

## 03_fase3.md (5 eventos / 7 obras)
- char_wb → Sapkota et al. 2015. densas/estilometría → Schler, Koppel, Argamon y Pennebaker 2006 + Argamon, Koppel, Pennebaker y Schler 2007. menciones/hashtags/URL → Nguyen et al. 2013. log-odds → Monroe, Colaresi y Quinn 2008. NB-SVM → Wang y Manning 2012 + Mukherjee y Liu 2010.

## 04_fase4.md (5)
- LightGBM → Ke et al. 2017 (NeurIPS); early stopping y Optuna → (Ke et al., 2017). cadena de umbrales → Frank y Hall 2001. plug-in → Lipton et al. 2014.

## 04b_fase4_nn.md (7 eventos / 10 obras)
- TabularResNet → Gorishniy, Rubachev, Khrulkov y Babenko 2021. distancia ordinal → Frogner et al. 2015. SE → Hu, Shen y Sun 2018. SWA → Izmailov, Podoprikhin, Garipov, Vetrov y Wilson 2018. label smoothing → Szegedy et al. 2016 + Müller, Kornblith y Hinton 2019. cosenoidal → Loshchilov y Hutter 2017. Cramér-von Mises/W₂ → Cramér 1928 + von Mises 1931 + Székely y Rizzo 2013.

## 05_fase5.md (6 eventos / 8 obras)
- folds agrupados → Kaufman et al. 2012. PoE → Hinton 2002; pesos del log-pool → Heskes 1998. plug-in → Lipton et al. 2014. Cramér → Cramér 1928 + von Mises 1931; cadena de umbrales → Frank y Hall 2001.

## 06_fase6.md (8 eventos / 14 obras)
- ensemble de árboles → Ke et al. 2017; redes tabulares → Gorishniy et al. 2021; pérdidas ordinales → Frank y Hall 2001; transporte → Frogner et al. 2015; consenso → Hinton 2002; regla por clase → Lipton et al. 2014. near-duplicados → Kaufman et al. 2012. ordinales/Cramér → Frank y Hall 2001 + Cramér 1928. CatBoost → Prokhorenkova et al. 2018; FTRL → McMahan et al. 2013; CORAL → Cao, Mirjalili y Raschka 2020; ordinales → Frank y Hall 2001.

## anexo_c_glosario.md (15 eventos)
- puntos medios/Ridge → Nguyen et al. 2011. folds agrupados → Kaufman et al. 2012. miembros: Frank & Hall → Frank y Hall 2001; NB-SVM → Wang y Manning 2012. cuartos: LightGBM → Ke et al. 2017. plug-in → Lipton et al. 2014. PoE → Hinton 2002 + Heskes 1998. SWA → Izmailov et al. 2018. label smoothing → Szegedy et al. 2016 + Müller et al. 2019. Cramér → Cramér 1928 + von Mises 1931. ordinal: distancia esperada → Frogner et al. 2015; enlace acumulativo → McCullagh 1980. CORAL → Cao et al. 2020. FTRL → McMahan et al. 2013. char_wb → Sapkota et al. 2015. SVD/LSA → Hovy y Fornaciari 2018.

## fase2_5_auditoria_historial.md (1)
- fuga de autor/temporal → Kaufman et al. 2012.

## fase3_5_residual_v13v17.md (1)
- TF-IDF word + LinearSVC (vista estructural) → Sapkota, Solorio, Montes-y-Gómez y Ramírez-de-la-Rosa 2013 (PAN@CLEF).

## fase4_1_v18_v29.md (4)
- árbol boosting → Ke et al. 2017; cadena de umbrales → Frank y Hall 2001; meta-modelo LightGBM → (Ke et al., 2017). cumulative-link → Cao et al. 2020.

## fase4_3_fracasos.md (3)
- CatBoost → Prokhorenkova et al. 2018; FTRL-Proximal → McMahan et al. 2013; cumulative-link → Cao et al. 2020.

## fase4_5_v44_v52.md (9)
- Wasserstein-1 → Frogner et al. 2015; Cramér-von Mises → Cramér 1928 + von Mises 1931. enlace acumulativo → McCullagh 1980. PoE → Hinton 2002. CatBoost oblivious → Prokhorenkova et al. 2018. Squeeze-and-Excitation → Hu, Shen y Sun 2018. suavizado → Szegedy et al. 2016 + Müller et al. 2019. SNN/SELU → Klambauer, Unterthiner, Mayr y Hochreiter 2017. paradoja de Simpson → Simpson 1951.

## fase4_intro.md (1), fase5_4_matematicos.md (1) y fase5_5_calibracion.md (1)
- plug-in → Lipton et al. 2014 (fase4_intro y fase5_4). near-duplicados → Kaufman et al. 2012 (fase5_5).

## anexo_b_referencias.md: 20 marcas [VERIFICAR] reemplazadas por datos del bib
- B.2 (11): Ke 2017, Prokhorenkova 2018, Gorishniy 2021, Hu 2018, Izmailov 2018, Szegedy 2016, Müller 2019, Loshchilov 2017, Klambauer 2017, Hinton 2002, Heskes 1998 (URL/DOI del bib; en Ke también título y nota de ficha).
- B.3 (5): Frank 2001 (DOI), Cao 2020 (DOI), Cramér 1928 (DOI), von Mises 1931 (sin DOI/URL en la ficha), Frogner 2015 (URL). B.4/B.9 (2): Wang y Manning 2012 (URL ACL). B.5 (2): Kaufman 2012 (DOI), Simpson 1951 (DOI).
- [VERIFICAR] conservado (8): McCullagh 1980, Rizzo y Székely 2016, McMahan 2013, Cimino 2020, Morgan-Lopez (AJPH) 2017, Pardo 2015, Rosenthal 2011 y la nota de B.8.

## anexo_a_inventario.md — Sin cambios: es tabla de inventario, sin prosa de técnicas; no se tocaron tablas.

## Sin base en el repo (no agregado)
- TF-IDF (origen), LinearSVC/LogReg/SGD/ComplementNB/Ridge/MLP como algoritmos de librería, Optuna, bootstrap pareado (no aparece "Efron" ni autores), Nelder-Mead (solo el nombre, sin autores), Platt scaling (solo "Platt por clase"), Jaccard, bagging (Breiman), destilación, mutual information, Bonferroni, HistGradientBoosting, XGBoost, mixup, multi-sample dropout, TabNet/TabTransformer/MHSA, focal loss, Nystroem, RankGauss/QuantileTransformer, Rocchio, Power-Mean/rank-averaging, BatchNorm (Ioffe y Szegedy 2015 no está en el bib), CRISP-ML: el repositorio no da ficha ni cita con autores; se omitió por la regla 5.
- Referencias de dominio sin mención en prosa: Morgan-Lopez 2017, PAN 2013-2019 (Rangel) y Verhoeven (TwiSty/CSI) quedan solo en el anexo B.
- Duda reportada: en ke2017 (8 autores) se aplicó "et al." según la regla ≤4; el ejemplo del brief lo lista completo.
