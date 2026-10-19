# Citas inline — segunda pasada: herramientas estándar (2026-10-17)

Fichas del coordinador; inline con (Autores, «Título», Venue, Año) en la primera mención por archivo y (Autores, Año) después. >4 autores → "et al." (convención de la 1ª pasada). Sin duplicar citas de `citas_agregadas.md`; no se tocaron fences, tablas, cifras ni encabezados.

## 00_portada.md (2) — TF-IDF → Salton y Buckley 1988; CRISP-ML(Q) → Studer et al. 2021.
## 01_fase1.md (2) — TF-IDF → Salton y Buckley 1988; bootstrap → Efron 1979.
## 02_fase2.md (1) — Jaccard → Jaccard 1912.
## 03_fase3.md (1) — LinearSVC → Fan et al. 2008 (prosa §3.2; tabla de miembros intacta).
## 04_fase4.md (2) — Optuna → Akiba et al. 2019 (completa en V29; corta en V30).
## 04b_fase4_nn.md (1) — BatchNorm («normalización por lotes») → Ioffe y Szegedy 2015.
## 05_fase5.md (1) — bootstrap → Efron 1979.
## 06_fase6.md (1) — scikit-learn → Pedregosa et al. 2011 (§6.2, única mención general).
## anexo_c_glosario.md (14) — Jaccard → Jaccard 1912; miembros: LinearSVC → Fan et al. 2008, SGD → Bottou 2010, ComplementNB → Rennie et al. 2003, Ridge → Hoerl y Kennard 1970, Rocchio → Rocchio 1971; bootstrap → Efron 1979 (+corta); BatchNorm → Ioffe y Szegedy 2015; Ridge ordinal → (Hoerl y Kennard, 1970); TF-IDF → Salton y Buckley 1988 (+corta); LinearSVC → (Fan et al., 2008); Optuna → Akiba et al. 2019.
## fase2_5_auditoria_historial.md (3) — bootstrap → Efron 1979; Optuna → Akiba et al. 2019; Jaccard → Jaccard 1912.
## fase3_5_residual_v13v17.md (2) — TF-IDF → Salton y Buckley 1988; LinearSVC → Fan et al. 2008 (fusionada con la cita Sapkota 2013 existente).
## fase4_1_v18_v29.md (6) — Optuna → Akiba et al. 2019 (+corta); TF-IDF → Salton y Buckley 1988; destilación → Hinton, Vinyals y Dean 2015; bootstrap → Efron 1979; bagging → Breiman 1996.
## fase4_3_fracasos.md (4) — TF-IDF → Salton y Buckley 1988; SGD → Bottou 2010; bootstrap → Efron 1979; LinearSVC → Fan et al. 2008.
## fase4_5_v44_v52.md (8) — bootstrap → Efron 1979; Multi-Sample Dropout → Inoue 2019; mixup → Zhang et al. 2018; XGBoost → Chen y Guestrin 2016; BatchNorm → Ioffe y Szegedy 2015; TabTransformer → Huang et al. 2020; self-attention/MHSA → Vaswani et al. 2017; TabNet («SE-TabNet») → Arik y Pfister 2021.
## fase5_4_matematicos.md (2) — Optuna → Akiba et al. 2019; Jaccard → Jaccard 1912.
## fase5_5_calibracion.md (1) — Jaccard → Jaccard 1912.
## anexo_b_referencias.md — añadida subsección B.10 con las 20 fichas citadas (nota: no provienen del `references.bib`).
## anexo_a_inventario.md — sin cambios (solo tablas).

Total: 51 citas inline en 16 archivos (45 completas + 6 repeticiones cortas).

## No mencionadas en prosa (no citadas)
- Nelder-Mead y Platt scaling: sin menciones en `secciones/`. focal loss: solo en tabla de anexo_a. Nyström: solo en tabla de fase4_5 (§V45). TabNet se citó vía «SE-TabNet».

## Dudas
- >4 autores con "et al." (Fan 5, Akiba 5, Studer 7, Vaswani 8, Pedregosa 15), convención de la 1ª pasada; en B.10 van completos.
- En el glosario, la cita completa de bootstrap se ancló en la primera mención (entrada PoE) y la entrada «bootstrap pareado» lleva forma corta.
- fase3_5: la cita de LinearSVC se fusionó con la paren de Sapkota 2013 para no apilar paréntesis.
