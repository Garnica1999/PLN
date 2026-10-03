# P48 - Trazabilidad del EDA original

Generado por `experiments/v48_00_eda_traceability.py` (V48-00 (P0 trazabilidad de los hallazgos del EDA)).

Fecha 2026-09-28T00:44:32+00:00. **0 modelos, 0 submissions, 0 consultas a Kaggle.**

Pregunta: El EDA original se convertingio de forma suficientemente completa en hipotesis experimentales con mecanismo concreto?

Regla de cierre aplicada: mecanismo probado + control adecuado + evaluacion valida.
El mismo tema no implica la misma familia.

## 1. Cobertura del informe EDA original

- hallazgos declarados en `fase2_hallazgos_clave.csv`: **6**
- con hipotesis no vacia: **3**
- sin hipotesis: **3**

| hallazgo | hipotesis declarada en el EDA |
|---|---|
| El corpus no contiene emojis Unicode (normalizacion ASCII). | H6 (descarta la subfamilia de emojis) |
| La mencion inicial es el fenomeno mas asociado con TR dentro de HS=1. | H3 |
| El odio con objetivo colectivo explicito es casi siempre TR=0. | D-005 (semantica de TR) |
| La deduplicacion por near-duplicado no altera el orden de los efectos. | Transversal (D-004) |
| Desbalanceo moderado en las tres etiquetas (maximo 2.34:1). | H9 |
| Existe estructura jerarquica fuerte: 0 violaciones de TR=1 con HS=0. | H8 |

## 2. Matriz de trazabilidad

| hallazgo del EDA | mecanismo derivado | exp | cerrada | hueco |
|---|---|---:|:-:|---|
| La mencion inicial es el fenomeno mas asociado con TR de | feature de mencion inicial como termino propio | 20 | si | cubierta |
| La referencia colectiva sobrerepresenta falsos negativos | feature de referencia colectiva y su interacci | 11 | si | cubierta |
| El insulto lexico sobrerepresenta falsos positivos de HS | feature de insulto lexico y su peso en la clas | 32 | si | cubierta |
| La negacion altera la interpretacion de TR y AG | feature de negacion y su interaccion con la po | 18 | si | cubierta |
| La cita ocupa una fraccion medible del texto y se asocia | fraccion de tokens dentro de comillas y sus at | 46 | si | cubierta |
| La segunda persona se asocia a HS y a TR | conteo y densidad de segunda persona | 10 | si | cubierta |
| La longitud distingue clases y correlaciona con el error | longitud como feature y como condicion de bina | 27 | si | cubierta |
| La elongacion y el leet marcan registro coloquial | normalizacion y conteo de elongation y leet | 19 | si | cubierta |
| La puntuacion interna y las risas matizan la agresividad | puntuacion interna y risas como feature de AG | 11 | si | cubierta |
| La atribucion y el retweet aparecen en la distribucion d | marcadores de atribucion y presencia de RT | 5 | si | cubierta |
| Los hashtags y las URLs senalan registro digital y no to | conteo de hashtags y URLs, e interaccion con l | 25 | si | cubierta |
| Los perfiles de POS por texto separan las tres etiquetas | perfil de distribucion POS como representacion | 9 | si | cubierta |
| La estructura de etiquetas tiene combinaciones casi-impo | regla dura o suave sobre la combinacion de eti | 46 | si | cubierta |
| Los grupos de near-duplicate y pseudo-autor cambian la v | agrupacion para la validacion cruzada | 42 | si | cubierta |
| La rareza de la etiqueta condiciona el error | poblacion de etiqueta rara como condicion | 1 | NO | abierta |

## 3. Detalle por hallazgo

### La mencion inicial es el fenomeno mas asociado con TR dentro de HS=1

- artefacto fuente: `reports/eda/fase2_rasgos_TR.csv`
- evidencia historica: V de Cramer 0.512
- mecanismo derivado: feature de mencion inicial como termino propio
- experimentos encontrados: 20 (12 declaran cierre, 1 abiertos)
- ids: PRE-04; PA-01; E2-09-EMO-LEX; E2-13-META-ENT-RIDGE; E2-13-META-ENT-LR; E2-13-ENT-SOLO-LR; E2-16-MEJOR; E2-20-ESPECIALISTAS; AUD-02; V11-02; V12-04; V15-00; V15-01; V29-01; ...
- resultado registrado: {"f1_media_columnas": 0.7577, "f1_macro6": 0.8071, "f1_micro": 0.7543, "f1_samples": 0.304, "exact_match": 0.7008, "f1_por_etiqueta": {"HS": 0.7371, "TR": 0.8045, "AG": 0.7316}, "std_pliegues": 0.0183
- cerrada: **True** (16 veredictos sustantivos: 12 declaran cierre y 1 declaran apertura)
- hueco restante: **cubierta**

### La referencia colectiva sobrerepresenta falsos negativos de HS

- artefacto fuente: `reports/eda/error_perfil_rasgos.csv`
- evidencia historica: sobrerrepresentacion FP vs TN 1.3665
- mecanismo derivado: feature de referencia colectiva y su interaccion con HS
- experimentos encontrados: 11 (10 declaran cierre, 0 abiertos)
- ids: HIB-H12; HIB-H14; E2-05-CH-TGT; E2-05-CH-TGT-DIST; E2-06-TXT-CH; E2-06-TXT-TGT-SPARSE; E2-09-EMO-LEX; E2-16-MEJOR; E2-20-ESPECIALISTAS; E2-21-ESPACIOS; AUD-02
- resultado registrado: {"f1_media_columnas": 0.7836, "f1_macro6": 0.8275, "f1_micro": 0.7812, "f1_samples": 0.3228, "exact_match": 0.7221, "f1_por_etiqueta": {"HS": 0.7734, "TR": 0.8159, "AG": 0.7614}, "std_pliegues": 0.014
- cerrada: **True** (11 veredictos sustantivos: 10 declaran cierre y 0 declaran apertura)
- hueco restante: **cubierta**

### El insulto lexico sobrerepresenta falsos positivos de HS

- artefacto fuente: `reports/eda/error_perfil_rasgos.csv`
- evidencia historica: tasa FP 0.8108 con insulto
- mecanismo derivado: feature de insulto lexico y su peso en la clase positiva
- experimentos encontrados: 32 (23 declaran cierre, 0 abiertos)
- ids: HIB-H11; HIB-H14; E2-05-CH-TGT; E2-05-CH-TGT-DIST; E2-06-TXT-CH; E2-06-TXT-TGT-SPARSE; E2-09-EMO-EST; E2-09-EMO-LEX; E2-09-EMO; E2-09-TXT-EMO; E2-15-NB-WORD; E2-15-NB-CHAR; E2-15-NBSVM; E2-15-NBSVM-SVC; ...
- resultado registrado: {"f1_media_columnas": 0.7713, "f1_macro6": 0.8173, "f1_micro": 0.7701, "f1_samples": 0.3259, "exact_match": 0.7106, "f1_por_etiqueta": {"HS": 0.7619, "TR": 0.7846, "AG": 0.7676}, "std_pliegues": 0.012
- cerrada: **True** (30 veredictos sustantivos: 23 declaran cierre y 0 declaran apertura)
- hueco restante: **cubierta**

### La negacion altera la interpretacion de TR y AG

- artefacto fuente: `reports/eda/fase2_rasgos_TR_continuos.csv`
- evidencia historica: presente en la caracterizacion de TR
- mecanismo derivado: feature de negacion y su interaccion con la polaridad
- experimentos encontrados: 18 (14 declaran cierre, 1 abiertos)
- ids: E2-03-POS; E2-03-POS-TXT; E2-04-DEPS; E2-04-MORPH; E2-04-SYN3; E2-04-DEPS-TXT; E2-04-MORPH-TXT; E2-04-SYN3-TXT; E2-07-CITA; E2-07-CITA-TXT; E2-07-CITA-NEG-TXT; E2-08-NEG; E2-08-NEG-TXT; AUD-02; ...
- resultado registrado: {"corrected_cv": 0.5637100546249322, "f1_por_etiqueta": {"HS": 0.5687352710133543, "TR": 0.5720476706392199, "AG": 0.5503472222222222}, "pliegues": [0.6013440719323072, 0.5939713064713065, 0.553284698
- cerrada: **True** (17 veredictos sustantivos: 14 declaran cierre y 1 declaran apertura)
- hueco restante: **cubierta**

### La cita ocupa una fraccion medible del texto y se asocia a HS

- artefacto fuente: `reports/eda/fase2_rasgos_TR.csv`
- evidencia historica: frac_tokens_cita medida por fila
- mecanismo derivado: fraccion de tokens dentro de comillas y sus atributos
- experimentos encontrados: 46 (37 declaran cierre, 5 abiertos)
- ids: PRE-00; PRE-01; PRE-02; PRE-03; PRE-04; PRE-05; PRE-06; PRE-07; PRE-08; PRE-09; HIB-H11; HIB-H14; B5-DENSE; E2-05-CH-TGT; ...
- resultado registrado: {"f1_media_columnas": 0.7587, "f1_macro6": 0.8087, "f1_micro": 0.7557, "f1_samples": 0.3028, "exact_match": 0.7013, "f1_por_etiqueta": {"HS": 0.7429, "TR": 0.8026, "AG": 0.7306}, "std_pliegues": 0.014
- cerrada: **True** (46 veredictos sustantivos: 37 declaran cierre y 5 declaran apertura)
- hueco restante: **cubierta**

### La segunda persona se asocia a HS y a TR

- artefacto fuente: `reports/eda/fase2_tests_categoricos.csv`
- evidencia historica: sobrerrepresentacion FP vs TN 1.1757 en HS
- mecanismo derivado: conteo y densidad de segunda persona
- experimentos encontrados: 10 (7 declaran cierre, 1 abiertos)
- ids: HIB-H12; V8-00; V8-01; V8-09; V12-01; V12-02; V12-03; V12-05; V13-01; V31-04
- resultado registrado: {"f1_media_columnas": 0.7836, "f1_macro6": 0.8275, "f1_micro": 0.7812, "f1_samples": 0.3228, "exact_match": 0.7221, "f1_por_etiqueta": {"HS": 0.7734, "TR": 0.8159, "AG": 0.7614}, "std_pliegues": 0.014
- cerrada: **True** (10 veredictos sustantivos: 7 declaran cierre y 1 declaran apertura)
- hueco restante: **cubierta**

### La longitud distingue clases y correlaciona con el error

- artefacto fuente: `reports/eda/longitudes_por_etiqueta.csv`
- evidencia historica: fase2_longitudes_por_clase.csv
- mecanismo derivado: longitud como feature y como condicion de binarizacion
- experimentos encontrados: 27 (17 declaran cierre, 3 abiertos)
- ids: DL-01; DL-02; DL-03; E2-05-CH-TGT; E2-05-CH-TGT-DIST; E2-06-TXT-CH; E2-06-TXT-TGT-SPARSE; AUD-02; E-CORP-06; E-CORP-08; EBD-19; V6-03; V6-06; V8-11; ...
- resultado registrado: {"f1_oof_media_columnas": 0.794854, "f1_por_pliegue": [0.812911, 0.781653, 0.753066, 0.797838, 0.829036], "std_pliegues": 0.026168, "f1_por_etiqueta": {"HS": 0.7838014369693012, "TR": 0.82173498570066
- cerrada: **True** (25 veredictos sustantivos: 17 declaran cierre y 3 declaran apertura)
- hueco restante: **cubierta**

### La elongacion y el leet marcan registro coloquial

- artefacto fuente: `reports/eda/fase2_textometria_por_clase.csv`
- evidencia historica: registro coloquial
- mecanismo derivado: normalizacion y conteo de elongation y leet
- experimentos encontrados: 19 (12 declaran cierre, 2 abiertos)
- ids: PRE-06; REP-01; REP-02; REP-03; REP-04; REP-05; REP-06; DL-02; E2-10-ORT-A; E2-10-ORT-ABCD; AUD-03; E-CORP-01; E-CORP-06; EBD-22; ...
- resultado registrado: {"f1_media_columnas": 0.7586, "f1_macro6": 0.8086, "f1_micro": 0.7555, "f1_samples": 0.3028, "exact_match": 0.7008, "f1_por_etiqueta": {"HS": 0.742, "TR": 0.8038, "AG": 0.73}, "std_pliegues": 0.0151, 
- cerrada: **True** (18 veredictos sustantivos: 12 declaran cierre y 2 declaran apertura)
- hueco restante: **cubierta**

### La puntuacion interna y las risas matizan la agresividad

- artefacto fuente: `reports/eda/fase2_textometria_por_clase.csv`
- evidencia historica: textometria de superficie
- mecanismo derivado: puntuacion interna y risas como feature de AG
- experimentos encontrados: 11 (8 declaran cierre, 0 abiertos)
- ids: PRE-07; HIB-H11; E2-09-EMO-EST; E2-10-ORT-ABCD; V6-04; V6-05; V7-03; V29-01; V29-04; V30-01; V41-01
- resultado registrado: {"f1_media_columnas": 0.7587, "f1_macro6": 0.8087, "f1_micro": 0.7557, "f1_samples": 0.3028, "exact_match": 0.7013, "f1_por_etiqueta": {"HS": 0.7429, "TR": 0.8026, "AG": 0.7306}, "std_pliegues": 0.014
- cerrada: **True** (8 veredictos sustantivos: 8 declaran cierre y 0 declaran apertura)
- hueco restante: **cubierta**

### La atribucion y el retweet aparecen en la distribucion de TR

- artefacto fuente: `reports/eda/fase2_rasgos_TR.csv`
- evidencia historica: caracterizacion de TR
- mecanismo derivado: marcadores de atribucion y presencia de RT
- experimentos encontrados: 5 (5 declaran cierre, 0 abiertos)
- ids: HIB-H12; E2-07-CITA; E2-07-CITA-TXT; E2-07-CITA-NEG-TXT; V8-03
- resultado registrado: {"f1_media_columnas": 0.7836, "f1_macro6": 0.8275, "f1_micro": 0.7812, "f1_samples": 0.3228, "exact_match": 0.7221, "f1_por_etiqueta": {"HS": 0.7734, "TR": 0.8159, "AG": 0.7614}, "std_pliegues": 0.014
- cerrada: **True** (5 veredictos sustantivos: 5 declaran cierre y 0 declaran apertura)
- hueco restante: **cubierta**

### Los hashtags y las URLs senalan registro digital y no toxicity

- artefacto fuente: `reports/eda/fase2_tests_categoricos.csv`
- evidencia historica: test categorico
- mecanismo derivado: conteo de hashtags y URLs, e interaccion con longitud
- experimentos encontrados: 25 (11 declaran cierre, 4 abiertos)
- ids: PRE-03; PRE-05; E2-13-META-ENT-RIDGE; E2-13-META-ENT-LR; E2-13-ENT-SOLO-LR; E2-16-MEJOR; E2-20-ESPECIALISTAS; AUD-02; V4-A01; E-CORP-01; E-CORP-06; V9-03; V28-03; V29-01; ...
- resultado registrado: {"f1_media_columnas": 0.7623, "f1_macro6": 0.8116, "f1_micro": 0.7596, "f1_samples": 0.3054, "exact_match": 0.7048, "f1_por_etiqueta": {"HS": 0.7497, "TR": 0.8019, "AG": 0.7352}, "std_pliegues": 0.017
- cerrada: **True** (21 veredictos sustantivos: 11 declaran cierre y 4 declaran apertura)
- hueco restante: **cubierta**

### Los perfiles de POS por texto separan las tres etiquetas

- artefacto fuente: `reports/eda/fase2_textometria_por_clase.csv`
- evidencia historica: textometria por clase
- mecanismo derivado: perfil de distribucion POS como representacion
- experimentos encontrados: 9 (2 declaran cierre, 0 abiertos)
- ids: PA-01; HOLD-01; EBD-01; V10-00; V10-01; V10-08; V11-00; V15-00; V29-02
- resultado registrado: {"f1_media_columnas": 0.7838245474334548, "f1_publicada_solo_near_dup": 0.7926588861781015, "diferencia": -0.008834, "std_pliegues": 0.01855536946824282, "resumen_grupos": {"n_tweets": 1731, "tweets_c
- cerrada: **True** (8 veredictos sustantivos: 2 declaran cierre y 0 declaran apertura)
- hueco restante: **cubierta**

### La estructura de etiquetas tiene combinaciones casi-imposibles

- artefacto fuente: `reports/eda/combinaciones_etiquetas.csv`
- evidencia historica: TR=1 solo con HS=1
- mecanismo derivado: regla dura o suave sobre la combinacion de etiquetas
- experimentos encontrados: 46 (32 declaran cierre, 2 abiertos)
- ids: TH-01; S03; ENS-04; OPT-01; E2-01-HS-TR-AG; E2-01-HS-TR-AG-STD; E2-01-HS-AG-TR; E2-01-HS-AG-TR-STD; E2-01-AG-HS-TR; E2-01-AG-HS-TR-STD; E2-14-MEJOR-M2; E2-14-MEJOR-META; E2-13-META-ENT-RIDGE; E2-13-META-ENT-LR; ...
- resultado registrado: {"f1_media_columnas_umbral_0.5": 0.7926588861781015, "f1_media_columnas_umbrales_optimos": 0.7989327027697396, "ganancia": 0.0062738165916380595, "macro6_umbral_0.5": 0.8349033705135338, "macro6_umbra
- cerrada: **True** (43 veredictos sustantivos: 32 declaran cierre y 2 declaran apertura)
- hueco restante: **cubierta**

### Los grupos de near-duplicate y pseudo-autor cambian la validacion

- artefacto fuente: `reports/eda/clusters_near_duplicados.csv`
- evidencia historica: clusters y pseudo_autores.csv
- mecanismo derivado: agrupacion para la validacion cruzada
- experimentos encontrados: 42 (20 declaran cierre, 0 abiertos)
- ids: HIB-H14; PA-01; HOLD-01; V2-BASE-M1_word_lr; V2-BASE-M2_union_lr_c2; V2-BASE-M3_union_svc; V2-BASE-M4_char_lr; V4-00-SCAFFOLD; V4-A01; V4-F01-F04; E-CORP-01; E-CORP-06; E-CORP-08; E-CORP-09; ...
- resultado registrado: {"f1_media_columnas": 0.763, "f1_macro6": 0.8112, "f1_micro": 0.7607, "f1_samples": 0.3167, "exact_match": 0.7025, "f1_por_etiqueta": {"HS": 0.7572, "TR": 0.7943, "AG": 0.7374}, "std_pliegues": 0.0136
- cerrada: **True** (39 veredictos sustantivos: 20 declaran cierre y 0 declaran apertura)
- hueco restante: **cubierta**

### La rareza de la etiqueta condiciona el error

- artefacto fuente: `reports/eda/distribucion_etiquetas.csv`
- evidencia historica: distribucion de etiquetas
- mecanismo derivado: poblacion de etiqueta rara como condicion
- experimentos encontrados: 1 (0 declaran cierre, 1 abiertos)
- ids: V4-B01
- resultado registrado: p_true vs error por etiqueta r = -0.795/-0.810/-0.801; media de p_true vs n_errores r = -0.841; BCE media r = -0.136; cercania a la frontera r = +0.345; violacion del umbral r = +0.811; 441 ejemplos (
- cerrada: **False** (1 veredictos sustantivos: 0 declaran cierre y 1 declaran apertura)
- hueco restante: **abierta**

## 4. Resumen

- hallazgos auditados: 15
- con mecanismo cerrado: 14
- sin ningun experimento: 0
- abiertos: 1

