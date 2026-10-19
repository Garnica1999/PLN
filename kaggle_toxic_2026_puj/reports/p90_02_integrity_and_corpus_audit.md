# P90-02 - Auditoria integral de V03, corpus, EDA y trazabilidad experimental

Generado por `experiments/p90_02_auditoria_integral.py` y `experiments/p90_02b_informe_y_handoff.py`.
Commit auditado: `e43c594917fc643d0e300b73646aefb7dbd6b61d`. Fecha: 2026-10-10. Sin entrenamiento, sin test, sin holdout,
sin submissions y sin commits.

## 1. Resumen ejecutivo

La auditoria reproduce integramente la evidencia de V03 y del failure analysis P90-00 y la
clasifica. Resultados principales: (1) la cadena del texto raw -> casebook -> entrada de V03 no
tiene perdida: 0 diferencias en 441 tweets y
0 en 757 celdas; las formas sin diacriticos
("Espaa", "rabes") ya estan en la fuente autorizada (clasificacion `SOURCE_ALREADY_NORMALIZED`,
documentada en D-010). (2) `corrected_cv` = macro pooled = 0.7772114525165722
reproducido a 1e-16; los F1 por etiqueta reportados son los pooled y coinciden; la media entre
folds es un nivel de agregacion distinto y no la metrica canonica. (3) El casebook queda
VERIFIED: 441 tweets / 757 celdas alineados 1:1 con el OOF,
controles 757/757 reproducidos con 0 incumplimientos. (4) Las 19
slices se reclasifican en 3 familias (texto / lexico inducido in-sample / subconjuntos definidos
por errores); sus definiciones se reproducen exactamente y los subconjuntos condicionados por
error dejan de reportar F1 cero-por-construccion. (5) El corpus de desarrollo (1483 filas)
queda perfilado por completo. (6) El EDA historico se consolida en 48 lineas con
estado (37 CLOSED, 6 BLOCKED_BY_RULES,
3 DUPLICATE, 1 INVALIDATED,
1 COMPLEMENTARY). Recomendacion: **A. CLOSE_FAILURE_ANALYSIS**,
con los asuntos de integridad (opcion B) resueltos en esta auditoria.

## 2. Estado de Git

- Rama `main`; HEAD `e43c594917fc643d0e300b73646aefb7dbd6b61d` (P90-00); arbol limpio al iniciar; 8 commits por
  delante de `origin/main`; sin push.
- Commits P89 (`a07e080`) y P90-00 (`e43c594`) presentes en la genealogia.
- Esta tarea no realiza commits: los entregables quedan como archivos nuevos (y el notebook
  modificado) sin registrar en git.

## 3. Baseline

V03 congelado: TF-IDF word (1,2) + char_wb (3,5), min_df=2, sublinear_tf, lowercase;
LogisticRegression C=2.0, class_weight=balanced, solver=liblinear, random_state=42; umbrales
HS 0.44 / TR 0.51 / AG 0.46; `corrected_cv` = 0.7772114525165722;
leaderboard historico 0.78306. Hashes: OOF `559265816ce4f631...`, submission
`7d7478f271e6edfd...`, modelo `263ed5bd0fc0de7c...`, train
`3d40e7345b588860...`, particion `e5b62e5f22c4c63a...`.
Procedencia del OOF: features/v2_oof_base_corrected.npz, modelo 'M2_union_lr_c2' (fila 2 de 4; mismos ids, grupos y pliegues corregidos v2).

## 4. Alcance autorizado

Corpus de desarrollo autorizado: las 1483 filas con prediccion OOF (ids del artefacto
canonico), del archivo `data/raw/toxic_train.csv` (1731 filas; las 248 restantes son la
particion reservada, no leida). Exclusiones: test (no tocado), holdout (agotado, no tocado),
holdout2 (congelado, no tocado). El artefacto `oof_mod03b` (que cubre el train completo con
filas de la particion reservada) NO se abrio: la tabla historica del train-1731 se cita desde
`reports/error_analysis.md` sin recalcular. Limitacion: el escaneo de bytes no-ASCII de todo el
archivo se hizo como verificacion de integridad; el analisis de contenido se limito a las
1483 filas de desarrollo.

## 5. Fuentes utilizadas

| artifact_id | path | sha256[:16] | tracked | status |
|---|---|---|---|---|
| oof_canonico | features/oof_v03_final.npz | 559265816ce4f631 | SI | OK |
| train_raw | data/raw/toxic_train.csv | 3d40e7345b588860 | NO | OK |
| particion | data/processed/holdout_v2_ids.csv | e5b62e5f22c4c63a | NO | OK |
| grupos | data/processed/grupos_near_duplicados.csv | afabf9ae1c295ea8 | NO | OK |
| pipeline_v03 | experiments/baseline_v03_frozen/pipeline_v03.json | 89ee399ae849296f | SI | OK |
| hashes_v03 | experiments/baseline_v03_frozen/hashes.txt | 3e3b9039015a592d | SI | OK |
| submission_v03 | submissions/toxic_submission_v03_lr_union_thr.csv | 7d7478f271e6edfd | SI | OK |
| modelo_v03 | experiments/baseline_v03_frozen/model_v03.joblib | 263ed5bd0fc0de7c | SI | OK |
| casebook_md | reports/v03_error_casebook.md | 3fb2d120acd7f298 | SI | OK |
| casebook_csv | reports/v03_error_casebook.csv | b14d6ed21f011dc1 | SI | OK |
| cells_csv | reports/v03_error_cells.csv | 6a582e67d6450bcf | SI | OK |
| fold_csv | reports/v03_error_fold_summary.csv | ad16825ae8b9931d | SI | OK |
| slice_csv | reports/v03_error_slice_summary.csv | 0d19379e00e3c1f2 | SI | OK |
| confidence_csv | reports/v03_error_confidence_summary.csv | c729ff05b93d961b | SI | OK |
| sample_csv | reports/v03_error_review_sample.csv | 33272bbc92a65e81 | SI | OK |
| controls_csv | reports/v03_error_controls.csv | 75bc065a15e5a0b5 | SI | OK |
| provenance_json | reports/v03_error_provenance.json | 76438dec77a4ac55 | SI | OK |
| lexico_fase2 | reports/eda/fase2_lexico_inducido_agresividad.csv | abcb61f0d6f93f33 | SI | OK |
| familias_p57 | reports/p57_family_status.csv | e5de8a4de973ef22 | SI | OK |
| oportunidades_csv | reports/toxic_remaining_legal_opportunities.csv | 282a1603dd7bc21a | SI | OK |
| experiments_csv | experiments.csv | 5b2994fee24e3174 | SI | OK |
| decisions | DECISIONS.md | be31c9b70783f16a | SI | OK |

## 6. Auditoria de integridad textual

Cadena verificada: texto de origen (`data/raw/toxic_train.csv`) -> texto recuperado para el
casebook (`v03_error_casebook.csv` / `v03_error_cells.csv`) -> entrada real de V03 (texto
original + `lowercase` del vectorizador; `strip_accents=None` en los scripts de entrenamiento).

| chequeo | resultado | estado |
|---|---|---|
| identidad_raw_casebook | 0 diferencias | MATCH_CONFIRMED |
| identidad_raw_celdas | 0 diferencias | MATCH_CONFIRMED |
| ascii_del_desarrollo | 0 textos con caracteres > 127 | SOURCE_ALREADY_NORMALIZED |
| formas_sin_diacriticos_en_el_raw | {"Espaa": 28, "agresin": 4, "estacin": 3, "polica": 5, "rabes": 59, "inmigracin": 29} | SOURCE_ALREADY_NORMALIZED |
| vectorizadores_de_v03 | word={'analyzer': 'word', 'ngram_range': [1, 2], 'min_df': 2, 'sublinear_tf': True, 'lower | MATCH_CONFIRMED |
| documentacion_historica | D-010 declara que el corpus llega sin tildes ni enye y que train es 100 % ASCII | MATCH_CONFIRMED |

Clasificacion: **SOURCE_ALREADY_NORMALIZED**. Evidencia: las formas sin diacriticos aparecen en
la fuente autorizada (Espaa: 28, rabes: 59,
inmigracin: 29, agresin: 4,
estacin: 3, polica: 5 en
el desarrollo); 0 de 1483 textos tienen bytes > 127; D-010 documenta
que "los organizadores normalizaron a ASCII". No hay operacion de retirada de diacriticos en la
ruta de V03. Conclusion: no procede corregir ni reentrenar; la ausencia de diacriticos es una
propiedad del dato y una limitacion documentada.

## 7. Conciliacion de metricas

Matrices pooled (todas las predicciones OOF agregadas, umbrales congelados):

| etiqueta | TP | FP | FN | TN | F1 pooled |
|---|---|---|---|---|---|
| HS | 505 | 173 | 143 | 662 | 0.761689 |
| TR | 363 | 82 | 82 | 956 | 0.815730 |
| AG | 425 | 135 | 142 | 781 | 0.754215 |

Precision / recall / F1 (precision completa en el CSV):

| etiqueta | precision | recall | F1 |
|---|---|---|---|
| HS | 0.74483775 | 0.77932098 | 0.76168929 |
| TR | 0.81573033 | 0.81573033 | 0.81573033 |
| AG | 0.75892857 | 0.74955908 | 0.75421472 |

F1 por fold (n = tamano del fold):

| fold | HS | TR | AG |
|---|---|---|---|
| fold 1 (n=297) | 0.778626 | 0.827586 | 0.781818 |
| fold 2 (n=297) | 0.783582 | 0.827957 | 0.770563 |
| fold 3 (n=296) | 0.768116 | 0.815217 | 0.753247 |
| fold 4 (n=296) | 0.729323 | 0.782123 | 0.704348 |
| fold 5 (n=297) | 0.748031 | 0.826347 | 0.762791 |

Niveles de agregacion (no intercambiables):

| metrica | valor | definicion |
|---|---|---|
| corrected_cv (canonica) | 0.7772114525165722 | macro pooled: media de F1 pooled por etiqueta |
| media entre folds por etiqueta | {"HS": 0.761536, "TR": 0.815846, "AG": 0.754553} | media aritmetica de 5 F1 por etiqueta |
| media de f1_pliegues | 0.777312 | media de los 5 macro-F1 por fold |

**ORIGINAL / RECOMPUTED / EXPLANATION** de la discrepancia percibida: ORIGINAL: las cifras
citadas como matrices globales (TP 614/431/517) provienen de la tabla historica del TRAIN
completo de 1731 filas con MOD-03b en `reports/error_analysis.md` (umbral optimo), una
poblacion distinta. RECOMPUTED: sobre el desarrollo de 1483 filas las matrices pooled son
505/173/143/662, 363/82/82/956 y 425/135/142/781; sus F1 coinciden exactamente con los F1 por
etiqueta reportados (diferencias <= 1.1e-16). EXPLANATION: no hay discrepancia de datos sino
mezcla de poblaciones y de niveles de agregacion; `corrected_cv` es el macro pooled y la media
entre folds (0.777312) es otra cantidad. las cifras citadas como discrepantes (TP 614/431/517) proceden de la tabla historica del TRAIN completo de 1731 filas con MOD-03b (reports/error_analysis.md), una poblacion distinta del desarrollo de 1483 filas; las matrices del desarrollo son 505/173/143/662, 363/82/82/956 y 425/135/142/781, y sus F1 coinciden exactamente con los F1 por etiqueta reportados (0.761689/0.815730/0.754215). corrected_cv es el macro pooled y no la media entre folds (0.7773117), que es un nivel de agregacion distinto.

Por que el promedio de F1 entre folds difiere del F1 pooled: F1 es una razon no lineal; el
promedio de razones (con denominadores por fold) no es la razon de las sumas (denominador
agregado), y los folds tienen tamanos y prevalencias distintas (ver tabla). Criterio de cierre:
filas, ids, umbrales y predicciones coinciden; matrices y F1 por fold se reproducen;
`corrected_cv` se obtiene con su definicion; discrepancias abiertas: 0.

## 8. Validacion del casebook

| chequeo | esperado | observado | estado |
|---|---|---|---|
| filas_casebook | 441 | 441 | OK |
| filas_celdas | 757 | 757 | OK |
| ids_fuera_del_desarrollo | 0 | 0 | OK |
| probabilidades_alineadas | 0 desalineadas | 0 | OK |
| predicciones_alineadas | 0 desalineadas | 0 | OK |
| patrones_y_conteos | 0 desviaciones | 0 | OK |
| folds | 0 desviaciones | 0 | OK |
| celdas_alineadas | 0 desviaciones | 0 | OK |
| controles_reproducidos_1a1 | 757 celdas emparejadas | 757 reproducidos; sin control: 0 | OK |
| controles_cumplen_criterios | 0 incumplimientos | 0 | OK |
| controles_repetidos | documentar | 207 | INFORMATIVO |
| muestra_errores_y_controles | 100 + 185 | 100 + 185 | OK |
| muestra_motivos | 0 sin motivo | 0 | OK |

Recuentos verificados: 441 tweets con al menos un error (223
con una etiqueta, 120 con dos, 98 con las tres) y
1042 tweets sin error; 757 celdas tweet-etiqueta. Veredicto:
**VERIFIED** (alineacion 1:1 con el OOF canonico, sin ids duplicados ni fuera de alcance).

## 9. Validacion de la muestra y de los controles

| chequeo | valor | nota |
|---|---|---|
| controles_totales | 757 | 1:1 por celda de error |
| controles_unicos | 454 | un control puede repetirse |
| controles_incumplen_criterios | 0 | etiqueta+fold+y_true+direccion |
| muestra_errores | 100 | tope 100 |
| muestra_controles | 185 | controles de los errores seleccionados |
| muestra_no_es_aleatoria | CONFIRMADO | estratificada; no estimar prevalencias poblacionales |

Los controles se reprodujeron con el procedimiento determinista completo: para cada celda de
error, el candidato correcto con misma etiqueta+fold+y_true y vecino mas cercano por
|delta distancia| (desempate por id). Resultado: 757/757 reproducidos,
0 incumplimientos, 0 celdas sin control; 207 reutilizaciones de control
(un mismo tweet correcto puede emparejarse con varios errores). La muestra (100 errores + 185
controles) es estratificada, NO aleatoria: no admite estimaciones de prevalencia poblacional;
los controles 1:1 comparten etiqueta, fold y valor verdadero con su error (no son
independientes). La columna `matching_criteria` registra el |delta_distancia| efectivo, que
puede ser grande: el criterio real es "vecino mas cercano", no un umbral fijo.

## 10. Taxonomia de slices

| slice | familia | definicion | n_rows | limitacion |
|---|---|---|---|---|
| URLs | SLICE_TEXT_DEFINED | URL_PATTERN (06) | 458 | ninguno |
| hashtags | SLICE_TEXT_DEFINED | HASHTAG_PATTERN (06) | 144 | ninguno |
| menciones | SLICE_TEXT_DEFINED | MENTION_PATTERN (06) | 702 | ninguno |
| mencion_inicial | SLICE_TEXT_DEFINED | INITIAL_MENTION_PATTERN (06) | 605 | ninguno |
| elongaciones | SLICE_TEXT_DEFINED | ELONGATION_PATTERN (06) | 76 | ninguno |
| mayusculas_sostenidas | SLICE_TEXT_DEFINED | UPPER_WORD_PATTERN o >=4 mayusculas (06) | 864 | ninguno |
| coloquial | SLICE_TEXT_DEFINED | patron por token del grupo coloquial (adaptacion registrada) | 240 | ninguno |
| colectivos_superficie | SLICE_TEXT_DEFINED | cuantificadores colectivos o plural tras determinante (06+spaCy) | 363 | ninguno |
| segunda_persona | SLICE_TEXT_DEFINED | pronombres/verbos de segunda persona o imperativo (06+spaCy) | 680 | ninguno |
| imperativo | SLICE_TEXT_DEFINED | verbos en imperativo (06+spaCy) | 335 | ninguno |
| cita | SLICE_TEXT_DEFINED | texto contiene comillas dobles | 33 | ninguno |
| insultos | SLICE_LEXICON_INDUCED | lemas en el lexico inducido completo (in-sample) | 1114 | lexico inducido in-sample: diagnostico, no independi |
| insultos_nucleo | SLICE_LEXICON_INDUCED | lemas con z_score >= 2.5 (in-sample) | 901 | lexico inducido in-sample: diagnostico, no independi |
| interpersonal | SLICE_LEXICON_INDUCED | insulto + mencion inicial + sin referencia colectiva (region 06) | 458 | lexico inducido in-sample: diagnostico, no independi |
| sin_odio_explicito | SLICE_ERROR_CONDITIONED | FN_HS sin la subcadena "insulto" (IMPLICIT v4) | 143 | subconjunto definido por errores: F1 interno cero po |
| colectivos_v4 | SLICE_ERROR_CONDITIONED | errores de AG con AG verdadero 1 (COLLECTIVE v4) | 142 | subconjunto definido por errores: F1 interno cero po |
| afiliacion | SLICE_ERROR_CONDITIONED | FP_HS cuyo texto contiene @ (AFFILIATIVE v4) | 83 | subconjunto definido por errores: F1 interno cero po |
| interpersonal_v4 | SLICE_ERROR_CONDITIONED | errores de TR con TR verdadero 1 (DIRECTED v4) | 82 | subconjunto definido por errores: F1 interno cero po |
| news | SLICE_ERROR_CONDITIONED | error cuyo texto contiene "http" (NEWS v4) | 156 | subconjunto definido por errores: F1 interno cero po |

Reclasificacion frente al CSV historico (que etiquetaba como "rebanada_de_texto" tambien las de
lexico inducido): 11 slices de
texto, 3 de lexico inducido
in-sample (solo diagnostico: el detector uso las etiquetas de las mismas observaciones) y
5 subconjuntos definidos por
errores (taxonomia y recuentos; su F1 interno es cero por construccion y no se reporta).

## 11. EDA general del corpus autorizado

| metrica | valor |
|---|---|
| filas_autorizadas | 1483 |
| ids_unicos | 1483 |
| textos_vacios | 0 |
| duplicados_exactos_de_texto | 0 |
| caracteres_min_media_mediana_max | 10 / 104.7 / 92 / 306 |
| caracteres_p10_p90 | 41 / 193 |
| tokens_min_media_mediana_max | 1 / 17.2 / 15 / 57 |
| tweets_cortos_menos_25_tokens | 1182 |
| tweets_largos_25_o_mas | 301 |
| vocabulario_tipos | 8888 |
| vocabulario_tokens | 25519 |
| hapax | 7045 |
| top20_tokens | de:994; que:667; la:627; y:523; a:523; en:408; puta:362; el:302; no:285; un:236; es:229; los:226; te |
| con_url | 458 |
| con_hashtag | 144 |
| con_mencion | 702 |
| con_mencion_inicial | 605 |
| con_elongacion | 76 |
| con_mayusculas_sostenidas | 864 |
| con_puntuacion_repetida | 84 |
| con_simbolo_no_alfanumerico | 1291 |
| caracteres_no_ascii | 0 |
| grupos_near_dup_dev | 1479 |
| filas_en_grupos_no_unitarios | 6 |
| grupos_de_5_o_mas | 1 |

## 12. Estructura de las labels

Marginales:

| etiqueta | positivos | proporcion |
|---|---|---|
| HS | 648 | 0.436952 |
| TR | 445 | 0.300067 |
| AG | 567 | 0.382333 |

Ocho combinaciones (las dos no observadas se listan con 0):

| combinacion | n | proporcion |
|---|---|---|
| HS=0 TR=0 AG=0 | 813 | 0.548213 |
| HS=0 TR=0 AG=1 | 22 | 0.014835 |
| HS=1 TR=0 AG=0 | 79 | 0.053270 |
| HS=1 TR=0 AG=1 | 124 | 0.083614 |
| HS=1 TR=1 AG=0 | 24 | 0.016183 |
| HS=1 TR=1 AG=1 | 421 | 0.283884 |
| HS=0 TR=1 AG=0 | 0 | 0.000000 |
| HS=0 TR=1 AG=1 | 0 | 0.000000 |

Dependencias (con denominadores):

| probabilidad | numerador | denominador | valor |
|---|---|---|---|
| P(TR=1|HS=1) | 445 | 648 | 0.686728 |
| P(TR=1|HS=0) | 0 | 835 | 0.000000 |
| P(AG=1|HS=1) | 545 | 648 | 0.841049 |
| P(AG=1|HS=0) | 22 | 835 | 0.026347 |

Identidad estructural: TR=1 con HS=0 tiene **0 casos** en el desarrollo (445/445 TR positivos
tienen HS=1). Esta identidad esta documentada historicamente (D-001: "TR=1 solo ocurre cuando
HS=1"; D-044/D-045: "TR=1 implica HS=1 de forma determinista") y su explotacion modelistica
(factorizacion p(TR|X)=p(HS|X)*p(TR|HS=1,X), familia joint_labels) fue explorada y CERRADA.
AG=1 con HS=0 ocurre en 22 casos (2.6 por ciento condicional; 1.5 por ciento historico en el
train completo).

## 13. Calidad de los datos

Sin textos vacios, sin ids duplicados, sin duplicados exactos de texto, sin caracteres no ASCII
en el desarrollo. Longitudes: 10-306 caracteres (media
104.7, mediana 92); 1-57
tokens (media 17.21). Vocabulario: 8888 tipos,
25519 tokens. Limitaciones: la particion reservada no se inspecciono; el
analisis describe solo el desarrollo.

## 14. Perfil textual

Longitud y estructura (definiciones en el CSV): tweets cortos (<25 tokens) 1182 vs
largos 301; URLs 458, hashtags 144, menciones 702 (605 iniciales), elongaciones 76, mayusculas
sostenidas 864, puntuacion repetida 84. Tokens mas frecuentes del texto original: de, que, la,
y, a, en, **puta**, el, no, un, es, los, te, una, **perra**, lo, me, se, **eres**, tu (marcador
lexico de misoginia en posiciones altas del ranking; descriptivo, no causal). El lexico de
insultos del proyecto (inducido in-sample, z>=2.5) cubre 1114 de 1483 textos (901 con
nucleo).

## 15. Duplicados y grupos

Grupos near-dup historicos (`data/processed/grupos_near_duplicados.csv`, metadatos): 1479 grupos
distintos tocan el desarrollo; solo 6 filas (0.4 por ciento) pertenecen a
grupos no unitarios: los grupos 112 y 338
(3 filas dev cada uno; tamanos declarados 5 y 4, con el resto de sus filas fuera del desarrollo).
La dependencia por near-duplicados en desarrollo es practicamente nula; el mecanismo de grupos
se uso para el resampling del bootstrap historico y no se modifico. No se afirma que el ruido
de etiquetado explique los errores: no hay evidencia especifica.

## 16. Sintesis de errores por etiqueta

| etiqueta | TP | FP | FN | TN | precision | recall | F1 | errores | cerca<0.025 | medio | lejos>=0.10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HS | 505 | 173 | 143 | 662 | 0.744838 | 0.779321 | 0.761689 | 316 | 40 | 87 | 189 |
| TR | 363 | 82 | 82 | 956 | 0.81573 | 0.81573 | 0.81573 | 164 | 13 | 40 | 111 |
| AG | 425 | 135 | 142 | 781 | 0.758929 | 0.749559 | 0.754215 | 277 | 31 | 80 | 166 |

## 17. Errores conjuntos

| patron | n |
|---|---|
| HS_FP | 72 |
| HS_FP|TR_FP|AG_FP | 56 |
| HS_FN | 48 |
| HS_FN|AG_FN | 47 |
| AG_FN | 41 |
| HS_FN|TR_FN|AG_FN | 40 |
| AG_FP | 35 |
| HS_FP|AG_FP | 34 |
| TR_FN | 18 |
| TR_FN|AG_FN | 14 |
| HS_FP|TR_FP | 11 |
| TR_FP | 9 |
| HS_FN|TR_FN | 6 |
| TR_FP|AG_FP | 6 |
| HS_FN|TR_FN|AG_FP | 2 |
| TR_FN|AG_FP | 2 |

Intersecciones por pares de etiquetas y errores aislados:

| interseccion | n | de | nota |
|---|---|---|---|
| HS+TR | 115 | 316 HS y 164 TR | errores simultaneos; no implica causa co |
| HS+AG | 179 | 316 HS y 277 AG | errores simultaneos; no implica causa co |
| TR+AG | 120 | 164 TR y 277 AG | errores simultaneos; no implica causa co |
| HS_solo | 120 | errores de esa etiqueta | errores aislados |
| TR_solo | 27 | errores de esa etiqueta | errores aislados |
| AG_solo | 76 | errores de esa etiqueta | errores aislados |
| tres_etiquetas | 98 | todos los errores | 98 historicos verificados |

Errores de las tres etiquetas por estrato de distancia (98 tweets):

| estrato | n |
|---|---|
| HS_cerca_<0.025_tres_etiquetas | 8 |
| HS_lejos_>=0.100_tres_etiquetas | 67 |
| TR_cerca_<0.025_tres_etiquetas | 8 |
| TR_lejos_>=0.100_tres_etiquetas | 76 |
| AG_cerca_<0.025_tres_etiquetas | 5 |
| AG_lejos_>=0.100_tres_etiquetas | 75 |

Los errores triples son mayoritariamente lejanos al umbral (67-76 de 98 en HS/TR): no son un
problema de frontera. No se agrupan bajo una unica causa.

Hipotesis de FP/FN evaluadas (recuento, nivel de evidencia y contraejemplo declarado):

| hipotesis | evaluacion |
|---|---|
| FP_HS_con_marcadores_broma_afiliacion | 27/173 (15.6%) | SUPPORTED | contraejemplo: FP_HS sin marcadores con probabilidad >= 0.9 (p. ej. 21892) |
| FP_AG_con_marcadores_broma_afiliacion | 26/135 (19.3%) | SUPPORTED | misma evidencia que FP_HS; subconjunto, no explicacion completa |
| FN_AG_con_referencia_colectiva_superficie | 16/142 (11.3%) | TENTATIVE | lista ilustrativa; el slice spaCy de colectivos da el recuento sistematico |
| FN_HS_en_textos_con_comilla | 5/143 (3.5%) | TENTATIVE | QUOTED v4 tiene 16 errores en total; minoritario |
| errores_fronterizos_menos_0.025 | 84/757 (11.1%) | CONFIRMED | porcentaje directo; no implica causa de decision |
| errores_lejanos_mas_0.100 | 466/757 (61.6%) | CONFIRMED | limita el potencial de una solucion basada solo en thresholds |

## 18. Distancias a los thresholds

Frontera (<0.025): 84 de 757 celdas
(11.1 por ciento). Lejanos (>=0.10):
466 de 757 (61.6
por ciento). La distancia al umbral NO es una probabilidad calibrada ("confianza real"); el
resultado limita el potencial de una solucion basada exclusivamente en reajustar thresholds
(familia threshold_transfer CLOSED, P55/P56).

## 19. Resultados por slice

Slices de texto y de lexico inducido (F1 dentro / fuera; lexico in-sample = diagnostico):

| slice | familia | n | F1 HS in | F1 TR in | F1 AG in | F1 HS out | F1 TR out | F1 AG out |
|---|---|---|---|---|---|---|---|---|
| URLs | TEXT_DEFINED | 458 | 0.662651 | 0.682927 | 0.627737 | 0.794769 | 0.837027 | 0.794842 |
| hashtags | TEXT_DEFINED | 144 | 0.593407 | 0.5 | 0.5 | 0.774089 | 0.82448 | 0.76851 |
| menciones | TEXT_DEFINED | 702 | 0.832653 | 0.866774 | 0.839098 | 0.673435 | 0.696629 | 0.632035 |
| mencion_inicial | TEXT_DEFINED | 605 | 0.847561 | 0.870432 | 0.856209 | 0.677612 | 0.701389 | 0.63301 |
| elongaciones | TEXT_DEFINED | 76 | 0.710526 | 0.861538 | 0.742857 | 0.7648 | 0.812121 | 0.754967 |
| mayusculas_sostenidas | TEXT_DEFINED | 864 | 0.731988 | 0.762136 | 0.708263 | 0.794304 | 0.861925 | 0.805243 |
| coloquial | TEXT_DEFINED | 240 | 0.798354 | 0.854271 | 0.794521 | 0.753463 | 0.804631 | 0.744493 |
| colectivos_superficie | TEXT_DEFINED | 363 | 0.677419 | 0.637681 | 0.522613 | 0.787402 | 0.830694 | 0.803879 |
| segunda_persona | TEXT_DEFINED | 680 | 0.801187 | 0.820896 | 0.801325 | 0.720859 | 0.80791 | 0.699809 |
| imperativo | TEXT_DEFINED | 335 | 0.769759 | 0.813725 | 0.776471 | 0.75942 | 0.816327 | 0.747706 |
| cita | TEXT_DEFINED | 33 | 0.6875 | 0.666667 | 0.9 | 0.763524 | 0.816234 | 0.746017 |
| insultos | LEXICON_INDUCED | 1114 | 0.797054 | 0.827586 | 0.773832 | 0.346154 | 0.3 | 0.385965 |
| insultos_nucleo | LEXICON_INDUCED | 901 | 0.822976 | 0.847087 | 0.809176 | 0.515152 | 0.424242 | 0.440476 |
| interpersonal | LEXICON_INDUCED | 458 | 0.883072 | 0.889693 | 0.884477 | 0.669323 | 0.694362 | 0.628272 |

Subconjuntos definidos por errores (recuentos; F1 omitido por construccion):

| slice | n | celdas HS | celdas TR | celdas AG | tweets 1/2/3 etiquetas |
|---|---|---|---|---|---|
| sin_odio_explicito | 143 | 143 | 48 | 89 | 48/53/42 |
| colectivos_v4 | 142 | 87 | 54 | 142 | 41/61/40 |
| afiliacion | 83 | 83 | 41 | 50 | 27/21/35 |
| interpersonal_v4 | 82 | 48 | 82 | 58 | 18/22/42 |
| news | 156 | 112 | 39 | 102 | 84/47/25 |

Solapamiento: 91 pares entre las 14 slices no condicionadas por error; pares con
mayor interseccion:

| slice_a | slice_b | interseccion | jaccard |
|---|---|---|---|
| menciones | mencion_inicial | 605 | 0.861823 |
| insultos | insultos_nucleo | 901 | 0.808797 |
| mencion_inicial | interpersonal | 458 | 0.757025 |
| menciones | interpersonal | 458 | 0.652422 |
| URLs | mayusculas_sostenidas | 440 | 0.498866 |
| segunda_persona | imperativo | 335 | 0.492647 |

No se suman errores entre slices (no son excluyentes); "URLs" (458 filas, descriptiva) y "news"
(156 filas, definida por errores) describen poblaciones distintas y no se comparan como si
fueran la misma.

## 20. Sintesis del EDA historico

Matriz consolidada (48 lineas; CSV completo en `p90_02_historical_eda_matrix.csv`). Familias del
registro historico P57:

| family | status | best_result | cierre |
|---|---|---|---|
| embeddings | CLOSED | E2-19 y V38-01: supervised embeddings CLOSED, 0.595466. Fr | E |
| ordinal | CLOSED | V4-E01-E03 TRACK E = NO_SIGNAL; V13-02 0.777746 (delta +0. | A |
| joint_labels | CLOSED | 6 variantes E2-01-* (HS-TR-AG, HS-AG-TR, AG-HS-TR, con y s | A |
| PLS | INVALIDATED | V40-01 SUPERVISED_PLS 0.763469 (OVERFIT); V43-02 resultado | B |
| VW / optimizacion sparse online | CLOSED | V44-01: FTRL, Coin Betting y PiSTOL sobre la representacio | A |
| contextual_selection | CLOSED | P0 (no hacer nada) = V03. Todas las politicas observables  | A |
| threshold_transfer | CLOSED | P55 reporto 0.7781367700. P56 con anidamiento completo da  | A |
| data_expansion | CLOSED | P47: 1731 filas oficiales, 1483 en dev y 248 en holdout. 0 | E |
| augmentation | CLOSED | V6-08: espaciado, elongacion y repeticion de puntuacion. J | A |
| retrieval | CLOSED | V4-F01-F04: propagacion de etiquetas descartada, techo med | A |
| kernels | CLOSED | V4-X01 candidato D (svd300_rbf) corrected_cv 0.7904, 5/5 p | C |
| FM (factorization machines) | CLOSED | V25-01: k=4, lambda_v=1, 0.782699 bajo D-004. V6-02 Track  | A |
| Tsetlin | CLOSED | V26-03: 0.583397, cortado en K=200 por los criterios de la | A |
| BM25 | CLOSED | V37-01: 0.7744254 | A |
| hashing | CLOSED | V36-01: 0.748427 | A |
| SGD | CLOSED | V35-01: hinge -0.016177, modified_huber -0.033645 | A |
| label_specialization | CLOSED | V28-00 LABEL_SPECIALIZATION = CLOSED. v09_p5: C por etique | C |
| noise (label noise y reweighting) | CLOSED | V30-00 LABEL_NOISE = CLOSED. V4-B03 down_violacion LR 0.77 | A |
| fuzzy_matching | CLOSED | V29-03 REJECTED, FUZZY_MATCHING = INFORMATIONAL_ONLY | A |
| ensembles | CLOSED | ENS-02 anidado 0.791900, descartada por no superar MOD-03b | C |
| stacking y meta-modelos | CLOSED | S08 meta holdout 0.8465 frente a 0.8285 de V03, delta +0.0 | C |
| structural_features | CLOSED | V15-01: 0.775860, NO ADOPTADA. V14-00 0.660630. V32-05 url | A |

## 21. Resultados negativos

Verificados contra `p57_family_status.csv`, `p57_candidate_matrix.csv` y
`reports/toxic_remaining_legal_opportunities.csv`:

- FastText aprende senal, pero no supera el gate de adopcion (familia embeddings CLOSED).
- Word2Vec-control queda por debajo de FastText (documentado en la auditoria de oportunidades).
- Mayor cobertura no garantiza mejor F1 (familia data_expansion CLOSED).
- El corpus mixto no aporta una mejora suficiente.
- El stacking generico falla repetidamente (S08; v08_meta_v2 LB -0.00143).
- La propagacion por near-duplicate no resuelve el problema (familia retrieval CLOSED).
- BCE agregada no constituye un proxy adecuado de dureza.
- La distancia especifica por etiqueta resulta mas util para diagnosticar dificultad.
- Una pequena mejora de CV no garantiza mejor transferencia al leaderboard: cuatro candidatos
  enviados con delta CV positivo cayeron en LB (v23_01 -0.00368; v08_meta_v2 -0.00143;
  v20_01 -0.00643; y el caso de umbrales anidados con delta honesto negativo).

Candidatos historicos con veredicto externo:

| candidato | CV | LB | estado |
|---|---|---|---|
| v23_01 svd300_rbf (V4-X01 candidato D) | 0.7904 | 0.77938 | SI, RECHAZADO |
| v08_meta_v2 (meta de 44 features, Ridge alpha=3) | 0.7874 | 0.78163 | SI, RECHAZADO |
| v20_01 regla HS por interaccion de scores (V4-X0 | 0.7882 | 0.77663 | SI, RECHAZADO. Ademas RECHAZADO antes po |
| E2-13-META-ENT-RIDGE (meta + entropias) | 0.7885 | n/d | NO |
| V03 con umbrales anidados de P56 (T1 = 0,41/0,46 | 0.774547873 | n/d | NO |
| v25_01 FM k=4 | 0.782699 | n/d | NO |
| v16_ch2 correccion de prior con EM | n/d (validac | n/d | NO |

## 22. Lineas cerradas y duplicadas

48 lineas clasificadas: 37 CLOSED,
6 BLOCKED_BY_RULES, 3 DUPLICATE,
1 INVALIDATED, 1 COMPLEMENTARY.
Lineas no cerradas:

| estado | linea |
|---|---|
| INVALIDATED | PLS |
| BLOCKED_BY_RULES | Revisar y, si procede, corregir las etiquetas de TR de desarrollo |
| BLOCKED_BY_RULES | Propagar etiquetas train->test por near-duplicados |
| BLOCKED_BY_RULES | Correccion EM de prevalencia sobre probabilidades del test (H4) |
| BLOCKED_BY_RULES | Expandir los datos etiquetados (mas filas de entrenamiento) |
| DUPLICATE | Embeddings desde cero: mas corpus, dimensiones o hiperparametros |
| BLOCKED_BY_RULES | Transformers, LLM, embeddings preentrenados, etiquetas externas o pseudo-labeling |
| DUPLICATE | P86: Factorization Machine sobre la representacion de V03 |
| BLOCKED_BY_RULES | P87: revision humana de anotaciones en TR |
| DUPLICATE | P88: LightGBM sobre la matriz TF-IDF de V03 |
| COMPLEMENTARY | P90: casebook de errores de V03 |

No se reabre ninguna familia por interes descriptivo de los errores; las variantes de
hipotesis ya probadas se consideran DUPLICATE y no novedosas.

## 23. Posibles nuevas hipotesis

Dentro de las reglas vigentes no se identifica una hipotesis de modelado genuinamente nueva:
las familias del espacio TF-IDF/scores/modelos clasicos estan CLOSED o DUPLICATE, las lineas de
anotacion y datos estan BLOCKED_BY_RULES, y las mejoras locales de CV no transfieren (patron
repetido). Las unicas cuestiones abiertas son de revision humana de anotacion (bloqueada) y de
comprension de los errores lejanos (requiere autorizacion y mecanismo legal no medido). No se
propone ejecutar nada en esta fase.

## 24. Limitaciones metodologicas

- Los controles 1:1 no son independientes ni una muestra aleatoria (lectura cualitativa).
- El lexico inducido es in-sample: las slices de insulto no tienen independencia OOF.
- Los subconjuntos definidos por errores no estiman rendimiento de categorias.
- No hay anotacion humana independiente: ninguna hipotesis de anotacion se demuestra.
- La distancia al umbral no es una probabilidad calibrada.
- El artefacto del train completo con MOD-03b no se abrio (mezcla filas reservadas).
- Los juicios de registro (broma, sarcasmo, cita) son cualitativos y con contraejemplos.
- La reproduccion cubre un unico OOF de 5 folds; no se re-entreno nada (por regla).

## 25. Decisiones finales

- **Decision 1 - V03**: FROZEN. Sin modificaciones.
- **Decision 2 - Calidad del failure analysis**: casebook = VERIFIED; slice_analysis =
  PARTIALLY_VERIFIED (definiciones y cifras reproducidas exactamente; taxonomia de familias
  corregida y documentada; interpretacion de subconjuntos restringida a recuentos);
  metric_reconciliation = VERIFIED; text_provenance = VERIFIED.
- **Decision 3 - Corpus**: analizado el desarrollo autorizado (1483 filas); fuera: test,
  holdout y holdout2 (intactos). Limitaciones declaradas en las secciones 4 y 24.
- **Decision 4 - Interpretacion de errores**: CONFIRMED: identidad TR dentro de HS; distancias
  (84 frontera / 466 lejanos); alineacion del casebook. SUPPORTED: broma/afiliacion en FP
  (27/173 FP_HS, 26/135 FP_AG); errores de TR en textos sin destinatario; error lejano como
  region dominante. TENTATIVE: FN de AG con referencia colectiva superficial; cita/meta-discurso
  minoritario; limitacion de representacion/contexto. UNSUPPORTED: ambiguedad de anotacion como
  causa; sarcasmo como causa de FN; "el modelo carece de la informacion" sin experimento.
- **Decision 5 - Estado de la investigacion**: 37 CLOSED,
  6 BLOCKED_BY_RULES, 3 DUPLICATE,
  1 INVALIDATED, 1 COMPLEMENTARY;
  ninguna POTENTIALLY_NEW demostrada.
- **Decision 6 - Submissions**: new submissions = 0.
- **Decision 7 - Baseline y holdouts**: V03 FROZEN; holdout FROZEN; holdout2 FROZEN; test
  UNTOUCHED.
- **Decision 8 - Integridad**: metricas reproducidas SI; predicciones alineadas SI; texto
  trazado SI; slices correctamente interpretadas SI (con taxonomia corregida); corpus
  caracterizado SI; EDA historico consolidado SI; notebook actualizado de forma segura SI
  (seccion 136 con celda de verificacion ejecutada en aislamiento).
- **Decision 9 - Recomendacion cientifica**: **A. CLOSE_FAILURE_ANALYSIS** (principal). La
  opcion B (integridad) queda resuelta por esta auditoria; C no es necesaria; D no procede
  porque no hay hipotesis nueva demostrable.

## 26. Preguntas no resueltas

- Mecanismo de los 466 errores lejanos (>=0.10): sin causa demostrada; requiere revision
  humana o un experimento autorizado (fuera de alcance).
- Impacto cuantificable de la ausencia de diacriticos y emojis: no medido (requiere
  experimento; no se afirma).
- Regla de anotacion formal detras de TR=1 => HS=1: la identidad es empirica y esta documentada,
  pero no se encontro el texto normativo de los organizadores en el repositorio.
- Calibracion de probabilidades: la distancia al umbral no es confianza; una calibracion no
  se ha medido como mejora legal.

