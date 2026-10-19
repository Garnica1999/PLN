# P90-02 - Handoff autocontenido para segunda revision independiente (ChatGPT)

Commit auditado: `e43c594917fc643d0e300b73646aefb7dbd6b61d`. Fecha: 2026-10-10. Auditoria descriptiva: 0
entrenamientos, 0 submissions, 0 commits, test y holdouts intactos. Este documento contiene los
hallazgos completos; las tablas viven en `reports/p90_02_*.csv`.

## A. Estado del proyecto

- Baseline V03 FROZEN: TF-IDF word(1,2)+char_wb(3,5), LR C=2.0 balanced liblinear; umbrales
  0.44/0.51/0.46; `corrected_cv` = 0.7772114525165722; LB historico 0.78306.
- Git: rama `main`, HEAD `e43c594917fc643d0e300b73646aefb7dbd6b61d` (P90-00), arbol limpio al inicio, 8 commits sin
  push; esta auditoria no commitea.
- Test UNTOUCHED; holdout y holdout2 FROZEN; sin uso del leaderboard.

## B. Calidad de la medicion

- Integridad textual: SOURCE_ALREADY_NORMALIZED; 0 diferencias raw/casebook/celdas; 0 textos con
  bytes > 127 en el desarrollo (D-010: normalizacion ASCII de los organizadores).
- Conciliacion: `corrected_cv` = macro pooled = 0.7772114525165722
  (reproducido a 1e-16); F1 por etiqueta = pooled y coinciden con lo reportado; media entre
  folds = 0.777312 (nivel distinto, NO es la metrica canonica). Las cifras
  614/431/517 citadas son del train-1731 con MOD-03b (error_analysis.md), otra poblacion.
- Discrepancias abiertas: 0.

| etiqueta | TP | FP | FN | TN | F1 pooled |
|---|---|---|---|---|---|
| HS | 505 | 173 | 143 | 662 | 0.761689 |
| TR | 363 | 82 | 82 | 956 | 0.815730 |
| AG | 425 | 135 | 142 | 781 | 0.754215 |

| fold | HS | TR | AG |
|---|---|---|---|
| f1 (n=297) | 0.778626 | 0.827586 | 0.781818 |
| f2 (n=297) | 0.783582 | 0.827957 | 0.770563 |
| f3 (n=296) | 0.768116 | 0.815217 | 0.753247 |
| f4 (n=296) | 0.729323 | 0.782123 | 0.704348 |
| f5 (n=297) | 0.748031 | 0.826347 | 0.762791 |

## C. Perfil completo del corpus autorizado

- 1483 filas (ids unicos), sin vacios, sin duplicados exactos, sin ids duplicados.
- Longitud: 10-306 caracteres (media 104.7,
  mediana 92); 1-57 tokens
  (media 17.21); cortos (<25 tokens) 1182, largos 301.
- Vocabulario: 8888 tipos / 25519 tokens; top: de, que, la, y,
  a, en, puta, el, no, un, es, los, te, una, perra, lo, me, se, eres, tu.
- Estructura: URLs 458; hashtags 144; menciones 702 (605 iniciales); elongaciones 76; mayusculas
  sostenidas 864; puntuacion repetida 84; no-ASCII 0.
- Labels (positivos): HS 648, TR 445, AG 567.

| combinacion | n |
|---|---|
| HS=0 TR=0 AG=0 | 813 |
| HS=0 TR=0 AG=1 | 22 |
| HS=1 TR=0 AG=0 | 79 |
| HS=1 TR=0 AG=1 | 124 |
| HS=1 TR=1 AG=0 | 24 |
| HS=1 TR=1 AG=1 | 421 |

| dependencia | valor | denominador |
|---|---|---|
| P(TR=1|HS=1) | 0.686728 | 648 |
| P(TR=1|HS=0) | 0.000000 | 835 |
| P(AG=1|HS=1) | 0.841049 | 648 |
| P(AG=1|HS=0) | 0.026347 | 835 |

- Identidad: TR=1 => HS=1 determinista (0 violaciones; documentada en D-001/D-044); AG=1 con
  HS=0: 22 casos.
- Duplicados/grupos: 1479 grupos near-dup en desarrollo; solo 6 filas (0.4 por ciento) en 2
  grupos no unitarios (112 y 338). Sin evidencia de que el ruido de etiquetado explique errores.
- Limitaciones: desarrollo unicamente; sin anotacion humana independiente; categorias de
  registro cualitativas sin prevalencia inventada.

## D. Diagnostico de V03

| etiqueta | TP | FP | FN | TN | P | R | F1 | cerca | medio | lejos |
|---|---|---|---|---|---|---|---|---|---|---|
| HS | 505 | 173 | 143 | 662 | 0.744838 | 0.779321 | 0.761689 | 40 | 87 | 189 |
| TR | 363 | 82 | 82 | 956 | 0.81573 | 0.81573 | 0.81573 | 13 | 40 | 111 |
| AG | 425 | 135 | 142 | 781 | 0.758929 | 0.749559 | 0.754215 | 31 | 80 | 166 |

- 441 tweets con error / 757 celdas (223 /
  120 / 98 con 1/2/3 etiquetas); 1042
  tweets sin error.
- Distancias: 84/757 frontera (<0.025);
  466/757 lejanos (>=0.10). Triples: 67-76 de 98 lejanos.
- Patrones cuantificados: broma/afiliacion 27/173 FP_HS y 26/135 FP_AG (SUPPORTED);
  FN de AG con referencia colectiva 16/142 (TENTATIVE, lista ilustrativa); cita 5/143
  (TENTATIVE); errores de TR en textos sin destinatario (region observada).
- Contraejemplos: FP con p>=0.9 sin marcadores de broma (p. ej. 21892); controles correctos con
  @ y lenguaje fuerte (20609, 20347); la distancia no es confianza.

| patron de error conjunto | n |
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

Casos representativos verificados (13): tabla en `p90_02_representative_cases.csv`; muestra:

| tweet_id | fold | patron | distancias | texto (truncado) |
|---|---|---|---|---|
| 22628 | f2 | AG_FP | AG:0.505068 | @kleopatreus T eres bien perra |
| 23076 | f3 | HS_FP|TR_FP|AG_FP | HS:0.472303; TR:0.418366; AG:0.424273 | @CamilaArvayo Cllate todo te queda bien zorra te amo |
| 23839 | f5 | HS_FP|TR_FP | HS:4.6e-05; TR:0.108201 | @Barbijaputa A mi perra me la tienen frita!! |
| 21362 | f4 | HS_FP|TR_FP|AG_FP | HS:0.498901; TR:0.450505; AG:0.487944 | Cllate Zorra' el 'cllate perro' de este semestre |
| 20331 | f4 | AG_FN | AG:0.084236 | Os acordais de cuando decian los del lacito que preferian inmigr |
| 20678 | f5 | HS_FN|AG_FN | HS:0.012378; AG:0.079995 | Mierda para aquellos que saltan la valla de Melilla y Ceuta. Cab |
| 23074 | f2 | HS_FN|AG_FN | HS:0.013257; AG:0.138867 | Xdios odio a estos arabes q se extingan |
| 21973 | f5 | HS_FN|AG_FN | HS:0.027747; AG:0.106797 | Ms de 40.000 invasores negros africanos se preparan para invadir |
| 24508 | f5 | AG_FN | AG:0.022683 | El nefasto ministro de @interiorgob Marlaska acude a #Algeciras  |
| 20932 | f5 | HS_FN|TR_FN|AG_FN | HS:0.004134; TR:0.170708; AG:0.099325 | Gorda chaconda.. tu eres por mi polla la mujer perfecta.. quiero |
| 20956 | f4 | HS_FN|TR_FN|AG_FN | HS:0.132468; TR:0.058407; AG:0.158382 | @zvonimirsoldo @FOXSports_Chile No quera empezar el ao puteando  |
| 24269 | f5 | TR_FN | TR:0.486646 | El agressor de una mujer de 25 ha sido detenido. La agresin se p |
| 23724 | f2 | AG_FN | AG:0.046439 | Quintana ha explotado como una cucaracha. Que se jodan los sudac |

## E. Slices

Slices de texto y de lexico inducido (F1 dentro/fuera; las de lexico son in-sample y solo
diagnosticas):

| slice | familia | n | HS in | TR in | AG in | HS out | TR out | AG out |
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

Subconjuntos definidos por errores (recuentos; F1 omitido: cero por construccion):

| slice | n | celdas HS | celdas TR | celdas AG | tweets 1/2/3 |
|---|---|---|---|---|---|
| sin_odio_explicito | 143 | 143 | 48 | 89 | 48/53/42 |
| colectivos_v4 | 142 | 87 | 54 | 142 | 41/61/40 |
| afiliacion | 83 | 83 | 41 | 50 | 27/21/35 |
| interpersonal_v4 | 82 | 48 | 82 | 58 | 18/22/42 |
| news | 156 | 112 | 39 | 102 | 84/47/25 |

Solapamiento: 91 pares; no se suman errores entre slices.

## F. EDA historico

48 lineas clasificadas (37 CLOSED, 6
BLOCKED_BY_RULES, 3 DUPLICATE, 1
INVALIDATED, 1 COMPLEMENTARY). Familias P57:

| family | status | best_result |
|---|---|---|
| embeddings | CLOSED | E2-19 y V38-01: supervised embeddings CLOSED, 0.59 |
| ordinal | CLOSED | V4-E01-E03 TRACK E = NO_SIGNAL; V13-02 0.777746 (d |
| joint_labels | CLOSED | 6 variantes E2-01-* (HS-TR-AG, HS-AG-TR, AG-HS-TR, |
| PLS | INVALIDATED | V40-01 SUPERVISED_PLS 0.763469 (OVERFIT); V43-02 r |
| VW / optimizacion sparse online | CLOSED | V44-01: FTRL, Coin Betting y PiSTOL sobre la repre |
| contextual_selection | CLOSED | P0 (no hacer nada) = V03. Todas las politicas obse |
| threshold_transfer | CLOSED | P55 reporto 0.7781367700. P56 con anidamiento comp |
| data_expansion | CLOSED | P47: 1731 filas oficiales, 1483 en dev y 248 en ho |
| augmentation | CLOSED | V6-08: espaciado, elongacion y repeticion de puntu |
| retrieval | CLOSED | V4-F01-F04: propagacion de etiquetas descartada, t |
| kernels | CLOSED | V4-X01 candidato D (svd300_rbf) corrected_cv 0.790 |
| FM (factorization machines) | CLOSED | V25-01: k=4, lambda_v=1, 0.782699 bajo D-004. V6-0 |
| Tsetlin | CLOSED | V26-03: 0.583397, cortado en K=200 por los criteri |
| BM25 | CLOSED | V37-01: 0.7744254 |
| hashing | CLOSED | V36-01: 0.748427 |
| SGD | CLOSED | V35-01: hinge -0.016177, modified_huber -0.033645 |
| label_specialization | CLOSED | V28-00 LABEL_SPECIALIZATION = CLOSED. v09_p5: C po |
| noise (label noise y reweighting) | CLOSED | V30-00 LABEL_NOISE = CLOSED. V4-B03 down_violacion |
| fuzzy_matching | CLOSED | V29-03 REJECTED, FUZZY_MATCHING = INFORMATIONAL_ON |
| ensembles | CLOSED | ENS-02 anidado 0.791900, descartada por no superar |
| stacking y meta-modelos | CLOSED | S08 meta holdout 0.8465 frente a 0.8285 de V03, de |
| structural_features | CLOSED | V15-01: 0.775860, NO ADOPTADA. V14-00 0.660630. V3 |

Lineas no cerradas: 11 (revision de anotacion TR, propagacion near-dup,
correccion EM de prevalencia, expansion de datos, transformers/LLM: BLOCKED_BY_RULES; FM y
LightGBM sobre la representacion de V03: DUPLICATE).

Resultados negativos clave (verificados en los informes): FastText no supera el gate; Word2Vec
por debajo; mas cobertura no mejora F1; corpus mixto no aporta; stacking generico falla; la
propagacion near-dup no resuelve; BCE agregada no es proxy de dureza; la distancia por etiqueta
diagnostica mejor; CV no garantiza LB (4 candidatos enviados con delta CV positivo cayeron:
v23_01 -0.00368, v08_meta_v2 -0.00143, v20_01 -0.00643).

| candidato | CV | LB | estado |
|---|---|---|---|
| v23_01 svd300_rbf (V4-X01 candidato D) | 0.7904 | 0.77938 | SI, RECHAZADO |
| v08_meta_v2 (meta de 44 features, Ridge alph | 0.7874 | 0.78163 | SI, RECHAZADO |
| v20_01 regla HS por interaccion de scores (V | 0.7882 | 0.77663 | SI, RECHAZADO. Ademas RECHAZADO an |
| E2-13-META-ENT-RIDGE (meta + entropias) | 0.7885 | n/d | NO |
| V03 con umbrales anidados de P56 (T1 = 0,41/ | 0.77454787 | n/d | NO |
| v25_01 FM k=4 | 0.782699 | n/d | NO |
| v16_ch2 correccion de prior con EM | n/d (valid | n/d | NO |

Complementariedad (diagnostico, citado de `reports/p66_complementarity.md`): la correccion
residual de MI_arbol rescata 13/16/24 celdas HS/TR/AG con dano 10/16/27 (neto +3/+0/-3);
rescate de alta confianza total: 6 (0/4/2); bootstrap 2000 remuestreos: delta medio +0.0027270,
IC95 [-0.0044312, +0.0102109] incluye cero; brecha vs barajado +0.0041134; veredicto historico
PROMISING, no adoptado. Ninguna solucion complementaria transferible demostrada.

## G. Integridad y reproducibilidad

- Artefactos fuente y hashes: ver seccion 5 del informe (`p90_02_artifact_inventory.csv`).
- Texto: verificacion byte a byte (3 representaciones, 0 diferencias); vectorizadores con
  strip_accents=None; entrada real = texto original + lowercase.
- Numerico: matrices pooled, F1 por fold, media de folds, macro pooled; todo reproducido;
  controles 757/757; casebook 441/757 alineado; muestra 100+185.
- Tests ejecutados: hashes de 5 artefactos; pred == proba >= umbrales; particion dev == ids
  OOF; validacion de casebook/celdas/controles/muestra; 19/19 definiciones de slices
  reproducidas; checks ASCII (0 bytes > 127) y AST de operaciones y ficheros de los scripts.
- Limitaciones de acceso: el artefacto del train completo con MOD-03b no se abrio (mezcla
  filas reservadas); el contenido del test/holdout no se inspecciono.

## H. Interpretacion cientifica

- Demostrado: no hay perdida de texto en el pipeline (propiedad del dato); `corrected_cv` es el
  macro pooled; el casebook y sus controles son fieles al OOF; la identidad TR=>HS; la mayoria
  del error esta lejos de los umbrales (466/757).
- Respaldo parcial: broma/afiliacion como subconjunto de los FP (15.6 / 19.3 por ciento, no la
  mayoria); region TR sin destinatario; limitacion de representacion/contexto (consistente con
  los errores lejanos, sin test causal).
- Hipotesis: FN de AG con referencia colectiva superficial (lista ilustrativa); cita/meta-discurso
  minoritario; calibracion como mejora potencial no medida.
- No afirmable: ambiguedad de anotacion como causa; sarcasmo como causa de FN; que el modelo
  carezca de la informacion necesaria; prevalencias poblacionales a partir de la muestra.

## I. Decisiones finales

V03 FROZEN; casebook VERIFIED; slices PARTIALLY_VERIFIED (taxonomia corregida); metricas
VERIFIED; texto VERIFIED; submissions 0; test/holdouts intactos; recomendacion **A.
CLOSE_FAILURE_ANALYSIS** con B resuelta. Preguntas pendientes: mecanismo de los 466 errores
lejanos; impacto de la normalizacion ASCII; regla de anotacion formal TR=>HS; calibracion como
mejora legal no medida.

