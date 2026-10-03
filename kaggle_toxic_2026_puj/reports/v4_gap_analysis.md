# Advanced Gap Analysis v4 - KAGGLE_TOXIC_2026_PUJ

Fecha: 2026-09-25. Autor: orquestador ML (revision de artefactos en disco, no de
memoria). Spec: `docs/advanced_research_v4_spec.md`. Referencia: **V03**
(LB 0.78306; `corrected_cv` = **0.777211** sobre el desarrollo de 1.483 tweets,
verificado en `reports/baseline_v03_v4_check.json`).

## 1. Proposito y metodo

Antes de experimentar, cada idea de la spec v4 se clasifica contra la evidencia
que el proyecto ya genero, para no repetir lineas bajo otro nombre (§84) y para
separar lo genuinamente nuevo de lo ya cerrado. Fuentes consultadas en disco:

`experiments.csv` (128 filas), `DECISIONS.md` (D-001…D-024), `RUNLOG.md`,
`reports/advanced_gap_analysis_v2.md` (24 brechas), `reports/research_backlog_v2.md`
(24 lineas, 24/24 cerradas), `reports/spec_compliance_v2.md` (matriz FASE 0-99),
`reports/final_candidate_screening.md` (125 experimentos categorizados),
`reports/error_intersection_v2.csv`, `reports/error_by_label_combination_v2.csv`,
`reports/model_complementarity_v2.csv` (496 parejas de 32 canales),
`reports/feature_catalog_v2.csv` (225 features), `reports/rule_audit_v2.md`,
`reports/target_encoding_audit_v2.md`, `experiments/23_candidatos_finales.py` y
`experiments/25_submissions_candidatos.py`.

## 2. Taxonomia de estados

| Estado | Significado |
|---|---|
| `NEW` | nunca se probo con evidencia en el proyecto |
| `NOT_TESTED` | se menciono en un plan pero no se ejecuto (brecha declarada) |
| `TESTED_NEGATIVE` | se ejecuto y fue negativo con numero |
| `TESTED_BUT_BADLY_INTEGRATED` | se ejecuto con una integracion que el propio analisis v2 senalo como no representativa |
| `TESTED_BUT_UNCERTAIN` | se ejecuto con IC que incluye 0 o sin consistencia entre pliegues |
| `TESTED_AND_CLOSED` | se ejecuto, fue negativo y se cerro con decision documentada |
| `TRANSFER_FAILED` | funciono localmente pero no transfirio al leaderboard |

## 3. Matriz de ideas de la spec v4

| Spec | Idea | Estado | Evidencia en disco | Brecha real que queda | Tratamiento en v4 |
|---|---|---|---|---|---|
| 7-8 | Consistencia de etiquetas en clusters de texto identico o casi identico | `NOT_TESTED` | `data/processed/grupos_near_duplicados.csv` (grupos train-test, umbral 0.85); EDA: deduplicar en train elimina 4 filas | Nunca se calculo `purity`, entropia, numero de combinaciones ni desacuerdo por cluster **dentro de train** | TRACK A: `experiments/27_label_consistency_v4.py` → `reports/label_consistency_v4.csv` |
| 9-13 | `hardness_score` OOF, hard-example mining, reweighting, ruido vs difficult | `NOT_TESTED` | `reports/error_intersection_v2.csv` cuantifica errores, no dureza ni ruido | No existe ninguna medida de dureza por ejemplo ni de conflicto textual | TRACK B: `experiments/28_hardness_v4.py` + `29_reweighting_v4.py` |
| 14 | Perdidas robustas en modelos lineales | `NOT_TESTED` | Todo el proyecto usa BCE logistica de sklearn | Ninguna alternativa a la perdida estandar | TRACK B (linea secundaria, acotada) |
| 15-19 | Weighting supervisado (chi², MI, log-odds, BM25, binario, augmented TF) | `TESTED_BUT_UNCERTAIN` parcial | `E2-15` NB-SVM 0.7612 con `ensemble_gain` +0.0038 e IC [-0.008,+0.016]; barrido de TF-IDF/sublinear en `03_variantes_representaciones.py` | Solo se probo una instancia (log-count ratio) y la geometria de la matriz nunca se repondro por clase | TRACK C: `experiments/30_supervised_weighting_v4.py` |
| 20-21 | Multigranularidad y fragmentos internos (subword-like) | `TESTED_AND_CLOSED` como union; `NOT_TESTED` como canal de morfologia | Union word(1,2)+char_wb(3,5) es la representation ganadora (REP-05/06, MOD-03b) | Sufijos/prefijos nunca se materializaron como canal (brecha FASE 13 de v2) | TRACK D: `experiments/34_structural_sparse_v4.py` |
| 22 | Sufijos y prefijos 2/3/4 | `NOT_TESTED` | `reports/spec_compliance_v2.md` FASE 13 = PARCIAL | Canal morfologico propio inexistente | TRACK D |
| 23-28 | Pattern n-grams, ventanas de contexto, target-insulto, atribucion, autor vs citado, ambito de negacion | `TESTED_AND_CLOSED` | `E2-05` (targeting, 0/5 pliegues), `E2-06` (colectivo), `E2-07` (cita, -0.0335), `E2-08` (negacion, -0.0115); todos con canal hstack y `union + canal` | Se evaluaron como **features binarias/densas concatenadas**, no como **tokens sparse de interaccion** | TRACK D solo si una hipotesis nueva lo justifica (§85, §88) |
| 29 | Regimen noticia | `TESTED_BUT_BADLY_INTEGRATED` | Caracterizado en rebanadas; modelo dedicado nunca entrenado (FASE 25 = COMPLETA variante) | La condicion de disparo (diferencia significativa de error) se documento pero no se formalizo | TRACK D: solo como hipotesis ligada a error medido |
| 30-31 | Emocion (global vs local) y afiliacion/broma | `TESTED_AND_CLOSED` | `E2-09-EMO` 0.6230, `EMO-LEX` 0.6205, `EMO-EST` 0.5897 (0/5 pliegues) | La relacion emblema x target como **canal** no se separo del bloque de emocion | No se repite (§87): solo si aparece en el backlog con hipotesis nueva |
| 32 | HS implicito por subtipo (estereotipo, deshumanizacion, amenaza) | `NOT_TESTED` | FASE 70 = PARCIAL; taxonomia de FN de HS en `reports/error_analysis.md` | Sin features dedicadas por subtipo | TRACK D (linea secundaria) |
| 33-34 | Colectivo y patrones de generalizacion | `TESTED_AND_CLOSED` | `E2-06` (colectivo + predicado negativo); recall de rebanada mejoro en dev y no transfirio | Ninguna linea extra sin hipotesis | No se repite (§89): se documenta como cerrada |
| 35-39 | Modelo conjunto de etiquetas, estructura, combinaciones raras, dependencia como feature, compartido + interaccion | `TESTED_BUT_BADLY_INTEGRATED` para cadenas/meta; `NEW` para el modelo conjunto | `E2-01` cadenas 0.7737-0.7744 (cerrada); `E2-02`/`E2-18` meta 0.7853/0.7874 (`TRANSFER_FAILED`, LB 0.78163) | Las cadenas inyectan scores en el texto y el meta opera sobre 44 features; **nunca se entreno un modelo conjunto sobre las combinaciones** ni se marginalizo | TRACK E: `experiments/31_joint_label_v4.py` |
| 40 | Multi-task desde cero | `TESTED_AND_CLOSED` | `DL-01` SVD300+MLP 0.7949 OOF sin sonda; `DL-02` char CNN 0.7410; `DL-03` BiLSTM 0.7364 | Multitask con regularizador de consistencia no probado | Sin prioridad (§64) |
| 41 | Hard negative classification (subproblemas) | `TESTED_BUT_BADLY_INTEGRATED` | `E2-20` especialistas por rebanada: 11/21 celdas significativas pero -0.045 dentro de su rebanada | Los subproblemas propuestos (insult non-hate, implicit hate, collective hate) no se definieron | TRACK B/D solo con evidencia de error |
| 42-43 | Cost-sensitive learning por etiqueta; sample weight x class weight | `NOT_TESTED` | Todo el proyecto usa `class_weight='balanced'` fijo | Nunca se barro el espacio de pesos de clase | TRACK C: `experiments/30_supervised_weighting_v4.py` |
| 44 | Interacciones en espacio sparse | `NOT_TESTED` como tokens; `TESTED_BUT_BADLY_INTEGRATED` como producto | `reports/features_densas_resultados.csv` (hibridos -0.009 a -0.032) | Las interacciones nunca se representan como **tokens** de patron | TRACK D |
| 45-47 | Error-centered features, errores universales, correcciones por modelo | `TESTED_BUT_BADLY_INTEGRATED` | `AUD-01` (98/103 fallan las tres), `AUD-02` (por combinacion), `AUD-03` (496 parejas de canales) | Existe la materia prima pero no un artefacto de correcciones v4 ni features ligadas a error | P7: `experiments/35_discovery_v4.py` |
| 48 | Correlacion de errores | `TESTED_AND_CLOSED` como analisis | `AUD-03` con disagreement y correccion mutua de 496 parejas | Ninguna | Se reutiliza el artefacto; no se repite |
| 49-51 | Late fusion, regularizacion del meta, **calibracion de las entradas del meta** | `TESTED_AND_CLOSED` para late fusion; `NEW` para el shift | `advanced_research_v2.md` §4.3 deja el shift como hipotesis abierta: el meta se entrena con scores OOF de bases de 4/5 y predice con bases de train completo | El mecanismo concreto de la no transferencia nunca se midio | TRACK H: `experiments/32_meta_shift_v4.py` (prioritario) |
| 52-55 | Shift train-test no supervisado, representaciones robustas, robustez a duplicados | `NOT_TESTED` | Auditoria de Fase 0 (5 duplicados en test, 10 test en grupos near-dup con train) | Nunca se compararon distribuciones no supervisadas train vs test | TRACK F: `experiments/35_discovery_v4.py` |
| 56-58 | Vecindad, retrieval no supervisado, propagacion de etiquetas por near-duplicado | `NOT_TESTED` | Grupos near-dup conocidos | Solo 10 tweets de test tienen near-duplicado en train: el techo de impacto es ~0.9 % del test | TRACK F: analisis y, si la regla lo permite, canal separado |
| 59-61 | LM class-conditional (unigram/bigram/char), perplejidad, perfil de estilo | `NOT_TESTED` (NB-SVM es otra cosa) | `E2-15` uso log-count ratio como **pesos de features**, no como likelihood de texto | La verosimilitud class-conditional como feature nunca se computo | TRACK G: `experiments/33_class_conditional_lm_v4.py` |
| 62-63 | Clasificadores alternativos y no lineales sobre espacio reducido | `TESTED_BUT_BADLY_INTEGRATED` | `MOD-01` LinearSVC 0.7849; NB 0.7567; `B5-DENSE`/LightGBM/XGBoost/CatBoost en familia densa (negativos) | Ridge, SGD y PassiveAggressive no se probaron; no lineales **sobre SVD** tampoco (DL-01 era MLP) | TRACK G (acotado) |
| 64-65 | DL y DL robusto a ruido | `TESTED_AND_CLOSED` / `TRANSFER_FAILED` | `DL-01`..`DL-03` | Nada nuevo | No se ejecuta salvo evidencia de P1 |
| 92-93 | Revision bibliografica 2022-2026 | `NOT_TESTED` (focalizada) | `reports/estado_arte.md` (45 refs, 2019-2021) | Sin revision de ruido, reweighting, weighting supervisado, BM25 y modelos conjuntos | P0: revision en paralelo |

## 4. Ideas genuinamente nuevas (priorizadas)

1. **Consistencia de etiquetas y contradicciones textuales** (spec 7-8) — nunca
   medida; ataca directamente la ambiguedad de anotacion ya sospechada.
2. **Dureza de ejemplos y reweighting OOF** (spec 9-13, 42-43) — mecanismo de
   aprendizaje nuevo en el proyecto, no una variante de modelo.
3. **Calibracion de las entradas del meta** (spec 50) — unica linea con evidencia
   previa (dev +0.0081, holdout +0.0180) cuyo mecanismo de fallo esta identificado
   y sin medir.
4. **Modelo conjunto de etiquetas** (spec 35-37, 39) — clase de modelo no
   entrenada; ataca la senal mas fuerte del corpus (phi 0.74-0.83) de una forma
   distinta a cadenas y meta.
5. **Weighting supervisado por clase** (spec 15-19) — repondera la geometria de la
   matriz antes del clasificador, con antecedente mixto (NB-SVM complementario pero
   no significativo).
6. **Verosimilitud class-conditional** (spec 59-61) — familia de features nueva.
7. **Sufijos/prefijos y tokens de interaccion** (spec 22, 44) — brechas declaradas
   de v2, representacion sparse (no densa).
8. **Subtipos de HS implicito** (spec 32) — brecha FASE 70.

## 5. Lo que no se repite y por que

- POS/dependencias/morfologia como bloque denso, emocion global, targeting
  binario, cita/atribucion, negacion, colectivo y generalizacion: `TESTED_AND_CLOSED`
  con numero (spec §85-§89 obligan a no repetirlos bajo el mismo formato).
- Cadenas de etiquetas: cerradas (ganancia +0.002, 1/5 pliegues).
- Concatenar features densas al TF-IDF: degradacion documentada (-0.009 a -0.032).
- Optuna como metodo de seleccion: mejoro OOF y empeoro LB (OPT-01, LB 0.77602).
- Deep learning desde cero: DL-01..03 no superaron al lineal.

## 6. Brechas heredadas de v2 que v4 si ataca

| Brecha v2 | Spec v4 | Track |
|---|---|---|
| FASE 13 (morfologia: sufijos/prefijos) | 21-22 | D |
| FASE 70 (HS implicito por subtipo) | 32 | D |
| FASE 54 (MI/redundancia) | 15-18 | C |
| FASE 73-75 (scores dedicados) | 12, 24, 33 | B/D |
| E2-12 (rareza/entropia) | 52-53 (diagnostico) | F |
| E2-22 (umbral por incertidumbre) | 49 (metas) | H |
| Hipotesis abierta §4.3 (shift de calibracion) | 50 | H |
