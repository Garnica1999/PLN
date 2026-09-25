# Especificación maestra — Fase v4 «Advanced Research v4»

Proyecto: `KAGGLE_TOXIC_2026_PUJ`
Autor de la spec: Carlos (documento maestro); transcripción estructurada por el
orquestador ML el 2026-09-25.
Documento relacionado: `docs/advanced_research_v2_spec.md` (fase v2, archivada).
Referencia congelada: **V03**, leaderboard **0.78306**.

> Nota de fidelidad: este archivo es una transcripción estructurada de la
> especificación maestra de v4 (108 secciones, numeración original conservada).
> Las secciones se reproducen con su título original y su contenido operativo; la
> numeración permite auditar cada requisito contra la matriz de cumplimiento
> (`reports/spec_compliance_v4.md`).

## 0. Objetivo

Encontrar **alternativas realmente nuevas** a V03 = 0.78306, es decir nuevas
fuentes de señal o nuevas formas de explotar la señal existente que no hayan
sido descartadas experimentalmente. No se busca «otro modelo de clasificación»
sino «otra manera de representar o aprender la señal».

La fase termina únicamente cuando: (A) exista una mejora suficientemente
respaldada, o (B) las nuevas hipótesis razonables se hayan investigado y
cerrado con evidencia (§107). No se declara irreducible el problema porque las
primeras alternativas fallen (§102).

## 1. Restricciones duras (no negociables)

Aplican a toda la fase; se auditan por candidato (§91):

| Regla | Contenido |
|---|---|
| NO_TRANSFORMERS | sin transformers, LLM ni modelos preentrenados de ningún tipo |
| NO_PRETRAINED_EMBEDDINGS | sin embeddings preentrenados (spaCy solo simbólico: POS, lemas, dependencias, morfología; sin `.vector`, `.tensor`, `similarity()`) |
| NO_PRETRAINED_CLASSIFIER | todo clasificador se entrena desde cero con datos de la competencia |
| NO_EXTERNAL_TRAINING_CORPUS | sin corpus externos (D-002) |
| NO_EXTERNAL_LABELS | sin etiquetas externas |
| NO_TEST_FITTING | el test solo se usa con `transform`/`predict`; nunca para fitting, vocabulario, IDF, selección ni calibración |
| NO_TEST_THRESHOLD_TUNING | umbrales se eligen solo con OOF o holdout |
| NO_PSEUDO_LABELING | sin autoentrenamiento con predicciones propias |
| LEXICONS_DOCUMENTED | todo lexicón inducido se genera dentro del fold y se documenta su origen (D-007) |
| NO_REPEATING | antes de cada experimento se consulta `experiments.csv` y `DECISIONS.md`; si la idea ya fue probada y la implementación no es genuinamente distinta, no se ejecuta (§84) |
| V03_INMUTABLE | V03 no se modifica, no se sobreescribe y sigue reconstruible byte a byte |

Restricciones de presupuesto y deinterpretación:

| Regla | Contenido |
|---|---|
| MULTIPLICIDAD | con más de 100 experimentos previos, un `+0.002` no es un descubrimiento; se exigen consistencia entre pliegues, mecanismo, complementariedad y estabilidad (§72) |
| WINNERS_CURSE | 0.78306 es el máximo observado de 12 sondas ruidosas; el nivel esperado de la familia es ~0.776–0.780; no se trata como objetivo reproducible (§2, §3) |
| NO_SEL_ECCION_POR_OOF | «mayor OOF = candidato final» está prohibido: Optuna y el meta mejoraron local ypeaaron en LB (§98) |
| HOLDOUT | se respeta el holdout congelado (248 tweets); 4 de 5 evaluaciones usadas; la restante se reserva para el único mejor finalista (§70, D-025) |
| LB_PROBE | sin envíos automáticos; un candidato solo se marca `READY_FOR_KAGGLE_PROBE` y Carlos lo envía a mano (§78, §81, §101) |

## 2. Mapa de las 108 secciones

Tipo: **R** = restricción/protocolo, **P** = proceso, **I** = idea/hipótesis,
**E** = entregable, **S** = sección de cierre.

| Nº | Sección | Tipo | Estado heredado de v2 / tratamiento en v4 |
|---|---|---|---|
| 0 | Rol del investigador | R | Adoptado:Senior NLP, CL, HSD, multi-label, IR, estilometría, label noise |
| 1 | Competencia | R | Kaggle `tweet-toxic-2026-puj`, F1, deadline 2026-10-19 |
| 2 | Baseline vigente (V03) | R | Congelada; no se modifica (§2 y verificador `26_scaffold_v4.py`) |
| 3 | Resultados históricos como evidencia | R | 14 observaciones heredadas; se verification en `reports/v4_gap_analysis.md` |
| 4 | Nueva filosofía (datos/representación/función/estructura) | P | Estructura de los 8 tracks (§73) |
| 5 | Primera acción: auditoría v4 | P | P0 (`experiments/26_scaffold_v4.py` + informe `baseline_v03_v4_check.json`) |
| 6 | Gap analysis con taxonomía | E | `reports/v4_gap_analysis.md` (P0) |
| 7 | Control de duplicados y consistencia | I | **NUEVO**: no se había calculado purity/entropía por cluster |
| 8 | No eliminar ruido automáticamente | P | Tres estrategias (baseline / downweight / remove extremos) |
| 9 | Label noise analysis (OOF) | I | **NUEVO**: `hardness_score` solo OOF |
| 10 | Hard example mining | I | **NUEVO**: HARD_FP, HARD_FN, ALL_MODELS_FAIL, LABEL_CONFLICT, DISAGREEMENT |
| 11 | Iterative sample reweighting | I | **NUEVO**: dos fases, pesos derivados solo de OOF |
| 12 | Label-specific hard examples | I | **NUEVO**: `hardness_HS/TR/AG` con pesos independientes |
| 13 | Label noise vs true hard case | E | `reports/hardness_vs_noise_v4.md` (P1) |
| 14 | Robust loss | I | **NUEVO** en modelos lineales (label smoothing, clipped, GCE) |
| 15 | Supervised feature weighting | I | **NUEVO**: chi², MI, information gain, log odds |
| 16 | Class-specific feature weighting | I | **NUEVO**: `w_j(label)` con smoothing |
| 17 | Supervised TF-IDF | I | **NUEVO**: TF-IDF × peso discriminativo vs binario vs log-count ratio |
| 18 | Mutual information por etiqueta | I | **NUEVO**: MI(word/char, etiqueta) como reponderación |
| 19 | BM25 / IR weighting | I | **NUEVO**: sublinear, binario, augmented TF, k1/b |
| 20 | Multigranularidad sin transformers | I | partly cerrada (unión word+char ya es la ganadora) |
| 21 | Subword-like (fragmentos internos) | I | **NUEVO** como canal (brecha FASE 13 de v2) |
| 22 | Sufijos y prefijos 2/3/4 | I | **NUEVO** (brecha FASE 13) |
| 23 | Pattern n-grams simbólicos | I | parcialmente cerrada en P2–P4; se reevalúa solo con hipótesis nueva |
| 24 | Context window features | I | **NUEVO** como tokens sparse de ventana (P5 lo hizo con densas) |
| 25 | Target-insult relation | I | cerrada en v2 (E2-05/06), salvo scores separados de especificidad/directness/collectivity |
| 26 | Attribution / quotation | I | cerrada en v2 (E2-07) |
| 27 | Author vs quoted speech | I | cerrada en v2 (variante de E2-07) |
| 28 | Negation scope | I | cerrada en v2 (E2-08) |
| 29 | News / attribution regime | I | caracterizada, sin modelo dedicado |
| 30 | Emotion revisit (local) | I | cerrada en v2 (E2-09) salvo relación emoción×target |
| 31 | Affiliation / humor | I | cerrada en v2 (E2-09-EST) |
| 32 | Implicit hate | I | **NUEVO** por subtipo (brecha FASE 70) |
| 33 | Collective target | I | cerrada en v2 (E2-06) |
| 34 | Generalization patterns | I | cerrada en v2 (E2-06) |
| 35 | Joint label model | I | **NUEVO** como clase de modelo (potencias + multinomial) |
| 36 | Structured label model | I | **NUEVO**: P(HS,TR,AG\|X) con interacciones sin reglas duras |
| 37 | Rare label combinations | I | **NUEVO**: 110 (n=28) y 001 (n=26) con smoothing/prior jerárquico |
| 38 | Label dependency as feature | I | cerrada en v2 (E2-01/E2-02) |
| 39 | Shared + label interaction model | I | **NUEVO**: coeficientes compartidos + específicos + interacciones |
| 40 | Multi-task learning desde cero | I | DL-01..03 ya negativos; solo si hay hipótesis nueva |
| 41 | Hard negative classification | I | **NUEVO** como subproblemas (insult non-hate, implicit hate, collective) |
| 42 | Cost-sensitive learning | I | **NUEVO**: `class_weight='balanced'` nunca se barrió por etiqueta |
| 43 | Sample weight × class weight | I | **NUEVO** |
| 44 | Feature interaction en espacio sparse | I | **NUEVO** como tokens de interacción justificados |
| 45 | Error-centered feature engineering | P | solo con hipótesis ligada a un error medido |
| 46 | Universal errors | P | análisis: ¿todos fallan? ¿hay información no representada? |
| 47 | Model-specific corrections | E | `reports/correction_opportunities_v4.csv` |
| 48 | Correlación de errores | I | AUD-03 ya lo cubre (496 parejas) |
| 49 | Late fusion revisit | I | cerrada en v2 salvo fuente con complementariedad demostrada |
| 50 | Calibration de las entradas del meta | I | **NUEVO y prioritario**: hipótesis abierta en `advanced_research_v2.md` §4.3 |
| 51 | Meta model regularization | I | comparada en v2 (Ridge/LR) |
| 52 | OOD / hardness analysis | I | **NUEVO** (solo features no supervisadas, sin labels de test) |
| 53 | Train-test representation shift | I | **NUEVO** diagnóstico |
| 54 | Robust representations | I | condicionada a F; comparación con F |
| 55 | Duplicate-robust representation | I | **NUEVO**: desempeño en únicos vs near-duplicados |
| 56 | Nearest-neighbor analysis | I | **NUEVO**: acuerdo de etiqueta del vecino más cercano |
| 57 | Retrieval no supervisado | I | **NUEVO**: cosine, char sim, Jaccard en TRAIN |
| 58 | Near-duplicate label propagation | I | **NUEVO**: solo 10 tweets de test tienen near-duplicado en train; decidir legalidad antes |
| 59 | Statistical LM desde cero | I | **NUEVO**: unigram/bigram/char bigram, P(text\|clase) |
| 60 | Class-conditional perplexity-like | I | **NUEVO**: log-likelihood, margen, perplejidad |
| 61 | Language style profile | I | **NUEVO** como canal complementario (denso ya falló) |
| 62 | Alternative classifiers | I | SVC/NB ya barridos; **nuevos**: Ridge, SGD, PassiveAggressive |
| 63 | No lineales sobre representación reducida | I | secundario; HGB/ET sobre SVD sin probar |
| 64 | Deep learning desde cero | R | última prioridad; sin BERT/RoBERTa/LLM |
| 65 | Label-noise robust DL | I | solo si P1 encuentra evidencia |
| 66 | Feature selection no es el objetivo | R | buscar geometría, no menos features |
| 67 | Ablación completa | P | V03 / +mecanismo / +mecanismo 2 / combinación |
| 68 | Métricas | P | F1 HS/TR/AG, macro, micro, exact match, P, R, fold mean, fold std |
| 69 | Corrected validation | R | `StratifiedGroupKFold`, grupos near-dup ∪ pseudo-autor (D-016) |
| 70 | Frozen holdout | R | 248 tweets; 4/5 usadas |
| 71 | Bootstrap | P | por tweet, pareado, IC95 |
| 72 | Multiplicidad | R | >100 experimentos; no tratar `+0.002` como descubrimiento |
| 73 | Research tracks | P | ocho tracks A–H (§4 de este plan) |
| 74 | Research backlog | E | `reports/research_backlog_v4.md` (P0) |
| 75 | Experiment log | E | `experiments.csv` con filas `V4-*` |
| 76 | Decision log | E | `DECISIONS.md` D-025… |
| 77 | Versionamiento | R | V04, V05… sin sobrescribir V03; guardar modelo, vectorizador, thresholds, OOF, hash |
| 78 | Submissions | P | no automáticas; solo con hipótesis, evidencia, sin leakage, reproducida, documentada |
| 79 | Submission manifest | E | `submissions/submission_manifest.csv` |
| 80 | Reproducibility | P | reconstrucción desde TRAIN + doble ejecución + SHA256 |
| 81 | No gastar LB sin hipótesis | R | el LB es evaluación externa |
| 82 | Nueva condición de éxito | P | mejora corrected CV / folds / bootstrap / explicación / complementariedad / transferencia |
| 83 | Resultado negativo | P | registrar NO_SIGNAL, REDUNDANT, UNSTABLE, LEAKAGE_RISK, OVERFIT, IMPLEMENTATION_ISSUE, BAD_INTEGRATION |
| 84 | Regla contra repetición | R | consultar registro antes de cada experimento |
| 85 | Regla sobre features densas | R | la degradación no implica inutilidad; probar late fusion antes de cerrar la familia |
| 86 | Regla sobre POS | R | POS patterns como sparse, no como bloque denso |
| 87 | Regla sobre emoción | R | si se revisa, hacerlo local a target/insult |
| 88 | Regla sobre targeting | R | distancia, orden, contexto, sintaxis, interacción; no binario |
| 89 | Regla sobre label dependency | R | joint structured output y soft priors; no solo cadenas |
| 90 | Regla sobre label noise | R | no corregir etiquetas solo por predicciones; exigir inconsistencia textual |
| 91 | Rule audit | P | auditoría por candidato |
| 92 | Web / estado del arte | P | revisión 2022–2026 antes de cerrar C, E y G |
| 93 | Transferibilidad de la literatura | E | tabla paper/año/dataset/idea/adaptación/experimento |
| 94 | Matriz de cobertura | E | `reports/signal_coverage_v4.csv` |
| 95 | Error taxonomy v4 | E | `reports/error_taxonomy_v4.md` |
| 96 | Correction matrix | E | `reports/correction_matrix_v4.csv` |
| 97 | Final candidates | P | máximo 3 con corrected CV, folds, bootstrap, holdout si corresponde, error analysis, leakage, reproducibilidad |
| 98 | No seleccionar por OOF únicamente | R | ver winner's curse |
| 99 | Candidate ranking | E | `reports/v4_candidate_comparison.md` comparativo, no numérico arbitrario |
| 100 | Final submission generation | P | refit → transform → predict → thresholds previos → CSV → validate → hash → reproducir |
| 101 | Leaderboard probe | P | solo `READY_FOR_KAGGLE_PROBE`, envío manual |
| 102 | Si ningún candidato es suficiente | P | mantener V03; no fabricar mejora; cerrar líneas con evidencia |
| 103 | Actualización del informe | E | `reports/advanced_research_v4.md` y `FINAL_REPORT.md` al final, preservando histórico |
| 104 | Project state | E | actualizar los campos de estado |
| 105 | Primera acción (24 pasos) | P | ejecutada en P0–P8 de este plan |
| 106 | Principio absoluto | R | no «más features» ni «mejor OOF»: evidencia de generalización |
| 107 | Objetivo final | P | mejor modelo posible + señal nueva + validación corregida + error analysis + auditoría + submission reproducible |

## 3. Los ocho tracks (sección 73)

| Track | Nombre | Contenido | Prioridad v4 |
|---|---|---|---|
| A | Data integrity / label noise | clusters de consistencia, contradicciones, ruido de etiqueta | 1 (primera acción) |
| B | Sample hardness / reweighting | hardness OOF, hard-example mining, reweighting, class weights | 1 |
| C | Representation weighting | chi², MI, log-odds, BM25, binario, augmented TF, weighted TF-IDF | 2 |
| D | Structural sparse representations | sufijos/prefijos, interacciones sparse, ventanas locales, HS implícito | 4 (solo subítems nuevos) |
| E | Joint label structure | multinomial sobre combinaciones, marginalización, interacciones, combos raras | 2 |
| F | Robustness / distribution shift | shift train↔test no supervisado, vecindad, retrieval, robustez a duplicados | 3 |
| G | Alternative statistical models | LM class-conditional, perplejidad, clasificadores alternativos, no lineales sobre SVD | 3 |
| H | Final fusion | late fusion con complementariedad demostrada; **calibración de las entradas del meta** (shift de tamaño de entrenamiento) | 2 (shift) / cierre |

## 4. Secuencia de ejecución (fase a fase)

| Fase | Contenido | Script | Entregables |
|---|---|---|---|
| P0 | Scaffold, verificación de V03 y del protocolo, gap analysis, backlog, cobertura, literatura | `experiments/26_scaffold_v4.py` | `baseline_v03_v4_check.json`, `v4_gap_analysis.md`, `research_backlog_v4.md`, `signal_coverage_v4.csv`, `estado_arte_v4.md` |
| P1 | TRACK A + B (diagnóstico) | `27_label_consistency_v4.py`, `28_hardness_v4.py` | `label_consistency_v4.csv`, `hardness_v4.*`, `hardness_vs_noise_v4.md` |
| P2 | TRACK B (experimento) + C + cost-sensitive | `29_reweighting_v4.py`, `30_supervised_weighting_v4.py` | `reweighting_v4.csv`, `supervised_weighting_v4.csv` |
| P3 | TRACK E (modelo conjunto) | `31_joint_label_v4.py` | `joint_label_v4.csv` |
| P4 | TRACK H (shift del meta) | `32_meta_shift_v4.py` | `meta_shift_v4.csv` |
| P5 | TRACK G (LM class-conditional + clasificadores) | `33_class_conditional_lm_v4.py` | `class_conditional_lm_v4.csv` |
| P6 | TRACK D (subítems nuevos) | `34_structural_sparse_v4.py` | `structural_sparse_v4.csv` |
| P7 | TRACK F + discovery | `35_discovery_v4.py` | `correction_opportunities_v4.csv`, `correction_matrix_v4.csv`, `error_taxonomy_v4.md`, shift train↔test |
| P8 | Cierre: candidatos, auditorías, reproducibilidad, memoria, entregables | `36_candidatos_v4.py`, `37_auditoria_v4.py` | `v4_candidate_comparison.md`, `rule_audit_v4.md`, `advanced_research_v4.md`, `spec_compliance_v4.md` |

Cada fase sigue la secuencia §4: OBSERVACIÓN → HIPÓTESIS → MÉTODO →
EXPERIMENTO → RESULTADO → ERROR ANALYSIS → DECISIÓN, y cada línea se cierra con
número en `experiments.csv` y decisión en `DECISIONS.md`.
