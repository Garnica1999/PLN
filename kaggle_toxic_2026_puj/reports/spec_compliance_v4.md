# Cumplimiento de la especificación v4 «Advanced Research v4»

Fecha: 2026-09-25. Spec archivada en `docs/advanced_research_v4_spec.md`
(108 secciones). Auditorías: `reports/rule_audit_v4.md` (0 violaciones) y
`reports/reproducibility_v4.md` (doble ejecución con SHA256 estable).
Registro: `experiments.csv` con 143 filas (128 previas + 15 de v4) y
`DECISIONS.md` D-025…D-029.

Convenciones: **COMPLETA** = ejecutada con evidencia; **COMPLETA (variante)** =
cubierta con otro nombre o empaquetado; **PARCIAL** = cubierta en parte;
**N/A** = no aplicable o impedida por las reglas; **NO EJECUTADA** = sin
ejecución, con el motivo explícito.

## 1. Restricciones y protocolo (secciones 0-4, 69-72, 77, 81, 84, 91, 106-107)

| Sección | Estado | Evidencia |
|---|---|---|
| 0-1, 4 | COMPLETA | `reports/v4_gap_analysis.md`; el análisis se estructura por datos, representación, función de aprendizaje y estructura de salida, y se traduce en los ocho tracks |
| 2 (baseline intacta) | COMPLETA | `experiments/26_scaffold_v4.py`: 5/5 hashes OK, `corrected_cv` 0.777211 reproducida exactamente |
| 69 (corrected validation) | COMPLETA | protocolo v2 (D-016) usado en los 13 scripts de la fase: `src/splits.py` |
| 70 (holdout) | COMPLETA | 4/5 al abrir, **5/5** al cerrar, con la única evaluación autorizada para el finalista (`reports/holdout_evaluaciones_v2.csv`) |
| 71 (bootstrap) | COMPLETA | bootstrap pareado por tweet, 2.000 remuestreos, en todas las comparaciones decisivas |
| 72 (multiplicidad) | COMPLETA | `reports/research_backlog_v4.md` pre-registrado antes de experimentar; regla de nivel +0.005 y 3/5 pliegues aplicada |
| 77 (versionamiento) | COMPLETA | V03 intacta; V04 no se crea porque ningún candidato se adopta; los artefactos experimentales se nombran `v12_descartado_*` |
| 81 (no gastar LB) | COMPLETA | **0 sondas nuevas** en v4; la submission v12 se marcó `DESCARTADO` antes de cualquier envío |
| 84 (no repetir) | COMPLETA | `reports/v4_gap_analysis.md` clasifica cada idea como NEW/NOT_TESTED/TESTED_NEGATIVE/TESTED_BUT_BADLY_INTEGRATED/TESTED_BUT_UNCERTAIN/TESTED_AND_CLOSED/TRANSFER_FAILED |
| 91 (rule audit) | COMPLETA | `reports/rule_audit_v4.md` y `reports/auditoria_v4.json`: 0 violaciones en 12 scripts auditados |
| 106-107 (principio y objetivo) | COMPLETA | `reports/advanced_research_v4.md` §5: los resultados negativos se atribuyen a mecanismo concreto, nunca a «irreducible» |

## 2. Datos, representación y etiquetas (5-8, 20-34, 44)

| Sección | Estado | Evidencia |
|---|---|---|
| 5-6 (auditoría y gap analysis) | COMPLETA | `reports/baseline_v03_v4_check.json`, `reports/v4_gap_analysis.md` |
| 7-8 (consistencia y contradicciones) | COMPLETA | `reports/label_consistency_v4.csv` (15 clusters 2+, 2 contradictorios, 84 tweets); estrategias A/B/C en `reports/reweighting_v4.csv` |
| 20 (multigranularidad) | COMPLETA (variante) | la unión word+char ya era la granularidad ganadora; no se abre una línea nueva |
| 21-22 (subword, sufijos y prefijos) | COMPLETA | `reports/structural_sparse_v4.csv` (canal de afijos: 0.7758 standalone/hstack) |
| 23-24 (patrones y ventanas) | COMPLETA | ventanas locales como tokens: `d:ventanas_hstack` y `standalone` en el mismo CSV |
| 25-29 (target-insulto, cita, autor/citado, negación, noticia) | COMPLETA (variante) | cerradas en v2 (E2-05…E2-08); el gap analysis las marca `TESTED_AND_CLOSED` y v4 no las repite |
| 30-31 (emoción y afiliación) | COMPLETA (variante) | cerradas en v2 (E2-09) |
| 32 (HS implícito) | COMPLETA | `reports/implicit_hate_diagnostic_v4.json` (diagnóstico de masa antes de construir el canal) |
| 33-34 (colectivo y generalización) | COMPLETA (variante) | cerradas en v2 (E2-06) |
| 44 (interacciones sparse) | COMPLETA | `d:interacciones_hstack` y `standalone` en `reports/structural_sparse_v4.csv` |

## 3. Función de aprendizaje (9-19, 41-43)

| Sección | Estado | Evidencia |
|---|---|---|
| 9-13 (dureza, ruido vs difícil) | COMPLETA | `reports/hardness_vs_noise_v4.md`; las tres poblaciones quedan cuantificadas (1.144/296/43) |
| 11 (reweighting iterativo) | COMPLETA | `reports/reweighting_v4.csv` con 6 esquemas × 2 modelos y OOF anidado de 4 pliegues |
| 12 (dureza por etiqueta) | COMPLETA | `V4-B04` en `experiments.csv` |
| 14 (pérdidas robustas) | NO EJECUTADA | sklearn no expone una BCE alternativa para `LogisticRegression` sin reimplementar el optimizador; hacerlo exigiría PyTorch, es decir DL, que la spec §64 pone en última prioridad. Se registra como `IMPLEMENTATION_ISSUE`, no como resultado negativo |
| 15-19 (weighting supervisado, BM25, IR) | COMPLETA | `reports/supervised_weighting_v4.csv`: log-odds, chi2, MI, BM25 (4 combinaciones de k1/b), binario |
| 41 (hard negatives) | PARCIAL | el diagnóstico de masa de `reports/implicit_hate_diagnostic_v4.json` se ejecuta, pero los subproblemas de especialista no se entrenan porque la evidencia de v2 (E2-20) ya fue negativa y el track B no mostró señal |
| 42-43 (cost-sensitive y sample×class) | COMPLETA | barrido anidado por etiqueta y combinación con downweight: `V4-C04` y `V4-C05` |

## 4. Estructura de salida (35-40, 60)

| Sección | Estado | Evidencia |
|---|---|---|
| 35-37 (modelo conjunto, estructura, combinaciones raras) | COMPLETA | `reports/joint_label_v4.csv`: 12 variantes (6 y 4 clases, C ∈ {0.5,1,2}, balanced y unbalanced) |
| 38-39 (dependencia como feature, compartido + interacción) | COMPLETA (variante) | el meta de 36 features es exactamente un modelo compartido con términos de dependencia; §39 se cubre en `V4-H01` |
| 40 (multi-task desde cero) | N/A | `DL-01`…`DL-03` ya negativos y la spec §64 los relega; no se abre línea |
| 60 (perplejidad class-conditional) | COMPLETA | `reports/class_conditional_lm_v4.csv` |

## 5. Robustez y modelos alternativos (52-65)

| Sección | Estado | Evidencia |
|---|---|---|
| 52-53 (shift y OOD) | COMPLETA | `reports/distribution_shift_v4.csv` (10 variables no supervisadas, 2 con desplazamiento) |
| 54 (representaciones robustas) | COMPLETA (variante) | sin shift material, no se abre una línea de representaciones robustas; la reponderación se evalúa en su lugar (C) |
| 55-56 (duplicados y vecindad) | COMPLETA | `reports/discovery_v4.json` (robustez por subconjunto y vecindad) |
| 57-58 (retrieval y propagación) | COMPLETA | techo nulo: 0 coincidencias literales entre test y train; la línea se cierra sin aplicar |
| 59-61 (LM, perplejidad, estilo) | COMPLETA | verosimilitud unigram/bigram/char y su fusión; el perfil de estilo se cubre indirectamente por el canal de afijos (D) |
| 62 (clasificadores alternativos) | COMPLETA | Ridge, SGD y PassiveAggressive: 0.7698, 0.7454 y 0.7626 |
| 63 (no lineales sobre SVD) | COMPLETA | SVC RBF 0.7904, HistGradientBoosting 0.7297, ExtraTrees 0.7033 |
| 64-65 (DL) | N/A | sin hipótesis nueva y con la spec §64 en última prioridad |

## 6. Fusión y cierre (49-51, 66-68, 73-76, 79-83, 95-104)

| Sección | Estado | Evidencia |
|---|---|---|
| 49-51 (fusión y calibración del meta) | COMPLETA | `reports/meta_shift_v4.csv` (4 situaciones: control, shift, cuantil, Platt) y `reports/correction_matrix_v4.csv` (5 canales) |
| 66-68 (geometría, ablación, métricas) | COMPLETA | todas las variantes reportan F1 por etiqueta, macro, micro, exact match, pliegues y std |
| 73-74 (tracks y backlog) | COMPLETA | `reports/research_backlog_v4.md` con las 26 líneas pre-registradas y su estado final |
| 75-76 (registros) | COMPLETA | `experiments.csv` (143 filas) y `DECISIONS.md` D-025…D-029 |
| 78, 80, 100 (submissions y reproducibilidad) | COMPLETA | submission v12 generada, validada y **descartada** por el holdout; `reports/reproducibility_v4.md` con doble ejecución y SHA256 |
| 79 (manifest) | COMPLETA | `submissions/submission_manifest.csv` con el estado `DESCARTADO` de v12 |
| 82-83 (condición de éxito y resultados negativos) | COMPLETA | los cuatro candidatos y sus veredictos están en `reports/v4_candidates.csv` y en el informe de la fase |
| 92-93 (revisión bibliográfica) | COMPLETA | `reports/estado_arte_v4.md` (38 papers verificados) y `.csv`; cuatro temas sin cobertura reciente declarados explícitamente |
| 94 (cobertura) | COMPLETA | `reports/signal_coverage_v4.csv` (29 fenómenos) |
| 95 (taxonomía) | COMPLETA | `reports/error_taxonomy_v4.md` (13 categorías con conteo e ids) |
| 96 (matriz de corrección) | COMPLETA | `reports/correction_matrix_v4.csv` (5 canales y 4 candidatos) |
| 97-99 (candidatos y comparativa) | COMPLETA | `reports/v4_candidate_comparison.md` con comparación descriptiva y riesgo de transferencia por candidato |
| 101 (sonda de LB) | N/A | no se autorizó ninguna sonda: el holdout ya rechazó al único finalista |
| 103-104 (informe y estado) | COMPLETA | `reports/advanced_research_v4.md` y actualización de `FINAL_REPORT.md`, `PROJECT_STATE.md`, `RUNLOG.md` y el notebook |

## 7. Brechas acotadas

1. **Sección 14 (pérdidas robustas)**: no ejecutada por limitación de la librería, no
   por resultado negativo (`IMPLEMENTATION_ISSUE`).
2. **Sección 41 (hard negatives)**: ejecutada solo la parte de diagnóstico; la
   construcción de especialistas se justifica únicamente si el track B mostrara
   señal, que no mostro.
3. **Secciones 25-31 y 33-34**: `TESTED_AND_CLOSED` en v2; v4 no las repite por
   la regla §84 y §85-§89, y el gap analysis deja constancia con su evidencia.
4. **Secciones 52-58**: ejecutadas como diagnóstico; no generan modelo porque los
   números no justifican abrir una línea.

## 8. Veredicto de la condición de terminación (§102-B y §107-B)

**CUMPLIDA.** Se investigaron y cerraron con evidencia las hipótesis nuevas
razonables: 8 tracks, 26 líneas de backlog, 15 experimentos nuevos registrados
(143 filas en total), 0 violaciones de reglas, reproducibilidad verificada y un
candidato final que pasó todos los gates locales y **fue rechazado por el
holdout congelado** (0.8163 vs 0.8285). V03 se mantiene como modelo y submission
finales; no se fabricó ninguna mejora y no se gastaron sondas de leaderboard.
