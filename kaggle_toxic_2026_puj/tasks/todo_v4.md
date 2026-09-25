# TODO v4 - KAGGLE_TOXIC_2026_PUJ (Advanced Research v4)

Actualizado: 2026-09-25. Rama: `proyecto-kaggle-toxic-2026-v4`.
Spec: `docs/advanced_research_v4_spec.md`. Plan: `tasks/plan_v4.md`.
Backlog pre-registrado: `reports/research_backlog_v4.md`.

Leyenda: `[ ]` pendiente, `[~]` en curso, `[x]` hecho, `[!]` cerrado con evidencia negativa.

## P0 - Scaffold, verificacion y pre-registro

- [x] Rama `proyecto-kaggle-toxic-2026-v4` desde `main`
- [x] Spec maestra archivada en `docs/advanced_research_v4_spec.md`
- [x] Plan y TODO de la fase (`tasks/plan_v4.md`, `tasks/todo_v4.md`)
- [x] Verificacion de V03 y del protocolo (`experiments/26_scaffold_v4.py`, veredicto OK)
- [x] Gap analysis con taxonomia de estados (`reports/v4_gap_analysis.md`)
- [x] Backlog v4 pre-registrado (26 lineas) y matriz de cobertura (`reports/signal_coverage_v4.csv`)
- [x] Memoria actualizada: D-025, RUNLOG, `experiments.csv` (fila V4-00), PROJECT_STATE
- [~] Revision bibliografica 2022-2026 (`reports/estado_arte_v4.md`, en curso en paralelo)

## P1 - TRACK A y B (diagnostico)

- [ ] V4-A01 clusters de consistencia de etiquetas (exacto, normalizado, Jaccard, cosimilitud) -> `reports/label_consistency_v4.csv`
- [ ] V4-A02 relacion contradiccion-error por etiqueta
- [ ] V4-B01 `hardness_score` OOF por etiqueta y global
- [ ] V4-B02 informe difficult-vs-noise (`reports/hardness_vs_noise_v4.md`)
- [ ] Gate: cifras de contradicciones y de su asociacion con el error; sin tocar el test

## P2 - TRACK B (experimento) y TRACK C

- [ ] V4-B03 reweighting mild/medium/clipped en LR y LinearSVC
- [ ] V4-B04 dureza por etiqueta
- [ ] V4-B05 tres estrategias de manejo de contradicciones
- [ ] V4-B06 perdidas robustas (si es viable sin hacks)
- [ ] V4-C01 log-odds suavizado por clase aplicado a la matriz TF-IDF
- [ ] V4-C02 chi2 y MI por etiqueta
- [ ] V4-C03 BM25 (k1, b), binario y augmented TF
- [ ] V4-C04 jilla de `class_weight` por etiqueta
- [ ] V4-C05 interaccion sample weight x class weight
- [ ] Gate: cada linea con `corrected_cv`, pliegues, bootstrap y decision

## P3 - TRACK E (modelo conjunto de etiquetas)

- [ ] V4-E01 multinomial sobre las 6 combinaciones + marginalizacion
- [ ] V4-E02 smoothing/prior jerarquico para 110 y 001
- [ ] V4-E03 coeficientes compartidos + interacciones regularizadas
- [ ] Gate: comparacion contra 3 binarios independientes y contra el meta v2

## P4 - TRACK H (calibracion de las entradas del meta)

- [ ] V4-H01 simulacion de shift por tamaño de entrenamiento y alineacion de scores
- [ ] V4-H02 meta restringido con complementariedad demostrada
- [ ] Gate: el shift se cuantifica y su efecto en `corrected_cv` queda registrado

## P5 - TRACK G (modelos estadisticos alternativos)

- [ ] V4-G01 verosimilitud class-conditional (unigram/bigram/char) como feature
- [ ] V4-G02 Ridge, SGD y PassiveAggressive sobre la union
- [ ] V4-G03 no lineales sobre SVD (RBF/HistGradientBoosting/ExtraTrees)
- [ ] Gate: con la revision bibliografica cerrada (C, E y G)

## P6 - TRACK D (subtemas no cubiertos por v2)

- [ ] V4-D01 sufijos y prefijos 2/3/4 como canal sparse
- [ ] V4-D02 interacciones como tokens sparse
- [ ] V4-D03 ventanas locales como tokens
- [ ] V4-D04 subtipos de HS implicito
- [ ] Gate: solo con hipotesis ligada a un error medido

## P7 - TRACK F y discovery

- [ ] V4-F01 shift no supervisado train vs test
- [ ] V4-F02 robustez a duplicados (unicos vs repetidos)
- [ ] V4-F03 acuerdo de etiqueta del vecino mas cercano
- [ ] V4-F04 techo de impacto de la propagacion por near-duplicado
- [ ] `reports/correction_opportunities_v4.csv`, `reports/correction_matrix_v4.csv`, `reports/error_taxonomy_v4.md`
- [ ] Gate: ningun artefacto usa labels de test

## P8 - Cierre

- [ ] Seleccion de maximo 3 candidatos con evidencia completa
- [ ] `reports/v4_candidate_comparison.md` (comparativo, no ranking numerico)
- [ ] Auditoria de reglas v4 y de reproducibilidad (doble ejecucion + SHA256)
- [ ] Submissions solo si pasan gates; `submissions/submission_manifest.csv` actualizado
- [ ] `reports/advanced_research_v4.md` y `reports/spec_compliance_v4.md`
- [ ] Memoria: D-026..., RUNLOG, PROJECT_STATE, `experiments.csv`
- [ ] Seccion nueva en `FINAL_REPORT.md` y actualizacion del notebook con re-ejecucion
- [ ] PR a `main` desde `proyecto-kaggle-toxic-2026-v4`

## Fuera de alcance (decidido)

- Envios automaticos a Kaggle (los hace Carlos a mano)
- Transformers, LLM, embeddings o clasificadores preentrenados
- Pseudo-labeling, corpus o etiquetas externas
- Mas de una evaluacion del holdout y mas de una sonda de LB por candidato
- Repetir lineas cerradas en `reports/v4_gap_analysis.md`
