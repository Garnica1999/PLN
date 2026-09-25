# Tareas de la fase v5.1

## P0 - Contabilidad (hecha)

- [x] Migrar el worktree a `/home/carlos/Git/PLN` con el WIP de AGE aparcado en stash
- [x] Verificar sha256 de los 5 corpus y de los 10 `.vec` tras la migracion
- [x] Verificar que el control V03 reproduce 0.777211 exacto desde la nueva ubicacion
- [x] `training_signature` canonica (13 campos) en `experiments/embedding_11_auditoria_firmas_v5_1.py`
- [x] Anadir `processed_sha256` y `corpus_version` al manifiesto de corpus
- [x] Inventario de las 10 firmas de v5, con verificacion de unicidad
- [x] Calcular la firma de las 9 propuestas: 0 colisiones
- [x] Tabla A (`embedding_architecture_results_v5_1.csv`) y Tabla B
      (`embedding_downstream_evaluations_v5_1.csv`)
- [x] Auditoria de reglas de la seccion 36 delegada en el script de auditoria de v5
- [x] **D-035** registrada

## P1 - vector_size (hecha, cerrada por inercia)

- [x] `experiments/embedding_12_vector_size_v5_1.py`: entrenados 100d y 300d (2/9)
- [x] embedding-only (mean+LR, mean+SVC) y fusion L2 con V03, con el 200d reevaluado
      como control (reproduce v5 exacto)
- [x] bootstrap pareado, matriz de complementariedad y 11 slices
- [x] cobertura: 0.6839 por tipo y 0.8833 por token (train+test, como en v5)
- [x] **Decision: CERRADA** (inerte, dispersion 0.0017). No se exploran 400d, 500d, 768d
- [x] **D-036**: el gate pasa a ser de dos niveles

## P2-P5 - window, min_count, subwords, negative (pendiente)

- [ ] `embedding_13_window_v5_1.py` (2/9)
- [ ] `embedding_14_min_count_v5_1.py` (2/9)
- [ ] `embedding_15_subwords_v5_1.py` (2/9)
- [ ] `embedding_16_negative_v5_1.py` (1/9)
- [ ] una decision registrada por familia

## P6-P10 - cierre (pendiente)

- [ ] comparacion final y heatmap de arquitectura
- [ ] complementariedad: salvage, damage, slices, solapamiento con char_wb
- [ ] seccion v5/v5.1 en `notebooks/10_final_pipeline.ipynb` con `TRAIN_FROM_SCRATCH=False`
- [ ] validacion del notebook completo por ejecucion
- [ ] `FINAL_REPORT.md`, `PROJECT_STATE.md`, `DECISIONS.md`
- [ ] auditoria final y veredicto GO / NO-GO

## Fuera de alcance

- Escalar el corpus (100M, 250M, 500M): la curva 5M-50M ya no muestra tendencia y la
  seccion 12 de la spec no autoriza presupuesto automatico para ello.
- Reentrenar Word2Vec: ya cumple su papel de control del mecanismo subword.
- Reabrir poolings, clasificadores o umbrales: cerrados en v5 salvo hipotesis nueva.
- Usar el holdout o generar submissions.
