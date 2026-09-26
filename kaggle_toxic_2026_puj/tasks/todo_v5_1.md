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

## P2 - window (hecha, cerrada por inercia)

- [x] `embedding_13_window_v5_1.py`: ventanas 3 y 7 (2/9), control B0 reproduce v5
- [x] ws3 0.7830 (+0.0057), ws5 0.7818, ws7 0.7806; embedding solo 0.6694 (mejor de v5)
- [x] **Decision: CERRADA**. No ventanas 1, 2, 9 ni 15 (**D-037**)
- [x] El script define la maquinaria de familia que reutilizan P3, P4 y P5

## P3 - min_count (hecha, cerrada por inercia)

- [x] `embedding_14_min_count_v5_1.py`: min_count 2 y 5 (2/9), control B0 reproduce v5
- [x] mc2 0.7771 (-0.0001), mc3 0.7818 (mejor), mc5 0.7792; dispersion 0.0047
- [x] **Decision: CERRADA**. No min_count 1 ni 8 (**D-038**)
- [x] Hallazgo: 6.059/6.059 tipos con vector en los tres puntos; las subpalabras ya cubren
      el mecanismo de min_count

## P4 - subwords (hecha, cerrada)

- [x] `embedding_15_subwords_v5_1.py`: intervalos 3-5 y 3-7 (2/9), control 3-6 reproduce v5
- [x] 3-5 0.7791, 3-6 0.7818 (optimo), 3-7 0.7767; orden monotono
- [x] **Decision: CERRADA** con la regla de dispersion revisada (**D-039**)

## P5 - negative sampling (hecha, cerrada por inercia)

- [x] `embedding_16_negative_v5_1.py`: 5 negativos (1/9), control con 10 reproduce v5
- [x] neg5 0.7811 (+0.0039) frente a 0.7818; dispersion 0.0006, la mas baja de las cinco
- [x] **Decision: CERRADA** (**D-040**). Presupuesto 9/9 y **reserva 0/1 intacta**
- [ ] `embedding_14_min_count_v5_1.py` (2/9)
- [ ] `embedding_15_subwords_v5_1.py` (2/9)
- [ ] `embedding_16_negative_v5_1.py` (1/9)
- [ ] una decision registrada por familia

## P6 - comparacion final (hecha)

- [x] `embedding_17_comparacion_final_v5_1.py`: tabla maestra, heatmap, curva de escala
- [x] 14 configuraciones, 0 adoptables; embedding aislado rango 0.0297 vs fusion 0.0063
- [x] **D-041**: conclusion de representacion y de tarea, y no se escala el corpus

## P7 - complementariedad (hecha)

- [x] `embedding_18_complementariedad_v5_1.py`: salvage/damage de 4 variantes
- [x] bootstrap pareado de la interaccion ortografica (sin significacion, con poca potencia)
- [x] ejemplos de filas y slices consolidadas
- [x] **D-042**: la ventaja de WS-3 esta en romper menos, no en arreglar mas

## P8-P10 - cierre (pendiente)
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
