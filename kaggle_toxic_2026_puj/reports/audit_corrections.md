# P12 - Registro de correcciones (auditoria forense TOXIC 2026 PUJ)

Regla aplicada: **el arbol congelado NO se modifica**. Cada correccion se
registra con su patch exacto para que Carlos (o una fase posterior autorizada)
la aplique con re-validacion. Original y corregido conviven; ningun resultado
historico se sustituye en silencio.

| Bug_ID | Archivo(s) | Comportamiento original | Comportamiento correcto | Afecta a | Reproduccion | Estado |
|---|---|---|---|---|---|---|
| B-01 | `experiments.csv` (6 filas, lineas 147-152 de la version final) | 6 copias de una fila corrupta de 9 campos con `experiment_id=KAGGLE_TOXIC_2026_PUJ` (lote EBD-01..06, commit ca275b9) | 6 filas validas con ids EBD-01..06 y sus campos/metricas (los datos existen en `reports/embedding_downstream_*` y `embedding_intrinsic_eval.md`) | registro de la fase v5; trazabilidad EBD-01..06 | `audit/p02_experiments_forensic.py` | PROPUESTA (no aplicada) |
| B-02 | `experiments.csv` | No existen filas EBD-11 ni EBD-12 (huecos internos de numeracion) | O documentar el salto o registrar las filas que correspondan | registro v5 | `audit/p02_findings.json` (huecos_numeracion) | PROPUESTA (no aplicada) |
| B-03 | `reports/v6_reproducibility.md` | 9 rutas `experiments/experiments/v6_*.py` (prefijo duplicado) | `experiments/v6_*.py` | documentacion v6 | `grep -c experiments/experiments/` | PROPUESTA (no aplicada) |
| B-04 | `reports/target_encoding_audit_v2.md` | "solo 2 de 5 evaluaciones usadas" del holdout | 5/5 usadas (log final `holdout_evaluaciones_v2.csv`) | documentacion v2 | F-27 | PROPUESTA (no aplicada) |
| B-05 | `DECISIONS.md` | 8 decisiones con `Conclusion final: pendiente` (D-001, D-003, D-004, D-005, D-007, D-008, D-010, D-017) | Redactar la conclusion final usando la evidencia posterior (postproceso descartado, spaCy anotador, D-004 superada por D-015/D-016, etc.) | registro de decisiones | `audit/p08_docs_decisions.py` | PROPUESTA (no aplicada) |
| B-06 | `experiments/v6_07_ruido_par.py` | La sospecha se calcula con las etiquetas de TODO el desarrollo (incluye el pliegue evaluado); pesos de entrenamiento con informacion del pliegue de validacion | Calcular `densidad`/`estructural` por pliegue con etiquetas solo del entrenamiento | V6-07 (F1-completo +0.000142 -> 0.0 fold-local; sin adopcion) | `audit/p04b_suspicion_foldlocal.json` | PROPUESTA (no aplicada; familia cerrada) |
| B-07 | `experiments/00_auditoria_datos.py` | El clustering de near-duplicados ajusta TF-IDF y une componentes sobre train+test (2885 textos) | Construir grupos solo con train (o documentar la eleccion como metadata estructural) | grupos de validacion de todas las fases | F-05; contrafactual ARI=1.0 (impacto nulo demostrado) | PROPUESTA (no aplicada; sin impacto) |
| B-08 | `features/features/oof_v03_final.npz` | Copia identica anidada del OOF (mismo sha256) | Eliminar el directorio duplicado | higiene de artefactos (v8) | F-14 | PROPUESTA (no aplicada) |
| B-09 | `submissions/` (v8) | Dos familias de nombres comparten numeracion v08..v12 (`toxic_submission_v08_meta_v2.csv` y `toxic_submission_v08_p1_fold_ensemble.csv`, etc.) | Renumerar la familia de v8 a partir de v17 | trazabilidad de submissions | F-33; `audit/p11_submissions.py` | PROPUESTA (no aplicada) |
| B-10 | `reports/final_candidates.csv` | `delta_vs_v03` calculado contra la referencia redondeada 0.7772 (desvio hasta 1.1e-5) | Usar 0.777211 (o documentar la referencia del redondeo) | fines v2 | F-17 | PROPUESTA (no aplicada) |
| B-11 | historia git (commit ae0084e) | El mensaje afirma un "digesto normalizado c3aa4198 estable" del notebook que no existe en ningun artefacto | Documentar el metodo del digesto o retirar la afirmacion | documentacion v2 | F-31; `audit/p08_docs_decisions.py` | DOCUMENTADO (UNKNOWN) |
| B-12 | `submissions/leaderboard_log.csv` | La columna `diferencia` significa "LB - local" en 12 filas y "LB - LB(V03)" en las 3 filas de v8 | Unificar semantica (columna extra o nota por fila) | analisis de transferencia | F-33; `audit/p11_submissions.py` | PROPUESTA (no aplicada) |
| B-13 | `reports/audit_documentation.csv` (auditoria) | - | - | n/a | n/a | RESUELTO en la auditoria (D-013 verificado como actualizado) |
| B-14 | `notebooks/10_final_pipeline.ipynb` (celdas 29-30) | La celda 30 afirma y comprueba '12 sondas de leaderboard'; el log final tiene 15 (v8 anadio 3) -> `AssertionError` y el notebook ya NO se ejecuta end-to-end | Esperar 15 sondas (o recuento dinamico) y actualizar la prosa de la celda 29 | entregable del curso; EBD-26 (GO) quedo desactualizado | audit/out_p06/wave2.log; celda 30 del notebook | PROPUESTA (no aplicada) |

## Notas de aplicacion

- Ninguna correccion altera resultados historicos: B-01/B-02 son de registro,
  B-03/B-04/B-05/B-11 de documentacion, B-06/B-07 son mejoras de protocolo con
  impacto medido (1.4e-4 y 0.0) y B-08/B-09/B-10/B-12 de higiene/trazabilidad.
- Si se aplican B-06 y B-07 en una fase futura, re-ejecutar los controles v6 y
  el OOF corregido para confirmar que 0.777211 no cambia (la auditoria predice
  que no cambia: contrafactuales ya medidos).
