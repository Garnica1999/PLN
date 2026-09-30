# V32 - E2 (P2) - Pre-registro congelado: challenger CatBoost sobre la matriz 279

**Fecha**: 2026-09-30. **Estado**: CONGELADO antes de ejecutar (commit `9a2fbfd`).
**Baseline primario recomputado**: 0.6159732 (referencia secundaria nested 0.6154643).

## La pregunta y la hipotesis

> El boosting con ordered boosting (CatBoost) sobre la matriz 279 congelada produce
> un challenger con informacion complementaria al campeon V30 + plug-in, mas alla de
> la familia boosting ya usada (LightGBM del pool)?

Ciclo 2 de la reapertura V32 (tras cerrar P1 como NO_ATRIBUIBLE, D-081). Auditoria por
operacion: CatBoost 1.2.10 no aparece en el historial y v20 lo registro como nunca
ejecutado sobre matriz_279.

## Materiales congelados (SHA256)

| Artefacto | SHA256 (16) |
|---|---|
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| manifest_plugin | `8d635e6a546b5a0e...` |
| matriz_279_bytes | `175b6d1ee242eef9...` |
| fuente_v19_b1_boosting.py | `841aa5fdf1e13c57...` |
| fuente_v32_01_surrogate_macrof1.py | `6d4176837024c792...` |
| fuente_v30_02_ensemble.py | `be11ee23e0d06947...` |
| fuente_age_umbrales_por_clase.py | `a9cb47ec4c37f3fe...` |

Versiones: catboost 1.2.10, lightgbm 4.6.0, numpy 1.26.4.

## Brazos

| Brazo | Config | Rol |
|---|---|---|
| A1_catboost | CatBoost MultiClass: 3000 iter, lr 0.03, depth 6, l2 3.0, Bernoulli 0.8, border 254, Plain/SymmetricTree, seed 20261010, 32 hilos | decisional |
| A2_lgbm_twin | LightGBM con la MISMA matriz/folds/split/ES(150) macro-F1 y el ancla afinada V29/V30 (56 hojas, depth 7, lr 0.03, min 54, bag 0.78, col 0.6, l1 0.015, l2 0.0005, deterministic, 32 hilos) | control de atribucion (decisional) |
| A3_barajado | CatBoost A1 con etiquetas del fold-train permutadas (20261011) | control negativo (no decisional) |

## Entrenamiento congelado

- Split interno 90/10 estratificado del fold-train (semilla 20261010+fold); el
  fold-val jamas se usa para parar/seleccionar; solo predict_proba para el OOF.
- Early stopping con od_wait/early_stopping = 150 sobre TotalF1 (CatBoost) y
  macro-F1 feval (LGBM); use_best_model; la equivalencia TotalF1 <-> macro-F1 se
  verifica en el sanity.
- Guarda de truncamiento: best_iteration >= iterations-150 => flag; A1 capado a
  MARGINAL; A2 se registra como control debilitado.
- Determinismo: semillas fijas; 32 hilos fijos; sonda de reentreno del fold 0 de A1 y
  de A2 con tolerancias declaradas (0 exacto; <=1e-9 declarada; >1e-9 rango y
  delta_gate > rango; rango >= 0.0008876 => no concluyente).
- Ejecucion resumible por fold (checkpoint JSON); calibracion tras fold 0 de A1
  (proyeccion 9 x t_fold0 > 20 h => parar) y timeout global 24 h.

## Integracion y evaluacion (mecanismo P6/P7, sin cambios)

- Combinador logistico anidado [L2(v30) || L2(p_brazo)] (12 features); LogReg C=1.0 por
  fold; decision argmax(p_comb * w) con los pesos del plug-in CONGELADOS.
- Standalone macro-F1 del OOF por brazo; complementariedad rescate/dano/neto y alta
  confianza (max p del campeon >= 0.6); diagnosticos sin regla.

## Gates (todos obligatorios)

1. baseline recomputado en la misma corrida (aborta si |delta| > 5e-7);
2. delta_int(A1) >= +0.0015;
3. folds positivos de A1 >= 4/5;
4. control favorable: delta_int(A1) > delta_int(A2) Y bootstrap pareado de la
   diferencia (F1_combA1 - F1_combA2) con IC95 sin cero;
5. control negativo: si delta_int(A3) >= delta_int(A1) -> NO_ATRIBUIBLE;
6. bootstrap pareado 2000 (semilla 20261010) del delta de A1 sin cero;
7. sin leakage (asserts del runner) y sin truncamiento de A1 para PROMISING.

## Arbol de decision

| Resultado | Estado | Siguiente paso |
|---|---|---|
| gates completos | PROMISING | segunda particion congelada (semilla 987654) |
| delta > 0 sin gates | MARGINAL | no se adopta; cierre documentado |
| delta <= 0 | CLOSED | se cierra P2 (ultimo candidato priorizado; P3 solo con autorizacion) |

## Compliance y limites

- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin
  leaderboard, sin submission; un script nuevo; campeon intacto.
- El gemelo LGBM es un control interno reentrenado desde cero (no reutiliza pesos del
  pool); auditoria AST del runner (0 prohibidos, 0 lecturas de test/holdout).
- Referencia de variabilidad de entrenamiento: 0.0008876.
