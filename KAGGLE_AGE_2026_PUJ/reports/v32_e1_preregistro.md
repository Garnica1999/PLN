# V32 - E1 - Pre-registro congelado: challenger con perdida surrogate de macro-F1

**Fecha**: 2026-09-30. **Estado**: CONGELADO antes de ejecutar (commit `9c5cca2`).
**Baseline primario recomputado**: 0.6159732 (referencia secundaria nested 0.6154643).

## La pregunta y la hipotesis

> Entrenar un modelo con una perdida que **directamente** optimiza macro-F1 (surrogate
> diferenciable) produce un challenger con informacion complementaria al campeon
> V30 + plug-in, y mejora el macro-F1 corregido al integrarse con el mecanismo establecido?

Es la unica operacion priorizada que nunca se implemento (familia 'objetivo': solo
cross-entropy en 95 scripts). P2 (CatBoost) y P3 (FTRL) solo se evaluan si P1 falla.

## Materiales congelados (SHA256)

| Artefacto | SHA256 (16) |
|---|---|
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| manifest_plugin | `8d635e6a546b5a0e...` |
| matriz_279_bytes | `175b6d1ee242eef9...` |
| fuente_v19_b1_boosting.py | `841aa5fdf1e13c57...` |
| fuente_v30_02_ensemble.py | `be11ee23e0d06947...` |
| fuente_age_umbrales_por_clase.py | `a9cb47ec4c37f3fe...` |

## Dispositivo y arquitectura

- Entrenamiento en **GPU (ROCm)** con guardian D-007: `require_gpu(wait acotado)`; si la
  GPU esta ocupada, el script avisa y se detiene (sin fallback silencioso a CPU).
- GPU detectada: `AMD Radeon RX 9070 XT`; torch `2.14.0+rocm7.2`; matmul precision `highest`.
- Red: StandardScaler (fit solo fold-train) -> Linear(279,128) -> GELU -> Dropout(0.1)
  -> Linear(128,6) -> softmax. Orden canonico de clases (demostrado por diagonal).

## Perdida (soft-macro-F1)

```text
p = softmax(z), clamp [1e-6, 1-1e-6]; objetivos one-hot float32
TP_c = sum(p * y);  FP_c = sum(p * (1-y));  FN_c = sum((1-p) * y)
P = TP/(TP+FP+1e-6);  R = TP/(TP+FN+1e-6);  F1 = 2PR/(P+R+1e-6)
L = 1 - media_c F1_c
```

## Entrenamiento congelado

- Lote 8192; pasos por epoca **calculados** (ceil(n/lote)); optimizador **AdamW** lr=0.001,
  wd=0.0001; max 60 epocas; grad-clip 5.0; shuffle con generador por
  (brazo, fold, epoca) con semilla fija.
- **Early stopping estricto (decision de Carlos)**: split interno 90/10 estratificado del
  fold-train (semilla 20261010+fold), paciencia 8, se restauran los mejores pesos por
  macro-F1 interna. El fold-val jamas se usa para parar.
- Guardas: loss no finita -> salta paso; >=3 en una epoca o >=10 acumuladas -> aborta brazo.
- Sanity-check de 1 lote (100 pasos): PASS si loss finita y reduccion >= 0.30.
- Determinismo: seeds fijos; `torch.use_deterministic_algorithms(True)` (fallback
  warn_only registrado si ROCm lo exige); `torch.set_float32_matmul_precision('highest')`;
  8 hilos CPU; sonda de determinismo sobre fold 0 del brazo A1.

## Brazos

| Brazo | Perdida | Rol |
|---|---|---|
| A1_surrogate | soft-macro-F1 | decisional |
| A2_ce | cross-entropy media (gemelo identico) | control de atribucion |
| A3_barajado | soft-macro-F1 con etiquetas permutadas (20261011) | control negativo no decisional |

## Integracion y evaluacion (mecanismo P6/P7, sin cambios)

- Combinador logistico anidado sobre [L2(v30_prob) || L2(p_brazo)] (12 features); LogReg
  C=1.0 fit por fold; decision `argmax(p_comb * w)` con los pesos del plug-in CONGELADOS.
- Standalone: macro-F1 del argmax del OOF por brazo (global y por fold).
- Complementariedad vs campeon: rescate/daño/neto; alta confianza = max p del campeon
  >= 0.6.
- Diagnosticos: `argmax_sin_regla` y delta vs nested. No se re-derivan los pesos.

## Gates (todos obligatorios)

1. baseline recomputado en la misma corrida (aborta si |delta| > 5e-7);
2. delta_int(A1) >= +0.0015;
3. folds positivos de A1 >= 4/5;
4. control favorable: delta_int(A1) > delta_int(A2) (comparacion pre-registrada);
5. control negativo: si delta_int(A3) >= delta_int(A1) -> NO_ATRIBUIBLE;
6. bootstrap pareado 2000 (semilla 20261010) sin cero;
7. sin leakage (asserts del runner: alineacion, OOF por fold, split interno).

## Arbol de decision

| Resultado | Estado | Siguiente paso |
|---|---|---|
| gates completos | PROMISING | segunda particion congelada (semilla 987654) antes de cualquier otra cosa |
| delta > 0 sin gates | MARGINAL | no se adopta; cierre documentado o P2 con justificacion |
| delta <= 0 | CLOSED | se cierra P1; solo entonces se evalua P2 (CatBoost) |

## Compliance y limites

- 0 embeddings (ambos brazos), 0 transformers, 0 LLM, 0 datos externos; sin test, sin
  holdout, sin leaderboard, sin submission; un script nuevo; campeon intacto.
- Las activaciones ocultas nunca se exportan como features (solo las 6 probabilidades).
- Referencia de variabilidad de entrenamiento: 0.0008876.
