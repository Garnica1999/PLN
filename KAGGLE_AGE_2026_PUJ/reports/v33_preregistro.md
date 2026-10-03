# V33 - E1 - Pre-registro congelado: FTRL-Proximal softmax, una pasada online

**Fecha**: 2026-09-30. **Estado**: CONGELADO antes de ejecutar (commit `7e59b06`).
**Baseline primario recomputado**: 0.6159732 (nested 0.6154643).
**Ancla G3 - macro-F1 trivial** (clase 0): 0.0485421218068709; gemelo SGD de D-083: 0.4934789; uniforme K=6: log(6) = 1.791759.

## La pregunta y el diagnostico que la motiva

> El aprendizaje online proximal con la semantica correcta (una pasada, softmax
> multiclase, alpha derivado) aporta informacion complementaria al campeon?

D-083: la implementacion FTRL de P3 era exacta pero la configuracion (OvR, alpha 0.05, 10 pasadas proximales) DEGENERO (log_loss de train 11.72 -> 23.03; standalone 0.040169 bajo el trivial 0.048542). Causa: `sigma` inflado ~10x por alpha=0.05 y diez pasadas en un regimen donde McMahan calibro alpha con T ~ 1e6-1e9.

## Diseno congelado

- **A1 (decisional)**: FTRL-Proximal **softmax multiclase** (K=6, un solo juego de pesos; p suma 1), **una sola pasada online**, orden = fold-train barajado una vez con `default_rng(20261010+fold)` y reutilizado por el control.
- `alpha = 3/(2*sqrt(T))` con T = n_train_fold: para T = 135468 sale alpha ~ 0.004075 (rango 0.00404-0.00409). **Derivado, cero valores evaluados**: no es un grid. Con alpha=0.05 el desplazamiento del intercepto seria 36.8, 36x mayor.
- lambda1 0.0, lambda2 1.0, beta 1.0, intercepto x0=1.0, sin power_t, float64, softmax sin temperatura.
- **A2 (control)**: `SGDClassifier(loss='log_loss', learning_rate='optimal', alpha=1e-6, max_iter=1, tol=None)` con la MISMA matriz, el mismo barajado y la misma 1 pasada. Asimetria declarada (alpha 1e-6 frente a 0.004) y por eso la sonda `|sigma*w|/|g|` es **obligatoria** (si no se publica, veredicto capado a MARGINAL).
- La hipotesis 'solo alpha tiene palanca' queda como HIPOTESIS del arquitecto, no como evidencia.

## Puerta de convergencia fail-fast (G1-G6)

- Corre dentro del fold 0, **antes** de los folds 1-4 y **antes** de integrar; se aplica **igual a A1 y a A2**.
- G1 banda de log_loss multiclase de train [0.25, 0.60] (anclaje: uniforme log 6).
- G2 log_loss en 25/50/100 % de la pasada no creciente (tolerancia 1e-6).
- G3 macro-F1 standalone en el fold-val >= 0.30 (anclas: 0.0485421 y 0.4934789).
- G4 frecuencia predicha por clase en [0.10, 0.30] (proporciones reales del train).
- G5 determinismo: max|delta p| = 0.0 entre dos corridas identicas.
- G6 filas que suman 1: max|sum_c p_c - 1| <= 1e-9.
- Si falla cualquiera: veredicto **NO_ATRIBUIBLE_BY_GATE** (nodo nuevo del arbol), **sin fallback y sin segundo alpha**; el fallo NO toca delta. El fold 0 queda registrado como consumido-diagnostico.

## Sondas obligatorias antes de entrenar (exit 4)

1. Vectorizado == lazo ingenuo (1e-12). 2. Raiz analitica con lambda 0 (1e-9). 3. TotalF1:average=Macro == macro-F1 (1e-6). 4. Sin NaN/inf. 5. **NUEVA**: log_loss multiclase propio == sklearn (1e-9) y filas que suman 1 (D-083 solo probo el OvR).
6. Sonda de refit: los sha256 de vocabulario e idf de los 5 folds deben reproducirse bit a bit.

## Gates de la serie (heredados sin cambios)

baseline +-5e-7; delta_int(A1) >= +0.0015; folds+ >= 4/5; control delta(A1) > delta(A2); IC95 pareado 2000 (semilla 20261010) sin cero; sin leakage; determinismo concluyente; sonda de atribucion publicada.
Caps: sin sonda de atribucion o con determinismo no concluyente el veredicto no puede ser PROMISING. Arbol: NO_ATRIBUIBLE_BY_GATE / PROMISING (segunda particion solo con autorizacion) / MARGINAL / CLOSED.

## Materiales congelados (SHA256)

| Artefacto | SHA256 (16) |
|---|---|
| corpus_archivo | `39c6c8270366fe15...` |
| corpus_textos | `53c1a348a7954f3e...` |
| vectorizador | `9bae157614458dbd...` |
| ancla_d083_json | `aa791c89b0b89cea...` |
| manifest_canonico | `8881a9f53e8613aa...` |
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| canonico_ids | `d69ecb339f1dc57d...` |
| canonico_cand_pred | `2fd8ab0292e8b736...` |
| canonico_cand_prob | `57be29623543c261...` |

## Novedad y lo que NO se hace

- Novedad: FTRL-Proximal **softmax multiclase propio**, semantica online de una pasada, un solo juego de pesos, sobre la TF-IDF por fold. No es embedding, no es vector aprendido, no es representacion nueva, no usa datos externos.
- Se registra en artefactos NUEVOS escritos por el runner (`reports/v33_novelty_registry.csv`, fila anexada en `reports/age_complete_experiment_history.csv`) y en el acta D-084; **no se edita ningun script ya ejecutado**.
- **Via B DESCARTADA**: B1 (grafo coseno kNN transductivo) es ILEGAL (usa features de test sin etiqueta para modificar predicciones de train; su forma legal es kNN = M09 cerrado); B2 (masking/spans) descartada por analogia con C04; B3 (cabeza multitarea) refutada por algebra con tronco lineal.

## Compliance y limites

- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin leaderboard, sin submission; un script nuevo; campeon V30 + plug-in (LB 0.63575) intacto.
- AST audit de imports Y literales (vowpal, vw, torch, transformers, gensim, fasttext, sentence-transformers, word2vec, glove, spacy-vectors) y 0 rutas de test/holdout/submission/leaderboard.
- Coste esperado ~13-16 min de CPU; RAM con tope 20 GB medido con `ru_maxrss`; calibracion de fold 0; checkpoints por fold; escritura atomica.
- Referencia de variabilidad de entrenamiento: 0.0008876.

## Enmienda pre-ejecucion 1 (2026-09-30)

- **G1**: banda de log_loss multiclase de train `[0.25, 0.60]` -> `[0.05, 1.6125835223052496]`. Motivo: la banda congelada era inalcanzable para un
  modelo sano (en D-083 el gemelo sano dio log_loss multiclase 1.3568 > 0.60) y cerraba el
  experimento por construccion. Derivacion: suelo 0.05 = detector de humo; techo
  0.9*log(6) = 1.6125835223052495 (mejorar el uniforme >= 10 % relativo).
- **A2**: `shuffle` no estaba congelado (default de sklearn True) -> `False`, para que el control
  consuma el orden congelado tal cual (el default aplicaba su propia permutacion encima).
- Sin resultados: enmienda **pre-ejecucion** (ningun fold entrenado). Resto del protocolo intacto.
