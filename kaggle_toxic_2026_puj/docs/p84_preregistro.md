# P84: preregistro de la ejecucion del plan de exploracion

Fecha: 2026-10-02.
Referencia: /home/carlos/.opencode/plan/kaggle_toxic_exploracion_modelos_f1_080.md
Estado: Gate 0 completado. Sin submissions. Sin uso de test ni de holdout.

## 1. Hashes congelados (Gate 0)

| Artefacto | sha256 |
|---|---|
| data/raw/toxic_train.csv | 3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f |
| features/oof_v03_final.npz | 559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab |
| data/processed/grupos_near_duplicados.csv | afabf9ae1c295ea8f203120bf28578cb4bdb4e8f16b89178d820baeaac62e622 |
| data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt | 00b839da6fbc377ee5375acc9a236e831e46ecf92cc544188b81aef1a826e871 |
| tools/fastText/build/fasttext | 59c51cc53802bcad9e7832145f6525a281d22d42e7c476bfbac545c0c392137f |

Nota fastText: el binario no existia en este worktree; se reconstruyo el 2026-10-02 desde
`tools/fastText` con el Makefile oficial (mismo arbol de fuente que EBD-00 declaro como
commit 1142dc4c4ecb, MIT). No se descargo ningun modelo ni vector.

## 2. Verificaciones de particiones y control (Gate 0)

- `read_train`: 1731 filas, 5 columnas.
- `grupos_fusionados`: 1645 grupos unicos.
- `holdout_v2`: dev 1483, holdout 248, interseccion 0.
- `pliegues_corregidos` sobre dev: tamanos [297, 297, 296, 296, 297], sin grupos
  compartidos entre entrenamiento y validacion en ninguno de los 5 pliegues.
- Identidad V03: `features/oof_v03_final.npz`, 1483 filas, pliegues 1..5, umbrales
  [0.44, 0.51, 0.46]; `corrected_cv` guardado y recalculado desde `proba`/`pred`:
  0.7772114525165721 (coincidencia exacta).
- `f1_pliegues` de V03: [0.79601011, 0.79403395, 0.77886003, 0.73859801, 0.77905650].
- Fuente del npz: `features/v2_oof_base_corrected.npz`, modelo `M2_union_lr_c2`.
- Entorno: /home/carlos/venv, Python 3.14.7, sklearn 1.8.0, torch 2.14.0+rocm7.2 con
  GPU disponible; `tools/verificar_caracteres.py` en 0 errores antes de esta fase.

## 3. Convenciones de comparacion (fijadas antes de ejecutar)

- Referencia: probabilidades OOF de V03 del npz y umbrales congelados 0.44/0.51/0.46.
- Delta por pliegue y delta medio; numero de pliegues mejorados.
- IC95 bootstrap pareado (10.000 remuestreos, semilla 20260926) sobre filas del desarrollo.
- Rescate/dano: celdas (fila, etiqueta) que pasan de 0->1 (rescate) y de 1->0 (dano)
  frente a la prediccion de V03; tambien recuento neto.
- `ensemble_gain` (convencion P6): F1 de la media (proba_candidato + proba_v03)/2 a
  umbral 0.5 menos 0.771778476256122 (M2 a 0.5), con IC bootstrap pareado.
- Tasa de positivos predichos por etiqueta.
- Brecha train-validacion cuando el modelo tenga etapa de ajuste.

## 4. Registro de experimentos de la fase

| ID | Script | Pregunta | Gate | Estado |
|---|---|---|---|---|
| P84-01 | experiments/p84_01_dl01_corregido.py | DL-01 corregido: pool 1483 y epoca elegida en particion interna | +0.005, 4/5 pliegues, IC95 que excluya 0 | preregistrado |
| P84-02 | experiments/p84_02_fasttext_supervisado.py | fastText supervisado desde cero, fold-local, vs V03 | +0.005, 4/5, IC95 > 0; o ensemble_gain >= 0.0015 con IC > 0 | preregistrado |
| P84-03 | experiments/p84_03_sobremuestreo_aleatorio.py | sobremuestreo aleatorio fold-local frente a class_weight | +0.005, 4/5, IC95 > 0 | preregistrado |
| P84-04 | experiments/p84_04_sgns_corregido_anclado.py | SGNS suma vs 1/K vs aleatorio, con ancla externa | supera ambos controles y +0.005, 4/5, IC95 > 0; o ensemble_gain >= 0.0015 | preregistrado |
| P84-05 | experiments/p84_05_em_shrinkage_simulado.py | EM con shrinkage en simulacion OOF | +0.002, 4/5, IC95 > 0 y tasa +-0.005 de V03 | preregistrado |

## 5. Decisiones de politica para esta fase

- Overfitting en redes: early stopping estricto con seleccion de epoca en particion
  interna del entrenamiento exterior, como fija el plan aprobado. No se sobreentrena.
- Sin Optuna ni rejillas grandes; el ajuste interno solo se usa donde el diseno lo exige.
- El holdout y el test no se tocan; no se generan submissions sin autorizacion explicita.
- No se repiten familias cerradas: transformers/preentrenados/labels externas/
  pseudoetiquetado; focal y otras perdidas; ComplementNB/NB-SVM/BM25/term weighting;
  cadenas/powerset/stacking; calibracion/umbrales; SGD/hashing/VW; rejillas de corpus,
  ventana y pooling de embeddings no supervisados.
- Grafo de codigo y Graphiti: la generacion del grafo (2026-09-30) es anterior a las
  correcciones del 2026-10-02 y los valores recientes se leen de la fuente; Graphiti no
  contiene memoria de este proyecto.

## 6. Enmienda 1 (EXP-01, antes de codificar)

Aprobada por dl-specialist (APROBADO CON CAMBIOS). Cambios fijados antes de escribir
`experiments/p84_01_dl01_corregido.py`:

- E_j = `mejor_epoca` del mejor peso restaurado en la particion interna (no las epocas
  ejecutadas); E* = mediana de (E_0, E_1, E_2) por semilla.
- `partir_interna` controla prevalencia marginal por etiqueta (no la combinacion
  HSxTRxAG); se registran los intentos y se avisa si la tolerancia 0.07 no se cumple.
- `pos_weight = n_neg / max(n_pos, 1.0)` en float32 y device, con etiquetas del pliegue.
- AMP False con evidencia medida (BiLSTM 3.42 vs 7.48 s; CharCNN 0.41 vs 1.73 s), sin
  autocast ni GradScaler.
- Determinismo: `fijar_semillas` por split y refit; pre-vuelo de doble pasada (semilla 42,
  pliegue 1) con tolerancia max|delta p| <= 1e-6; fallback a CPU si falla; guardia de
  evaluacion doble con `torch.equal`.
- Contador: una unica pasada forward de `outer_val` por (semilla, pliegue); las tres
  lecturas de umbral reutilizan la misma matriz de probabilidades.
- D-057: brecha = media por pliegue y semilla de (F1 train a 0.5 - F1 validacion a 0.5)
  en el mismo harness; capacidad = parametros + dimension densa. Umbral efectivo de
  promocion +0.010 para candidatos que aumentan capacidad.
- Checkpoints finales por (semilla, pliegue) en `modelos/p84/` y publicacion de la
  dispersion de E_j.
- Semilla primaria: 42 (comparable con DL-01-SEEDS); estabilidad con 43 y 44.
- IC bootstrap por filas (no por grupos) queda declarado como posiblemente optimista; no
  se cambia sin una enmienda nueva.

## 7. Enmienda 2 (EXP-02, antes de ejecutar la corrida completa)

Aprobada por nlp-specialist (APROBADO CON CAMBIOS) y revisada por ml-reviewer
(APROBADO CON CAMBIOS MENORES, aplicados). Parametros fijados antes de la corrida:

- Script: `experiments/p84_02_fasttext_supervisado.py`, sha256 `bd06478f94a4b4f83703cf15d83d841362d4ecc4f6a172bfc0ead8c127802d55`.
- Tres clasificadores binarios fastText por pliegue (HS, TR, AG), entrenados solo con
  `outer_train`; archivo `__label__POS`/`__label__NEG`; texto con `lower()` y nada mas (no
  se reutiliza `normalizar_emb_normal`).
- Flags: `-dim 100 -lr 0.1 -minCount 2 -bucket 2000000 -loss softmax -wordNgrams 2
  -thread 1 -seed 20260926`; brazo principal `-minn 3 -maxn 6`; control `-minn 0 -maxn 0`.
- Epocas {5, 10, 20} elegidas en 3 splits internos (`partir_interna`, semillas
  20261101+100*k+7*j); E* = mediana; reentrenamiento sobre `outer_train` con E* y una
  unica prediccion de `outer_val`.
- `predict-prob` con `k=2`, parseando `__label__POS` por nombre.
- Umbrales: anidado (primario) elegido en `inner_val`, 0.20-0.80 paso 0.01; 0.5 y
  0.44/0.51/0.46 como controles.
- Gate standalone con `delta_global` (consistente con P84-01) y `delta_medio_pliegues`
  como diagnostico; complementariedad por `ensemble_gain >= 0.0015` con IC > 0.
- Expectativa pre-registrada: correlacion alta con los char n-grams de V03; posibilidad
  baja de mejora standalone; la via realista es la complementariedad. Sin rejilla nueva si
  falla.
- Determinismo: doble corrida identica (tolerancia 0) integrada y publicada.

## 8. Enmienda 3 (EXP-03, antes de ejecutar la corrida completa)

Aprobada por ml-reviewer (APROBADO CON CAMBIOS MENORES, aplicados). La spec de EXP-03
corrige el diseno original: con liblinear L2, la duplicacion de filas es reasignacion de
pesos por instancia; ROS cae dentro de familias cerradas (class_weight V4-C04 y reweighting
V4-B03). Se ejecuta como micro-verificacion de equivalencia, NO PROMOVIBLE.

- Script: `experiments/p84_03_ros_equivalencia.py`, sha256 `13e500f90702328495a57d09f8c10f79d49cd3a7efcb8ca0a9e172e27f84e481`.
- Brazos por pliegue: A control V03; B1 ROS r=1 con masa total n y C=2 (equivalencia
  esperada con A salvo redondeo entero); B2 ROS r=0.5 con masa n y C=2; B3 ROS r=1 sin
  normalizar con C=2 y `C_efectivo = C*n_neg*(1+r)/n` reportado.
- `n_pos' = round(r*n_neg)`; copias deterministas con semilla 20260926; vectorizador/IDF
  ajustados en `outer_train` antes de duplicar; validacion nunca duplicada.
- Equivalencia B1-A: `max|proba| <= 1e-3` y `|delta F1| <= 0.001`; si no, REFUTACION
  PARCIAL con causa cuantificada (redondeo entero o solver), nunca promocion. El humo de
  pliegue 1 ya mostro refutacion por redondeo (max|delta proba| 5.97e-2; delta F1 -0.0076;
  copias enteras 1/2 por positivo con desviacion de peso 0.29-0.56 frente a balanced).
- Decision pre-declarada: "ROS cerrado por equivalencia con reweighting y class_weight;
  no promovible (CLOSED BY EQUIVALENCE)". Ningun brazo va a Gate 2 ni a EXP-06.

## 9. Enmienda 4 (EXP-04, antes de ejecutar la corrida completa)

Aprobada por nlp-specialist (APROBADO CON CAMBIOS) y ml-reviewer (APROBADO CON CAMBIOS
MENORES, aplicados: umbral anidado solo sobre filas cubiertas; metricas de lectura
completas; firma de reanudacion v2 con sha256; ancla con reintentos acotados y verificacion
de finitud).

- Script: `experiments/p84_04_sgns_corregido_anclado.py`, sha256 `26ca470f0ee00c4b0d2d16039182fd52b1bdb04a402941859fc5b99ac66e5975`.
- Corpus: `data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt`
  (sha 00b839da...), EXTERNAL-ONLY.
- Ancla: identidad algebraica (`N_suma = K*N_media`; `loss_suma = K*loss_media + (K-1)*P`),
  gradiente por diferencias finitas, y binario oficial `skipgram -minn 0 -maxn 0 -dim 32
  -neg 10 -epoch 5 -thread 1 -seed 20260926 -minCount 1` con hasta 5 intentos por NaN del
  binario y verificacion `np.isfinite`; pureza nuestra >= oficial - 0.05; si falla,
  ANCLA NO LOGRADA (bloquea promocion).
- Brazos S (suma), M (media 1/K), R (aleatoria congelada); semillas {20261016, 20261017,
  20261018}; dim 200, ventana 5 dinamica, min_count 3, negativos 10, subsampling 1e-4,
  5 epocas, lote 256, lr 0.025*lote con decaimiento a 1e-3; flujo de datos identico entre
  S y M (hashes por epoca).
- Downstream: mean pooling OOV-excluido, StandardScaler en `outer_train`, LR C=2 balanced
  liblinear rs=42; lecturas congelada (primaria) y anidada (secundaria, solo filas
  cubiertas); S vs M, S vs R, S vs V03; `ensemble_gain`.
- Gate: S>M y S>R (delta>0, >=4/5, IC>0) y (S vs V03 +0.005/4/5/IC>0 o `ensemble_gain`
  >=0.0015 con IC>0). Se publica `ensemble_gain` de S como diagnostico aunque S no supere a M.
- Reanudacion: firma v2 (version + sha256 del script); parciales por (semilla, brazo).

## 10. Enmienda 5 (EXP-05, antes de ejecutar la corrida completa)

Aprobada por ml-reviewer (APROBADO CON CAMBIOS MENORES, aplicados: semilla global publicada;
precision/recall por etiqueta anadidas).

- Script: `experiments/p84_05_em_shrinkage_simulado.py`, sha256 `9c6d62c7a62917823586f7366f5f1dc1cbf53549a9bc120b18923868f7694251`.
- Simulacion analitica sobre el OOF de V03 (`features/oof_v03_final.npz`): pesos de
  importancia `w_i = rho` si `y_i=1`; F1 ponderada por etiqueta; rho en {0.85, 0.90, 0.94,
  1.00, 1.05}; supuesto p(x|y) invariable declarado.
- EM con shrinkage: `rho_usado = exp(s*log rho_hat)`, `s = n_eff/(n_eff+m)`, m {0,25,50,100};
  rejilla t 0.20-0.80; criterio minimax sobre las 5 rho; desempate menor m y t mas cercano
  al congelado. El EM de aplicacion se estima en las filas del pliegue evaluado (adaptacion
  sin etiquetas del destino), declarado; la seleccion (m,t) es LOFO y no ve el pliegue
  evaluado.
- Controles: descomposicion vs solo-umbral; sanidad rho=1; guardarrail de tasa +-0.005.
- Gate: delta medio >= +0.002, >=4/5 pliegues LOFO positivos, IC95 bootstrap por filas con
  pesos (10.000, semilla 20260926) que excluya cero, guardarrail cumplido y componente EM
  positiva; si falla, cierre de la familia EM/threshold.
- Sin test ni holdout; sin submissions.

## 11. Enmienda 3bis (EXP-03, correccion del conteo de rescate/dano antes de reejecutar)

La auditoria adversarial final detecto que `p84_03_ros_equivalencia.py` usaba `pred` en
lugar de `pred_ref` al contar celdas frente al brazo de referencia (linea ~929), forzando
rescate=dano=0. Corregido por ml-reviewer: la referencia correcta se usa en todos los call
sites; el texto "predicciones identicas" se calcula de los datos.

- Script: `experiments/p84_03_ros_equivalencia.py`, sha256 `2a83741691a4d7064b14bd74999481ad339cbb03739d735a49e0ad8865a31e19` (sustituye al de la
  Enmienda 3).
- Recuentos corregidos (recomputados desde el npz de la corrida previa): B1 61 celdas
  (32 rescates/29 danos), B2 446 (207/239), B3 63 (34/29) sobre 4449.
- Se reejecuta la corrida completa para regenerar informe y fila del CSV; la decision
  CLOSED BY EQUIVALENCE no cambia.
