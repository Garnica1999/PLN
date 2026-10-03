# V23 P0: auditoria historica de las tecnicas de small-data NLP

Worktree `/home/carlos/Git/PLN-v23`, rama `v23`, base `ba87c68` (V22 cerrado).
Fuente: lectura de codigo, no de resumenes. `experiments.csv` tiene **242 registros** y
todo lo de abajo se verifico contra los scripts, no solo contra el registro.

## 1. La tabla que pide el encargo

| tecnica | historico | implementacion exacta | resultado | diferencia respecto V03 | realmente cerrado |
|---|---|---|---|---|---|
| FTRL / FTRL-Proximal | **0 scripts** | -- | -- | -- | **no existe** |
| online logistic regression | **0 scripts** | -- | -- | -- | **no existe** |
| Passive-Aggressive | V4-G03 | `PassiveAggressiveClassifier` en `33_class_conditional_lm_v4.py` | **0.762605**, 1/5 | -0.0146 | **si** |
| coordinate descent sparse | mucho | `liblinear` en todas partes, 113 registros | es el de V03 | es V03 | **si** |
| L1 | V21 P1-03 | `penalty="l1"`, C=4.0 | 0.783642, **3/5** | +0.0064 | **si** |
| L2 | V03 | `penalty` por defecto, C=2.0 | 0.777211 | es V03 | **si** |
| **Elastic Net** | **0 scripts** | -- | -- | -- | **no existe** |
| supervised feature reweighting | V4-B03 | log-odds, chi2, MI, en `30_supervised_weighting_v4.py` | 0.7571 / 0.6622 / 0.7378 | **-0.020 a -0.115** | **si, y en contra** |
| delta-IDF | **0 scripts** | -- | -- | -- | **no existe** |
| BM25 | V4-C | `esquema_bm25` en `30_supervised_weighting_v4.py`, 4 configs | 0.774974-0.775532 | **-0.0017 a -0.0022** | **si** |
| log-entropy weighting | **0 scripts** | -- | -- | -- | **no existe** |
| chi-square weighting | V4-C | idem, `chi2_por_etiqueta` | 0.662192 | **-0.1150** | **si** |
| mutual-information weighting | V4-C | idem, `mi_por_etiqueta` | 0.737764 | -0.0394 | **si** |
| information gain | **0 scripts** | -- | -- | -- | **no existe** |
| Rocchio | **0 scripts** | -- | -- | -- | **no existe** |
| linear discriminant sparse | V4-G02, 25 registros | `RidgeClassifier` (no LDA) | 0.769810, 2/5 | -0.0074 | **si** |
| calibrated linear | `v7_00_gap_audit.py` | **solo como patron de busqueda**, no implementado | -- | -- | **no se ejecuto** |
| character-only | 39 registros | char_wb(3,5) + LR, MOD-04 | 0.787 | -0.006 | **si** |
| word-only | word(1,2) + LR | parte de la union de V03 | -- | -- | **si, es subconjunto** |
| word+char interaction | 84 registros | `hstack` word(1,2)+char_wb(3,5) | 0.777211 | **es V03** | **si, es V03** |
| supervised feature selection | `v8_00_gap_audit.py` | **solo patron de busqueda** | -- | -- | **no se ejecuto** |
| low-rank sparse classifier | V4-G, DL-01 | `TruncatedSVD(300)` + SVC RBF / MLP | **0.790410**, 5/5 | **+0.0132** | **NO auditado** |
| NB-SVM / log-count ratio | V21 P1-03 | canonico de Wang y Manning, 6 variantes | 0.763839 | -0.0134 | **si** |
| deep learning desde cero | DL-01..03 | MLP 0.7949, CharCNN 0.7410, BiLSTM 0.7364 | -- | mixto | **si** |
| embedding denso | V5/V5.1/V21/V22 | 14 configs, 5 corpora, + ABTT | maximo 0.783642 | +0.0064 | **si** |
| stacking / meta | 15 registros | Ridge/LogReg sobre probabilidades OOF | 0.7678 | -0.0094 | **si** |
| ensembles | ENS-01..04, S04 | mediana y rank-average | 0.7751 LB | -0.0234 LB | **si** |

**Resumen de la auditoria**: de las 22 tecnicas del encargo, **cinco no existen en el
proyecto** (FTRL, online logistic, Elastic Net, delta-IDF, log-entropy, information gain,
Rocchio), tres solo se mencionaron en auditorias sin ejecutarse (calibrado, feature
selection, mutual information como filtro), y el resto esta cerrado **con resultados
medidos**, no con suposiciones.

## 2. Hallazgo 1: toda la reponderacion supervisada generica ya esta cerrada, y en contra

`30_supervised_weighting_v4.py` implementa la familia completa de la Candidata B, con
protocolo fold-local y bootstrap pareado de 2.000 remuestreos. 15 variantes:

| variante | corrected_cv | pliegues |
|---|---:|---:|
| geo:v03_sublinear (**control**) | **0.777211** | -- |
| geo:bm25 (4 configuraciones) | 0.774974-0.775532 | 1-2/5 |
| geo:binario | 0.773802 | 1/5 |
| peso:log_odds (C de 2 a 200) | 0.743496-0.757139 | 0/5 |
| peso:mutual_information | 0.737764 | 0/5 |
| peso:chi2 | 0.662192 | 0/5 |
| clase:barrio_anidado_por_etiqueta | 0.781189 | 3/5 |

**Ninguna reponderacion generica supera al control**, y BM25, que se supone mas robusto
para recuperacion de informacion, pierde en las cuatro configuraciones. El unico elemento
de la familia que supera al control de forma amplia es el `class_weight` por etiqueta
(+0.0040), que no es una reponderacion de features sino del coste de la funcion de perdida.

La explicacion es coherente con lo que V5 ya diagnostico: con 1.483 ejemplos y 27.431
features, la varianza ya domina, y anadir una segunda capa de Supervision sobre las
features no hace mas que duplicar informacion que el clasificador lineal ya consume.

## 3. Hallazgo 2: el mayor DeltaOOF de la historia existe, y nunca llego al leaderboard

`33_class_conditional_lm_v4.py` probo modelos no lineales sobre una compresion SVD de la
union de V03. Resultado, con umbrales congelados de V03:

| variante | corrected_cv | delta V03 | pliegues | IC95 |
|---|---:|---:|---:|---|
| **svd:rbf_svc** | **0.790410** | **+0.013199** | **5/5** | [-0.001124, +0.014835] |
| alt:ridge | 0.769810 | -0.0074 | 2/5 | -- |
| alt:passive_aggressive | 0.762605 | -0.0146 | 1/5 | -- |
| alt:sgd | 0.745404 | -0.0318 | 0/5 | -- |
| svd:hist_gb | 0.729722 | -0.0475 | 0/5 | -- |
| svd:extra_trees | 0.703331 | -0.0739 | 0/5 | -- |

**0.790410 es +0.0132 sobre V03 con 5 de 5 pliegues**, el mayor delta OOF medido en el
proyecto, por encima de L1 (+0.0064) y de la mejor fusion de embeddings (+0.0057).

**Ninguna submission de las 21 pertenece a esta familia.** El registro lo anota como
"Candidato D (svd300_rbf) ... con evidencia fragile (IC95 del delta toca 0)" y la linea
murio ahi.

### 3.1 No hay leakage: verificado

Es la comprobacion que V20 y V21/V22 hicieron siempre, y aqui sale limpia:

- `TruncatedSVD` se ajusta **dentro** del pliegue: `svd.fit_transform(izquierda)` y
  `svd.transform(derecha)`, linea 434-436.
- El `SVC` se ajusta en `reducido_entrenar` y predice en `reducido_validacion`, linea 448.
- No hay ninguna estructura de peso ni de seleccion que dependa de datos fuera del pliegue.

### 3.2 Si hay una debilidadKnown, y es corregible

`SVC` no expone `predict_proba` salvo con `probability=True`, asi que el script aplica **una
sigmoidal sobre `decision_function`** en vez de Platt scaling. Eso es una calibracion
ad-hoc: deforma las probabilidades y por tanto desplaza el punto de operacion de los
umbrales congelados de V03. Es la unica objeccion seria a la cifra, y es exactamente el
punto donde una reimplementacion puede mejorar.

## 4. Hallazgo 3: el segundo mejor DeltaOOF tiene leakage, y por eso no es candidato

`combo:downweight_violacion_x_clase` (V4-C05) da **0.785416, +0.008205, 5/5, IC95
[-0.000225, +0.011688]**, y el propio informe lo marca `cumple_criterio_D016: True` y
`decision_familia: ADOPTAR`. Es el unico resultado del proyecto que cumple el criterio de
adopcion.

**Pero no es una candidata valida.** `experiments/30_supervised_weighting_v4.py` lineas
594-604 carga `dureza["error_score"]` de `features/hardness_v4.npz`, que `28_hardness_v4.py`
linea 181 calcula como `np.clip(-margen_favor, 0, None)`, es decir, **la magnitud de la
violacion del umbral de la prediccion OOF de V03**, sobre **todo el desarrollo y una sola
vez, antes de los pliegues**. Ese array se usa despues como `sample_weight` en el
entrenamiento de cada pliegue.

La consecuencia tecnica: el peso de una fila de entrenamiento depende de una prediccion
OOF que fue producida por un modelo que si contenia filas de la validacion de ese mismo
pliegue. Es informacion de la validacion entrando en el entrenamiento por la puerta del
`sample_weight`. El script ya habia corregido un sesgo de seleccion parecido (los
multiplicadores de `class_weight` se elegian sobre el mismo corrected_cv que despues se
reportaba) y en el mismo comentario, pero **el `sample_weight` de dureza se quedo fuera de
esa correccion**.

Por eso su +0.0082 no es una medida valida de nada, y por eso no debe mezclarse con `rbf_svc`
en ninguna tabla comparativa. Se deja registrado, no se ejecuta.

## 5. Conclusion de P0

1. La Candidata B (supervised feature reweighting) esta **cerrada y en contra**: 15
   variantes, ninguna gana al `sublinear_tf` de V03. No se ejecuta.
2. La Candidata C (online / small-data) esta **cerrada en contra**: Passive-Aggressive
   0.7626 y SGD 0.7454, ambos muy por debajo. El `liblinear` de V03 es mejor.
3. FTRL, online logistic, Elastic Net, Rocchio, delta-IDF y log-entropy **no existen en el
   proyecto**, pero los dos ultimos carecen de hipotesis mecanica en este problema, y FTRL
   e online logistic son variaciones de un mismo estimador que liblinear ya aproxima bien.
4. **La unica via con diferencia algoritmica real, no cerrada y con mejor senal del
   proyecto es `svd:rbf_svc`**: un kernel RBF sobre la compresion densa de la union de
   V03. Es genuinamente distinto de V03 (lineal sobre sparse) y de todo lo demas
   (no lineal sobre denso, con regularizacion implicita por kernel en vez de por norma).
5. V23 debe auditar esa candidata en terminos de perfil de celdas, que es la cifra que en
   V21 demostro que decide la transferencia, y intentar corregir su unica debilidad
   conocida (la sigmoidal en lugar de Platt).
