# Pre-registro congelado - ciclo final: familias restantes (P3 / P4 / P5)

**Fecha**: 2026-09-30. **Estado**: CONGELADO antes de ejecutar (commit y hashes en
`experiments/age_linguistic_11_remaining_preregistro.json`).

## Contexto y decision del usuario

Las tres familias quedaron sin ejecutar en los ciclos anteriores con estas
clasificaciones:

| Familia | Clasificacion previa (auditorias por operacion) |
|---|---|
| P3_lemma_POS | PARTIALLY_COVERED (auditoria del ciclo P6: lemas solo como agregados en v14a; incremento lexical cubierto; incremento POS cerrado en P2) |
| P4_morfologia | PARTIALLY_COVERED (ratios en v14a; sufijos redundantes con char_wb medido en V13) |
| P5_dependencias | GENUINELY_NEW por operacion; sin justificacion independiente tras P7 (revision unica del gap) |

**Decision del usuario (2026-09-30)**: ejecutar las experimentaciones restantes de la
serie, con el mismo protocolo congelado de P6/P7, para cerrar el expediente con evidencia
propia de cada familia. Se registra como override explicito de la recomendacion de no
ejecutar (se documenta en D-080).

## Materiales congelados (SHA256)

| Artefacto | SHA256 (16) |
|---|---|
| anotacion_md_parquet | `dacbc3055ccde0d9...` |
| anotacion_lg_parquet | `81287d541c082ea8...` |
| dep_edges_md_parquet | `3983212647c7f48f...` |
| dep_edges_lg_parquet | `ed32f5d2a91d360c...` |
| corpus_train | `39c6c8270366fe15...` |
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| manifest_plugin | `8d635e6a546b5a0e...` |

Pesos congelados del plug-in: `[1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]`
Baseline primario recomputado: **0.6159732** (brazo A = ablacion sin canal, D-076 cond.4).

## Documentos con secuencia vacia por canal (congelado; uniforme 1/6 declarado)

| Anotador | R3 | R4 (tras excluir '_') | R5 |
|---|---|---|---|
| md | 104 | 4343 | 104 |
| lg | 104 | 4901 | 104 |

## Representaciones (congeladas, sin grid)

| ID | Definicion | Vectorizador |
|---|---|---|
| R3_lemma_POS | tokens lemma_POS de la columna congelada lemma_pos_seq (querer_VERB) | TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=5, sublinear_tf=True, lowercase=True, token_pattern=r'(?u)\S+') |
| R4_morfologia | tokens de morph_seq con los vacios '_' EXCLUIDOS (bundles tipo Tense=Past|Person=1, en orden; los bigramas son adyacencia en la secuencia FILTRADA) | identico a R3 |
| R5_dependencias | tokens 'dep->POS_cabeza' por token (nsubj->VERB, det->NOUN; raiz 'ROOT->POS' como codificacion propia) del artefacto derivado determinista dep_edges (mismo modelo congelado; validacion 0/0 contra lo almacenado) | identico a R3 |

Aprendiz (los tres): LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', tol=1e-3), ajustado SOLO en el train de cada fold; OOF en
validacion. Orden de ejecucion congelado: **R3 -> R4 -> R5**.

## Integracion y control (pre-registrados)

- Integracion: combinador logistico anidado sobre [p_V30 || p_canal] L2 por fila (12 features), decision argmax(w_plug-in * p_combinada); identico a P6/P7.
- Control: secuencias del canal permutadas DENTRO DE CADA FOLD (semilla 20261031; mismo mapeo md/lg); pipeline completo; NO comparable numericamente con los barajados globales de P2/P6 (declarado).
- Filas con secuencia vacia: probabilidad uniforme 1/6 (convencion P2/P6), declarado.
- Anotadores: md y lg (mandato D-076).

## Gates (identicos a la serie; por familia y anotador)

- delta >= **+0.0015** sobre el baseline primario;
- folds positivos >= **4/5**;
- delta real > delta barajado (mismo canal y anotador);
- IC95 bootstrap pareado (2.000, semilla 20261010) sin cero;
- sin leakage (canal OOF por fold; combinador solo con train).

Veredicto por familia: PROMISING si algun anotador pasa los cinco; MARGINAL si positivo
por debajo del gate; CLOSED si <=0 o el control empata/gana. Si algo pasara: segunda
particion congelada (semilla 987654) y jamas submission sin autorizacion (D-076 cond.5).

## Revisiones de agentes (puntos adoptados)

| Agente | Punto | Decision |
|---|---|---|
| Kaggle Specialist | re-analisis legal bajo D-076; registrar artefacto hijo en el registro; ROOT->POS como codificacion propia | adoptado |
| Kaggle Specialist | declarar barajado intra-fold y su no-comparabilidad con P2/P6 | adoptado |
| Revisor ML | excluir '_' en R4; contabilizar vacios (uniformes) | adoptado |
| Revisor ML | R5: re-validar dep_seq/n_tokens en el runner; trampas declaradas | adoptado |
| Revisor ML | R3 vs P7 limpio; vs P2 no comparable (protocolos distintos) | registro |
| Revisor ML | asserts de orden de clases, pesos, SHA de particion y parquet | adoptado |
| Revisor ML | alternativa de rasgos atomicos en R4 | NO adoptada: otra hipotesis, no un brazo extra |

## Que NO se hara

- No hay grids, ni variantes de n-gramas, ni escalas, ni combinadores, ni semillas extra.
- No se toca el test, ni el holdout, ni el leaderboard. Sin submission automatica.
- No se reabren P2, P6 ni P7. El campeon (V30 + plug-in, LB 0.63575) permanece intacto.

## Interpretacion pre-registrada

- Un delta plano con barajado igual o mejor es un cierre valido de la familia, no un
  fracaso: significa que el canal no aporta atribucion positiva bajo el mecanismo.
- Si una familia falla: 'el canal X no aporta mejora al campeon bajo el mecanismo
  evaluado' (no una afirmacion sobre toda la informacion linguistica).
