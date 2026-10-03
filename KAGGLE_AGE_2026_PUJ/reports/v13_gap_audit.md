# V13 · P0 — Gap audit

**Entrega previa a cualquier entrenamiento.** No se ha entrenado ningún modelo: solo se leen artefactos, se mapea la literatura contra las features reales de V12 y se decide qué merece ejecutarse.

**Regla de falsación**: cada variable de la literatura se mapea contra nombres de columna leídos de `features/`. Si un nombre declarado no existe, el script aborta. Por eso "nunca probado" es una afirmación verificada, no una opinión.

## 1. Inventario real del historial

**72 experimentos registrados** en `experiments.csv`, agrupados por bloque:

| Bloque | Etapa | Experimentos | IDs |
|---|---|---|---|
| 2 | B - baseline | 4 | E-001, B0, B1, B5 |
| 2 | B - representacion | 14 | E-002a, E-002b, E-002c, E-003a... |
| 3 | D - modelos clasicos | 4 | E-009, E-010, E-011, E-012 |
| 3 | D-F | 8 | E-014, E-015, E-016, E-017... |
| 4 | Fase avanzada A | 1 | ADV-01 |
| 4 | Fase avanzada B | 1 | ADV-02 |
| 4 | Fase avanzada C | 1 | ADV-03 |
| 4 | Fase avanzada D | 1 | ADV-04 |
| 4 | Fase avanzada E | 1 | ADV-05 |
| 4 | Fase avanzada F | 1 | ADV-06 |
| 4 | Fase avanzada G | 1 | ADV-07 |
| 5 | Fase avanzada v2 | 3 | PH1-L1, PH2-L6, PH3-V11 |
| 5 | Fase 10 vía A: retrieval | 1 | PH10-A |
| 5 | Fase 10 vía B: POS spaCy | 1 | PH10-B |
| 5 | Fase 11 ruido de etiqueta | 1 | PH11-RUIDO |
| 5 | Fase residual A | 1 | PH12-AUD |
| 5 | Fase residual B | 1 | PH13-ERR |
| 5 | Fase residual C | 1 | PH14-CONS |
| 5 | Fase residual D | 1 | PH15-CALIB |
| 5 | Fase residual D5 | 1 | PH15B-SUBSETS |
| 5 | Fase residual G | 1 | PH16-ORD-GREEDY |
| 5 | Fase residual F | 1 | PH18-LIMPIEZA |
| 5 | Fase residual H | 1 | PH19-HOLDOUT-VISTA |
| 5 | Fase residual §13 multivista | 1 | PH21-VISTA |
| 5 | Fase residual §13 refuerzo | 1 | PH22-VISTA2 |
| 5 | Fase residual I (integridad) | 2 | PH24-DIAG, PH26-ESTABILIDAD |
| 5 | Fase residual I/J | 1 | PH25-FRESCA |
| 5 | Fase residual I (cierre) | 1 | PH27-V12REPRO |
| 5 | Capa de decisión (preparación) | 1 | PH30-COBERTURA |
| 5 | Capa de decisión A/E/G/F | 1 | PH31-DECISION |
| 5 | Capa de decisión C/D/H/B | 1 | PH32-CORRECCION |
| 5 | Capa de decisión I | 1 | PH33-CORRECTOR |
| 5 | Capa de decisión (cierre) | 1 | PH34-INFORME-DEC |
| 5 | L2 residuales | 1 | PH5-L2 |
| 5 | L3 contenido vs estilo | 1 | PH6-L3 |
| 5 | L7 y combinador | 1 | PH7-L7 |
| 5 | Candidato v12 | 1 | PH8-V12 |
| 5 | Fase 9 combinadores | 1 | PH9-COMB |
| 6 | P0 auditoria de cobertura | 1 | PH36-COBERTURA-LEX |
| 6 | P1 catalogo de lexicones | 1 | PH37-CATALOGO-LEX |
| 6 | P2-P5 bloques, control, redundancia y ablacion | 1 | PH38-BLOQUES-LEX |
| 6 | P6 fusion con V12 | 1 | PH39-FUSION-LEX |
| 6 | P7 analisis de error | 1 | PH40-ANALISIS-ERROR-LEX |

Experimentos cuya decisión contiene "adoptar/candidato": **8** (ADV-02, ADV-04, PH2-L6, PH7-L7, PH14-CONS, PH21-VISTA, PH19-HOLDOUT-VISTA, PH25-FRESCA).

## 2. Las 18 variables del artículo de referencia, mapeadas a V12

Fuente: Moreno-Sandoval, Pomares-Quimbaya y Alvarado-Valencia (2021), *Computational Social Networks* 8:16, doi:10.1186/s40649-021-00097-w. Es el antecedente metodológico directo de esta competencia (corpus PAN@CLEF, perfiles de autor, variable demográfica).

| Var | Descripción | Estado en V12 | Columnas que la satisfacen |
|---|---|---|---|
| V1 | Avg_character (media de caracteres por palabra) | **EXISTE** | `avg_word_len`, `wlen_mean` |
| V2 | Kurtosis_character (curtosis de la longitud de palabra) | **EXISTE** | `wlen_kurtosis` |
| V3 | Lexical_diversity | **EXISTE** | `ttr`, `cttr`, `hapax_ratio`, `root_ttr`, `maas`, `yule_k`, `simpson_d`, `char_entropy` |
| V4 | Label_word (media de palabras por publicacion) | **EXISTE** | `n_words` |
| V5 | Kurtosis_avg_character | **EXISTE** | `wlen_cv`, `wlen_std` |
| V6 | Kurtosis_label_word | **PARCIAL** | `std_word_len` |
| V7 | Skew_avg_character (asimetria de la longitud) | **EXISTE** | `wlen_skew` |
| V8 | Skew_label_word | **PARCIAL** | `wlen_range`, `wlen_iqr` |
| V9 | Label_emoji | **INERTE** | `n_emoji`, `has_emoji` |
| V10 | Label_hashtag | **EXISTE** | `n_hashtag` |
| V11 | Label_mention | **EXISTE** | `n_mention`, `starts_mention` |
| V12 | URL (informacion complementaria) | **EXISTE** | `n_url` |
| V13 | Label_retweets | **EXISTE** | `is_rt` |
| V14 | Person_1_singular (yo) | **AUSENTE** | — |
| V15 | Person_2_singular (tu) | **AUSENTE** | — |
| V16 | Person_3_singular (el/ella) | **AUSENTE** | — |
| V17 | Person_1_plural (nosotros) | **AUSENTE** | — |
| V18 | Person_3_plural (ellos/ellas) | **AUSENTE** | — |

**Verificación**: 18 de 18 mapeos verificados contra los archivos reales; 14 miembros del ensamble construyendo los 78 scores.

**Conclusión del mapeo**: de las 18 variables, **14 ya existen** en V12 (o son inertes por el corpus sin emojis) y **5 están ausentes**: las cinco de pronombres de persona. El artículo declara que la 3ª persona singular es la variable más predictiva de su modelo.

## 3. Las 7 familias de estilo del curso

| Familia | Estado en V12 | Qué falta o por qué se cierra | Pista |
|---|---|---|---|
| Palabras funcionales | **PARCIAL** | solo stopword_ratio y content_ratio; falta el vector de tasas | P18 |
| Sintaxis: n-gramas de POS | **CERRADO** | D-026: corpus ASCII degrada el etiquetado (-0,0032) | - |
| Arboles SFL (conjunciones por funcion) | **AUSENTE** | el curso indica que NO requiere POS; es la via no cerrada tras D-026 | P19 |
| Morfologia (prefijos, sufijos) | **CERRADO** | redundante: char_wb(3-5) ya lo cubre | - |
| Medidas de complejidad (longitud, entropias) | **EXISTE** | wlen_* y *_entropy | - |
| Palabras inestables (traduccion ida-vuelta) | **PROHIBIDO** | requiere modelos de traduccion externos (§3) | - |
| Idiosincrasias | **CERRADO** | D-039: fase de lexicones cerrada | - |

## 4. Las 10 dimensiones de la taxonomía NLP (Zhou y Zafarani, 2020)

| Dimensión | Estado en V12 | Detalle | Pista |
|---|---|---|---|
| Quantity / Complexity / Diversity | **EXISTE** | n_chars, n_words, ttr y familia de entropias | - |
| Informality (typos) | **PROHIBIDO** | requiere diccionario externo para detectar faltas | - |
| Incertidumbre (modales, certeza, tentativa) | **AUSENTE** | no hay ningun vocabulario modal | P20 |
| Subjectivity (verbos subjetivos, de reporte) | **AUSENTE** | no existe | P21 |
| Non-immediacy (pronombres, pasiva) | **PARCIAL** | solo pasiva prohibida; pronombres ausentes | P17 |
| Specificity (temporal, espacial, exclusivo) | **AUSENTE** | no existe | P21 |
| Sentiment / swear / netspeak | **CERRADO** | reabriria la linea de lexicones (§1, D-039) | - |
| Readability (Flesch, Gunning-Fog, ARI) | **AUSENTE** | no hay indice de legibilidad | P23 |

## 5. Inventario de pistas y decisión de ejecución

| ID | Pista | Mecanismo | ¿Probada antes? | IDs | Mejor resultado | Por qué sigue abierta | Complementariedad esperada | Coste | Riesgo de leakage | Prioridad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | Gap audit | inventario verificable de lo cerrado y lo nuevo | no aplica | - | - | - | define el resto | 15 min | ninguno | 0 |
| P15 | Matriz de error por miembro | diagnostico de que miembro sabe y donde | parcial | PH13, phase20 | diagnostico existente sin desglose por miembro | nunca se respondio que miembro concentra el conocimiento complementario | ALTO: 66,9% de los errores tienen algun miembro correcto | 20 min | ninguno | 1 |
| P1 | Evidencia de edad por token (P(clase|token) in-fold) | supervision dentro del train fold | no | - | - | las fases previas solo usaron P(clase|entidad) para mention/hashtag/RT, nunca por palabra | ALTO: el unico canal supervisado por palabra que no existe | 45 min | ALTO si se calcula antes de los folds (§5, §38) | 1 |
| P17 | Pronombres de persona (V14-V18) | categoria sintactica sin POS | no | - | - | V12 no tiene ninguna variable de pronombres; el articulo los declara predictivos | MEDIO-ALTO: correlaciona con edad y es ortogonal a los n-gramas de caracter | 20 min | ninguno (determinista) | 1 |
| P19 | Arboles SFL (conjunciones por funcion) | funcion gramatical sin POS | no | - | - | unico camino de sintaxis que no necesita etiquetado POS, que si esta cerrado | MEDIO: la funcion gramatical de las conjunciones varia con la edad | 25 min | ninguno (determinista) | 1 |
| P20 | Incertidumbre y modalidad | vocabulario cerrado propio | no | - | - | no existe ningun bloque de marcadores de modalidad | MEDIO: la incertidumbre verbal cambia con la edad | 20 min | ninguno (determinista) | 1 |
| P3 | Especialistas binarios de frontera | modelo binario por par adyacente | parcial | E-028a, advanced_07 | routing duro 0,0000; compuerta -0,0036 | se evaluo el enrutamiento sobre probabilidades ya entrenadas, no un especialista binario nuevo | MUY ALTO: 10.122 + 7.252 cruces de frontera | 60 min | MEDIO: el especialista debe entrenarse dentro del fold | 1 |
| P24 | Forma de la distribucion de evidencia | medidas de tendencia y dispersion sobre P1 | no | - | - | V12 solo aplica esas medidas a longitudes, no a la evidencia por token | MEDIO: distingue evidencia concentrada de evidencia difusa | 25 min | ALTO: hereda el de P1 | 2 |
| P4 | Boundary gating | compuerta entrenada sobre la proximidad a la frontera | parcial | phase33-C4, advanced_07 | correccion de frontera neta 0 | se modifico la salida de V12; no se entreno una compuerta con especialista nuevo | ALTO si P3 aporta | 40 min | MEDIO: umbral aprendido, no fijo sobre la validacion | 2 |
| P5 | Entrenamiento pairwise | objetivo pairwise en vez de post-proceso | no | - | - | Luce, Bradley-Terry y projection ordinal fueron capas de decision sobre probabilidades | ALTO: entrena el modelo con la estructura ordinal en vez de corregirla despues | 60 min | ninguno extra | 2 |
| P18 | Palabras funcionales (vector de tasas) | vocabulario cerrado propio | parcial | PH7 (stopword_ratio) | stopword_ratio y content_ratio | solo 2 agregados; falta el vector de tasas que usa authorship attribution clasico | MEDIO-ALTO: es el metodo canonico y V12 no lo tiene | 30 min | ninguno (determinista) | 2 |
| P21 | Subjetividad y especificidad | vocabulario cerrado propio | no | - | - | no existe | MEDIO | 30 min | ninguno (determinista) | 2 |
| P2 | Evidencia supervisada de hashtag y URL | P(clase|hashtag|dominio) in-fold | parcial | PH7, phase14 | menciones supervisadas +0,0016 (familia J/K/L) | se aprendio P(clase|entidad) para mention/hashtag/RT pero sin estructura de URL ni soporte | MEDIO: solo_url tiene 58-59% de error | 40 min | ALTO si se calcula antes de los folds | 3 |
| P6 | CentROIDES de clase en el espacio TF-IDF | geometria de proximidad a prototipos | no | - | - | nunca se calcularon centroides de clase en el espacio de V12 | MEDIO: la decision lineal no es la misma que la proximidad al prototipo | 25 min | ALTO: los centroides deben salir del train fold | 3 |
| P22 | Queismo y dequeismo | bigramas deterministas | no | - | - | el curso lo señala como patron de escritura; nunca se midio | BAJO-MEDIO | 20 min | ninguno (determinista) | 3 |
| P26 | PCA in-fold del bloque nuevo | reduccion de dimensionalidad antes del modelo | parcial | fase 4-11 | SVD en el meta | se aplico SVD a representaciones, no al bloque de features nuevas | BAJO (metodo) | 15 min | ALTO: el PCA debe ajustarse dentro del fold | 3 |
| P8 | Topologia de menciones | posiciones y distancias entre menciones | parcial | estructuradas pos_mention_* | posiciones first/last/mean ya existen | faltan densidad, distancia entre menciones y fraccion previa | BAJO-MEDIO | 20 min | ninguno (determinista) | 4 |
| P9 | Topologia de URL y hashtag | estructura de composicion del tweet | parcial | estructuradas pos_url_*/pos_hashtag_* | posiciones ya existen | falta la proporcion de texto ocupada por URL/hashtag/mencion | BAJO-MEDIO | 20 min | ninguno (determinista) | 4 |
| P10 | Estilometria residual | solo features ausentes, <=30 | parcial | E-001..E-028 | familia completa ya construida (35+87) | debe abrirse solo con lo que P0 demuestre ausente | BAJO | 20 min | ninguno (determinista) | 4 |
| P11 | Vista dual con menciones normalizadas | canal complementario | no | - | - | el canal de menciones existe, pero no una vista con @usuario -> @MENTION | MEDIO | 40 min | ALTO: un vectorizador para la vista nueva esta prohibido | 4 |
| P12 | Micro-lexicon derivado del target | seleccion supervisada de tokens in-fold | no | - | - | distinto de los lexicones generales cerrados en D-039: aqui la evidencia sale del train | MEDIO-ALTO | 45 min | ALTO: la seleccion debe usar OOF interno | 5 |
| P23 | Legibilidad en espanol | indices clasicos con heuristica de silabas | no | - | - | no existe | BAJO | 20 min | ninguno (determinista) | 5 |
| P16 | Tipologia de error en fronteras | slices deterministas | parcial | PH13 | segmentos | falta el cruce de slices con las dos fronteras centrales | diagnostico | 20 min | ninguno | diagnostico |
| P25 | Significancia por atributo | que variable explica que grupo de edad | no | - | - | nunca se hizo | diagnostico | 20 min | ninguno (supervisado, solo train) | diagnostico |
| P13 | Hard-boundary training | especialista sobre ejemplos difficulties del train | no | - | - | nunca se intento | CONDICIONAL: solo si P3 da senal | 90 min | MUY ALTO: exige OOF interno; con diagnostico R-2 | condicional |
| P14 | Combinacion de positivos independientes | V12 + evidencia + especialista | no | - | - | - | CONDICIONAL: solo con dos positivos independientes | 60 min | MEDIO | condicional |

## 6. Dónde está el mayor potencial de complementariedad

- **Techo de información**: 13.869% de las filas no las acierta **ningún miembro** del ensamble. Ahí ninguna capa ni especialista puede aportar: es el límite duro.
- **Agregación**: 26.7% de las filas tienen **algún miembro correcto** y el ensamble falla. Ese es el terreno de P1, P3, P5, P15 y P14.
- **Fronteras centrales**: 10.122 cruces en `18-24|25-34` y 7.252 en `25-34|35-49` (fase 34). Es el terreno de P3, P4, P13 y P16.
- **Evidencia negativa acumulada**: las tres familias de la fase de lexicon dieron +0,0001, +0,0000 y -0,0009, y su combinación -0,0018; las 33 capas de decision no adoptaron ninguna. Los netos de complementariedad fueron +52, -339, -152, +47. Es decir: **mover filas no es ganar aciertos**, y por eso P15 debe preceder a cualquier modelo nuevo.

**Conclusion de potencial**: la unica palanca con masa propia es **informacion que V12 no tiene** (P1, P17, P19, P20, P21) o **un objetivo que V12 no optimiza** (P3, P5). Todo lo que sea recolocar probabilidades ya esta demostrado como ruido.

## 7. Anidamiento de validacion requerido

| Requiere nested (in-fold) | No requiere |
|---|---|
| P1, P2, P6, P12, P13, P24, P26 | P17, P19, P20, P18, P21, P22, P23, P8, P9, P10, P11 |

Motivo: las pistas de la columna izquierda calculan alguna magnitud que depende de la etiqueta (`P(clase|token)`, centroides, seleccion de tokens). Si se calcularan antes de crear los folds, la validacion de la fold de evaluacion habria visto su propia etiqueta (§5 y §38 del brief).

## 8. Artefactos de V12 reutilizables

| Artefacto | Contenido | Uso en V13 |
|---|---|---|
| `features/oof_ensemble_v12.npy` | probabilidades OOF de V12 | referencia, rescate/daño, gating |
| `features/oof_argmax_v12.npy` y `oof_argmax_v12__*.npy` | argmax OOF de V12 y de los bloques probados en la fase de lexicones | base de la matriz de error por miembro |
| scores de los 14 miembros | 78 probabilidades por miembro | P15: qué miembro sabe y con qué confianza |
| `features/age_structured_features.parquet` | 87 features estructuradas | mapeo de redundancia |
| `features/dense_train.parquet` | 35 features densas |idem|
| `features/dup_clusters.parquet` | clústeres de near-duplicados | folds agrupados (D-003) |
| `src/decision_layer.py` | evaluador LOFO, metricas, fronteras, segmentos, rescate/daño | motor de todos los experimentos |
| `src/cv.py` | `make_folds(..., scheme="group")` | unico split permitido |

## 9. coste y riesgo agregados de la fase

- **Experimentos previstos**: 23 pistas, de las cuales 3 son diagnosticas y 2 condicionales. Dentro del presupuesto de 24 principales + 8 reservas.
- **Coste de la cabecera (P1, P17, P19, P20)**: ~2 h en total, casi todo CPU y matrices cacheadas.
- **Coste de P3/P4/P5**: ~2,5 h (reentrenar el combinador de produccion cuesta ~2,5 min por evaluacion de 5 folds).
- **Riesgo de leakage dominante**: las estadisticas supervisadas in-fold (P1, P2, P6, P12, P24, P26) y el nested de P13.
- **Riesgo de compliance**: prohibitions yaumbered en §3; se verificara de forma experimental en `reports/v13_compliance.md`, no solo declarada.

## 10. Orden final de ejecucion

1. **P0** (este informe) — sin entrenar.
2. **P15** — diagnostico: decide si P3 y P4 se dirigen a una familia concreta de modelos.
3. **P1, P17, P19, P20** — las cuatro familias de informacion nueva (nadie se solapa).
4. **P3** — especialistas de las dos fronteras centrales.
5. **P24, P4, P5** — forma de la evidencia, gating y entrenamiento pairwise.
6. **P18, P21** — vocabulario cerrado adicional.
7. **P2, P6, P22, P26**.
8. **P8/P9, P10, P11, P12, P23**.
9. **P13** solo si P3 aporta, con el diagnostico R-2 completo. **P14** solo con dos positivos independientes.

## 11. Prediccion falsable pre-registrada

El artículo de referencia concluye que sus features lingüisticas **ayudan a los modelos clasicos y perjudican a los profundos**. V12 promedia LogReg y MLP, asi que la prediccion es: **el bloque nuevo sube la rama LogReg y se diluye en la MLP**. Se medira por cabeza en cada experimento que añada features. Si no se cumple, se registra como evidencia contra la literatura y se prioriza la combinacion solo con la cabeza que se beneficia.

## 12. Referencias

1. Moreno-Sandoval, L.G., Pomares-Quimbaya, A., Alvarado-Valencia, J.A. (2021). *Celebrity profiling through linguistic analysis of digital social networks.* Computational Social Networks, 8:16. doi:10.1186/s40649-021-00097-w
2. Material del curso `Clasificacion Texto.pdf` (78 pag.): atribucion de autoria, author profiling, 7 familias de estilo, arboles SFL.
3. Zhou, L., Zafarani, R. (2020). *A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities.* ACM Computing Surveys. doi:10.1145/3395046

_Generado por `scripts/v13_p0_gap_audit.py`. Sin entrenamiento._
