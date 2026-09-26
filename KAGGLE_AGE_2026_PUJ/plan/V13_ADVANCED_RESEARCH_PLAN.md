# Plan vivo — V13 ADVANCED RESEARCH (KAGGLE_AGE_2026_PUJ)

> **Se actualiza al cerrar cada fase.** Creado: 2026-09-25 · Rama: `kaggle-age-2026-puj-v13`
> (base `kaggle-age-2026-puj-lexicons`, commit `c8a2c30`)

## Referencia congelada

| Artefacto | Valor | Estado |
|---|---|---|
| V12 corrected CV | **0,6004** ± 0,0042 (per fold 0,6007 / 0,6050 / 0,6032 / 0,6004 / 0,5928) | intocable |
| Holdout histórico | 0,6077 | **agotado**, solo referencia documental |
| `submissions/age_submission_v12_reproducible.csv` | **LB 0,62188** | **no tocar** |
| Errores OOF | 67.610 | 34,7% techo de información · 66,9% agregación |
| Fronteras centrales | `18-24\|25-34` 10.122 cruces · `25-34\|35-49` 7.252 cruces | foco de la fase |

## Pregunta científica (§40)

¿V12 falla por **A** (no posee información), **B** (la tiene y no la usa), **C** (solo existe en
una familia de modelos), **D** (la capa de decisión no puede explotarla) o **E** (hace falta un
especialista explícito)?

## Reglas fijadas (pre-registradas)

| # | Regla |
|---|---|
| R-1 | **Vocabulario cerrado válido SI y SOLO SI lo construimos nosotros desde cero**: palabras funcionales, árboles SFL, modalidad, subjetividad, especificidad, patrones gramaticales. Cada lista versionada con procedencia, SHA256, nº de entradas, sin datos externos, sin etiquetas de edad, sin embeddings |
| R-2 | **P13 se implementa con diagnóstico explícito** de underfitting, overfitting y leakage (§4 del plan) |
| R-3 | Sin push, sin PR, sin merge, sin modificar remotes, sin tocar `age_submission_v12*.csv` |
| R-4 | Test = transform/predict/submission **nada más**. Holdout intacto. Sin pseudo-labels, sin embeddings, sin vectorizador aprendido nuevo, sin corpus/labels/lexicones externos |
| R-5 | Toda estadística supervisada `P(clase\|token)` se calcula **dentro del train fold**, nunca antes de crear los folds |
| R-6 | Gate: Δ ≥ +0,003 **y** ≥4/5 folds **y** IC95 bootstrap que excluya 0 **y** no explicable por una sola fold **y** segunda partición para los 2–3 finalistas |
| R-7 | Doble métrica siempre: macro-F1 global **y** accuracy/F1 de `18-24↔25-34` y `25-34↔35-49` |
| R-8 | Control de permutación cuando la hipótesis dependa de asociación con la etiqueta |

## Predicción falsable pre-registrada (del artículo de referencia)

> Moreno-Sandoval et al. (2021) conclude que sus features lingüísticas **ayudan a los modelos
> clásicos y perjudican a los profundos**. V12 es un promedio LogReg+MLP, así que el bloque nuevo
> **debe subir la rama LogReg y diluirse en la MLP**. Se medirá por cabeza (misma comprobación DD-8
> de la fase de lexicones). Si no se cumple, se documenta como evidencia contra la literatura.

## Estado de fases

| Fase | Contenido | Entregable | Estado |
|---|---|---|---|
| **P0** | Gap audit: inventario, cerradas, nuevas, nested, reutilizables, potencial, coste, leakage, orden | `reports/v13_gap_audit.md` + `.csv` | ⏳ en curso |
| **P15** | Diagnóstico: matriz de error por miembro (**antes** de P3/P4) | `reports/v13_member_error_matrix.csv` | ⬜ **siguiente** |
| **P1** | Evidencia de edad por token, in-fold | `reports/v13_hypothesis_registry.csv` | ⬜ |
| **P17** | Pronombres de persona (V14–V18 del artículo) | idem | ⬜ |
| **P19** | Árboles SFL (sin POS) | idem | ⬜ |
| **P20** | Incertidumbre y modalidad | idem | ⬜ |
| **P3** | Especialistas binarios de frontera | idem | ⬜ |
| **P24** | Forma de la distribución de evidencia | idem | ⬜ |
| **P4** | Boundary gating | idem | ⬜ |
| **P5** | Entrenamiento pairwise | idem | ⬜ |
| **P18** | Palabras funcionales | idem | ⬜ |
| **P21** | Subjetividad + especificidad | idem | ⬜ |
| **P2** | Evidencia supervisada de hashtag/URL | idem | ⬜ |
| **P6** | CentROIDES de clase en el espacio TF-IDF | idem | ⬜ |
| **P22** | Queísmo / dequeísmo | idem | ⬜ |
| **P26** | PCA in-fold del bloque nuevo | idem | ⬜ |
| **P8/P9** | Topología de menciones / URL / hashtag | idem | ⬜ |
| **P10** | Estilometría residual (≤30, solo ausentes) | idem | ⬜ |
| **P11** | Vista dual (menciones normalizadas) | idem | ⬜ |
| **P12** | Micro-lexicón derivado del target | idem | ⬜ |
| **P23** | Legibilidad en español | idem | ⬜ |
| **P16** | Tipología de error en fronteras | idem | ⬜ |
| **P25** | Significancia por atributo (diagnóstico) | idem | ⬜ |
| **P13** | Hard-boundary training (**condicional**, con diagnóstico R-2) | idem | ⬜ |
| **P14** | Combinación (**solo** con dos positivos independientes) | idem | ⬜ |
| **Cierre** | Informes §31 + memoria | `reports/v13_final_research.md` | ⬜ |

**Presupuesto**: ≤24 experimentos principales + 8 reservas.

## Orden de ejecución (inmutable salvo evidencia)

1. **P0** gap audit — sin entrenar nada
2. **P15** diagnóstico del error por miembro (decide si P3/P4 se dirigen a una familia concreta)
3. **P1 · P17 · P19 · P20** — las cuatro familias de información nueva
4. **P3** especialistas de frontera (prioridad máxima del error)
5. **P24 · P4 · P5** — forma de la evidencia, gating, pairwise
6. **P18 · P21** — vocabulario cerrado adicional
7. **P2 · P6 · P22 · P26**
8. **P8/P9 · P10 · P11 · P12 · P23**
9. **P13** condicional · **P14** solo con dos positivos independientes

## Hallazgos de P0 (verificados, no opinados)

- **72 experimentos** inventariados por bloque y etapa.
- **18/18 mapeos verificados** contra los archivos reales de `features/`: el script aborta si un
  nombre de columna declarado no existe.
- De las **18 variables** del artículo de referencia: **14 ya existen** en V12 (2 inertes por el
  corpus sin emojis) y **5 AUSENTES**: las cinco de **pronombres de persona** (V14–V18). El
  artículo declara la 3ª persona singular como su variable más predictiva.
- De las **7 familias de estilo** del curso: 3 abiertas (funcionales, SFL, y las 5 de complejidad
  medidas), 4 cerradas (POS, morfología, palabras inestables, idiosincrasias).
- De las **10 dimensiones** de la taxonomía NLP: 3 abiertas (incertidumbre, subjetividad,
  especificidad), 5 cerradas o prohibidas, 2 ya existentes.
- **7 pistas exigen nested validation** (P1, P2, P6, P12, P13, P24, P26) porque calculan
  magnitudes dependientes de la etiqueta.
- Potencial: solo hay dos palancas con masa propia — **información que V12 no tiene** (P1, P17,
  P19, P20, P21) u **objetivo que V12 no optimiza** (P3, P5). Recolocar probabilidades ya está
  demostrado como ruido (33 capas + 3 familias de lexicon, todas con rescate ≈ daño).

## Matriz de cobertura de las técnicas del PDF (cerradas sin ejecutar)

| Técnica del PDF | Decisión | Motivo |
|---|---|---|
| % de palabras sin tilde | **INERTE** | corpus ASCII puro sin tildes (D-009): la feature sería constante |
| Errores ortográficos (hayga, lledo) | **PROHIBIDO** | requiere diccionario externo |
| Concordancia / gerundio / "donde" | **CERRADO** | requiere POS o parser (D-026) |
| Neologismos y verbos inventados | **CERRADO** | requiere diccionario; y reabre la línea de lexicones (D-039) |
| N-gramas de POS | **CERRADO** | D-026, −0,0032 |
| Morfología (prefijos/sufijos) | **CERRADO** | redundante: `char_wb(3-5)` lo cubre |
| Palabras inestables (traducción ida-vuelta) | **PROHIBIDO** | requiere modelos de traducción externos |
| Idiosincrasias (slang colombiano) | **CERRADO** | D-039 |
| Sentiment / swear words / netspeak | **CERRADO** | reabriría la línea de lexicones |
| Voz pasiva | **CERRADO** | requiere parser |
| Preprocesamiento con stopwords / stemming | **CERRADO** | implicaría un vectorizador nuevo (§4) |
| Ponderación booleano / frecuencia / tf-idf | **CERRADO** | ya en V12 |
| Feature selection | **CERRADO** | ya evaluado (0,0000) |
| "Característica mágica" (largo, TTR, hapax) | **CERRADO** | el propio curso los declara obsoletos; ya existen |
| Medidas de complejidad y diversidad | **CERRADO** | ya existen y son superiores |
| Breakeven point | **ADOPTADO como métrica de reporte** de fronteras | coste cero, sin riesgo |
| V1–V13 léxico/simbólico/participación | **CERRADO** | ya existen en V12 |
| **V14–V18 pronombres de persona** | **P17 ABIERTA** | no existen; el artículo los declara predictivos |
| **Palabras funcionales** | **P18 ABIERTA** | solo 2 agregados actuales |
| **Árboles SFL** | **P19 ABIERTA** | no existen y no requieren POS |
| **Uncertainty / Subjectivity / Specificity** | **P20 · P21 ABIERTAS** | no existen |
| **Queísmo / dequeísmo** | **P22 ABIERTA** | bigramas deterministas |
| **Readability en español** | **P23 ABIERTA** | índice clásico, heurística de sílabas |
| Tendencia central y dispersión (media, mediana, moda, skew, std, curtosis) | **P24 ABIERTA** aplicadas a la distribución de evidencia por token | en V12 solo sobre longitudes |
| PCA / MCA sobre el bloque nuevo | **P26 ABIERTA** | in-fold |
| Modelo de significancia por atributo | **P25 ABIERTA** | diagnóstico de interpretabilidad |

## Referencias

- **Moreno-Sandoval, L.G., Pomares-Quimbaya, A. y Alvarado-Valencia, J.A. (2021).** *Celebrity
  profiling through linguistic analysis of digital social networks.* Computational Social Networks,
  8:16. DOI 10.1186/s40649-021-00097-w. — 18 variables (V1–V18) en 5 grupos; corpus PAN@CLEF2019;
  F1 0,65 fama / 0,88 género / 0,37 año nac. / 0,57 ocupación; **+0,14 accuracy media** con modelos
  clásicos; **empeora en modelos profundos**.
- **Material del curso** `Clasificacion Texto.pdf` (78 págs.): atribución de autoría y author
  profiling, 7 familias de características de estilo, árboles SFL, y las variables V1–V18 del
  artículo anterior.
- **Zhou, L. y Zafarani, R. (2020).** *A Survey of Fake News: Fundamental Theories, Detection
  Methods, and Opportunities.* ACM Computing Surveys. DOI 10.1145/3395046 — taxonomía de 10
  dimensiones de features textuales.

---
_Este archivo es el plan vivo de la fase. Se actualiza al cerrar cada pista._
