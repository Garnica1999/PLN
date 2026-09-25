# Especificación maestra — Fase v5 «Embedding from Scratch»

Proyecto: `KAGGLE_TOXIC_2026_PUJ`
Spec de origen: approved por el usuario el 2026-09-25 (44 secciones, numeración
conservada) y transcrita aquí de forma estructurada.
Decisiones de gobierno: **D-030** (alcance de datos externos no etiquetados).
Spec relacionada: `docs/advanced_research_v4_spec.md` (fase v4, ya cerrada).

> Nota de fidelidad: este archivo es una transcripción estructurada de la
> especificación aprobada. Se conservan los números de sección originales para
> que la matriz de cumplimiento pueda auditar cada requisito, y se registran
> también las decisiones explícitas que el usuario tomó tras la revisión del
> plan (secciones 1-32 de «DECISIÓN DEL USUARIO»).

## 0. Contexto y alcance autorizado

| Ítem | Contenido |
|---|---|
| Problema | Clasificación multi-etiqueta HS/TR/AG sobre 1.731 tweets de train y 1.154 de test; texto corto (mediana ≈ 91 caracteres), corpus ASCII, etiquetas muy relacionadas y `TR=1 ⇒ HS=1` |
| Referencia congelada | **V03** (union TF-IDF word(1,2) + char_wb(3,5) → 27.431 features → LogisticRegression balanced C=2.0 liblinear → umbrales 0.44/0.51/0.46). LB **0.78306**, `corrected_cv` **0.7772**. La submission `submissions/toxic_submission_v03_lr_union_thr.csv` **no se sustituye** automáticamente |
| Autorización del profesor | Uso de **grandes corpus externos de texto NO etiquetado** para entrenar desde cero el modelo de embeddings. **No** permite: importar embeddings, pesos o modelos existentes; usar labels externas; entrenar el clasificador final con textos externos; usar datasets externos de hate speech como datos supervisados |
| Restricción vigente | La representación debe entrenarse **completamente desde cero**; el tokenizer/vocabulario, cuando corresponda, se construye desde cero; cada corpus externo debe documentar procedencia, versión, licencia y hash; no se incorpora ninguna fuente nueva silenciosamente |

Texto normativo de la autorización (se registra en `DECISIONS.md` como D-030,
**sin inventar evidencia escrita**):

> El profesor autorizó para esta competencia el uso de datos externos no
> etiquetados exclusivamente para entrenar desde cero una representación de
> embeddings/vectorización. Los datos externos no pueden utilizarse para
> entrenamiento supervisado, clasificación, generación de labels,
> pseudo-labeling, ajuste de thresholds ni ninguna otra forma de aprendizaje
> sobre HS/TR/AG.

Regla de interpretación adicional del usuario: si durante el desarrollo aparece
una interpretación ambigua sobre si una fuente concreta entra en la
autorización, **no se asume permitida**: se registra la duda y se detiene esa
fuente hasta resolverla.

## 1. Hipótesis

- **H-EMB**: una representación distribuida de palabras y subpalabras, entrenada
  desde cero sobre corpus español externo grande y no etiquetado, aporta
  información complementaria a TF-IDF word+char y mejora HS/TR/AG bajo el
  protocolo corregido.
- **H0-EMB**: la representación distribuida no aporta información independiente
  suficiente para superar o mejorar robustamente V03.

## 2. Modelo, corpus y evaluación (mapa de secciones 3-38)

| Sección | Contenido | Decisión operativa |
|---|---|---|
| 3 | Hipótesis central (H-EMB / H0-EMB) | se evalúan ambas; H0 es un resultado válido y documentado |
| 4 | Arquitectura principal: FastText skip-gram, `vector_size=200`, `window=5`, `min_count=3`, `min_n=3`, `max_n=6`, `negative=10`, `sample=1e-4`, `epochs=5`, `sg=1`; exploración acotada de `vector_size {100,200,300}`, `window {3,5,7}`, subwords `(3,5)/(3,6)/(3,7)`, `negative {5,10}`, `min_count {2,3,5}` **sin producto cartesiano** | baseline FastText primero; el ajuste de arquitectura se hace **después** de fijar el corpus (§33) |
| 5 | Fuentes prioritarias: Wikipedia ES, OSCAR, CEREAL, Common Crawl | **Wikipedia ES** (principal) y **mC4-es** (web principal). OSCAR solo con acceso legítimo; HPLT/CEREAL opcionales y solo si aportan margen real. Sin scraping irregular |
| 6 | Escalas: 10M → 50M → 100M → 250M → 500M tokens; 1B solo con evidencia | **techo autorizado 500M**; cada expansión requiere evidencia previa |
| 7 | Cobertura del vocabulario TOXIC: `type_coverage`, `token_coverage`, `OOV_rate`, `frequency_weighted_coverage` | obligatorio antes de ampliar corpus |
| 8 | Cobertura por grupo relevante: insultos, targeting, colectivos, negación, coloquial, elongaciones, hashtags, menciones, noticia, formas raras | cada grupo con `n_types`, `n_tokens`, `coverage`, `OOV`, `mean_frequency` |
| 9 | Distinguir **word-OOV** de **representation-OOV** (subwords) | obligatorio: mide si los subwords ya resuelven la deformación ortográfica |
| 10 | Corpus Size Learning Curve con arquitectura/tokenizer/seed/downstream constantes | obligatoria en 10M/50M/100M; 250M/500M solo con evidencia |
| 11 | Criterio de saturación: requiere **dos incrementos consecutivos** sin mejora material | no se declara saturación con un solo punto |
| 12 | Criterio de `min_count` (1/2/3/5) con tabla `min_count → vocab_size → cobertura → F1 → coste` | solo si el vocabulario relevante está bloqueado |
| 13 | Separar H-DOC (más texto) de H-VOC (más diversidad léxica): `Δ vocabulary / Δ tokens` | obligatorio en cada expansión |
| 14 | Diversidad por fuente: volumen, vocabulario, diversidad, URLs, menciones, hashtags, formas raras, distribución de frecuencias | se identifica si la fuente aporta volumen o información |
| 15 | Corpus por dominio: GENERAL, WEB, MIXTO, y INFORMAL si es legal y útil | general/web/mixto en P4; informal solo con fuente legítima |
| 16 | Dos normalizaciones: **EMB-NORMAL** y **EMB-ASCII** (lowercase, transliteración reproducible, URLs, menciones, hashtags, formas informales, preservar elongaciones; **sin** corrección ortográfica ni reconstrucción de tildes) | ambas obligatorias |
| 17 | FastText: analizar words + character n-grams sobre OOV, errores, elongaciones, sufijo/prefijo, morfología | objetivo: comprobar utilidad subléxica, no superioridad universal |
| 18 | Word2Vec como control desde cero; si FastText no supera, investigar el motivo | control implementado con **fastText `minn=0, maxn=0`** y registrado como `WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF` |
| 19 | GloVe: **no** por defecto; solo si hay señal que justifique una tercera geometría | no implementado por defecto |
| 20 | Validación intrínseca sin labels externos de hate speech: vecinos, coseno, estabilidad, palabras raras, variantes, vocabulario TOXIC | sin afirmar que un vecino es «correcto» por labels |
| 21 | Pooling: **A** mean, **B** TF-IDF weighted mean, **C** mean+max | se comparan antes de usar redes |
| 22 | Clasificadores: primero LogisticRegression, luego LinearSVC, luego MLP pequeño solo si hay señal | sin asumir que una red es mejor |
| 23 | Comparación principal: `V03` vs `FastText`, `Word2Vec`, `V03+FastText`, `V03+Word2Vec` | V03 es la única referencia |
| 24 | Fusión: primero TF-IDF + vector de documento → clasificador; después probabilidades OOF de V03 + embedding → meta-modelo | toda fusión con OOF; sin stacking in-sample |
| 25 | Complementariedad: matriz de 2×2 (V03 correcto/incorrecto × candidato correcto/incorrecto) por rebanadas (insulto interpersonal, afiliación, colectivo, noticia, cita, odio implícito, hashtags, menciones, URLs) | una mejora pequeña con gran rescate se considera más interesante |
| 26 | Protocolo corregido: grupos = near-duplicado ∪ pseudo-autor, `StratifiedGroupKFold`, `src/splits.py`, referencia 0.7772 | no usar 0.7989 como referencia primaria |
| 27 | Anti-fuga: corpus externo sin labels; domain adaptation solo **in-fold** (`embedding_train_k` = externo + textos de train del fold k) | la adaptación de dominio es secundaria |
| 28 | **Holdout agotado**: no se usa ni se crea otro | jerarquía: `corrected_cv` → leaderboard |
| 29 | Criterio de éxito: **A** +0.005 sobre V03; **B** complementariedad robusta; **C** señal estructural fuerte aunque la F1 sea marginal | distinguir `adopted` / `candidate` / `exploratory` |
| 30 | Escalado solo con condición 1-4 (cobertura, ganancia marginal, mejora downstream, nuevos contextos/diversidad); registrar **`RATIONALE_FOR_CORPUS_EXPANSION`** | obligatorio antes de cada expansión |
| 31 | Separar «más tokens» de «mejor corpus»: `tokens_added`, `new_types_added`, `new_relevant_types_added`, `new_ngrams_added`, `domain_diversity_gain`, `downstream_delta` | obligatorio |
| 32 | Experimento controlado: cada incremento mantiene arquitectura, hiperparámetros, seed, downstream y preprocessing | identificadores `E-CORP-01..05` |
| 33 | Arquitectura vs tamaño: primero escala, luego arquitectura; nunca atribuir a una sola causa | obligatorio |
| 34 | Experimentos priorizados `EBD-01..EBD-17` y, solo con evidencia, `EBD-18..EBD-22` | ver `reports/research_backlog_v5.md` |
| 35 | Reproducibilidad del corpus: `source`, `url`, `version`, `license`, `download_date`, `file_count`, `document_count`, `raw_size`, `clean_size`, `token_count`, `vocabulary_size`, `sha256`, `preprocessing_version` + `corpus_manifest.json` | obligatorio |
| 36 | Reproducibilidad del embedding: arquitectura, `vector_size`, `window`, `min_count`, `min_n`, `max_n`, `negative`, `sample`, `epochs`, `seed`, `corpus_version`, `training_time`, `vocabulary_size`, `embedding_hash` | obligatorio |
| 37 | Auditoría: 0 usos de transformers/BERT/RoBERTa/`from_pretrained`/embeddings preentrenados/labels externas/labels de test/toxic test en el corpus/labels de hate speech externas | la librería puede implementar el algoritmo; los pesos se entrenan desde cero |
| 38 | Artefactos mínimos (ver §5 de este archivo) | obligatorio |
| 39 | No repetir estrategias cerradas (BM25, chi², MI, log-odds, NB-SVM, densas, POS, dependencia, targeting, emoción, stacking genérico, chains, label powerset, especialistas, hard-example mining, feature engineering arbitrario) | se consulta `PROJECT_STATE.md`, `DECISIONS.md`, `FINAL_REPORT.md`, `advanced_research_v2.md`, `advanced_research_v4.md`, `experiments.csv` |
| 40 | No generar submissions automáticamente; solo con evidencia suficiente, comparación reproducible, complementariedad cuantificada, sin leakage y decisión documentada | el envío sigue siendo del usuario |
| 41 | Análisis de error contra V03 para cualquier embedding prometedor (FP/FN por etiqueta, errores conjuntos, rebanadas débiles) | pregunta: ¿el embedding modifica la frontera donde la representación sparse no tiene señal? |
| 42 | Doce preguntas de resultado que la fase debe responder | §5 de este archivo |
| 43 | Regla de cierre (condiciones A-E): consistently inferior; señal sin alcance; saturación clara; cobertura alta con rendimientos decrecientes; complementariedad sin transferencia | se cierra y se documenta |
| 44 | Principio final: es investigación de **representación distribuida**, no «más features + más modelos + más stacking»; medir por separado corpus → cobertura → calidad → complementariedad → downstream → transferencia | obligatorio |

## 3. Gobierno y restricciones de ejecución (decisiones del usuario)

| Tema | Decisión |
|---|---|
| Autorización (D-030) | Registrar **solo** la información comunicada por el usuario; **no** inventar correo, foro o captura archivada. Ante ambigüedad sobre una fuente: detener esa fuente y registrar la duda |
| Fuentes | Wikipedia ES (P1) → mC4-es (P2) → mixto (P3) → HPLT/CEREAL/OSCAR (P4, solo con margen real) |
| Escala | 10M → 50M → 100M → decidir → 250M → decidir → 500M. **No** superar 500M sin nueva decisión explícita del usuario |
| Implementación | fastText C++ oficial (g++/cmake/make). Word2Vec-control con `minn=0, maxn=0`. PyTorch solo como respaldo/control. **No** instalar gensim |
| Corpus en disco | Pipeline `download → raw → clean → dedup → normalize → tokenize → shards → manifest → training`; streaming/chunks; no cargar todo en RAM; no conservar intermedios innecesarios |
| Anti-fuga | Auditar que no entran labels de TOXIC, datasets de hate speech, `toxic_test.csv`, labels de test ni textos del test; distinguir «texto para el embedding» de «labels para supervisión» |
| Uso del train de TOXIC | Variante principal **EXTERNAL-ONLY**; domain adaptation in-fold solo tras demostrar señal |
| Normalizaciones | EMB-NORMAL y EMB-ASCII, sin correctores ortográficos ni información semántica externa |
| GPU | No es requisito de fastText; **no** trasladar artificialmente el algoritmo a GPU |
| Holdout | Completamente congelado y agotado; no se crea otro |
| Leaderboard | Sin submissions automáticas; no gastar sonda por `+0.001`; el envío es del usuario |
| V03 | No se modifica ni se sobrescribe; no se borra evidencia histórica; no se modifica retrospectivamente D-024 ni decisiones anteriores |
| Repositorio | Rama independiente `proyecto-kaggle-toxic-2026-v5`; PR al finalizar; **sin** merge automático; **sin** push automático sin autorización explícita en la sesión |

## 4. Estructura de directorios de la fase

```
data/external/corpora/        # corpus externos (raw, clean, shards) — ignorado por git
data/external/embeddings/     # modelos y vectores entrenados — ignorado por git
features/embeddings/          # vectores de documento y OOF del downstream
models/embeddings/            # modelos de clasificación sobre embeddings
experiments/embedding_*.py    # scripts de la fase
reports/embedding_*.md|csv    # informes y resultados
docs/embedding_from_scratch_v5_spec.md
```

## 5. Preguntas que la fase debe responder (§42)

1. ¿Un embedding distribuido entrenado desde cero es viable con este problema?
2. ¿FastText supera a Word2Vec?
3. ¿Los subwords son útiles para este corpus?
4. ¿Cuántos tokens son necesarios?
5. ¿El vocabulario actual es suficiente?
6. ¿Hace falta más vocabulario?
7. ¿Hace falta más documentos?
8. ¿Qué fuente aporta información marginal?
9. ¿El registro web mejora la representación?
10. ¿La normalización ASCII ayuda?
11. ¿El embedding aporta información complementaria a V03?
12. ¿Puede mejorar los errores que V03 no resuelve?

## 6. Secuencia de fases aprobada

| Fase | Contenido | Gate |
|---|---|---|
| **P0** | Compliance: spec, D-030, compilación de fastText C++ validada con corpus de juguete, esquema del `corpus_manifest.json`, auditoría de librerías, plan y backlog | fastText compila y entrena; autorización registrada; auditoría en verde |
| **P1** | Corpus escala 1 (10M tokens) de Wikipedia ES + diagnóstico de cobertura TOXIC (§7-§9) | cobertura medida con números; manifest con hashes y licencias; cero fuga test→corpus |
| **P2** | Baseline FastText 10M + Word2Vec-control + intrínseca + downstream (pooling × clasificador) | primer veredicto: ¿hay señal? ¿se justifica 50M? |
| P3 | Curva 10M → 50M → 100M (→250M/500M con condiciones de §30) | saturación con dos incrementos consecutivos |
| P4 | Composición (general/web/mixto) y normalizaciones (NORMAL/ASCII) | ¿aporta el registro web? ¿cambia el ASCII? |
| P5 | Arquitectura (vector_size, window, subwords, min_count, negative) con corpus fijo | solo lo que mejore consistentemente |
| P6 | Fusión con V03 (hstack y meta OOF) + complementariedad + error analysis | complementariedad cuantificada |
| P7 | Cierre: candidato si cumple §29, auditorías, informes, memoria, notebook, PR | condición de cierre A-E de §43 |

**Alcance autorizado en esta sesión: P0, P1 y P2.** No se descarga 500M tokens
antes de completar el diagnóstico de P1/P2.
