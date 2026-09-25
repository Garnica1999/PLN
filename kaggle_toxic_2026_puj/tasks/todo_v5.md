# TODO v5 - KAGGLE_TOXIC_2026_PUJ (Embedding from Scratch)

Estado: **P0 completado (GO)**. Alcance autorizado en esta sesion: **P0, P1, P2**.

Leyenda: `[x]` hecho | `[~]` en curso | `[ ]` pendiente | `[!]` bloqueado.

## P0 - Compliance, scaffold y pre-registro

- [x] `experiments/embedding_00_scaffold_v5.py` verificado con venv
- [x] Spec maestra transcrita en `docs/embedding_from_scratch_v5_spec.md` (§0-§44)
- [x] **D-030** registrada con la transcripcion literal de la autorizacion y sin
      inventar evidencia escrita
- [x] fastText C++ compilado (commit `1142dc4c4ecb`, MIT, binario con sha256 en el
      manifiesto) y validado entrenando
- [x] Control `WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF` validado
      (`minn=0, maxn=0`; el `NaN` observado con un corpus de 25 tipos es un
      artefacto del corpus de juguete, no del modo)
- [x] Esquema de `data/external/corpora/corpus_manifest.json` + copia declarativa
      `reports/corpus_manifest_v5.json`
- [x] Auditoria de recursos en `reports/embedding_compliance.md` (0 usos de
      transformers / sentence-transformers / GloVe / `from_pretrained`)
- [x] Referencia V03 reproduce `corrected_cv` 0.777211 desde el OOF canonico
- [x] `tasks/plan_v5.md` y este todo pre-registrados
- [x] Fila `EBD-00-SCAFFOLD` en `experiments.csv`

## P1 - Corpus 10M (Wikipedia ES) y cobertura TOXIC

- [ ] Descarga por tramo del dump multistream (rangos HTTP) sin bajar los 5.19 GB
      completos; registrar URL, fecha y sha256 del tramo
- [ ] Extraccion de texto plano (parser iterativo del XML, sin cargar en RAM)
- [ ] Limpieza y deduplicacion (exacta y por hash de shingles)
- [ ] Normalizacion **EMB-NORMAL** (lowercase, URLs, menciones, hashtags,
      preservacion de elongaciones, **sin** correctores ortograficos)
- [ ] Tokenizacion y escritura de **shards** hasta 10M tokens
- [ ] `corpus_manifest.json` actualizado con `document_count`, `token_count`,
      `vocabulary_size`, `sha256`, `preprocessing_version`
- [ ] **Auditoria de fuga**: 0 textos de `toxic_test.csv` (y de test) en el corpus
- [ ] Cobertura TOXIC (§7): `type_coverage`, `token_coverage`, `oov_rate`,
      `frequency_weighted_coverage` (+ OOV listado)
- [ ] Cobertura por grupo (§8): insultos, targeting, colectivos, negacion,
      coloquial, elongaciones, hashtags, menciones, noticia, formas raras
- [ ] Distinguir **word-OOV** de **representation-OOV** (§9) usando
      `print-word-vectors` en modo subwords
- [ ] Informes: `reports/embedding_corpus_audit.md`,
      `reports/embedding_vocabulary_coverage.md`

## P2 - Baseline FastText 10M + control + downstream

- [ ] `E-CORP-01`: corpus 10M construido y auditado
- [ ] `EBD-01`: FastText 200d / ws5 / 3-6 / neg10 / minCount3 / 5 epocas /
      seed fija; doc vector = media de tokens in-vocab
- [ ] `EBD-02`: Word2Vec-control identico con `-minn 0 -maxn 0`
- [ ] `EBD-03`: pooling B (TF-IDF-weighted) y C (mean+max)
- [ ] `EBD-04`: LinearSVC sobre los vectores de documento
- [ ] `EBD-05`: cobertura sublexica (los tipos word-OOV reciben vector via subwords)
- [ ] Intrinseca (§20): vecinos, cosenos, estabilidad entre 2 semillas, sin
      afirmar \"correctitud\" con labels externos
- [ ] Downstream con el protocolo corregido: `corrected_cv`, bootstrap pareado vs
      V03, estabilidad por pliegue, y `metricas_multietiqueta` como metrica primaria
- [ ] Informes: `reports/embedding_intrinsic_eval.md`,
      `reports/embedding_downstream_results.csv`
- [ ] **Veredicto de escala**: ¿se justifica 50M? Si si, escribir
      `RATIONALE_FOR_CORPUS_EXPANSION` **antes** de descargar

## P3 - Curva de aprendizaje (condicionada)

- [ ] `E-CORP-02` 50M y `E-CORP-03` 100M con arquitectura fija
- [ ] 250M / 500M solo si se cumple una condicion de §30
- [ ] Saturacion declarada solo con **dos** incrementos consecutivos sin mejora

## P4 - Composicion y normalizaciones (condicionada)

- [ ] Wikipedia vs mC4-es vs mezcla, mediendo `tokens_added` y `new_types_added`
- [ ] EMB-NORMAL vs EMB-ASCII

## P5 - Arquitectura (condicionada)

- [ ] `vector_size {100,200,300}`, `window {3,5,7}`, subwords `(3,5)/(3,6)/(3,7)`,
      `negative {5,10}`, `min_count {2,3,5}` sin producto cartesiano

## P6 - Fusion y complementariedad (condicionada)

- [ ] `hstack` TF-IDF + vector de documento antes que stacking
- [ ] Meta-modelo solo con OOF de V03
- [ ] Matriz de complementariedad 2x2 y analisis de error por rebanadas

## P7 - Cierre (condicionada)

- [ ] Candidato `adopted` / `candidate` / `exploratory` segun §29
- [ ] Audiencias finales: `experiments/embedding_*_auditoria_v5.py` (0 violaciones)
- [ ] Informes, memoria, notebook y PR (sin merge automatico)

## Fuera de alcance (decidido)

- Corpus etiquetado externo, hate speech externo, labels externas, pseudo-labeling
  y ajuste de umbrales con datos externos (D-030).
- GloVe salvo senal que lo justifique (spec 19); gensim (no hay wheel para 3.14);
  transformers y cualquier embedding preentrenado.
- Uso del holdout (agotado 5/5), uso del test para entrenar o seleccionar,
  submissions automaticas, push/PR/merge sin autorizacion explicita.
- Repetir estrategias cerradas en v2/v4 (ver `tasks/plan_v5.md`).
