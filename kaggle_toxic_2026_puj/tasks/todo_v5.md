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

- [x] Descarga por tramo del dump multistream (rangos HTTP) sin bajar los 5.19 GB
      completos; registrar URL, fecha y sha256 del tramo
- [x] Extraccion de texto plano (parser iterativo del XML, sin cargar en RAM)
- [x] Limpieza y deduplicacion (exacta y por hash de shingles)
- [x] Normalizacion **EMB-NORMAL** (lowercase, URLs, menciones, hashtags,
      preservacion de elongaciones, **sin** correctores ortograficos)
- [x] Tokenizacion y escritura de **shards** hasta 10M tokens
- [x] `corpus_manifest.json` actualizado con `document_count`, `token_count`,
      `vocabulary_size`, `sha256`, `preprocessing_version`
- [x] **Auditoria de fuga**: 0 textos de `toxic_test.csv` (y de test) en el corpus
- [x] Cobertura TOXIC (§7): `type_coverage`, `token_coverage`, `oov_rate`,
      `frequency_weighted_coverage` (+ OOV listado)
- [x] Cobertura por grupo (§8): insultos, targeting, colectivos, negacion,
      coloquial, elongaciones, hashtags, menciones, noticia, formas raras
- [x] Distinguir **word-OOV** de **representation-OOV** (§9) usando
      `print-word-vectors` en modo subwords
- [x] Informes: `reports/embedding_corpus_audit.md`,
      `reports/embedding_vocabulary_coverage.md`

## P2 - Baseline FastText 10M + control + downstream

- [x] `E-CORP-01`: corpus 10M construido y auditado
- [x] `EBD-01`: FastText 200d / ws5 / 3-6 / neg10 / minCount3 / 5 epocas /
      seed fija; doc vector = media de tokens (pooling A)
- [x] `EBD-02`: Word2Vec-control identico con `-minn 0 -maxn 0`
- [x] `EBD-03`: pooling B (TF-IDF-weighted) y C (mean+max): **peores que A**
- [x] `EBD-04`: LinearSVC: 0.6584 frente a 0.6603 de LR, sin diferencia relevante
- [x] `EBD-05`: cobertura sublexica de los 14 tipos word-OOV mas frecuentes
- [x] `EBD-06` (control no pre-registrado, justificado por P1/P2): **suelo
      aleatorio** 0.5404 y **estabilidad entre semillas** (solapamiento top-10 0.792)
- [x] Intrinseca (§20): vecinos de 12 palabras en ambos modelos y de 14 variantes
      ortograficas, sin labels externos
- [x] Downstream con el protocolo corregido: `corrected_cv`, IC95 pareado vs V03,
      estabilidad por pliegue y `metricas_multietiqueta` como metrica primaria
- [x] Equivalencia de la normalizacion con P1 verificada token a token (2.885
      textos, 0 diferencias)
- [x] Informes: `reports/embedding_intrinsic_eval.md`,
      `reports/embedding_downstream_results.csv`
- [x] **Veredicto de escala**: el embedding por si solo queda **0.1169 por debajo de
      V03** con el IC95 completamente negativo. Como sustituto esta refutado. La
      hipotesis de **complementariedad** (que es la central de la fase) **sigue sin
      medir**: exige la fusion de la seccion 24 (P6), no autorizada en esta sesion.
      Escalar a 50M **no** esta justificado por la evidencia disponible y queda
      condicionado a la prueba de complementariedad.

## P3-P5 - Complementariedad y corpus de registro (ejecutados)

- [x] `EBD-07`: fusion hstack TF-IDF + 200d L2, umbrales congelados de V03
      (el control reproduce 0.777211 exacto)
- [x] `EBD-08`: fusion con bloque denso estandarizado (descartada: -0.0871)
- [x] `EBD-09`: meta-modelo anidado Ridge (cerrado: -0.0094, IC95 negativo)
- [x] Matriz de complementariedad 2x2 y metricas por 11 rebanadas
- [x] `E-CORP-06/07`: corpus web mC4-es 10M + mixto 20M, cobertura y contadores §31
- [x] `EBD-10`: FastText sobre corpus web y mixto, y revision de la fusion
- [x] Decision registrada en **D-031** y **D-032**

**Resultado**: la cobertura sube mucho (0.6839 -> 0.8413 por tipo; elongaciones
0.039 -> 0.221) y el embedding **aislado** mejora (0.6530 -> 0.6658), pero la
aportacion complementaria **no crece** (+0.0046 wiki, +0.0040 web, +0.0004 mixto,
todos con IC95 que toca el 0). La palanca de corpus queda **agotada** y la fase se
cierra **sin adoptar** el candidato (D-032).

## P7 - Cierre

- [x] Estado por linea y condiciones de cierre de la §43 evaluadas en D-032
- [x] Artefactos reproducibles: scripts, informes, CSV, figuras, manifiesto y
      memoria actualizados
- [x] Sin submissions y sin sondas de LB
- [x] `RATIONALE_FOR_CORPUS_EXPANSION` **no** llega a redactarse: ninguna
      expansion cumple las condiciones de la §30
- [x] `experiments/embedding_06_auditoria_v5.py`: auditoria final de la fase
- [x] Informe de cierre de la fase: `reports/embedding_phase_closeout_v5.md`
- [ ] Seccion de la fase v5 en `FINAL_REPORT.md`: pendiente (se redacta al abrir el PR)

## Fuera de alcance (cerrado en D-030, D-031 y D-032)

- Corpus etiquetado externo, hate speech externo, labels externas, pseudo-labeling
  y ajuste de umbrales con datos externos (D-030).
- Corpus etiquetado de HuggingFace con licence no revisada; OSCAR sigue gated.
- GloVe salvo senal que lo justifique (spec 19); gensim (no hay wheel para 3.14);
  transformers y cualquier embedding preentrenado.
- Escalar el corpus mas alla de 20M tokens: ninguna expansion cumple la §30
  (D-031 y D-032).
- Apilamiento (stacking) de probabilidades: cerrado tras cinco intentos
  fallidos en el proyecto.
- Uso del holdout (agotado 5/5), uso del test para entrenar o seleccionar,
  submissions automaticas, push/PR/merge sin autorizacion explicita.
- Repetir estrategias cerradas en v2/v4 (ver `tasks/plan_v5.md`).
