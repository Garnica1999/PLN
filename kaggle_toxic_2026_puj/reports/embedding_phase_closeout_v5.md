# Cierre de la fase v5 «Embedding from Scratch»

Proyecto: `KAGGLE_TOXIC_2026_PUJ`
Rama: `proyecto-kaggle-toxic-2026-v5` (worktree `/home/carlos/Git/PLN-v5`, base v4 `5d06ce7`)
Decisiones de gobierno: **D-030** (alcance de los datos externos), **D-031**
(estado y palanca), **D-032** (cierre).
Auditoria final: **GO** (`reports/embedding_auditoria_v5.json`).

## 1. Veredicto en una linea

Una representacion distribuida de palabras y subpalabras, entrenada **desde cero**
sobre corpus espanol externo no etiquetado, **no aporta** informacion util a esta
competencia: como sustituto de V03 es peor en 0,11 de F1 y como complemento adds
una cantidad constante de ~+0,004 que no llega a ser significativa y que no crece
ni con mas tokens ni con un registro de texto distinto.

## 2. Que se pregunto y que se respondio

| Pregunta (spec) | Respuesta | Evidencia |
|---|---|---|
| ¿Es viable un embedding desde cero aquí? (§42.1) | **No como sustituto**: 0,6530 frente a 0,777211 con IC95 completamente negativo | `embedding_fusion_v5.json` |
| ¿FastText supera a Word2Vec? (§42.2) | **Sí, y de forma medible**: +0,0138 downstream y cambio de geometría (vecinos morfológicos frente a distribucionales) | `embedding_downstream_results.csv`, `embedding_intrinsic_eval.md` |
| ¿Sirven los subpalabras? (§42.3) | **Sí como representación, no como disponibilidad**: todo tipo word-OOV obtiene vector, pero su vecino no es identificable con 10M tokens | `embedding_vocabulary_coverage.md` |
| ¿Cuántos tokens hacen falta? (§42.4) | **10M bastan para el efecto observable**; el efecto marginal no crece con 2,5x más cobertura | `embedding_web_results_v5.md` |
| ¿El vocabulario actual basta? (§42.5-6) | La cobertura sube de 0,6839 a 0,8413 por tipo con registro web, y **eso no mejora la aportacion** | `embedding_corpus_web_audit.md` |
| ¿Hace falta más vocabulario o más documentos? (§42.7) | **Ninguno de los dos**: la limitación no es de cobertura | D-032 |
| ¿Qué fuente aporta? (§42.8-9) | El registro web aporta cobertura, no senal downstream | D-032 |
| ¿Normalización ASCII? (§42.10) | **Medida y no decisive**: solo el 8,5 % de los tipos ausentes son pliegues de acento | `embedding_vocabulary_coverage.md` §1.1 |
| ¿Aporta complementariamente a V03? (§42.11) | **Muy poco y sin significance**: +0,0046 con IC95 que toca el 0, concentrado en elongaciones, URLs y hashtags | `embedding_fusion_complementarity_v5.md` |
| ¿Puede mejorar los errores que V03 no resuelve? (§42.12) | **Marginalmente**: 8 celdas de 4.449 que solo la fusión acierta | matriz 2x2 |

## 3. Los tres hallazgos que cambian la lectura de la fase

1. **La cobertura no era la palanca.** Toda la fase se diseño en torno a medir
   cobertura (§7-§9) y esa medicion resulto ser una condicion necesaria pero **no
   suficiente**: el corpus web sube la cobertura por tipo de 0,6839 a 0,8256 y la
   union a 0,8413, `elongaciones` de 0,039 a 0,221, y sin embargo la fusion se
   queda igual (+0,0046, +0,0040, +0,0004). El ~+0.004 proviene de una senal
   debil y distinta que TF-IDF char(3,5) ya captura en su mayor parte.

2. **La escala de los bloques decide qué senal sobrevive.** Con la union dispersa
   TF-IDF y 200 dimensiones densas en un LR regularizado con L2, estandarizar el
   bloque denso hace que **domine** al disperso y la fusion cae a 0,6901 (-0,0871
   con los mismos datos). Con normalización L2 por documento la fusion llega a
   0,7818. No es una propiedad de la representacion sino de cómo se combinan
   bloques de escalas distintas, y se fijo L2 como convencion de la fase.

3. **Los controles evitaron dos trampas.** El control `V03-TFIDF` reproducjo
   0,777211 **exacto** en la misma corrida, lo que valida la comparacion celda a
   celda; y el **suelo aleatorio** (0,5404) permitio leer 0,6603 como "+0,120 sobre
   el azar" y no como "0,66 de performance". Sin esos dos controles, la conclusion
   de la fase habria sido distinta y equivocada.

## 4. Estado por linea

| Linea | Estado | Numero |
|---|---|---|
| Embedding como sustituto de V03 | **REFUTADO** | 0,6658 (mejor) vs 0,777211 |
| Complementariedad (fusion hstack L2) | **CANDIDATE, no adoptado** | +0,0046, IC95 [-0,0032, +0,0124] |
| Subpalabras | **CONFIRMADAS** (geometria y +0,0138) | P2 |
| Corpus de registro web | **UTIL para cobertura, neutro downstream** | P4/P5 |
| Escala de corpus | **AGOTADA** (ni tamaño ni registro mueven el resultado) | D-031, D-032 |
| Stacking / meta-modelo | **CERRADO** (-0,0094, IC95 negativo) | EBD-09 |
| Pooling TF-IDF-weighted y mean+max | **DESCARTADOS** | EBD-03 |
| LinearSVC | sin diferencia relevante | EBD-04 |
| GloVe, transformers, embeddings preentrenados | **no usados** (0 usos, auditoria) | `embedding_auditoria_v5.json` |
| Techo de 500M tokens | **sin usar** y sin `RATIONALE_FOR_CORPUS_EXPANSION` | D-032 |

## 5. Lo que queda vivo

`FUSION-FASTTEXT-WIKI-10M` (0,7818) queda registrado como artefacto reproducible
con su matriz de complementariedad, sus 11 rebanadas y los cuatro modelos con su
`sha256`. **No se genera submission**: no hay holdout (agotado 5/5) y gastar una
sonda del leaderboard para +0,004 con IC95 tocando el 0 no esta justificado.

## 6. Consecuencia para el modelo vigente

**V03 se mantiene**: `corrected_cv` 0,777211, LB 0,78306, umbrales 0.44/0.51/0.46,
submission `toxic_submission_v03_lr_union_thr.csv` intacta (1.155 lineas,
verificada por la auditoria). La fusion L2 queda documentada como la unica
aportacion medible de la fase, sin adoptar.

## 7. Leccion transferible

La cobertura de vocabulario era el parametro que toda la fase daba por relevante,
y resulto no predecir el comportamiento downstream. El error no fue medirla: fue
**suponer que su mejora se traducía en mejora downstream**, y no contrastarlo. La
prueba que lo detecto fue comparar el embedding **aislado** frente a su **fusion**
en la misma corrida: si la cobertura fuera la palanca, las dos habrian subido};
solo subio la primera. Esa comparacion de control es la que conviene hacer siempre
que se cambien las variables de entrada de una representacion.

## 8. Inventario

- Scripts: `experiments/embedding_00..06_*_v5.py` (7, incluido el auditor).
- Informes: `reports/embedding_compliance.md`, `embedding_corpus_audit.md`,
  `embedding_vocabulary_coverage.md(.csv)`, `embedding_intrinsic_eval.md`,
  `embedding_fusion_complementarity_v5.md`, `embedding_web_results_v5.md`,
  `embedding_corpus_web_audit.md`, `embedding_coverage_registro_v5.csv`.
- Datos: `reports/embedding_*.json` (10), figuras `fig_41..46_*_v5.png` (6).
- Manifiesto: `data/external/corpora/corpus_manifest.json` con dos corpus
  construidos (E-CORP-01 Wikipedia 10M, E-CORP-06 mC4-es 10M) y el mixto de 20M.
- Memoria: `DECISIONS.md` (D-030 a D-032), `PROJECT_STATE.md`, `RUNLOG.md`,
  `experiments.csv` (157 filas), `tasks/plan_v5.md`, `tasks/todo_v5.md`.
- Cumplimiento: 0 usos prohibidos, 0 fugas, holdout 5/5 intacto, 0 submissions,
  0 sondas de leaderboard.

## 9. Nota de trazabilidad: tiempos de entrenamiento

En los informes de P2 y P5 el campo `segundos` de los modelos aparece como **0.0**.
Eso **no** significa que no se entrenaran: significa que el script **reutilizo** modelos
ya entrenados en una corrida anterior (el registro de those corridas se sobrescribio al
relanzar con `tee`). La evidencia de que el entrenamiento ocurrio son las huellas en
disco y la medicion directa:

| Modelo | Corpus | `.bin` | `.vec` | Tipos aprendidos | Hora |
|---|---|---|---|---|---|
| `fasttext_baseline` | Wikipedia 10M | 307,1 MB | 167,4 MB | 100.098 | 16:55:19 |
| `fasttext_seed_control` | Wikipedia 10M | 307,1 MB | 167,5 MB | 100.098 | 16:56:53 |
| `word2vec_control` | Wikipedia 10M | 154,5 MB | 170,1 MB | 100.098 | 16:57:54 |
| `fasttext_web_10m` | mC4-es 10M | 317,1 MB | 178,3 MB | 106.632 | 17:25:14 |
| `fasttext_mixto_20m` | mixto 20M | 404,5 MB | 273,2 MB | 163.228 | 17:28:33 |
| `word2vec_web_10m` | mC4-es 10M | 164,5 MB | 180,2 MB | 106.632 | 17:29:42 |

Medicion directa del entrenamiento con los parametros exactos de la fase
(200d, ws5, neg10, `minCount` 3, 5 epocas, 16 hilos, semilla 42, `bucket` 200.000),
ejecutada aparte para no alterar las huellas registradas:

- **tiempo real: 1 min 22 s** de reloj para 5 epocas sobre 10M tokens;
- **rendimiento: 40.099 palabras/s por hilo x 16 hilos = 641.584 palabras/s**;
- `avg.loss` converge a 2,149 y el vocabulario queda en **100.098 tipos** tras el
  poda de `min_count=3` sobre los 271.209 tipos crudos del corpus.

Que sea rapido es correcto y esperado: fastText es una de las implementaciones de
embeddings mas rapidas que existen (C++ optimizada con muestreo negativo y SGD
paralelo), y **10M tokens es un corpus diminuto** para esa herramienta. La rapidez
tiene dos lecturas y conviene no confundir la buena con la mala:

1. La buena: el techo autorizado de **500M tokens habria costado ~65 minutos** de
   entrenamiento (500M x 5 epocas / 641k palabras/s). La decision de D-031 y D-032 de
   **no** escalar el corpus se tomo por evidencia, **no** por coste de computo.
2. La mala: 10M tokens son pocos para aprender embeddings de calidad, y eso forma
   parte de por que los resultados downstream son debiles. La fase midio esa limitacion
   en lugar de suponerla, pero conviene tenerla presente al leer los numeros.
