# PLAN v5 - KAGGLE_TOXIC_2026_PUJ (Embedding from Scratch)

Decision de gobierno: **D-030** (los corpus externos no etiquetados se usan
**exclusivamente** para aprender la representacion distribuida; prohibido todo
uso supervisado).
Spec maestra transcrita: `docs/embedding_from_scratch_v5_spec.md` (44 secciones).

## Referencia y protocolo (no se modifican)

| Referencia | Valor |
|---|---|
| Modelo de referencia | **V03** (union TF-IDF word(1,2) + char_wb(3,5) -> LR C=2.0 balanced) |
| Metrica primaria | `f1_media_columnas` (media de la F1 binaria positiva de HS, TR, AG; D-009) |
| Protocolo | `corrected_cv`: grupos = near-duplicado UNION pseudo-autor, `StratifiedGroupKFold(5)` sobre el desarrollo (`src/splits.py`) |
| Referencia a superar | `corrected_cv` **0.777211** (n = 1.483, std de pliegues 0.0161) |
| LB de referencia | **0.78306** (0 sondas nuevas disponibles sin autorizacion) |
| Holdout | **agotado (5/5)**: no se usa ni se crea otro |
| Particiones | train 1.731 | test 1.154 | desarrollo 1.483 | holdout congelado 248 |
| Restricciones | sin transformers / sentence-transformers / GloVe / embeddings preentrenados / labels externas / `toxic_test.csv` en el corpus |

## Hipotesis

- **H-EMB**: una representacion distribuida (FastText) entrenada desde cero
  sobre corpus espanol externo no etiquetado aporta informacion complementaria a
  TF-IDF word+char y mejora HS/TR/AG bajo el protocolo corregido.
- **H0-EMB**: la representacion distribuida no aporta informacion independiente
  suficiente. Es un resultado valido y se documenta igual que H-EMB.

## Fases y gates

| Fase | Contenido | Gate de salida |
|---|---|---|
| **P0** | Spec transcrita, D-030, fastText C++ compilado y validado, esquema del `corpus_manifest.json`, auditoria de recursos, plan y todo | fastText entrena; referencia V03 reproduce 0.777211; 0 violaciones |
| **P1** | Corpus escala 1: 10M tokens de Wikipedia ES (tramo parcial del dump multistream), pipeline `download -> raw -> clean -> dedup -> normalize -> tokenize -> shards -> manifest`; auditoria de cobertura TOXIC (§7-§9) | cobertura medida con numeros (type, token, OOV, frequency-weighted, subword) + manifest con hashes y licencias + 0 textos de test en el corpus |
| **P2** | FastText baseline (200d, ventana 5, subwords 3-6, negative 10, min_count 3, 5 epocas, seed fija) + Word2Vec-control (`minn=0, maxn=0`) sobre 10M; intrinseca y downstream (pooling A/B/C x LR/LinearSVC) | primer veredicto: ¿hay señal? ¿se justifica 50M? |
| **P3** | Curva 10M -> 50M -> 100M (250M/500M solo con §30 cumplida) | saturacion declarada solo con **dos** incrementos consecutivos sin mejora |
| **P4** | Composicion (general / web / mixto) y normalizaciones (EMB-NORMAL / EMB-ASCII) | respuesta documentada al aporte de cada fuente |
| **P5** | Arquitectura (vector_size, window, subwords, negative, min_count) con corpus fijo | solo lo que mejore consistentemente |
| **P6** | Fusion con V03 (`hstack` y meta con OOF) + matriz de complementariedad + analisis de error | complementariedad cuantificada |
| **P7** | Cierre: candidato si cumple §29, auditorias, informes, memoria, notebook, PR | condicion de cierre A-E de §43 |

**Alcance autorizado en esta sesion: P0, P1 y P2.** La decision de escalar a
50M/100M se toma con la evidencia de P1 (cobertura) y P2 (downstream), y exige
`RATIONALE_FOR_CORPUS_EXPANSION` escrito antes de descargar.

## Experimentos pre-registrados

| Id | Fase | Pregunta | Estado |
|---|---|---|---|
| `EBD-00-SCAFFOLD` | P0 | ¿el entorno, el compliance y la build de fastText son reproducibles? | completado (GO) |
| `E-CORP-01` | P1 | ¿que cobertura del vocabulario TOXIC alcanza 10M tokens de Wikipedia ES? | pendiente |
| `E-CORP-02/03/04/05` | P3 | ¿que aporta 50M / 100M / 250M / 500M tokens? | condicionados |
| `EBD-01` | P2 | ¿el vector de documento de FastText (media) + LR supera V03 en `corrected_cv`? | pendiente |
| `EBD-02` | P2 | ¿Word2Vec-control supera o queda por debajo de FastText? | pendiente |
| `EBD-03` | P2 | ¿el pooling TF-IDF-weighted o mean+max mejora la media? | pendiente |
| `EBD-04` | P2 | ¿LinearSVC sobre vectores de documento mejora a LR? | pendiente |
| `EBD-05` | P2 | ¿la cobertura **subléxica** rescata los tipos word-OOV del corpus TOXIC? | pendiente |
| `EBD-06..17` | P2-P6 | ver `tasks/todo_v5.md` | pre-registrados |

## Criterios de adopcion (§29)

- **A**: `corrected_cv` >= V03 + 0.005 con estabilidad razonable.
- **B**: complementariedad robusta (matriz 2x2 con rescate claro en rebanadas
  donde V03 falla).
- **C**: senal representacional fuerte aunque la F1 sea marginal.

Se distingue `adopted` / `candidate` / `exploratory` y **no se presenta una
mejora exploratoria como mejora final**.

## Jerarquia de evidencia y prudencia

`corrected_cv` -> leaderboard. El `corrected_cv` **no es un selector fiable**
(cuatro casos consecutivos de mejora local sin transferencia en v2 y v4: Optuna
0.8039 -> LB 0.77602; meta v2 0.7874 -> LB 0.78163; candidatos v09-v11;
candidato C 0.7882 -> holdout 0.8163 vs 0.8285). Por eso ninguna conclusion de
v5 se apoya en un solo numero: se exigen intervalo de bootstrap, estabilidad por
pliegue y analisis de complementariedad.

## Reglas de higiene del experimento

1. Un experimento = un script nuevo `experiments/embedding_<NN>_<descripcion>.py`;
   nunca se edita ni se borra un script previo.
2. Toda ejecucion con `/home/carlos/venv/bin/python` y queda en
   `experiments/*.log`.
3. Corpus y vectores pesados viven en `data/external/` y
   `data/external/embeddings/` (ignorados por git); la trazabilidad va en
   `reports/corpus_manifest_v5.json` (hashes, licencias, conteos).
4. El holdout no se toca. El test solo se lee para diagnostico de cobertura, nunca
   para entrenar ni para seleccionar.
5. Sin submissions automaticas: cualquier candidate se entrega como artefacto y
   el envio lo decide Carlos.
6. V03 no se modifica ni se sobrescribe.
7. Sin emojis en codigo ni en documentos; respuestas en espanol.

## Fuera de alcance (cerrado en v2, v4 y D-030)

BM25, chi2, informacion mutua, log-odds, NB-SVM, POS, dependencias
sintacticas, targeting, emocion, stacking generico, chains, label powerset,
especialistas por etiqueta, hard-example mining y cualquier feature engineering
arbitrario ya cerrado. No se repiten: se consulta `PROJECT_STATE.md`,
`DECISIONS.md`, `FINAL_REPORT.md`, `reports/advanced_research_v4.md` y
`experiments.csv`.
