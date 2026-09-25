# Auditoria final - KAGGLE_TOXIC_2026_PUJ

Fecha: 2026-09-23. Auditor: ml-qa-engineer (Fase 14).
Alcance: leakage, reglas, reproducibilidad, submissions y entregables academicos.
Entorno de verificacion: `/home/carlos/venv/bin/python` (Python 3.14.7, sklearn
1.8.0, numpy 1.26.4, pandas 2.3.3). Los scripts de verificacion se conservan en
`/tmp/opencode/` (`audit_submissions.py`, `audit_leakage.py`, `audit_grupos.py`,
`audit_near_dup_full.py`, `audit_dups.py`, `audit_pseudo.py`,
`audit_modelo_final.py`, `audit_final_checks.py`) y la copia aislada de
re-ejecucion en `/tmp/opencode/repro_kaggle_toxic/`. Ningun archivo del
proyecto fue modificado por esta auditoria, salvo este informe.

---

## 1. LEAKAGE AUDIT

**Estado: APROBADO CON OBSERVACIONES**

### 1.1 Ajuste por pliegue (a) y vocabulario de test (b)

- Revision de codigo de los 11 scripts. El patron es consistente: el
  vectorizador/escalador/SVD se ajusta con `fit_transform` SOLO sobre los
  indices de entrenamiento del pliegue y se aplica con `transform` a la
  validacion (02:199-213; 04:312-320; 03:976-985; 05:668-675; 07:506-511;
  08:1259-1283; 09:668-696; 10:470-487; 11:887-897). Los `fit_transform` sobre
  `train["text"]` completo (p. ej. 02:257, 04:596) son el reajuste final con
  todo el train para generar la submission: es correcto y no incluye test.
- Verificacion empirica del artefacto final: el vocabulario de
  `models/mod03b_lr_union.joblib` es identico al de un `FeatureUnion` ajustado
  desde cero SOLO con train (word 4.024 + char_wb 23.407 = 27.431 features;
  `vocab_word == ref_word` y `vocab_char == ref_char` -> `True`). Si el test
  hubiera entrado en algun fit, la dimension/vocabulario no coincidiria.
- El test solo se usa con `transform`/`predict` (04:620-621). El script de DL
  no carga el test (0 ocurrencias de `read_test` en `11_dl_desde_cero.py`,
  confirmado por el propio docstring y por grep).

### 1.2 Corpus/etiquetas externas (c) y embeddings preentrenados (d)

- `data/external/lexicons/` esta vacio; no hay descargas de corpus externos;
  grep de `transformers|bert|gpt|llama|huggingface|sentencepiece|gensim|
  word2vec|fasttext|torch.hub|from_pretrained|load_state_dict` en todo el
  codigo: 0 usos reales (solo menciones en docstrings/comentarios). No hay
  HatEval ni etiquetas externas (D-002 respetada).
- spaCy se usa solo como anotador (POS/lema): no hay `.vector`, `.tensor` ni
  `similarity()` en ningun script (D-003 respetada).
- DL: `nn.Embedding` inicializado y entrenado desde cero, vocabularios
  construidos por pliegue (`vocabulario_caracteres`/`vocabulario_palabras`,
  `11_dl_desde_cero.py:547-587`); `load_state_dict` carga el mejor checkpoint
  propio (11:1225), no pesos preentrenados.

### 1.3 Seleccion de features/umbrales dentro de la validacion (e)

- Umbrales: se optimizan SOLO sobre probabilidades OOF (04:436-461), jamas
  sobre test. El propio artefacto declara la cota optimista
  (`advertencia_sesgo_optimista` en `reports/fase13a_refit_umbral_metricas.json`)
  y la verificacion anidada del revisor (+0.0014, 2/5 pliegues).
- Features densas (H11/H12/H14): lexicos inducidos por pliegue
  (`lexicos_inducidos_por_pliegue: true`), StandardScaler ajustado por pliegue.
  El modelo final no usa features densas.

### 1.4 Grupos de near-duplicado y fuga por duplicados (f, g) - recalculo independiente

- Recalculo completo e independiente: TF-IDF char_wb 4-4 + coseno sobre las
  2.885 filas (train+test). Resultado: 22 pares con coseno >= 0.85, **0 pares
  en grupos distintos** del archivo `data/processed/grupos_near_duplicados.csv`
  y 4 pares train-test (documentados). Los grupos guardados son conservadores
  (el grupo 112 agrupa textos de "JAJAJA+insulto" con difflib 0.29-0.60: sobre-
  agrupacion, nunca infra-agrupacion).
- Pliegues reconstruidos (`StratifiedGroupKFold(5, shuffle=True,
  random_state=42)`, grupos + estratificacion por HSxTRxAG):
  **0 grupos cruzan train/validacion** en los 5 pliegues y **0 textos de
  validacion tienen copia exacta en train** (normalizado).
- Duplicados exactos normalizados en todo el corpus: solo 1 grupo de 6 textos
  identicos dentro de test ("#¿nombre?", ids 22557/20477/24998/20081/22614/24635),
  ya documentado en Fase 0; **0 duplicados exactos train-test**.
- **OBSERVACION (Importante)**: el "control de robustez adicional con
  pseudo-autor (primera mencion) como grupo" declarado en D-004 **no se
  ejecuto** como validacion. Verificacion independiente: de los 17
  pseudo-autores con >=3 tweets (77 tweets, 4.4 % de train), **17/17 tienen
  tweets en mas de un pliegue** (p. ej. @youtube en los 5). Esto deja una via de
  optimismo acotada del OOF (memorizacion de estilo/tema del autor) que debe
  declararse como limitacion en el informe final; no invalida el resultado (el
  contraste v02 vs v03 en el LB es pareado y los umbrales son la unica palanca
  adoptada) pero contradice el protocolo declarado.

---

## 2. RULE AUDIT

**Estado: APROBADO**

Contraste con `docs/competition_rules.md`:

| Regla | Verificacion | Resultado |
|---|---|---|
| Sin transformers/LLM (RoBERTa/BERT) | grep en todo el codigo; 0 usos | CUMPLE |
| Sin embeddings preentrenados | sin gensim/torch.hub/from_pretrained; spaCy solo anotador; embeddings DL desde cero | CUMPLE |
| Vectorizacion entrenada desde cero | vocabulario del joblib identico al fit solo-train | CUMPLE |
| Lexicones documentados si se usan | ningun lexicon externo usado; lexicos inducidos solo de train y por pliegue; `reports/lexicon_catalog.csv` NO existe (consistente con no usar lexicones externos, D-007) | CUMPLE |
| El test no se usa para entrenamiento | solo `transform`/`predict`; DL no carga test | CUMPLE |
| Formato de submission | 8/8 archivos validados contra la plantilla oficial (columnas `id,HS,TR,AG`, 1.154 filas, ids identicos y en el mismo orden que `data/raw/submission.csv`, valores en {0,1}, sin nulos, ids unicos) | CUMPLE |
| Sin envios automaticos | 0 llamadas a submit/upload/Kaggle CLI en el codigo. Unica llamada a la API de Kaggle: `src/data.py:download_competition`, que descarga los CSV oficiales de datos (no envia submissions) | CUMPLE |

Nota: el uso de PyTorch esta permitido por el profesor ("incluyendo ...
PyTorch") y las redes se entrenaron desde cero con vocabulario por pliegue.

---

## 3. REPRODUCIBILITY AUDIT

**Estado: APROBADO**

Copia aislada en `/tmp/opencode/repro_kaggle_toxic/` (datos, src, scripts 02 y
04, experiments.csv y leaderboard originales). Ejecucion con
`/home/carlos/venv/bin/python`:

### 3.1 `experiments/02_baseline_tfidf_lr.py` (B1)

- `f1_media_columnas` OOF = **0.7587146098126788** -> identico al publicado
  (diferencia 0.0). Con post-proceso = 0.7577843027027181. Dimension 4.024.
- Submission v01 generada: **byte a byte identica** a la del proyecto
  (sha256 `0c02c8aad81a8458...`).

### 3.2 `experiments/04_refit_submissions_v02_v03.py` (MOD-03b + umbrales)

- `f1_media_columnas` OOF = 0.792659, referencia 0.7926588861781015,
  **diferencia +0.000000000** (tolerancia 1e-6).
- Umbrales reproducidos: **HS 0.44 / TR 0.51 / AG 0.46** (coinciden con D-014).
- Dimension 27.431; positivos en test v02 {HS 439, TR 338, AG 395} y v03
  {HS 506, TR 334, AG 422}, identicos a `experiments.csv` (S02/S03).
- Submissions v02 y v03 generadas: **byte a byte identicas** a las del proyecto
  (sha256 `228068c6ce0475e8...` y `7d7478f271e6edfd...`).
- Carga del artefacto final: `models/mod03b_lr_union.joblib` reproduce la
  submission v03 exactamente (0 filas distintas) y v02 con umbral 0.5 (0 filas
  distintas).

### 3.3 Semillas, dependencias y rutas

- Semillas: `random_state=42` en todos los scripts clasicos; DL con 42/43/44.
  Los resultados son deterministas (dos ejecuciones de 04 en la copia: misma
  metrica y mismas submissions).
- `requirements.txt`: los 9 paquetes principales verificados coinciden con el
  venv (scikit-learn 1.8.0, numpy 1.26.4, pandas 2.3.3, scipy 1.17.1, joblib
  1.5.3, nltk 3.9.4, spacy 3.8.13, optuna 4.8.0, lightgbm 4.6.0). torch queda
  fuera del pin (documentado como especifico de ROCm).
- Rutas: todos los scripts usan `PROJECT_ROOT = Path(__file__).resolve().parents[1]`
  y rutas relativas; la copia aislada funciono sin editar nada.
- Artefactos que cargan: 6/6 joblib (`b1_tfidf_lr`, `mod03b_lr_union`,
  `fase10_optuna_lr_union`, `fase13c_ens04_m2m3`, `fase13d_sonda_v06`,
  `fase13d_sonda_v07`), `models/dl_mlp.pt` (torch, state_dict propio), 5/5 JSON
  de metricas y 3/3 NPZ OOF (`oof_mod03b`, `oof_bases`, `oof_ens04_v04`) con
  formas correctas (1731 x 3).

---

## 4. SUBMISSION AUDIT

**Estado: APROBADO**

8/8 submissions en `submissions/` pasan todos los checks: columnas exactas
`id,HS,TR,AG`; 1.154 filas; ids identicos y en el mismo orden que
`data/raw/submission.csv` y que `toxic_test.csv`; ids unicos; sin nulos;
valores enteros en {0,1}. Detalle (sha256 truncado, tasa de positivos):

| Submission | sha256 | HS | TR | AG | LB |
|---|---|---|---|---|---|
| v00 plantilla | 0dfb4d07 | 1.000 | 1.000 | 1.000 | 0.51494 |
| v01 baseline | 0c02c8aa | 0.3735 | 0.2790 | 0.3102 | 0.75677 |
| v02 union u=0.5 | 228068c6 | 0.3804 | 0.2929 | 0.3423 | 0.77993 |
| **v03 union umbrales** | **7d7478f2** | 0.4385 | 0.2894 | 0.3657 | **0.78306** |
| v04 ensemble | c37cf1f5 | 0.4653 | 0.3016 | 0.4047 | 0.77510 |
| v05 optuna | a7f68a86 | 0.4099 | 0.2972 | 0.3579 | 0.77602 |
| v06 C=1.0 | fc943e68 | 0.4454 | 0.2825 | 0.3865 | 0.77227 |
| v07 min_df=5 | 4ba99f4d | 0.4515 | 0.2920 | 0.3536 | 0.77034 |

- `leaderboard_log.csv`: 8 filas; la aritmetica `diferencia = kaggle - local`
  es correcta en las 7 filas con modelo local (v00 no tiene local: es sonda).
- v00 tiene 1.153 combinaciones de etiqueta repetidas: esperado (todo 1,1,1),
  no es defecto.
- **No hay envios automaticos**: no existe ninguna llamada a la API/CLI de
  envio de Kaggle en el codigo (los 8 envios los hizo Carlos manualmente, como
  consta en `leaderboard_log.csv` y RUNLOG).

---

## 5. ACADEMIC AUDIT

**Estado: APROBADO CON OBSERVACIONES (entregables de Fase 14 pendientes)**

| Elemento | Evidencia | Estado |
|---|---|---|
| EDA explicado | `reports/eda_resumen.md` (253 lineas, 8 secciones, 26 figuras en `reports/figuras/`, 43 artefactos en `reports/eda/`) | OK |
| Hipotesis explicitas | `reports/hipotesis.md` H1-H10 completas (enunciado, literatura, falsacion); H11-H15 en `reports/error_analysis.md` (seccion 7, con experimento y criterio) | OK |
| Experimentos documentados, incluidos los fallidos | `experiments.csv` 50 filas: parte A (PRE-01..09 descartadas), post-proceso (MOD-04), representaciones (REP-01..04), modelos (MOD-02, MOD-03a_W12), densas (HIB-H11/H12/H14/UNION, B5-DENSE), ensembles (ENS-01..04, S04), robustez (R1/R2, S05-S07), DL (DL-02/03 por debajo de B1) | OK |
| Literatura citada | `reports/estado_arte.md` (45 referencias) + `references.bib` (553 lineas) | OK |
| Limitaciones documentadas | seccion 8 de `eda_resumen.md`; plateau D-014 (banda 0.77034-0.78306 = 0.99 SE con n=1154) y cota optimista local-LB (-0.013 a -0.029) | OK |
| **Notebook entregable (30 % de la nota)** | `notebooks/` esta **vacio**; D-011 planifica `notebooks/10_final_pipeline.ipynb` | **FALTA** |
| **FINAL_REPORT.md** | no existe en la raiz | **FALTA** |

### Brechas academicas detectadas

1. **[Importante] Notebook final**: no existe. Debe cubrir, con narrativa
   OBSERVACION -> HIPOTESIS -> METODO -> RESULTADO -> INTERPRETACION -> DECISION:
   parsing/encoding, EDA, near-duplicados, preprocesamiento (parte A), union
   word+char, LR C=2.0, umbrales, ensembles/densas/Optuna descartados, DL desde
   cero y el plateau. Es el 30 % de la nota y es el mayor riesgo restante.
2. **[Importante] FINAL_REPORT.md**: no existe. Debe integrar la decision D-014,
   la tabla del leaderboard, las limitaciones (incluida la brecha de
   pseudo-autores de la seccion 1.4) y el cierre de hipotesis.
3. **[Importante] Estado de hipotesis desactualizado**: `PROJECT_STATE.md`
   (linea 84) dice "Sin evaluar: H4, H5, H7, H10", pero H4 (REP-05), H7 (PRE-09)
   y H10 (ensembles) si se evaluaron despues. **H5 (sentimiento/emocion, E05) y
   E02b/E04 nunca se ejecutaron** (0 filas en `experiments.csv`) y no hay
   documento de cierre; el informe final debe declararlo como hipotesis no
   probada (limitacion de alcance, no de validez).
4. **[Sugerencia] `reports/lexicon_catalog.csv` ausente**: coherente con no usar
   lexicones externos, pero D-007 lo exige "si se usan"; conviene anotar en el
   informe final que no se uso ninguno.
5. **[Sugerencia] DL-01 sin sonda de LB**: DL-01 alcanzo OOF 0.7949 (media de 3
   semillas 0.7947 +/- 0.0011) frente a 0.7927 de MOD-03b. Por D-012/D-014 la
   mejora no es adoptable sin sonda de leaderboard y la fase competitiva esta
   cerrada; el informe final debe registrar explicitamente esta decision (no es
   una omision silenciosa) y su caracter de requisito metodologico.

---

## Veredicto: **GO** (con condiciones)

El pipeline y el modelo final estan **verificados**: sin fuga critica, sin
violacion de reglas, reproducibles bit a bit y con submissions en formato
valido. Las brechas encontradas son de documentacion/entrega, no de validez
cientifica. Condiciones para cerrar:

1. Crear `notebooks/10_final_pipeline.ipynb` y `FINAL_REPORT.md` antes del
   deadline (2026-10-19 23:00 UTC), incorporando las brechas de la seccion 5 y
   la limitacion de pseudo-autores.
2. Corregir las lineas desactualizadas de `PROJECT_STATE.md` (H4/H5/H7/H10) y
   registrar el cierre de H5/E02b/E04/E05 como "no ejecutadas".
3. No volver a sondear el leaderboard (D-014 vigente).

### Plan de rollback

- **Submission de respaldo: `submissions/toxic_submission_v03_lr_union_thr.csv`**
  (sha256 `7d7478f271e6edfd...`, LB 0.78306). Es la submission vigente y la de
  respaldo: si el notebook o el informe requirieran regenerar artefactos, v03 se
  reconstruye exactamente desde `models/mod03b_lr_union.joblib` (verificado: 0
  filas distintas) o re-ejecutando `experiments/04_refit_submissions_v02_v03.py`
  en copia aislada (verificado byte a byte).
- Fallback secundario: v02 (LB 0.77993, mismo modelo sin umbrales optimos,
  control metodologico). Ultimo recurso: v01 (LB 0.75677).
- Si se detectara corrupcion de datos/artefactos: re-ejecutar en orden
  `00_auditoria_datos.py` (regenera grupos), `04` (regenera modelo y
  submissions) y re-validar formato con el script de auditoria; no requiere
  reentrenar nada nuevo ni volver a enviar (v03 ya esta en el leaderboard).
- Rollback del sondeo: las variantes descartadas (v04-v07) permanecen en disco
  como evidencia; no deben reenviarse.
