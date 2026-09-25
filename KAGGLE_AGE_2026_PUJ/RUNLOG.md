# RUNLOG — KAGGLE_AGE_2026_PUJ

Bitácora cronológica. Una entrada por acción relevante: qué se hizo, con qué, resultado,
artefactos y decisiones tomadas.

---

## 2026-09-22 — Sesión 1 (Bloque 1, arranque)

| # | Acción | Comando / artefacto | Resultado |
|---|---|---|---|
| 1 | Creación de estructura del proyecto | `mkdir -p KAGGLE_AGE_2026_PUJ/{...}` | 9 carpetas + 2 subniveles de reports |
| 2 | Credencial Kaggle | `~/.kaggle/kaggle.json` con token `KGAT_…`, `chmod 600` | `600` verificado (`-rw-------`) |
| 3 | Escritura de memoria del proyecto | `.gitignore`, `README.md`, `DECISIONS.md` (D-001..D-007), `PROJECT_STATE.md`, `RUNLOG.md`, `experiments.csv` | Pendiente `requirements.txt` |
| 4 | Verificación GPU (read-only) | `rocm-smi --showpids --showuse --showmemuse` | Sin PIDs KFD; línea base ≈7% uso / 13% VRAM; `rocm-smi` en `/opt/rocm/bin/` |
| 5 | Módulos base | `src/config.py`, `src/data.py`, `src/gpu.py`, `src/cv.py`, `src/audit.py`, `src/submission.py`, `src/nbutil.py` | Pipeline CPU-first; guardián de GPU implementado (D-007) |
| 6 | Descarga real de datos | `python -m src.data --download --verify --save-interim` | ZIP sha256 `89a8d0586405e55ebbc4e04d2b4d2c68d7e9c4c7d890f4e6712ab8736ff9438e` (15.403.318 bytes); 0 registros malformados; parquet en `data/interim/` |
| 7 | Notebook de auditoría | `scripts/fase01_auditoria.py` → `notebooks/01_data_audit.ipynb` | Ejecutado en 130,8 s, 24 celdas, 0 errores; 4 figuras + 2 tablas + `reports/data_dictionary.csv` |

### INCIDENTE 2026-09-22 — Clustering de near-duplicados espurio (corregido)
- **Síntoma**: el control estratificado del baseline de memorización daba 17% de acierto en
  filas cubiertas (≈ azar), incoherente con la pureza observada.
- **Diagnóstico**: el agrupamiento por firma de los primeros 60 caracteres unía tweets
  distintos que comparten una apertura genérica; un clúster llegó a 190 tweets con las 6
  etiquetas mezcladas (pureza 67%).
- **Corrección**: `src/cv.build_dup_clusters` reescrito a igualdad de texto normalizado **o**
  Jaccard ≥ 0,8 sobre conjuntos de tokens con *blocking* por tokens raros (4,8 M
  comparaciones, 7 s, presupuesto no agotado).
- **Evidencia corregida**: 4,26% de test con near-duplicado en train; 92,8% de clústeres
  puros; LOO en clústeres multi: 75,7% (88,4% en pares).
- **Consecuencia**: la cuantificación del leakage pasa a ser interpretable (ver abajo).

### Resultados del gate A (parcial — datos)
- Memorización con CV **agrupado**: cobertura 0,0% (por construcción) → sin leakage.
- Memorización con CV **estratificado** (control): cobertura 3,99%, acierto en cubiertas
  79,5% → **macro-F1 "gratis" de 0,061** que el split aleatorio regalaría.
- Decisión D-003 confirmada con evidencia: validación agrupada por clúster en todas las fases.
- Ruido de etiqueta: 203 textos (643 filas, 0,38%) con etiquetas contradictorias.
- `id` no es proxy de clase (desviación máxima 1,25 pp entre deciles).

## 2026-09-22 — Sesión 1 (Fases 1-3)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 8 | Fase 1 — estado del arte (subagente `ml-architect`, background) | `reports/state_of_art.csv`, `references.bib`, `reports/tables/dataset_provenance.md` | 30 trabajos (29 verificados con DOI/URL); procedencia del corpus **irresoluble**; expectativas macro-F1 0,30-0,45 |
| 9 | Fase 2 — EDA cuantitativo | `scripts/fase02_eda.py` → `notebooks/02_eda.ipynb` (26 celdas, 82 s, 0 errores) | 38 features densas; ε² máximo 0,126 (`n_chars`); 4 figuras + 5 tablas; `features/dense_{train,test}.parquet` |
| 10 | Fase 3 — análisis lingüístico | `scripts/fase03_linguistic.py` → `notebooks/03_linguistic_analysis.ipynb` (17 celdas, 10 s, 0 errores) | términos distintivos, hashtags, preview de n-gramas; 12 hipótesis en `reports/hipotesis_gate_a.md` |

### INCIDENTE 2026-09-22 — Contaminación del conteo de emoticonos (corregido)
- **Síntoma**: `n_emoticon` marcaba 140.667 coincidencias, con `:/` como match dominante.
- **Diagnóstico**: el patrón capturaba el `:/` de `http://` en las URLs.
- **Corrección**: el conteo se hace sobre el texto sin URLs y con fronteras de palabra
  (`src/textometry.py`); caché de features densas invalidada y regenerada.

### HALLAZGO 2026-09-22 — El corpus es ASCII puro (afecta al modelado)
- 96 caracteres distintos y **un solo carácter no-ASCII** (`¿`, 88 apariciones); cero emojis,
  cero tildes, cero `ñ`. El texto fue transliterado a ASCII por el proceso de exportación.
- Consecuencias registradas en **D-009**: features de emoji descartadas por varianza nula;
  variantes de normalización de tildes/emojis son no-ops; refuerza la apuesta por char n-grams.

### Evidencia lingüística destacada (Fase 3)
- `13-17`: follow, justin, bieber, amo, please, love + hashtags de fandoms (mtvstars, btsbbmas).
- `25-34`: photo, xbox, jajaja, sorteo, friends.
- `65-xx`: venezuela, maduro, gobierno, pueblo, revolucion + efemerides, serviciopublico.
- char 4-grams distintivos: `foll` (13-17) vs `cion`, `ndro` (65-xx).

## 2026-09-22 — Sesión 1 (Etapa B: primera baseline real)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 11 | E-001: TF-IDF word(1-2)+char_wb(3-5) + LinearSVC, CV agrupado 5 folds | `scripts/experimento_e001_wordchar_linearsvc.py`, `experiments/E-001_resultados.json`, fila en `experiments.csv` | **macro-F1 = 0,5344 ± 0,0050** · weighted 0,5335 · accuracy 0,5343 · 336.493 features · 232 s |
| 12 | Verificación anti-leakage (disparada por superar el umbral 0,50) | `scripts/verificacion_e001_leakage.py`, `experiments/E-001_verificacion_leakage.json` | **Superada**: permutación 0,1672 (= azar 1/6); exactitud 0,5363 en el tramo limpio (similitud < 0,85, 99,5% de las filas); vectorizadores ajustados solo con train de cada fold |

### Lectura del resultado E-001
- El 0,5344 es **generalizable**, no memorización: la única vía de fuga estructural
  (duplicados) está cerrada por el agrupamiento y el control de permutación cae al azar.
- Queda **por encima del rango 0,30-0,45** que sugería la literatura (PAN 2014/2016), lo que
  se explica porque este corpus es más separable: clases balanceadas por diseño, longitud y
  registro monotónicos con la edad, y bloques temáticos muy distintos (fandoms vs política).
  El mejor sistema de PAN 2016 (5 clases, Twitter) alcanzó 0,5258 de accuracy: nuestro 0,5343
  es coherente con ese techo, no un artefacto.
- **Implicación para el LB**: el conjunto de test conserva un 4,26% de filas con
  near-duplicado en train, por lo que el score de Kaggle podría ser *ligeramente superior* al
  CV agrupado (la memorización de esos duplicados sí suma en el LB). Es esperable una brecha
  pequeña y positiva, no negativa.

## 2026-09-22 — Sesión 1 (Envío v01 y calibración del leaderboard)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 13 | Submission v01 (calibración): E-001 ajustado sobre todo el train | `scripts/submission_v01_calibracion.py` → `submissions/age_submission_v01.csv` (validado: columnas, orden, ids, 6 etiquetas, sin NaN) | 112.892 filas; distribución 17,9k-19,6k por clase; 59 s |
| 14 | **Subida manual de Carlos a Kaggle** | leaderboard público | **F1 = 0,55235** |

### Interpretación del score del leaderboard (cierra la calibración)
- **Brecha CV↔LB = +0,01791** (LB 0,55235 − CV agrupado 0,53444).
- **Reconciliación cuantitativa**: el test conserva 4,26% de filas con near-duplicado en train
  y el acierto en filas cubiertas es ≈0,795 (medido en train); con una exactitud base de
  0,534 el efecto esperado es 0,0426 × (0,795 − 0,534) ≈ **+0,0111**. La brecha observada
  (+0,0179) es del mismo orden y signo: el resto se explica porque el acierto en duplicados
  exactos es mayor que el promedio de clústeres y porque el modelo lineal sí memoriza
  duplicados exactos en test. **Conclusión: la brecha es estructural y esperada, no un error
  del protocolo.** El CV agrupado sigue siendo la estimación honesta para comparar modelos.
- **Métrica oficial (D-002)**: macro (0,53444) y weighted (0,53346) difieren en 0,001, muy por
  debajo de la brecha con el LB, así que un solo envío no permite distinguirlas. Se mantiene
  **macro-F1 como métrica de optimización** (estándar multiclase) reportando ambas; la
  distinción es inmaterial para las decisiones de modelado.
- **Punto de partida oficial del leaderboard: 0,55235** (E-001).

## 2026-09-22 — Sesión 1 (Etapa B completa: representaciones y baselines)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 15 | Matriz de representaciones E-002..E-008 (14 configuraciones, modelo fijo LinearSVC) | `scripts/experimentos_representaciones.py`, `experiments/representaciones_resultados.json`, `reports/tables/representaciones_comparativa.csv` | Mejor: **E-008 (char_wb 3-6 + canal de menciones) = 0,5557 ± 0,0052** |
| 16 | Baselines B0, B1, B5 (B2/B3/B4 salen de la matriz) | `scripts/baselines_b0_b5.py`, `experiments/baselines_resultados.json`, `reports/tables/baselines_comparativa.csv` | B0 0,0485 · B1 0,5015 · B5 0,3567 |
| 17 | Notebooks de documentación de la etapa | `notebooks/04_feature_engineering.ipynb`, `notebooks/05_baselines.ipynb` | 0 errores; incluyen la decisión D-010 y el catálogo |
| 18 | Catálogo de features (familias A-H) | `src/feature_catalog.py` → `reports/feature_catalog.csv` | 43 filas; grupos E/F declarados como pendientes y G descartado (D-009) |
| 19 | Submission v02 con la representación ganadora | `scripts/submission_v02_e008.py` → `submissions/age_submission_v02.csv` | Validada; **pendiente de subida manual** |

### HALLAZGO 2026-09-22 — La identidad de las cuentas mencionadas es una de las señales más fuertes
- Quitar **solo** las menciones: **−0,1251** de macro-F1 (0/5 folds a favor).
- Sustituirlas por `<user>`: −0,1237 (la pérdida no es el token, es la **identidad** de la cuenta).
- Quitar URLs: −0,0067 · quitar puntuación: −0,0080.
- Normalización completa: −0,1442; además deja 7.240 tweets vacíos (vs 104) y 13.137 con ≤1 token.
- **Canal dedicado de menciones** (TF-IDF sobre `@cuenta`): **+0,0144** sobre char solo (5/5 folds).
- Consecuencia: representación adoptada en **D-010**; el texto se mantiene crudo.

### Evidencia de la matriz de representaciones (macro-F1, CV agrupado)
`char_wb 3-6` 0,5413 · `char_wb 3-5` 0,5408 · `sin IDF` 0,5339 · `quitar URLs` 0,5277 ·
`quitar puntuación` 0,5264 · `char_wb 2-4` 0,5261 · `word 1-gram` 0,5025 · `word 1-2` 0,4971 ·
`word 1-3` 0,4941 · `marcadores <url>/<user>` 0,4107 · `quitar menciones` 0,4094 ·
`normalizado` 0,3902.

## 2026-09-22 — Sesión 1 (Envío v02 y Bloque 3: modelos, optimización y ensamble)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 20 | **Subida manual de v02 a Kaggle** | leaderboard | **F1 = 0,57018** (v01 = 0,55235) |
| 21 | E-009..E-012: modelos clásicos sobre la representación ganadora | `scripts/experimento_e009_modelos_clasicos.py`, `experiments/modelos_clasicos_resultados.json` | LinearSVC C=0.3 0,5624 · LogReg 0,5603 · SGD 0,5579 · ComplementNB 0,5370 |
| 22 | Fase 19: ajuste de decisión por clase (macro-F1) con protocolo LOFO | `scripts/fase19_optimizacion_macro_f1.py`, `experiments/fase19_pesos_por_clase.json` | **Resultado negativo**: ganancia +0,0006 → no se adopta (D-011) |
| 23 | E-013: barrido de regularización + márgenes OOF | `scripts/experimento_e013_barrido_c.py`, `reports/tables/barrido_c.csv`, `features/oof_E013_margenes.npy` | **C=0.2 óptimo: 0,5628 ± 0,0032** |
| 24 | Fase 20: ensamble (diversidad, voto suave, stacking LOFO) | `scripts/fase20_ensamble.py`, `experiments/fase20_ensamble.json` | **Stacking 0,5656** (gana al mejor individual en 5/5 folds); voto suave 0,5628 |
| 25 | Submission v03 con el ensamble | `scripts/submission_v03_ensamble.py` → `submissions/age_submission_v03.csv` | Validada; **pendiente de subida manual** |

### Interpretación del LB v02 = 0,57018
- Mejora sobre v01: **+0,01783** en el LB (CV: +0,0213).
- Brecha CV↔LB: +0,0179 (v01) y +0,0145 (v02) → estable, consistente con la memorización de
  near-duplicados del test (4,26% de las filas).
- Progreso acumulado del leaderboard: 0,55235 → 0,57018.

### Diversidad entre modelos (tasa de desacuerdo sobre OOF)
SVM↔LogReg 6,2% · SVM↔SGD 11,0% · SVM↔ComplementNB **25,2%** · LogReg↔ComplementNB 27,8% ·
SGD↔ComplementNB 29,7%. La diversidad de ComplementNB es la que habilita la ganancia del stacking.

## 2026-09-23 — Sesión 2 (v03 en el LB, análisis de errores y ensamble extendido)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 26 | **Subida manual de v03 a Kaggle** | leaderboard | **F1 = 0,58081** (v01 0,55235 · v02 0,57018) |
| 27 | Fase 17: análisis de errores del ensamble (OOF, LOFO) | `scripts/fase17_error_analysis.py`, `experiments/fase17_error_analysis.json`, 2 figuras + 5 tablas | Clases medias débiles (18-24 0,476; 25-34 0,449); confusión adyacente; longitud domina el error |
| 28 | E-014 (canal de hashtags) y E-015 (regresión ordinal) | `scripts/experimento_e014_e015_hipotesis_error.py` | hashtags 0,5644 · ordinal 0,3973 (débil solo, útil en el stack) |
| 29 | E-016 (char 3-6 con 700k features) y E-017 (word LogReg OOF) | `scripts/experimento_e016_e017.py` | **E-016 0,5658** (mejor individual: el tope de 300k truncaba vocabulario) · E-017 0,5015 |
| 30 | Fase 20b: ensamble extendido (7 combinaciones, 9 fuentes) | `scripts/fase20b_ensamble_extendido.py`, `experiments/fase20b_ensamble_extendido.json` | **0,5748 ± 0,0048** con 5 base + hashtags + ordinal + word (+0,0092 sobre la Fase 20) |
| 31 | Submission v04 con el ensamble extendido | `scripts/submission_v04_ensamble_extendido.py` → `submissions/age_submission_v04.csv` | Validada; **pendiente de subida manual** |

### Composición del ensamble extendido (9 fuentes → meta LogReg con StandardScaler)
`svm` (C=0.2, char 300k) · `svm700k` (C=0.2, char 700k) · `logreg` (OvR) · `sgd` · `cnb` ·
`hashtags` (SVM + canal de hashtags) · `ordinal` (Ridge sobre el punto medio) · `ordinal_soft` ·
`word` (LogReg sobre word 1-2). Comparativas LOFO: 4 base 0,5660 · +hashtags+ordinal 0,5682 ·
5 base 0,5703 · 5 base+hashtags+ordinal 0,5725 · **+word 0,5748**.

### Aportes marginales medidos (todos con LOFO)
- Vocabulario char de 700k: +0,0030 en el modelo individual (0,5628 → 0,5658).
- Canal de hashtags: +0,0016 individual (0,5628 → 0,5644) y +0,0019 en el ensamble.
- Modelo de palabra como miembro diverso: **+0,0023** en el ensamble (0,5725 → 0,5748).
- Señal ordinal (E-015): neutra sola, +0,0003 en el ensamble; se conserva por su bajo costo.

## 2026-09-23 — Sesión 2 (v04 en el LB y meta-modelo enriquecido)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 32 | **Subida manual de v04 a Kaggle** | leaderboard | **F1 = 0,59178** (progreso 0,55235 → 0,57018 → 0,58081 → 0,59178) |
| 33 | E-018: meta-modelo enriquecido con las 35 features densas | `scripts/experimento_e018_meta_enriquecido.py`, `experiments/e018_meta_enriquecido.json` | **0,5784 ± 0,0044** (+0,0036 sobre la Fase 20b); las interacciones no añaden nada más |
| 34 | Submission v05 (meta enriquecido) | `scripts/submission_v05_meta_enriquecido.py` → `submissions/age_submission_v05.csv` | Validada; **pendiente de subida manual** |

### Interpretación de E-018
Darle al meta-modelo la longitud, el retuit, las menciones y la textometría le permite
**condicionar la combinación según el tipo de tweet**, que es exactamente el sesgo de error
detectado en la Fase 17 (52,5% de error en el quintil más corto). Las interacciones explícitas
longitud×ordinal no añaden nada por encima de eso: el meta-modelo ya las aprende.

### Brechas CV↔LB observadas (cuatro mediciones)
+0,0179 · +0,0145 · +0,0152 · +0,0170 → media ≈ **+0,0162**, desviación ≈ 0,0015. El CV agrupado
es un estimador insesgado del LB con un desplazamiento estable y explicado (near-duplicados del
test). Esta es la evidencia que sostiene la selección de modelo por CV.

### Verificaciones previas (misma sesión, sin escritura a disco)
- Metadatos de la competencia vía API interna de Kaggle (id 167385, cierre 2026-10-19
  23:00 UTC, métrica F1 Score, `rowIdColumnName=id`, 10 envíos puntuados).
- Esquema real de los tres CSV y estadísticas de duplicados/leakage (detalle en
  `PROJECT_STATE.md` y en `01_data_audit.ipynb`).
- Estado de la GPU y de los procesos con handles en `/dev/kfd` (base para D-007).

## 2026-09-23 — Sesión 2 (Fase 14 en CPU, E-020 y submission v06)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 35 | **Subida manual de v05 a Kaggle** | leaderboard | **F1 = 0,59518** (progreso 0,55235 → 0,57018 → 0,58081 → 0,59178 → 0,59518) |
| 36 | Verificación de GPU (D-007) | `src/gpu.py`, `experiments/gpu_events.log` | **GPU OCUPADA** (uso 42% sostenido, sin PIDs de cómputo propios): se devuelve el control y la fase se ejecuta en CPU |
| 37 | E-019: MLP sobre densas y sobre densas+SVD(64) | `scripts/experimento_e019_modelos_densos.py`, `experiments/e019_modelos_densos.json` | 0,3381 y 0,4071 (muy por debajo de 0,5658 de los lineales dispersos); la SVD explica solo 7,8% de la varianza |
| 38 | E-020: ¿aporta el MLP al ensamble? | `scripts/experimento_e020_mlp_en_ensamble.py`, `experiments/e020_mlp_en_ensamble.json` | **0,5784 → 0,5851 (+0,0066)**: el MLP aporta por diversidad, no por exactitud |
| 39 | Submission v06 | `scripts/submission_v06_mlp_ensamble.py` → `submissions/age_submission_v06.csv` | Validada; **pendiente de subida manual** |

### Lectura de E-019/E-020
El resultado negativo como clasificador (0,4071) y positivo como miembro del ensamble (+0,0066)
es la evidencia más clara del proyecto sobre **dónde está el valor de un modelo no lineal** cuando
no se pueden usar embeddings: no en su exactitud, sino en la decorrelación de sus errores. La
caída de la SVD (7,8% de varianza en 64 componentes) explica por qué la vía densa pierde tanta
información en este corpus.

## 2026-09-24 — Sesión 3 (cierre: documentación, pipeline final y auditorías)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 40 | **Subida manual de v06 a Kaggle** | leaderboard | **F1 = 0,60273** (progreso acumulado +0,0504) |
| 41 | Refactor del modelo final a módulo reutilizable | `src/pipeline.py` | `pipeline.run()` ejecuta todo el modelo final en una llamada |
| 42 | Notebooks de documentación 06, 07 y 08 | `scripts/fase06_07_08_documentacion.py` | 0 errores; documentan modelos clásicos, densos y análisis de errores |
| 43 | Notebook 09 (pipeline final) | `notebooks/09_final_pipeline.ipynb` | Ejecutado (~9 min); **reproducibilidad 100,0000%** contra la submission v06 |
| 44 | Notebook 99 (entrega consolidada) | `notebooks/99_entrega_consolidado.ipynb` | Artefacto calificado (30% de la nota); 0 errores |
| 45 | Auditorías finales (Fase 23) | `scripts/fase23_auditorias.py` → `reports/auditorias.md` | **APROBADO** en leakage, reproducibilidad, submission y trazabilidad académica |
| 46 | Registro de los experimentos faltantes | `experiments.csv` | 30 experimentos registrados, todos con hipótesis y resultado |
| 47 | Informe final generado desde los artefactos | `scripts/fase_final_reporte.py` → `FINAL_REPORT.md` | 235 líneas con cifras inyectadas desde la evidencia (no transcritas a mano) |

### Estado del proyecto: COMPLETO
Todas las fases del plan (0-24) ejecutadas. Brecha CV↔LB calibrada en 6 mediciones
(+0,0165 ± 0,0014). El pipeline regenera la submission enviada con coincidencia exacta.

## 2026-09-24 — Sesión 3 (ronda de mejora: ordinal, centroides y cierre)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 48 | E-021: modelo ordinal acumulativo + centroides | `scripts/experimento_e021_ordinal_centroides.py` | ordinal **0,5527** · centroides 0,4017 (individuales) |
| 49 | E-022: ambos en el ensamble + meta LightGBM | `scripts/experimento_e022_ensamble_final.py` | **0,5860** con LogReg meta; LightGBM meta 0,5858 |
| 50 | E-023/E-024: especialistas por longitud | `scripts/experimento_e023_*.py`, `experimento_e024_*.py` | +0,0000 (el meta ya tiene la longitud) |
| 51 | E-025/E-026/E-027: BM25 y kNN sobre SVD | `scripts/experimento_e025_e026_vistas_nuevas.py` | BM25 0,4688 · kNN 0,3088 · aporte en el ensamble **−0,0002** |
| 52 | Pipeline actualizado a 12 miembros y submission v07 | `src/pipeline.py` → `submissions/age_submission_v07.csv` | CV **0,5860**; formato validado; 9 min |
| 53 | **Declaración de saturación** (D-016) | `DECISIONS.md` | últimas 4 rondas: +0,0003 · +0,0009 · +0,0000 · −0,0002 |

### Evidencia de saturación
Dentro de las restricciones (sin embeddings ni transformers) la representación dispersa sobre
texto crudo + canales léxicos + ensamble diverso alcanzó su techo. Las vías restantes exigirían
representaciones aprendidas (prohibidas) o un replanteamiento del problema (p. ej. supervisión
ordinal más fina), no más miembros en el ensamble.

## 2026-09-24 — Sesión 3 (ronda de ordinal fino y v08)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 54 | **Subida manual de v07 a Kaggle** | leaderboard | **F1 = 0,60372** (nuevo mejor; progreso +0,0514) |
| 55 | E-028: pares adyacentes + jerarquía gruesa | `scripts/experimento_e028_ordinal_fino.py` | fronteras: **0,7428** de precisión media; grupos: 0,7128 |
| 56 | Evaluación en el ensamble | `experiments/e028_en_ensamble.json` | pares **+0,0006** (0,5860 → 0,5866); grupos +0,0003 (redundantes) |
| 57 | Barrido del C del meta-modelo | — | C=1.0 ya era óptimo (0,5862/0,5866/0,5865 para 0,3/1/3) |
| 58 | Pipeline a 13 miembros y submission v08 | `src/pipeline.py` → `submissions/age_submission_v08.csv` | CV **0,5866**; formato validado; 9,4 min |

### Dificultad de las fronteras adyacentes (E-028a, precisión de un SVM entrenado solo con el par)
| Frontera | Precisión | Lectura |
|---|---|---|
| 18-24 \| 25-34 | 0,6856 | la más difícil: coincide con el par más confundido del análisis de errores |
| 25-34 \| 35-49 | 0,7112 | segunda más difícil |
| 13-17 \| 18-24 | 0,7153 | |
| 35-49 \| 50-64 | 0,7683 | |
| 50-64 \| 65-xx | 0,8099 | la más fácil: los extremos se separan mejor |

## 2026-09-24 — Sesión 4 (fase de investigación avanzada: segunda generación)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 59 | **Subida manual de v08 a Kaggle** | leaderboard | **F1 = 0,60376** (nuevo mejor) |
| 60 | Congelamiento del baseline | `experiments/baseline_frozen/` (11 archivos + hashes) | baseline real = v08 (el brief asumía v06; desviación documentada) |
| 61 | Gap analysis, matriz de cobertura y backlog | `reports/advanced_feature_gap_analysis.md`, `signal_coverage_matrix.csv`, `research_backlog.md` | 5 brechas de señal identificadas con evidencia |
| 62 | Features de segunda generación (160) | `src/advanced_features.py`, `scripts/advanced_02_features.py` | 87 locales + 73 supervisadas in-fold; tests de casos límite OK |
| 63 | Evaluación del aporte (LOFO) | `experiments/advanced_02_resultados.json` | **0,5866 → 0,5897 (+0,0031)**, positivo en 5/5 folds |
| 64 | Holdout congelado (15% de clústeres) | `experiments/advanced_03_holdout.json` | **0,5914 → 0,5966 (+0,0052)**: confirmado |
| 65 | Candidato `AGE_ADVANCED_V01` | `submissions/age_submission_v09.csv` (267 features del meta) | Formato validado; **pendiente de subida manual** |
| 66 | Informe de la fase y registro de envíos | `reports/advanced_research_report.md`, `reports/submissions_registry.csv` | 132 líneas; 9 envíos registrados |

## 2026-09-24 — Sesión 4 (confirmación del candidato en el leaderboard)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 67 | **Subida manual de v09 a Kaggle** | leaderboard | **F1 = 0,60807** (nuevo mejor; +0,0043 sobre v08) |

### Lectura del resultado
El candidato de segunda generación rindió en el leaderboard **mejor de lo que anticipaba su CV**:
- Δ CV (LOFO): +0,0031 · Δ holdout congelado: +0,0052 · **Δ leaderboard: +0,0043**.
- La mejora del holdout predijo mejor el comportamiento real que el CV de desarrollo, lo que
  respalda el diseño de validación por etapas del brief (§50): CV → holdout congelado → Kaggle.
- Octava medición de la brecha CV↔LB: +0,0184 (media acumulada +0,0169 ± 0,0013).

## 2026-09-24 — Sesión 5 (cierre de líneas avanzadas)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 68 | ADV-04: incertidumbre, acuerdo e interacciones | `scripts/advanced_04_incertidumbre.py` | **+0,0002** (línea cerrada); la std entre folds baja a 0,0036 |
| 69 | ADV-05: NB-SVM y selección de features | `scripts/advanced_05_nbsvm_seleccion.py` | NB-SVM 0,5641 · selección top-150 = 0,5896 (no aporta) · NB-SVM en el ensamble **+0,0007** |
| 70 | ADV-06: candidato v10 (273 features) | `scripts/advanced_06_submission_v10.py` → `submissions/age_submission_v10.csv` | CV **0,5904 ± 0,0047**; validado; **no enviado** (se reserva el último envío) |
| 71 | Política del último envío (D-018) | `DECISIONS.md` | 9 de ~10 envíos usados; el restante se reserva |

### Balance de la fase avanzada
- **Ganancia real**: v09 (features supervisadas de segunda generación) = **+0,0043 en el leaderboard**
  (0,60376 → 0,60807), la mayor de las últimas cinco rondas.
- **Líneas cerradas con evidencia**: incertidumbre (+0,0002), interacciones (+0,0001), selección de
  features (0,0000), NB-SVM como modelo (−0,0017 frente al SVM).
- **Líneas que requieren verificación de reglas**: retrieval train→test (no implementado).

## 2026-09-24 — Sesión 6 (investigación de la frontera 18-24|25-34)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 72 | ADV-07: diagnóstico y especialistas de frontera | `scripts/advanced_07_frontera.py`, `experiments/advanced_07_frontera.json` | Frontera medida: ensamble **0,486** · par-SVM binario 0,6856 · especialista con compuerta **−0,0036** · enrutamiento por margen **+0,0000** |
| 73 | Análisis de términos y ejemplos de la frontera | (diagnóstico en el informe) | `18-24` = métricas sociales y emoción; `25-34` = temas informativos/deportivos; **ambos retuitean el mismo contenido genérico** |
| 74 | Decisión de cierre (D-019) | `DECISIONS.md` | Techo de **información**, no de modelado: no se gasta envío |

### Conclusión de la investigación de frontera
El error en `18-24|25-34` no se corrige con más modelado: tres arquitecturas de especialización
fallaron, y en el umbral donde el especialista sería confiable **coincide con el ensamble** (0
correcciones, 0 daños en 3.571 activaciones). El perfil del error muestra que la señal que falta es
de **interacción social** (menciones y hashtags diferencian; los enlaces no), y el análisis de
términos explica por qué: las dos cohortes comparten tema y difieren poco en registro.

## 2026-09-24 — Sesión 7 (fase avanzada v2: diagnóstico, ponderación y candidato v11)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 75 | Congelar v09/v10 y gap analysis v2 | `experiments/baseline_v09_frozen/`, `reports/advanced_gap_analysis_v2.md`, `research_backlog_v2.md` | 16 archivos con hash; 22 fenómenos auditados |
| 76 | **L1 — intersección de errores** | `reports/error_intersection_analysis.csv`, `experiments/phase1_diagnostico.json` | **25,94%** de filas rescatables; rescatadores = modelos diversos |
| 77 | **L5 — error por tipo y longitud** | `reports/tables/phase1_error_por_tipo.csv`, `phase1_error_por_longitud.csv` | `solo_url` 61,1% vs `mencion_alta` 19,2%; longitud 54,5% → 35,1% |
| 78 | **L6 — ponderación condicional** | `experiments/phase2_ponderacion.json` | **MLP meta 0,5967 (+0,0061, 5/5 folds)**; segmentación dura negativa |
| 79 | Corrección de dos defectos | `scripts/phase3b_recorregido.py`, `src/advanced_features.py` | 0 columnas degeneradas; alineación del meta corregida |
| 80 | Confirmación y candidato v11 | `experiments/phase3_holdout_v11.json` → `submissions/age_submission_v11.csv` | Holdout **+0,0043** (> criterio +0,003); distribución sana |

## 2026-09-24 — Sesión 8 (v11 en el leaderboard)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 81 | **Subida manual de v11 a Kaggle** | leaderboard | **F1 = 0,61744** (nuevo mejor; +0,0094 sobre v09) |

### Lectura del resultado
- Δ leaderboard **+0,0094** > Δ CV **+0,0069** > Δ holdout **+0,0058**. La ganancia real superó a
  ambos estimadores internos, algo que no había ocurrido en las rondas previas (donde el LB quedaba
  entre el CV y el holdout).
- Novena medición de la brecha CV↔LB: **+0,0208**, por encima del rango anterior
  (+0,0145 a +0,0184). Interpretación plausible: el combinador no lineal explota mejor la
  memorización de near-duplicados del test, que es precisamente el componente que el CV agrupado
  bloquea por diseño. **No se cambia la metodología** (el CV sigue siendo el estimador honesto para
  comparar modelos), pero se registra la observación.
- Progreso acumulado: 0,55235 → **0,61744** (**+0,0651**).

## 2026-09-24 — Sesión 9 (L2, L3, L7 y candidato v12)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 82 | L2 — residualización por longitud | `scripts/phase5_residuales.py` | **−0,0019** (2/5 folds): el combinador ya aprende esa estructura. Cerrada |
| 83 | L3 — contenido vs estilo | `scripts/phase6_contenido_estilo.py` | estilo solo **0,2793** · contenido solo **0,5537** · aporte +0,0008. **Hipótesis refutada**: manda el contenido |
| 84 | L7 — NB-SVM de palabra + promedio de combinadores | `scripts/phase7_nbsvm_canal_meta.py` | NB palabra 0,5254 (+0,0009 al meta); **promedio MLP+LogReg 0,6005 (+0,0027)** |
| 85 | Confirmación y candidato v12 | `scripts/phase8_holdout_v12.py` → `submissions/age_submission_v12.csv` | Holdout **0,6077** (MLP 0,6034) → Δ +0,0043 **confirmado** |

### Hallazgo de la fase
La palanca no son las features: es el **combinador**. En esta fase, todas las líneas de features
nuevas quedaron en o por debajo del ruido, mientras que dos intervenciones sobre el combinador
(MLP como meta-modelo, y ahora el promedio de dos combinadores diversos) dieron las únicas
ganancias reales y confirmadas. La diversidad de sesgos paga en el segundo nivel igual que en el
primero.

## 2026-09-24 — Sesión 10 (v12 en el leaderboard y cierre de documentación)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 86 | **Subida manual de v12 a Kaggle** | leaderboard | **F1 = 0,62182** (+0,0044 sobre v11) |

### Lectura
- Δ leaderboard **+0,0044** ≈ Δ holdout **+0,0043** > Δ CV **+0,0038**. La confirmación en el holdout
  volvió a predecir el comportamiento real con precisión.
- Décima medición de la brecha CV↔LB: **+0,0213**; media acumulada **+0,0177 ± 0,0023**.
- Progreso acumulado del proyecto: 0,55235 → **0,62182** (**+0,0695**).
- Documentación sincronizada: `PROJECT_STATE.md`, `DECISIONS.md`, `FINAL_REPORT.md` regenerado desde
  los artefactos y `experiments.csv` (44 experimentos).

| 87 | Fase 9: tercer combinador | `scripts/phase9_combinadores.py` | MLP+LogReg 0,6005 óptimo; +Ridge 0,5999; +SGD 0,5676 → **línea cerrada** (D-025) |

## 2026-09-25 — Sesión 11 (las dos vías autorizadas, cerradas con evidencia)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 88 | Vía A: retrieval train→test (inferencia) | `scripts/phase10_dos_vias.py` | Cobertura de clúster in-fold **0,00%** (el CV agrupado la elimina); duplicados exactos 1,36% con **acierto de etiqueta 61,5%**; sobrescritura dura **−0,0009** → cerrada |
| 89 | Vía B: POS con spaCy `es_core_news_md` | `features/pos_features.parquet` | **−0,0032** → cerrada |

### Hallazgo de la sesión
El **techo del retrieval** en este corpus no lo fija la cobertura (4,26% de near-duplicados) sino la
**consistencia de etiqueta entre duplicados**: los tweets idénticos solo comparten etiqueta en el
61,5% de los casos, por debajo de la precisión del modelo. Memorizar contenido idéntico con
etiquetas contradictorias introduce más error del que corrige. Es un resultado sobre el **corpus**
(anotación con ruido), no sobre la técnica.

## 2026-09-25 — Sesión 12 (hipótesis del ruido de etiqueta)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 90 | PH11: exclusión de filas ruidosas del entrenamiento | `scripts/phase11_ruido_etiqueta.py` | contradictorias (643) **+0,0001** · duplicadas (2.123) **+0,0016** (3/5 folds) → mejora real pero bajo el umbral de +0,003 |

### Episodio metodológico
La primera ejecución usó la normalización agresiva (`normalize_for_dedup`), que colapsa miles de
tweets cortos distintos al mismo texto: los conflictos pasaron de **643 a 12.989 filas** y el
resultado salió −0,0042. Se detectó el artefacto, se corrigió la definición al criterio del audit
(minúsculas + espacios) y se repitió. **La elección de la normalización cambia el diagnóstico del
ruido** — queda registrado como lección.

## 2026-09-25 — Sesión 13 (entrega: notebook final, commit, PR y merge)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 91 | Notebook consolidado regenerado con las 12 fases | `scripts/fase12_notebook_final.py` → `notebooks/99_entrega_consolidado.ipynb` | 27 celdas, **0 errores**; incluye aserciones que verifican cada cifra contra los artefactos |
| 92 | Informe final regenerado | `FINAL_REPORT.md` | 252 líneas con las fases nuevas y las 15 líneas cerradas |
| 93 | Auditorías re-ejecutadas | `reports/auditorias.md` | **APROBADO** (leakage, reproducibilidad, submission, trazabilidad) |
| 94 | Commit y PR | rama `kaggle-age-2026-puj`, commit `3a518ac` | 255 archivos, 2,01 M líneas |
| 95 | Merge a `main` | PR **#26** fusionado (merge `27238e7`) | datos/features/modelos excluidos por `.gitignore` |

### Seguridad
El token de GitHub compartido en el chat **debe revocarse** tras esta sesión; no quedó escrito en
ningún archivo del repositorio (verificado con `grep`).

## 2026-09-25 — Sesión 14 (búsqueda de señal residual: FASE A, auditoría del estado)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 96 | Auditoría del estado real (FASE A) | `scripts/phase12_auditoria_estado.py` -> `experiments/phase12_auditoria_estado.json`, `reports/auditoria_estado_fase12.md` | 8 secciones, 607 s |
| 97 | Envíos verificados por API de Kaggle (solo lectura) | `experiments/phase12_auditoria_estado.json` -> `envios_kaggle` | **15 registrados, 12 puntuados** |

### Verificaciones de la FASE A (todas superadas)
- Alineación: etiquetas == OOF; 282.228 filas de clústeres con normalizado coincidente al 100,0000%.
- Matriz del meta de train **reconstruida exactamente** (169.336 x 273; diferencia máxima absoluta 0,0).
- Firmas de bloque train/test: 273 columnas; invariante exp-suma = 1 en los 8 bloques logarítmicos; cero columnas de varianza nula; cero NaN/Inf.
- CV LOFO del v12 reproducido: **0,6004 ± 0,0042** (por fold 0,6007 / 0,6050 / 0,6032 / 0,6004 / 0,5928, idénticos a `phase11`).
- **End-to-end**: el modelo v11 reconstruido sobre las matrices cacheadas reproduce `age_submission_v11.csv` con **acuerdo 100,0000%** -> la alineación semántica del lado test (riesgo tipo D-021) queda verificada.
- Holdout congelado: determinista (24.602 clústeres, 25.204 filas, 14,884%).
- Submission v12: 8/8 checks formales; distribución no degenerada.
- Envíos: 15 registrados (12 COMPLETE con puntaje, 3 ERROR de plataforma), incluido el degenerado 0,39563 del incidente D-021. D-004 (~10 envíos) queda obsoleto.

## 2026-09-25 — Sesión 15 (búsqueda de señal residual: FASE B, errores y desacuerdo)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 98 | OOF del v12 cacheado | `features/oof_ensemble_v12.npy` | macro-F1 0,6005 reproduciendo la referencia |
| 99 | Análisis de errores y desacuerdo (FASE B) | `scripts/phase13_error_desacuerdo.py` -> `experiments/phase13_error_desacuerdo.json`, `reports/phase13_error_desacuerdo.md` | 640 s; 5 tablas + figura |
| 100 | Corrección de bug de nombres `.npy.npy` en FASE B | `scripts/phase13_error_desacuerdo.py`, `scripts/phase15_calibracion_scores.py` | detectado por ejecución; corregido y re-ejecutado |

### Hallazgos de la FASE B
- **13,87%** de las filas: ningún miembro acierta (techo de la clase de modelos actual). **26,70%**: el ensamble falla y al menos un miembro acierta (rescatable).
- Gradiente de consenso monótono: quintil más bajo error 60,2% (rescatable 47,0%) -> quintil más alto 18,1% (6,3%).
- Familias que rescatan cuando el ensamble falla: ordinal 19,96%, prototipos 19,87%, denso 19,27%; el denso es el que más veces es **el único** que acierta (3,44%).
- Fronteras: `18-24|25-34` 42,48% del error total (exactitud dentro de la pareja 0,4963) y `25-34|35-49` 42,06% (0,4974).
- Peores segmentos: `solo_url` 59,9% de error, `muy_corto` 50,8%, sin menciones 51,0%.
- F1 por clase: 13-17 0,671 | 18-24 0,506 | 25-34 0,486 | 35-49 0,559 | 50-64 0,655 | 65-xx 0,726.

## 2026-09-25 — Sesión 16 (búsqueda de señal residual: FASE C, consenso y desacuerdo)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 101 | Bloques de consenso/desacuerdo/gating (FASE C) | `scripts/phase14_consenso_familias.py` -> `experiments/phase14_consenso_familias.json` | 46 features nuevas, 3 variantes, LOFO |
| 102 | Decisión D-028 | `DECISIONS.md` | Familia cerrada con evidencia |

### Resultado de la FASE C (referencia v12 = 0,6004 ± 0,0042)
| Variante | macro-F1 | Δ | folds positivos |
|---|---|---|---|
| +C1 acuerdo/entropía por familia (11) | 0,5986 | −0,0018 | 1/5 |
| +C1+C2 divergencias JS (18) | **0,6002** | −0,0002 | 3/5 |
| +C1+C2+C3 gating contextual (46) | 0,5998 | −0,0006 | 3/5 |

**Interpretación**: el consenso de votos es una función determinista de los scores de los miembros que
el combinador ya recibe; el MLP aproxima esas agregaciones por sí solo. Coherente con ADV-04 (+0,0002,
cerrado) y con D-020 (las ganancias del segundo nivel vinieron de diversidad de sesgos y de una vista
nueva, no de re-expresar los mismos scores).

## 2026-09-25 — Sesión 17 (búsqueda de señal residual: fases D, E, F, G y multivista)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 103 | Calibración y blending en logits (FASE D) | `scripts/phase15_calibracion_scores.py` | mejor +0,0003 (3/5) -> nulo (D-029) |
| 104 | Diversidad de subconjuntos de features (FASE D5) | `scripts/phase15b_diversidad_subsets.py` | 3 vías -0,0017; 4 vías -0,0028; producción reproducida 0,6004 exacta |
| 105 | Estructura v2: retuit, hashtag, extremos (FASE E) | `scripts/phase17_estructura_v2.py --bloque libres` | 25 features nuevas: +0,0002 (2/5) -> nulo (D-030) |
| 106 | Limpieza selectiva (FASE F) | `scripts/phase18_limpieza_selectiva.py --variantes F2,F3` | F2 -0,0001; F3 +0,0009 (3/5) -> cerrada (D-031) |
| 107 | Multivista con puerta de diversidad (§13) | `scripts/phase21_vista_estructural.py` | **estructura +0,0018 (4/5)**; extremos +0,0000; ambas pasan la puerta (rescate 18,2% y 18,8%) |
| 108 | Refuerzo de la vista estructural | `scripts/phase22_estructura_ampliada.py` | en curso |

### Bugs corregidos en la sesión
- `phase13`/`phase15`: nombres `.npy.npy` al usar `MEMBER_FILES` (extensión incluida).
- `phase16`: `probabilidad()` con extensión duplicada; `bloque` residual tras refactorizar G1.
- `phase18`/`phase19`: concatenación de cadenas numpy (`np.char.add`) y `representantes()`
  vectorizado (groupby por grupo era inviable: 120k grupos x 5 folds).
- `phase21`: `f1_score` con etiquetas mixtas (índices frente a cadenas).
- Nota metodológica: la concurrencia de procesos **no altera** los resultados (la referencia de
  producción se reprodujo idéntica dos veces bajo carga: 0,6007/0,6050/0,6032/0,6004/0,5928).

## 2026-09-25 — Sesión 18 (búsqueda de señal residual: multivista, holdout y candidato marginal)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 109 | Vista estructural y de extremos (§13) | `scripts/phase21_vista_estructural.py` | estructura **+0,0018 (4/5 folds)**, rescate 18,17%; extremos +0,0000 |
| 110 | Refuerzo de la vista (alfabeto ampliado) | `scripts/phase22_estructura_ampliada.py` | V1 +0,0005; V2 -0,0003; V3 +0,0006 -> la vista simple es el óptimo |
| 111 | Señal ordinal in-fold y greedy de miembros | `scripts/phase16_ordinal_greedy.py --bloques g1g2` | G1 -0,0003 (1/5); G2 -0,0234 |
| 112 | **Holdout congelado de la vista estructural** | `scripts/phase19_holdout_variante.py --variante VistaEstructura` | base 0,6077 (exacto) -> vista **0,6108 (+0,0031)** |
| 113 | Candidato marginal (archivo, sin enviar) | `scripts/phase23_submission_candidato_estructura.py` | 112.892 filas, 8/8 checks; cambia 10,47% de las filas frente a la v12 |
| 114 | Control de integridad del candidato | `scripts/phase24_diagnostico_matriz_test.py` | la reconstrucción del v12 desde la caché dio 93,88% -> **en diagnóstico** |

### Hallazgo principal de la fase
La **vista estructural** (secuencia de tipos de token, sin contenido) es la única fuente de señal nueva:
F1 individual 0,2285, rescate del 18,17% de los errores del v12 (acuerdo 27,96%), **+0,0018 en CV
(4/5 folds)** y **+0,0031 en el holdout congelado**. No cumple el gate pre-registrado de CV (+0,003),
por lo que **no se adopta** y no se envía nada (D-032, D-033).

## 2026-09-25 — Sesión 19 (cierre: integridad del artefacto, informe consolidado y veredicto)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 115 | Diagnóstico de la matriz de test (¿desalineación?) | `scripts/phase24_diagnostico_matriz_test.py` | **sin desalineación**: todos los bloques coinciden (max|Δ| 1,5e-05) |
| 116 | Control pareado en un proceso (fresca vs caché vs archivo) | `scripts/phase25_submission_fresca.py` | fresca vs caché **100,0000%**; ambas vs el archivo subido **93,8800%** |
| 117 | Estabilidad del ajuste final | `scripts/phase26_estabilidad_ajuste.py` | **100,0000%** de acuerdo entre dos ajustes idénticos |
| 118 | Candidato regenerado con la matriz fresca | `features/meta_features_test_v12_fresca.npy`, `submissions/age_submission_v13_candidato_estructura_NO_ADOPTADO.csv` | 112.892 filas, 8/8 checks |
| 119 | Informe consolidado de la fase | `scripts/phase20_informe_fase_residual.py` -> `reports/fase_residual_informe.md` | tabla de variantes + veredicto |

### Hallazgo de integridad (D-034)
La submission v12 enviada **no se reconstruye al 100%** desde el repositorio (93,88%), pero el
diagnóstico demuestra que **no** es por desalineación de matrices (fresca vs caché = 100,0000%) ni por
inestabilidad del ajuste (dos ajustes idénticos = 100,0000%) ni por falta de reproducibilidad del
protocolo (el holdout base se re-ajustó en 0,6077, idéntico a la fase 8). Es una diferencia entre el
artefacto subido y el estado committeado: **problema de trazabilidad, no de metodología**. Todas las
comparaciones de la fase son pareadas en el mismo proceso y quedan válidas.

## 2026-09-25 — Sesión 20 (cierre de la entrega: v12 reproducible, notebook e informe)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 120 | Artefacto v12 reproducible (cierre de D-034) | `scripts/phase27_reproduccion_v12.py` -> `submissions/age_submission_v12_reproducible.csv` | 112.892 filas, 8/8 checks; 93,88% de coincidencia con el archivo subido |
| 121 | Notebook consolidado con la Parte II | `scripts/phase28_notebook_final_v2.py` -> `notebooks/99_entrega_consolidado.ipynb` | **46 celdas, 0 errores**; aserciones contra 17 artefactos de la fase |
| 122 | Informe final ampliado (Parte II) | `scripts/phase29_amplia_reporte_final.py` -> `FINAL_REPORT.md` | 347 líneas (15-20: auditoría, errores, 27 variantes, vista estructural, integridad, cierre) |
| 123 | Commit local | rama `kaggle-age-2026-residual` (desde `main`), commit `03b95de` | 55 archivos; datos/features/modelos fuera por `.gitignore`; pendiente de push (requiere token nuevo) |

### Estado de la entrega
- Modelo final **sin cambios** (v12): CV 0,6004 · holdout 0,6077 · leaderboard 0,62182.
- **Nada enviado a Kaggle** en esta fase; el candidato marginal y el v12 reproducible quedan en
  `submissions/` con nombres explícitos.
- Pendiente: push/PR/merge a `main` con un token nuevo (el anterior quedó expuesto y debe estar revocado).
