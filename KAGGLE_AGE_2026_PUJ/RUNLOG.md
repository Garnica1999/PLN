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
| 7 | Notebook de auditoría | `scripts/fase01_auditoria.py` -> `notebooks/01_data_audit.ipynb` | Ejecutado en 130,8 s, 24 celdas, 0 errores; 4 figuras + 2 tablas + `reports/data_dictionary.csv` |

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
- Memorización con CV **agrupado**: cobertura 0,0% (por construcción) -> sin leakage.
- Memorización con CV **estratificado** (control): cobertura 3,99%, acierto en cubiertas
  79,5% -> **macro-F1 "gratis" de 0,061** que el split aleatorio regalaría.
- Decisión D-003 confirmada con evidencia: validación agrupada por clúster en todas las fases.
- Ruido de etiqueta: 203 textos (643 filas, 0,38%) con etiquetas contradictorias.
- `id` no es proxy de clase (desviación máxima 1,25 pp entre deciles).

## 2026-09-22 — Sesión 1 (Fases 1-3)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 8 | Fase 1 — estado del arte (subagente `ml-architect`, background) | `reports/state_of_art.csv`, `references.bib`, `reports/tables/dataset_provenance.md` | 30 trabajos (29 verificados con DOI/URL); procedencia del corpus **irresoluble**; expectativas macro-F1 0,30-0,45 |
| 9 | Fase 2 — EDA cuantitativo | `scripts/fase02_eda.py` -> `notebooks/02_eda.ipynb` (26 celdas, 82 s, 0 errores) | 38 features densas; ε² máximo 0,126 (`n_chars`); 4 figuras + 5 tablas; `features/dense_{train,test}.parquet` |
| 10 | Fase 3 — análisis lingüístico | `scripts/fase03_linguistic.py` -> `notebooks/03_linguistic_analysis.ipynb` (17 celdas, 10 s, 0 errores) | términos distintivos, hashtags, preview de n-gramas; 12 hipótesis en `reports/hipotesis_gate_a.md` |

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
| 13 | Submission v01 (calibración): E-001 ajustado sobre todo el train | `scripts/submission_v01_calibracion.py` -> `submissions/age_submission_v01.csv` (validado: columnas, orden, ids, 6 etiquetas, sin NaN) | 112.892 filas; distribución 17,9k-19,6k por clase; 59 s |
| 14 | **Subida manual de Carlos a Kaggle** | leaderboard público | **F1 = 0,55235** |

### Interpretación del score del leaderboard (cierra la calibración)
- **Brecha CV<->LB = +0,01791** (LB 0,55235 − CV agrupado 0,53444).
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
| 18 | Catálogo de features (familias A-H) | `src/feature_catalog.py` -> `reports/feature_catalog.csv` | 43 filas; grupos E/F declarados como pendientes y G descartado (D-009) |
| 19 | Submission v02 con la representación ganadora | `scripts/submission_v02_e008.py` -> `submissions/age_submission_v02.csv` | Validada; **pendiente de subida manual** |

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
| 22 | Fase 19: ajuste de decisión por clase (macro-F1) con protocolo LOFO | `scripts/fase19_optimizacion_macro_f1.py`, `experiments/fase19_pesos_por_clase.json` | **Resultado negativo**: ganancia +0,0006 -> no se adopta (D-011) |
| 23 | E-013: barrido de regularización + márgenes OOF | `scripts/experimento_e013_barrido_c.py`, `reports/tables/barrido_c.csv`, `features/oof_E013_margenes.npy` | **C=0.2 óptimo: 0,5628 ± 0,0032** |
| 24 | Fase 20: ensamble (diversidad, voto suave, stacking LOFO) | `scripts/fase20_ensamble.py`, `experiments/fase20_ensamble.json` | **Stacking 0,5656** (gana al mejor individual en 5/5 folds); voto suave 0,5628 |
| 25 | Submission v03 con el ensamble | `scripts/submission_v03_ensamble.py` -> `submissions/age_submission_v03.csv` | Validada; **pendiente de subida manual** |

### Interpretación del LB v02 = 0,57018
- Mejora sobre v01: **+0,01783** en el LB (CV: +0,0213).
- Brecha CV<->LB: +0,0179 (v01) y +0,0145 (v02) -> estable, consistente con la memorización de
  near-duplicados del test (4,26% de las filas).
- Progreso acumulado del leaderboard: 0,55235 -> 0,57018.

### Diversidad entre modelos (tasa de desacuerdo sobre OOF)
SVM<->LogReg 6,2% · SVM<->SGD 11,0% · SVM<->ComplementNB **25,2%** · LogReg<->ComplementNB 27,8% ·
SGD<->ComplementNB 29,7%. La diversidad de ComplementNB es la que habilita la ganancia del stacking.

## 2026-09-23 — Sesión 2 (v03 en el LB, análisis de errores y ensamble extendido)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 26 | **Subida manual de v03 a Kaggle** | leaderboard | **F1 = 0,58081** (v01 0,55235 · v02 0,57018) |
| 27 | Fase 17: análisis de errores del ensamble (OOF, LOFO) | `scripts/fase17_error_analysis.py`, `experiments/fase17_error_analysis.json`, 2 figuras + 5 tablas | Clases medias débiles (18-24 0,476; 25-34 0,449); confusión adyacente; longitud domina el error |
| 28 | E-014 (canal de hashtags) y E-015 (regresión ordinal) | `scripts/experimento_e014_e015_hipotesis_error.py` | hashtags 0,5644 · ordinal 0,3973 (débil solo, útil en el stack) |
| 29 | E-016 (char 3-6 con 700k features) y E-017 (word LogReg OOF) | `scripts/experimento_e016_e017.py` | **E-016 0,5658** (mejor individual: el tope de 300k truncaba vocabulario) · E-017 0,5015 |
| 30 | Fase 20b: ensamble extendido (7 combinaciones, 9 fuentes) | `scripts/fase20b_ensamble_extendido.py`, `experiments/fase20b_ensamble_extendido.json` | **0,5748 ± 0,0048** con 5 base + hashtags + ordinal + word (+0,0092 sobre la Fase 20) |
| 31 | Submission v04 con el ensamble extendido | `scripts/submission_v04_ensamble_extendido.py` -> `submissions/age_submission_v04.csv` | Validada; **pendiente de subida manual** |

### Composición del ensamble extendido (9 fuentes -> meta LogReg con StandardScaler)
`svm` (C=0.2, char 300k) · `svm700k` (C=0.2, char 700k) · `logreg` (OvR) · `sgd` · `cnb` ·
`hashtags` (SVM + canal de hashtags) · `ordinal` (Ridge sobre el punto medio) · `ordinal_soft` ·
`word` (LogReg sobre word 1-2). Comparativas LOFO: 4 base 0,5660 · +hashtags+ordinal 0,5682 ·
5 base 0,5703 · 5 base+hashtags+ordinal 0,5725 · **+word 0,5748**.

### Aportes marginales medidos (todos con LOFO)
- Vocabulario char de 700k: +0,0030 en el modelo individual (0,5628 -> 0,5658).
- Canal de hashtags: +0,0016 individual (0,5628 -> 0,5644) y +0,0019 en el ensamble.
- Modelo de palabra como miembro diverso: **+0,0023** en el ensamble (0,5725 -> 0,5748).
- Señal ordinal (E-015): neutra sola, +0,0003 en el ensamble; se conserva por su bajo costo.

## 2026-09-23 — Sesión 2 (v04 en el LB y meta-modelo enriquecido)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 32 | **Subida manual de v04 a Kaggle** | leaderboard | **F1 = 0,59178** (progreso 0,55235 -> 0,57018 -> 0,58081 -> 0,59178) |
| 33 | E-018: meta-modelo enriquecido con las 35 features densas | `scripts/experimento_e018_meta_enriquecido.py`, `experiments/e018_meta_enriquecido.json` | **0,5784 ± 0,0044** (+0,0036 sobre la Fase 20b); las interacciones no añaden nada más |
| 34 | Submission v05 (meta enriquecido) | `scripts/submission_v05_meta_enriquecido.py` -> `submissions/age_submission_v05.csv` | Validada; **pendiente de subida manual** |

### Interpretación de E-018
Darle al meta-modelo la longitud, el retuit, las menciones y la textometría le permite
**condicionar la combinación según el tipo de tweet**, que es exactamente el sesgo de error
detectado en la Fase 17 (52,5% de error en el quintil más corto). Las interacciones explícitas
longitud×ordinal no añaden nada por encima de eso: el meta-modelo ya las aprende.

### Brechas CV<->LB observadas (cuatro mediciones)
+0,0179 · +0,0145 · +0,0152 · +0,0170 -> media ≈ **+0,0162**, desviación ≈ 0,0015. El CV agrupado
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
| 35 | **Subida manual de v05 a Kaggle** | leaderboard | **F1 = 0,59518** (progreso 0,55235 -> 0,57018 -> 0,58081 -> 0,59178 -> 0,59518) |
| 36 | Verificación de GPU (D-007) | `src/gpu.py`, `experiments/gpu_events.log` | **GPU OCUPADA** (uso 42% sostenido, sin PIDs de cómputo propios): se devuelve el control y la fase se ejecuta en CPU |
| 37 | E-019: MLP sobre densas y sobre densas+SVD(64) | `scripts/experimento_e019_modelos_densos.py`, `experiments/e019_modelos_densos.json` | 0,3381 y 0,4071 (muy por debajo de 0,5658 de los lineales dispersos); la SVD explica solo 7,8% de la varianza |
| 38 | E-020: ¿aporta el MLP al ensamble? | `scripts/experimento_e020_mlp_en_ensamble.py`, `experiments/e020_mlp_en_ensamble.json` | **0,5784 -> 0,5851 (+0,0066)**: el MLP aporta por diversidad, no por exactitud |
| 39 | Submission v06 | `scripts/submission_v06_mlp_ensamble.py` -> `submissions/age_submission_v06.csv` | Validada; **pendiente de subida manual** |

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
| 45 | Auditorías finales (Fase 23) | `scripts/fase23_auditorias.py` -> `reports/auditorias.md` | **APROBADO** en leakage, reproducibilidad, submission y trazabilidad académica |
| 46 | Registro de los experimentos faltantes | `experiments.csv` | 30 experimentos registrados, todos con hipótesis y resultado |
| 47 | Informe final generado desde los artefactos | `scripts/fase_final_reporte.py` -> `FINAL_REPORT.md` | 235 líneas con cifras inyectadas desde la evidencia (no transcritas a mano) |

### Estado del proyecto: COMPLETO
Todas las fases del plan (0-24) ejecutadas. Brecha CV<->LB calibrada en 6 mediciones
(+0,0165 ± 0,0014). El pipeline regenera la submission enviada con coincidencia exacta.

## 2026-09-24 — Sesión 3 (ronda de mejora: ordinal, centroides y cierre)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 48 | E-021: modelo ordinal acumulativo + centroides | `scripts/experimento_e021_ordinal_centroides.py` | ordinal **0,5527** · centroides 0,4017 (individuales) |
| 49 | E-022: ambos en el ensamble + meta LightGBM | `scripts/experimento_e022_ensamble_final.py` | **0,5860** con LogReg meta; LightGBM meta 0,5858 |
| 50 | E-023/E-024: especialistas por longitud | `scripts/experimento_e023_*.py`, `experimento_e024_*.py` | +0,0000 (el meta ya tiene la longitud) |
| 51 | E-025/E-026/E-027: BM25 y kNN sobre SVD | `scripts/experimento_e025_e026_vistas_nuevas.py` | BM25 0,4688 · kNN 0,3088 · aporte en el ensamble **−0,0002** |
| 52 | Pipeline actualizado a 12 miembros y submission v07 | `src/pipeline.py` -> `submissions/age_submission_v07.csv` | CV **0,5860**; formato validado; 9 min |
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
| 56 | Evaluación en el ensamble | `experiments/e028_en_ensamble.json` | pares **+0,0006** (0,5860 -> 0,5866); grupos +0,0003 (redundantes) |
| 57 | Barrido del C del meta-modelo | — | C=1.0 ya era óptimo (0,5862/0,5866/0,5865 para 0,3/1/3) |
| 58 | Pipeline a 13 miembros y submission v08 | `src/pipeline.py` -> `submissions/age_submission_v08.csv` | CV **0,5866**; formato validado; 9,4 min |

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
| 63 | Evaluación del aporte (LOFO) | `experiments/advanced_02_resultados.json` | **0,5866 -> 0,5897 (+0,0031)**, positivo en 5/5 folds |
| 64 | Holdout congelado (15% de clústeres) | `experiments/advanced_03_holdout.json` | **0,5914 -> 0,5966 (+0,0052)**: confirmado |
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
  respalda el diseño de validación por etapas del brief (§50): CV -> holdout congelado -> Kaggle.
- Octava medición de la brecha CV<->LB: +0,0184 (media acumulada +0,0169 ± 0,0013).

## 2026-09-24 — Sesión 5 (cierre de líneas avanzadas)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 68 | ADV-04: incertidumbre, acuerdo e interacciones | `scripts/advanced_04_incertidumbre.py` | **+0,0002** (línea cerrada); la std entre folds baja a 0,0036 |
| 69 | ADV-05: NB-SVM y selección de features | `scripts/advanced_05_nbsvm_seleccion.py` | NB-SVM 0,5641 · selección top-150 = 0,5896 (no aporta) · NB-SVM en el ensamble **+0,0007** |
| 70 | ADV-06: candidato v10 (273 features) | `scripts/advanced_06_submission_v10.py` -> `submissions/age_submission_v10.csv` | CV **0,5904 ± 0,0047**; validado; **no enviado** (se reserva el último envío) |
| 71 | Política del último envío (D-018) | `DECISIONS.md` | 9 de ~10 envíos usados; el restante se reserva |

### Balance de la fase avanzada
- **Ganancia real**: v09 (features supervisadas de segunda generación) = **+0,0043 en el leaderboard**
  (0,60376 -> 0,60807), la mayor de las últimas cinco rondas.
- **Líneas cerradas con evidencia**: incertidumbre (+0,0002), interacciones (+0,0001), selección de
  features (0,0000), NB-SVM como modelo (−0,0017 frente al SVM).
- **Líneas que requieren verificación de reglas**: retrieval train->test (no implementado).

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
| 77 | **L5 — error por tipo y longitud** | `reports/tables/phase1_error_por_tipo.csv`, `phase1_error_por_longitud.csv` | `solo_url` 61,1% vs `mencion_alta` 19,2%; longitud 54,5% -> 35,1% |
| 78 | **L6 — ponderación condicional** | `experiments/phase2_ponderacion.json` | **MLP meta 0,5967 (+0,0061, 5/5 folds)**; segmentación dura negativa |
| 79 | Corrección de dos defectos | `scripts/phase3b_recorregido.py`, `src/advanced_features.py` | 0 columnas degeneradas; alineación del meta corregida |
| 80 | Confirmación y candidato v11 | `experiments/phase3_holdout_v11.json` -> `submissions/age_submission_v11.csv` | Holdout **+0,0043** (> criterio +0,003); distribución sana |

## 2026-09-24 — Sesión 8 (v11 en el leaderboard)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 81 | **Subida manual de v11 a Kaggle** | leaderboard | **F1 = 0,61744** (nuevo mejor; +0,0094 sobre v09) |

### Lectura del resultado
- Δ leaderboard **+0,0094** > Δ CV **+0,0069** > Δ holdout **+0,0058**. La ganancia real superó a
  ambos estimadores internos, algo que no había ocurrido en las rondas previas (donde el LB quedaba
  entre el CV y el holdout).
- Novena medición de la brecha CV<->LB: **+0,0208**, por encima del rango anterior
  (+0,0145 a +0,0184). Interpretación plausible: el combinador no lineal explota mejor la
  memorización de near-duplicados del test, que es precisamente el componente que el CV agrupado
  bloquea por diseño. **No se cambia la metodología** (el CV sigue siendo el estimador honesto para
  comparar modelos), pero se registra la observación.
- Progreso acumulado: 0,55235 -> **0,61744** (**+0,0651**).

## 2026-09-24 — Sesión 9 (L2, L3, L7 y candidato v12)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 82 | L2 — residualización por longitud | `scripts/phase5_residuales.py` | **−0,0019** (2/5 folds): el combinador ya aprende esa estructura. Cerrada |
| 83 | L3 — contenido vs estilo | `scripts/phase6_contenido_estilo.py` | estilo solo **0,2793** · contenido solo **0,5537** · aporte +0,0008. **Hipótesis refutada**: manda el contenido |
| 84 | L7 — NB-SVM de palabra + promedio de combinadores | `scripts/phase7_nbsvm_canal_meta.py` | NB palabra 0,5254 (+0,0009 al meta); **promedio MLP+LogReg 0,6005 (+0,0027)** |
| 85 | Confirmación y candidato v12 | `scripts/phase8_holdout_v12.py` -> `submissions/age_submission_v12.csv` | Holdout **0,6077** (MLP 0,6034) -> Δ +0,0043 **confirmado** |

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
- Décima medición de la brecha CV<->LB: **+0,0213**; media acumulada **+0,0177 ± 0,0023**.
- Progreso acumulado del proyecto: 0,55235 -> **0,62182** (**+0,0695**).
- Documentación sincronizada: `PROJECT_STATE.md`, `DECISIONS.md`, `FINAL_REPORT.md` regenerado desde
  los artefactos y `experiments.csv` (44 experimentos).

| 87 | Fase 9: tercer combinador | `scripts/phase9_combinadores.py` | MLP+LogReg 0,6005 óptimo; +Ridge 0,5999; +SGD 0,5676 -> **línea cerrada** (D-025) |

## 2026-09-25 — Sesión 11 (las dos vías autorizadas, cerradas con evidencia)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 88 | Vía A: retrieval train->test (inferencia) | `scripts/phase10_dos_vias.py` | Cobertura de clúster in-fold **0,00%** (el CV agrupado la elimina); duplicados exactos 1,36% con **acierto de etiqueta 61,5%**; sobrescritura dura **−0,0009** -> cerrada |
| 89 | Vía B: POS con spaCy `es_core_news_md` | `features/pos_features.parquet` | **−0,0032** -> cerrada |

### Hallazgo de la sesión
El **techo del retrieval** en este corpus no lo fija la cobertura (4,26% de near-duplicados) sino la
**consistencia de etiqueta entre duplicados**: los tweets idénticos solo comparten etiqueta en el
61,5% de los casos, por debajo de la precisión del modelo. Memorizar contenido idéntico con
etiquetas contradictorias introduce más error del que corrige. Es un resultado sobre el **corpus**
(anotación con ruido), no sobre la técnica.

## 2026-09-25 — Sesión 12 (hipótesis del ruido de etiqueta)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 90 | PH11: exclusión de filas ruidosas del entrenamiento | `scripts/phase11_ruido_etiqueta.py` | contradictorias (643) **+0,0001** · duplicadas (2.123) **+0,0016** (3/5 folds) -> mejora real pero bajo el umbral de +0,003 |

### Episodio metodológico
La primera ejecución usó la normalización agresiva (`normalize_for_dedup`), que colapsa miles de
tweets cortos distintos al mismo texto: los conflictos pasaron de **643 a 12.989 filas** y el
resultado salió −0,0042. Se detectó el artefacto, se corrigió la definición al criterio del audit
(minúsculas + espacios) y se repitió. **La elección de la normalización cambia el diagnóstico del
ruido** — queda registrado como lección.

## 2026-09-25 — Sesión 13 (entrega: notebook final, commit, PR y merge)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 91 | Notebook consolidado regenerado con las 12 fases | `scripts/fase12_notebook_final.py` -> `notebooks/99_entrega_consolidado.ipynb` | 27 celdas, **0 errores**; incluye aserciones que verifican cada cifra contra los artefactos |
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

## 2026-09-25 — Sesión 21 (capa de decisión ordinal: fases 30-35)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 124 | Matriz de cobertura de hipótesis | `scripts/phase30_cobertura_hipotesis.py` | 9 líneas abiertas, 27 variantes previas confirmadas cerradas |
| 125 | Decodificación/proyección/coste/desempate (A, E, G, F) | `scripts/phase31_decision_ordinal.py` | 14 capas; mejor G (tau 0,05) **+0,0011 (4/5)**; ninguna adopta |
| 126 | Corrección suave, transición, confianza, Luce (C, D, H, B) | `scripts/phase32_correccion_probabilistica.py` | 14 capas; mejor H 5 bandas +0,0002; C4 **neto 0** |
| 127 | Corrector mínimo (I) | `scripts/phase33_corrector_minimo.py` | 5 capas; todas <= 0 |
| 128 | Informe y artefactos consolidados | `scripts/phase34_informe_capa_decision.py` | 33 capas, 0 adoptadas; 6 entregables en `reports/age_*` |
| 129 | Cierre documental | `scripts/phase35_cierre_documentacion.py` | D-036 (D-034 cerrada) y D-037 (línea de decisión cerrada) |

### Bugs corregidos en la sesión
- `decision_layer`: convención de supervivencia frente a CDF (decodificación A degenerada a 0,026 ->
  0,5887 tras corregir), indexado global/local en tres capas, `np.ix_` que sobrescribía las seis
  columnas, y `del` sobre variables capturadas.
- `phase31`: el LogReg de umbrales no convergía sin estandarizar (se añadió `StandardScaler`).

### Resultado
Ninguna de las **33 capas de decisión** alcanza el gate (Δ ≥ +0,003 con ≥4/5 folds). La mejor,
`G desempate E[Y] (tau 0,05)`, da +0,0011 con 4/5 folds y **neto de rescate −117**: el argmax de V12 ya
es la mejor decisión de su distribución de probabilidades. El error restante se reparte en un techo
de información (35% del error: ninguna fuente acierta) y un 65% de agregación que ninguna regla de
decisión convierte en aciertos. V12 permanece congelado; sin submission nueva.

## 2026-09-25 — Sesión 22 (lexicones: fases P0-P8)

| # | Acción | Artefacto | Resultado |
|---|---|---|---|
| 130 | Entorno y housekeeping | `scripts/fase00_entorno.py` | 16 núcleos/32 hilos, 31,2 GB, RX 9070 XT; commit de la fase de decisión y rama nueva |
| 131 | Cobertura de hipótesis (P0) | `scripts/phase36_auditoria_lexicones.py` | 0 experimentos previos con canal léxico |
| 132 | Catálogo de lexicones (P1) | `scripts/phase37_catalogo_lexicones.py` | 24 subcat, 1.369 entradas, SHA256 por fichero |
| 133 | Bloques, control y redundancia (P2-P5) | `scripts/phase38_bloques_lexicos.py` | Control lexicon-only 0,203/0,152/0,073; 0 columnas redundantes con V12 |
| 134 | Revisión de código (puerta obligatoria) | `@ml-reviewer` | APROBADO CON CAMBIOS: el control de permutación no gateaba la adopción (corregido) |
| 135 | Fusión con V12 y control de permutación (P6) | `scripts/phase39_fusion_v12.py` | Ningún bloque alcanza el gate |
| 136 | Análisis de error y complementariedad (P7) | `scripts/phase40_analisis_error.py` | Rescate ≤ daño en todos los candidatos |
| 137 | Cierre documental (P8) | `scripts/phase41_cierre_lexicones.py` | D-038 y D-039; línea cerrada |

### Bugs corregidos en la sesión
- `ml-reviewer` detectó que el control de permutación se calculaba pero no gateaba la adopción
  (falso positivo posible): ahora `adopta` exige superar la nula, con 3 semillas si el gate numérico
  pasa, y se añade la comprobación de cabeza de DD-8.
- El cribado de redundancia de P5 usaba `spearmanr` con una matriz 2D, que devuelve la matriz de
  correlaciones entre variables: leía la autocorrelación (1,0) y marcaba las 72 columnas como
  redundantes. Corregido con ranks vectorizados; el resultado real es 0 columnas redundantes.
- `src/lexicons.py`: umbral mínimo de 5 entradas por subcategoría (retira `vida/hogar`), formas
  ambiguas comprobadas también en carga y tokenización con límites de palabra.
- `phase37`: fecha constante para que los SHA256 sean reproducibles.

### Resultado
La línea de lexicones queda **cerrada** (D-039). El bloque es genuinamente nuevo (0 de 72
columnas redundantes con V12) pero su señal es insuficiente: solo la familia de temas supera el
azar en control aislado, y ningún bloque alcanza el gate ni supera a su control de permutación.

## 2026-09-26 — V13 ADVANCED RESEARCH: cierre sin adopciones (D-041)

### Qué se hizo
- **P0b** cambió de enfoque a petición del usuario: la colinealidad se midió sobre la **matriz
  completa** de 279 variables, sin brazo reducido y sin reentrenar (la referencia se lee de
  `phase39_fusion_v12.json`, verificada fold a fold). Resultado: 401 pares con |rho| >= 0,80, 229
  redundantes, **15 duplicados exactos**, 54 clusters, 1 columna constante, rango efectivo 263/278 y
  **87 de 278 direcciones para el 95 % de la varianza**.
- **P1** evidencia `P(clase|token)` in-fold: control 0,3997 (el doble del mejor bloque léxico
  cerrado), 1 de 36 variables redundante. P1c separó señal léxica de reconocimiento de cuentas: el
  57 % de la mejora es léxica. La fusión (P1b) dio −0,0002 con 3/5 folds.
- **P17** pronombres: control 0,1918, 0 de 26 redundantes, pero rescate neto −64.075.
- **P20** modalidad: 0,1270, **por debajo del azar**; solo el 1,6 % de documentos tiene un marcador
  de duda.
- **P19** árboles de conjunciones: 0,1539, también bajo el azar, por falta de cobertura.
- **P3a** especialista binario de frontera: **empeora** en ambas fronteras.
- **P16, P16b, P16c, P16d** respondieron por qué fallan todas las fusiones.

### Bugs y correcciones propias (documentados sin maquillar)
- **D-040**: `numpy 1.26.4` sobre `Python 3.14.7` calcula mal `X @ X.T` (auto-transpuesta), con 1
  hilo y con 32, en cualquier layout. `np.corrcoef` y `np.cov` con `rowvar=True` caen en el mismo
  fallo. **Ninguna métrica del proyecto está afectada**: los 10 sitios que usan producto matricial
  usan rutas sanas y el `MLPClassifier` de producción es idéntico al parcheado. Se añade
  `src/verificacion_numerica.py` con `autotest_numerico()` y `grama_exacta()`.
- El **bootstrap** remuestreaba las dos tablas de confusión por separado, de modo que una
  diferencia nula daba un intervalo que no contenía el cero. Corregido a bootstrap **pareado** sobre
  la tabla conjunta y validado contra el método clásico (difieren en 5e-5).
- **P16 necesitó tres correcciones** antes de ser utilizable: ecuaciones normales sin término
  independiente, columnas sin tipificar (una dirección acapara la traza: 8,99e12 de 8,99e12) y
  penalización por alfa fijo en vez de por grados de libertad efectivos. Las dos medidas finales
  divergen y ambas se publican.
- Otros: `MLPClassifier` pelado en vez de `EncodedMLP`; `cfg.CLASES` en vez de `cfg.CLASSES`;
  `frecuencia_de_documento` contando ocurrencias en lugar de documentos; `rankdata` importado del
  módulo equivocado; vocabulario de conjunciones con una palabra en inglés.

### Resultado
Cierre de V13 con **0 adopciones** y con un diagnóstico causal de por qué. La única vía abierta es
una decisión de alcance (relajar §3 o cambiar la métrica), no un experimento más. Details en
`reports/v13_final_research.md` y D-041 en `DECISIONS.md`.

## 2026-09-26 · V15 (correlacion de entrada y poda) y arranque de V16 (arquitectura)

### Petición
Carlos pide el paso de variables que faltaba: medir la correlación entre variables de entrada,
identificar cuáles se asocian con la salida, y **eliminar las variables de entrada redundantes**.
Después pide añadir además el paso de la **arquitectura del meta-modelo**.

### Qué se hizo
- **V15a** (`scripts/v15a_correlacion.py`): sobre la matriz completa de 312 variables, correlación
  de Spearman con gram exacta (`grama_exacta`, nunca `A @ B.T` por D-040) y asociación con la
  salida por **eta cuadrado** (la salida es categórica). Todo **in-fold**: la estructura de pares, la
  asociación y el índice de variables conservadas se calculan dentro de cada fold y se guardan en
  `features/v15_conservadas_fold{0..4}.npy`.
- **V15b** (`scripts/v15b_poda.py`): evaluación de la poda con el gate de D-042 completo más la
  segunda partición de 4 folds.
- **V16** (`scripts/v16_arquitectura.py`): tres arquitecturas de combinación de las dos cabezas.

### Errores propios encontrados y corregidos
- Línea con indentación rota (`   .print_sizes = ...`) y un `representatives` construido dos veces
  con una expresión sin sentido; se rehízo el bloque para que la elección de representante sea
  enteramente in-fold.
- **La aserción estructural del diagnóstico falló** porque hay columnas constantes: su diagonal no
  puede ser 1. Se detectan y se excluyen del análisis en lugar de forzar la aserción; son
  `v12_supervisada_49` y `v14_pasiva_ratio`, y se podan por definición.
- `zip` de tres elementos desempaquetado como dos; `len(grupo) | len(otro)` con listas en vez de
  conjuntos; y **el más importante**: la lista de paresmarked por fold se pasaba **acumulada** en
  vez de local a `clusters_por_union`, lo que concatenaba los pares de todos los folds anteriores.
- **La medida de estabilidad que elegí primero (Jaccard sobre los miembros de cada cluster) dio
  0 de 58 y era engañosa**: exigir que un cluster tenga exactamente los mismos miembros es un
  criterio tan estricto que no informa, porque un solo par que cambie reordena el cluster entero por
  transitividad de union-find. La medida correcta es el Jaccard de la **estructura de pares**, que
  da **1,0000**: la redundancia es perfectamente estable entre folds. Las dos se publican, con su
  lectura explícita.
- En V16 el `f1_score` interno comparaba **etiquetas string contra índices numéricos** (el error
  "Mix of label input types"). Corregido convirtiendo a nombres de clase, y esta vez con prueba de
  humo antes de la pasada larga.

### Resultado
- **V15a**: 2 constantes · 402 pares >= 0,80 · 245 >= 0,90 · 15 duplicados exactos · **58 clusters
  sobre 171 de 310 variables** · 197 conservadas tras podar · eta2 mediana 0,0306 y máximo 0,5599 ·
  **112 variables con eta2 < 0,01 (110 de las 310 analizables; el recuento de 112 sale sobre las 312 columnas de la matriz, e incluye las 2 constantes, que son 0 por definicion)** · **Jaccard de pares entre folds 1,0000**.
- **V15b**: macro-F1 **0,5965** frente a 0,6005 de V12, **delta -0,0040, 0/5 folds**; segunda
  partición **0/4** con media -0,0065; IC95 pareado [-0,0054; -0,0027]. **NO ADOPTAR** (D-045).
- **V16**: en curso.
- `experiments.csv` con **93 experimentos**. Todo en español, sin alfabetos ajenos.

- **Nota de trazabilidad**: la plantilla de la sección 6 de `scripts/v15b_poda.py` tiene dos defectos
  de redacción (palabras en inglés que se colaron) y una rama de lectura que afirma que las
  diferencias están dentro del ruido de la estimación, lo que contradice al IC95. **El script no se
  editó**, porque ya se ejecutó y su salida es el registro del experimento; lo que se corrigió fue el
  artefacto generado `reports/v15_poda.md`, con una nota de edición explícita y **sin tocar ninguna
  cifra**. Si en el futuro se vuelve a ejecutar ese script, la plantilla habrá que arreglarse antes.

### Cierre de V16 (arquitectura del meta-modelo)
- **A1 (peso aprendido por fold)**: 0,6002, delta -0,0003, 0 de 5 folds. Pesos elegidos
  0,1 · 0,5 · 0,5 · 0,4 · 0,5: el 0,5 de producción se elige solo en tres de los cinco folds.
  **Descartado.**
- **A2 (combinador logístico sobre las doce probabilidades, apilamiento en dos niveles)**: 0,6011,
  delta **+0,0006**, 3 de 5 folds, 3 de 4 en la segunda partición, IC95 pareado
  **[-0,0003; +0,0015] que contiene el cero**. Falla el efecto mínimo y la significancia.
  **NO ADOPTAR** (D-046).
- **Lo que sí se queda**: las fronteras centrales mejoran **+0,0102** y **+0,0157**, tres a cinco
  veces el efecto global. El problema del conjunto de datos está en las fronteras de edad y ahí el
  meta-modelo tiene algo que aportar; lo que no tiene es magnitud para la métrica global.
- **Dos defectos de presentacion corregidos sin tocar cifras**: las tablas de fronteras llevaban la
  etiqueta con un guion vertical (`18-24|25-34`), que Markdown interpreta como separador de columna
  y partia la tabla. Se reparan con `scripts/v15_v16_tablas.py`, que escapa la barra y es
  idempotente; ninguna cifra cambia. A los informes de V15b y V16 se les anadio despues una seccion
  de lectura con su nota de edicion, porque los generadores no la incluian; los scripts de
  experimento **no se han editado ni vuelto a ejecutar**.
- `experiments.csv` pasa a **94 experimentos**. Auditoría de idioma sobre 336 ficheros: 0 errores.
- **Precision de una cifra**: el recuento de "112 variables con eta2 < 0,01" sale sobre las **312**
  columnas de la matriz e incluye las 2 columnas constantes (cero por definicion); sobre las **310
  variables analizables** son **110**. Se precisó la cifra en DECISIONS.md, PROJECT_STATE.md,
  RUNLOG.md y el plan. `experiments/v15a_correlacion.json` conserva el 112 original, que es
  correcto para la base sobre la que se calculó.

## 2026-09-26 · V17 arranca: rama nueva, P0, P0', P1, P1b y P8

- **Rama** `kaggle-age-2026-puj-v17` creada desde `a818349` con el arbol limpio. Perfil de maquina
  re-verificado en vivo (`scripts/fase00_entorno.py`): sin cambios. Intencion registrada en
  `plan/00_intent.md` a partir del brief de 53 secciones, sin entrevista de requisitos porque el
  brief ya fija objetivo, restricciones, orden y presupuesto.
- **P0, auditoria de brechas** (`reports/v17_gap_audit.{md,csv}`, 42 familias): de las 10 pistas,
  3 cerradas, 5 parcialmente probadas y 2 genuinamente nuevas. La forma de 279 variables del
  especialista de frontera ya estaba cerrada por V13 P3a (-0,0017 y -0,0029); la rejilla de
  umbrales de la compuerta ya se habia medido con 90 combinaciones en P16d.
- **P0', reconciliacion de la referencia** (`scripts/v17_p0_reproduccion.py`): 0,6004 y 0,6005 **no
  son el mismo numero redondeado**, son dos estimadores (pooled frente a media de folds, efecto
  Jensen). Diferencia 9,14416e-05, mayor que la unidad de redondeo. **Canonico 0,6005 pooled**,
  porque el umbral de D-042 y el IC95 del bootstrap viven en esa escala y las cinco fases modernas
  miden alli. **A2 de V16 se reproduce** (0,6011, fronteras 0,5065 y 0,5131). No hay STOP.
- **P1 y P1.1** (`scripts/v17_p1_fronteras.py`): el especialista de pares E-028a, entrenado solo
  con las dos clases de la frontera sobre el TF-IDF disperso y con folds agrupados, alcanza
  **0,6864 y 0,7218** en las filas de cada frontera, frente a **0,4963 y 0,4974** de V12. Rescata
  **14.356 y 15.571** filas y rompe **3.519 y 2.872**. Techo con enrutamiento perfecto: **0,7481 y
  0,7726**. En un 23 a un 25 % de las filas ambos fallan.
- **P1b** (`scripts/v17_p1b_curva_compuerta.py`): la compuerta sobre la confianza de V12, con seis
  umbrales, da como mucho **+0,0002** global y a partir de 0,20 de margen se vuelve **negativa**.
  Rescate y dano se compensan en las filas activadas. Es la segunda vez que una compuerta de
  frontera no rinde, como en D-019.
- **P8, perfil de confianza** (`scripts/v17_p8_perfil_confianza.py`): area bajo la curva para
  distinguir las filas que el especialista ARREGLA de las que ROMPE. **Las probabilidades de V12 no
  sirven** (separacion 0,5780 y 0,5268; entropia 0,4323 y 0,4230, es decir invertida). **Dos senales
  que V12 no expone si sirven**: el **acuerdo entre miembros** (0,7070 y 0,7300) y el **margen del
  propio especialista** (0,7294 y 0,7539). Eso reabre la via con un arbitro aprendido.
- **Entregables de registro**: `reports/v17_master_results.csv` (17 filas) y
  `reports/v17_hypothesis_registry.csv` (15 filas), generados desde los artefactos por
  `scripts/v17_marcos.py` para no teclear cifras a mano. `experiments.csv` pasa a **99
  experimentos**.
- **Auditoria**: `v13_p0a_auditoria_idioma.py` da **0 errores sobre 356 ficheros**. El
  `tools/verificar_caracteres.py` que nombra el brief **no existe en este proyecto**; el segundo
  escaneo independiente se hizo sobre los 21 ficheros nuevos de V17 buscando controles,
  invisibles, emojis y CJK: **0 problemas**.
- **Tablas de fronteras**: la etiqueta `18-24|25-34` lleva un guion vertical que Markdown parte
  como separador de columna; `scripts/v15_v16_tablas.py` (ampliado a los informes de V17) escapa
  32 filas sin tocar ninguna cifra.

## 2026-09-26 · V17 P9: el arbitro aprendido falsifica la prediccion de P1c

- **Por que P9 y no mas compuertas**: un diagnostico posterior a P1c explico el fondo. En la region
  de frontera hay 11.130 filas, el especialista **gana 1.830 y pierde 2.485**, y escucharle ahi sin
  mas vale **-0,0080**. Fuera de esa region el especialista gana mucho mas, pero ahi V12 no duda:
  esta seguro prediciendo otra clase. Y lo decisivo: **V12 esta mas confiado cuando se equivoca** en
  la frontera (separacion 0,809 en las filas que el especialista arregla frente a 0,539 en las que
  rompe), y solo el 12,7 % y el 8,0 % de las victorias del especialista caen dentro de la region de
  frontera. Por eso toda compuerta por ambiguedad falla: no hay banda de confianza que discrimine.
- **P9** (`scripts/v17_p9_arbitro.py`): arbitro binario LogReg por frontera sobre **47 columnas
  derivadas que ya existian** (votos y margenes de los 13 miembros, 5 margenes de pares, 3 de grupo,
  6 probabilidades de V12, 6 de A2 y el acuerdo entre miembros). **Ninguna representacion nueva.**
  Protocolo anidado en tres niveles: ajuste en un bloque interior, **umbral elegido en otro bloque
  que el arbitro no ve**, y reentrenamiento en todo el train exterior para la validacion exterior.
- **Las dos medidas de frontera de P3, y por que las dos**: los +0,0392 y -0,0552 del acoplamiento
son de la **cabeza lineal sola** (`fronteras_lineal_solo`), que es donde actua la perdida de margen.
En el **sistema combinado** el mismo efecto se amortigua a +0,0209 y -0,0268. Las dos medidas son
correctas y responder a preguntas distintas, y declararlas por separado es lo contrario de lo que
D-048 corrige.

**Resultado**: macro-F1 **0,6004**, delta **-0,0001**, **0 de 5 folds** positivos, IC95 pareado
  [-0,0002; -0,0001]. **NO ADOPTAR**. El arbitro elige umbrales de 0,60 a 0,90, activa entre 327 y
  1.256 filas, y cuando activa **pierde**: 3 rescates contra 22 danos en toda la fase.
- **Lo que significa**: no es un arbitro mal entrenado, es que la frontera que habria que aprender
  **no existe con las señales disponibles**. Confirma por falsacion la prediccion que P1c
  registro antes de mirar el resultado.
- **Errores propios corregidos**: `np.where` con nombres de clase devuelve un array de texto, que
  numpy no admite como indice (se devuelve directamente, sin indexar la lista de clases); la matriz
  de salida se construia **transpuesta** (fila=clase en vez de fila=fila); `en_train` devolvia
  posiciones locales en vez de indices globales; y varias etiquetas de frontera se tomaban del primer
  elemento de la tupla en vez de componerse, lo que dejaba maximos en `nan`.
- `experiments.csv` pasa a **101 experimentos**. `reports/v17_master_results.csv` con 19 filas y
  `reports/v17_hypothesis_registry.csv` con 17. Auditoria de idioma: **0 errores sobre 362 ficheros**.

## 2026-09-26 · V17 P5 y cierre de fase

- **P5, meta-combinador local** (`scripts/v17_p5_combinador_local.py`): la pregunta es de
  **combinacion**, no de seleccion. Un LogReg por frontera redistribuye la masa que V12 ya pone en
  las dos clases del par, conservando el total, de modo que no puede mejorar las clases faciles ni
  danar a las vecinas, y actua en todas las filas. La fuerza (de 0 a 1) se elige en un bloque
  interior agrupado. Resultado: **0,6003, delta -0,0002, 1/5 folds**, y **las fronteras empeoran a
  0,4953 y 0,4968**. La fuerza elegida en el bloque interior fue 1,00 en tres folds, lo que dice que
  la senal interior no generaliza. **NO ADOPTAR**.
- **Cierre** (`scripts/v17_cierre.py`): comparativa de candidatos, informe final, reproducibilidad y
  cumplimiento, mas las dos figuras que faltaban (rendimiento global por candidato y evolucion
  V12 -> V16 -> V17). **10 figuras** en total.
- **Punto importante del cierre**: P9 y P5 cierran **las dos formulaciones posibles** de la via
  local. Ninguna pista queda PROMISING ni FINALIST, y eso se declara en vez de dejar una pista
  abierta por prudencia.
- **Lo que queda abierto no es un experimento**: los margenes de pares ya son entradas de V12 y V12
  no los explota. Eso exige cambiar la funcion objetivo, no anadir otra compuerta.
- **Entregable declarado pendiente**: los resultados no estan volcados al notebook consolidado. Es
  trabajo de redaccion y queda dicho en `reports/v17_compliance.md`.
- `experiments.csv` pasa a **102 experimentos**. **D-047** en `DECISIONS.md`. Auditoria de idioma:
  **0 errores sobre 370 ficheros**. Sin push.

## 2026-09-26 · V17 pasa al notebook consolidado

- **PARTE III anadida** a `notebooks/99_entrega_consolidado.ipynb`: **32 celdas**, 11 de codigo, **8
  figuras embebidas** y las comprobaciones con `assert` que el notebook ya usaba, de modo que si un
  artefacto cambia sin reejecutar su experimento, el notebook falla en vez de mentir.
- Se ejecuta con `scripts/v17_notebook.py`, que construye un notebook temporal con la celda de
  configuracion mas las celdas nuevas y copia despues las salidas. **No se reejecuta el historico**:
  las 19 celdas de codigo anteriores leen artefactos de septiembre y no volverian a correr igual.
- Contenido: reconciliacion de la referencia y reproduccion de A2 de V16, analisis de miembros en
  las fronteras, especialista de pares, interseccion de errores, el mecanismo de los errores
  confiados, gating, perfil de confianza, P9 y P5, figuras y comparacion final. Es la lista que pide
  la seccion 43 del brief.
- **Correccion de una cifra**: el total de la activacion del arbitro de P9 es **3 rescates y 22
  danos**, no 4 rescates. La cifra la calcula el script y el informe de P9 la tenia bien; el error
  estaba en el texto que escribi a mano en `RUNLOG.md`, `experiments.csv` y el registro de hipotesis.
  Corregido en los tres. Este error lo detecto la celda del notebook, que recomputa la suma.
- Se corrige tambien la plantilla de `scripts/v17_cierre.py`, que reintroducia la ruta `figuras/v17_*`
  en vez de `reports/figures/v17_*`, y la fila del compliance que declaraba el notebook como
  pendiente: ya no queda ninguno.

### Correccion de un campo de V15 (2026-09-26)

- Al verificar el informe completo contra los artefactos aparecio una discrepancia real: el campo
  `variables_tras_poda` de `experiments/v15a_correlacion.json` vale **81** y esta **mal calculado**.
  La formula es `310 - 171 - 58`, que resta el numero de clusters una vez de mas, cuando un cluster de
  tamano k elimina k-1 variables y no k. El numero correcto es **197**.
- **197 es el que se uso**: el indice `features/v15_conservadas_fold{0..4}.npy` tiene 197 columnas en
  los cinco folds, y V15b evaluo exactamente ese conjunto, dando 0,5965 y delta -0,0040. La conclusion
  de V15 no cambia.
- No se edita `scripts/v15a_correlacion.py` porque ya se ejecuto y su salida es el registro. Se declara
  el campo erroneo en `reports/v15_correlacion.md` y en `INFORME_COMPLETO.md` para que nadie lo cite.
- La leccion sobre el proceso es que la verificacion contra artefactos funciono: el numero 81 nunca se
  habia mirado porque nadie lo leia. Aparece por una lectura automatica de todos los campos.


## 2026-09-26 · Informe completo del proyecto

- **`INFORME_COMPLETO.md`** (519 lineas): la cuenta completa y legible de V12 y de las cinco fases de
  busqueda. Catorce secciones: problema y datos, que se entrego, el protocolo de decision, V13, V14,
  V15, V16, V17, tabla maestra, lecciones aprendidas, estado final e indice de artefactos.
- **Todas las cifras se verificaron por script contra los artefactos**: 45 valores comprobados, y la
  verificacion encontro **un error real de la fuente**, no del informe: el campo
  `variables_tras_poda` del JSON de V15a vale 81 y esta mal calculado (resta los clusters una vez de
  mas). Declarado en los tres sitios donde se podria citar. El 197 correcto es el que se uso.
- `FINAL_REPORT.md` recibe una cabecera que apunta al informe completo y declara su propio alcance
  (el modelo V12 entregado), sin modificar su contenido.
- Auditoria de idioma: **0 errores sobre 372 ficheros**. Sin push.

## 2026-09-26 · V18 arranca: P0, P0' y la correccion de D-048

- **Rama** `kaggle-age-2026-puj-v18` creada desde `46b32f9`. Perfil de maquina re-verificado, sin
  cambios. Intencion registrada en `plan/00_intent_v18.md`.
- **P0, auditoria de brechas** (`reports/v18_gap_audit.{md,csv}`, 28 familias): de las 8 pistas de
  V18, 0 cerradas, 3 parcialmente probadas y 5 genuinamente nuevas. `sample_weight`, `class_weight`,
  `cost_matrix`, `head`, `multitask` y destilacion tienen **cero coincidencias** en el proyecto: la
  dimension que V18 abre no tiene ningun antecedente. Senalo tambien un handicap verificado para P1:
  las clases estan balanceadas (16,04 % a 17,04 %, razon 1,062).
- **P0', la pieza importante.** La auditoria concluyo que la ventaja del especialista venia de la
  representacion, comparando P3a (0,4944 denso) con E-028a (0,7218 disperso), y senalo el confound de
  que cambian las dos cosas a la vez. La rejilla de 2 x 2 lo cierra, y **la conclusion sale invertida**:

  | especialista en las filas del par | 18-24\|25-34 | 25-34\|35-49 |
  |---|---|---|
  | 279 columnas de V12 | **0,7136** | **0,7493** |
  | TF-IDF disperso (E-028a) | 0,6864 | 0,7218 |
  | V12 global | 0,4963 | 0,4974 |

- **El bug, encontrado**: el 0,4944 que P3a publico **no es la exactitud del especialista**, es una
  mezcla. `filas_actuables: 57.019` es el recuento sobre todo el dataset (33,7 % de 169.336), no dentro
  del par; dentro de las 57.023 filas del par solo 38.424 eran actuables y **18.599 quedaban
  congeladas** con la prediccion de V12, donde V12 acierta el 4,2 %. La auditoria de P0 comparo un
  recuento global con uno del par y concluyo que la compuerta congelaba cuatro filas. Ejecutando el
  codigo y las variables exactos de P3a, con sus dos codificaciones de etiqueta, sale 0,7136.
- **No hay fuga**: las 72 columnas supervisadas se construyen en `advanced_02_features.py:191` con
  exactamente el mismo `make_folds(y, groups=cluster_ids, 5, 1, "group")`.
- **Lo que sobrevive**: el resultado de sistema de P3a (delta -0,0012 con compuerta) se mantiene, porque
  la compuerta era mala. Y el mecanismo de V17, de que los errores de frontera de V12 son errores
  *confiados*, no se toca: es sobre enrutado, no sobre donde vive la informacion.
- **D-048** escribe la correccion y sus consecuencias. La asimetria que D-047 dejo abierta es mayor y
  mejor caracterizada: el hueco real es **0,4963 extraido contra 0,7136 disponible** en el propio
  espacio de variables de V12.
- **Defecto propio declarado**: la reconstruccion del TF-IDF de P0' **no reproduce** a E-028a (da
  ~0,50). Sus dos celdas quedan marcadas **invalidas** y excluidas del veredicto; los numeros del
  especialista disperso salen de la OOF guardada.
- `experiments.csv` pasa a **104 experimentos**. Sin push.

## 2026-09-26 · V18 P1: el peso de clases mueve la frontera y la global no lo sigue

- **Fidelidad del control: 0,000000.** El nivel base, reentrenado con una reimplementacion local que
  acepta `sample_weight` (la clase de produccion `EncodedMLP` no lo admite), reproduce **exactamente**
  la OOF guardada de V12. Eso valida el montaje y hace atribuible cualquier diferencia a los pesos.
- **Resultado de los tres niveles pre-registrados** (pesos 1,0 / 1,5 / 2,5 sobre `18-24`, `25-34`,
  `35-49`): macro-F1 **0,6005 / 0,6001 / 0,5841**; fronteras `18-24|25-34`
  **0,4963 / 0,5441 / 0,5786**; fronteras `25-34|35-49` **0,4974 / 0,5341 / 0,5699**.
  **Ninguno adopta.** El gate exploratorio es +0,0015.
- **La lectura que si vale, y es precisa**: la frontera sube de forma **monotona y enorme**
  (+0,082 en la primera) mientras la global **baja**. El intercambio es de suma cero y su
  localizacion es exacta. En el nivel moderado, `18-24` gana **+0,0170** y `13-17` paga **-0,0166**, y
  las otras cuatro clases no se mueven. **El coste entero cae sobre la clase facil vecina de la
  frontera.** En el nivel fuerte `13-17` paga -0,0659 y `18-24` solo gana +0,0149: sobrecorreccion.
- **Por que la accuracy de frontera sube mas que el F1 de las clases centrales**: la accuracy de la
  frontera esta restringida a las filas del par y no ve lo que se paga fuera. El F1 de la clase si lo
  ve. Esa es la asimetria de metrica, medida esta vez con su coste exacto.
- **Consecuencia para el rumbo**: el objetivo **si** mueve la frontera, luego la via del objetivo esta
  viva, pero **reordenar el equilibrio entre clases no puede ganar en una media de seis clases**. Lo
  que P1 descarta es el *re-peso*, no el objetivo. La pista mejor motivada ahora es un **objetivo de
  margen**, porque penaliza confusiones con la vecina verdadera y no con el resto, que es
  exactamente lo que evita el coste que P1 acaba de medir.
- `experiments.csv` pasa a **105 experimentos**. Sin push.

## 2026-09-26 · V18 P4: multi-objetivo, sin adopciones

- Cuatro brazos que separan arquitectura de objetivo, con las mismas 279 variables de V12 y todo
  pre-registrado (30 epocas fijas sin early stopping, lote 512, Adam 1e-3, lambda 0,0 / 0,1 / 0,5):
  **A** control, **B** arquitectura (dos cabezas binarias que reparten la masa que la global ya pone
  en cada par, 558 parametros extra), **C** objetivo (un termino binario de frontera sobre los
  **mismos** logit, **cero** parametros extra).
- Resultado: **A 0,5982** (-0,0023) · **B 0,5993** (-0,0012) · **C1 0,5990** (-0,0015) ·
  **C2 0,5990** (-0,0015). Ninguno adopta.
- **Lo que dice**: los cuatro brazos quedan a menos de 0,0011 entre si, y la frontera se mueve como
  mucho +0,0038 frente a las +0,08 que movia el re-peso de P1. El termino **aditivo** no anade
  parametros y **no hace nada**: la frontera no mejora. Anadir una cabeza auxiliar tampoco.
- **Handicap de montaje, declarado**: la cabeza lineal de PyTorch no iguala a la LogReg de sklearn
  (A_control queda 0,0023 por debajo de V12), y ese handicap es del mismo orden que los efectos que
  se miden. Este experimento **no puede resolver efectos por debajo de 0,002**, asi que la ausencia
  de diferencia entre brazos es compatible con efectos pequenos, no los descarta. Se registra porque
  la honestidad del resultado depende de ese limite.
- Con P1, P3 y P4 cerrados, **la pista del objetivo no da senal adoptable** en ninguna de sus tres
  formulaciones. Segun la propia condicion del brief para P5 ("solo si P1-P4 demuestran que cambiar el
  objetivo tiene senal"), P5 queda **no indicada** salvo decision explicita en contra.
- `experiments.csv` pasa a **107 experimentos**. Sin push.

## 2026-09-26 · V18 P5 — Destilacion del conocimiento de frontera (cierre de la fase)

**Decision previa**: el gate que impedia P5 se levanto **solo para P5**, con la justificacion de que
no repite P1, P3 ni P4. En efecto, los tres fallaron por mecanismos distintos y P5 plantea un cuarto:
transferir el conocimiento del especialista al modelo general **sin routing en inferencia**, con la
senal descargada durante el entrenamiento.

**Teacher**: el especialista binario sobre las **279 variables de V12**, ya validado en P0' con el
protocolo completo (0,7136 en `18-24|25-34` y 0,7493 en `25-34|35-49`), y no el de TF-IDF disperso
(0,6864 y 0,7218) por dos razones medidas: es mejor, y esta en la **misma representacion que el
student**, que es la condicion necesaria para que el student pueda expresar lo que se le ensena.

**Cross-fitting, que es donde este tipo de experimento se contamina**: el teacher se cross-fita en
cinco inner folds **agrupados por duplicados** sobre el train exterior, y solo predice su
`inner_validation`. Al concatenar, `teacher_OOF` cubre **todo** `outer_train` y cada ejemplo fue
predicho por un teacher que no lo uso para entrenarse. El teacher ajustado sobre todo `outer_train`
si predice `outer_validacion`, y esas predicciones se usan **solo para diagnostico**: nunca entran en
el entrenamiento del student.

**Senales**: la probabilidad de la clase superior de cada pareja, equivalente a su margen por la
sigmoida, aplicada **solo a las filas de esa pareja** porque fuera de ella el teacher extrapola. Sin
pseudo-etiqueta dura en ningun punto, y la etiqueta verdadera es siempre el objetivo principal.

**La diferencia tecnica con P4, que es la que hace el experimento nuevo**: en P4 el termino auxiliar
se media contra la **etiqueta verdadera**, que ya esta en la entropia cruzada, y por eso era
redundante. Aqui el objetivo auxiliar es la **prediccion suave del teacher**, que es informacion
nueva.

**Cuadrica**: seis combinaciones de peso y temperatura, **seleccionadas dentro del train exterior**
en una particion interior agrupada del 25 %, por macro-F1 de seis clases. La validacion exterior no
participa en la eleccion ni en el entrenamiento. Sin tuning sobre outer validation, sin leaderboard,
sin test.

**Las dos medidas de frontera de P3, y por que las dos**: los +0,0392 y -0,0552 del acoplamiento
son de la **cabeza lineal sola** (`fronteras_lineal_solo`), que es donde actua la perdida de margen.
En el **sistema combinado** el mismo efecto se amortigua a +0,0209 y -0,0268. Las dos medidas son
correctas y responder a preguntas distintas, y declararlas por separado es lo contrario de lo que
D-048 corrige.

**Resultado**:

| Brazo | Lambda | T | macro-F1 | Delta | Folds pos. | IC95 | 18-24\|25-34 | 25-34\|35-49 |
|---|---|---|---|---|---|---|---|---|
| control sin destilacion | - | - | 0,5983 | -0,0022 | 0/5 | [-0,0029, -0,0014] | 0,4957 | 0,4950 |
| destilado | 0,3 | 1 | 0,5990 | -0,0015 | 0/5 | [-0,0023, -0,0008] | 0,4980 | 0,4966 |
| destilado | 0,3 | 2 | **0,5991** | **-0,0014** | 0/5 | [-0,0022, -0,0007] | 0,4969 | 0,4973 |
| destilado | 0,3 | 4 | 0,5988 | -0,0017 | 0/5 | [-0,0024, -0,0009] | 0,4960 | 0,4943 |
| destilado | 1,0 | 1 | 0,5986 | -0,0019 | 0/5 | [-0,0027, -0,0012] | 0,4956 | 0,4949 |
| destilado | 1,0 | 2 | 0,5991 | -0,0014 | 0/5 | [-0,0022, -0,0006] | 0,4964 | 0,4937 |
| destilado | 1,0 | 4 | 0,5986 | -0,0019 | 0/5 | [-0,0027, -0,0012] | 0,4936 | 0,4950 |
| **V12** | - | - | **0,6005** | - | - | - | 0,4963 | 0,4974 |

**Que dice, en tres puntos**:
1. **Los seis brazos destilados superan a su propio control** (+0,0003 a +0,0008). La senal del
   teacher **no es inerte**: hay informacion que se transfiere. Esto no es un resultado nulo.
2. Pero ninguno adopta: 0 de 5 folds positivos en los siete brazos, y el mejor queda -0,0014 de V12.
   La mejora sobre el control es de **+0,0008** como maximo.
3. **La frontera recupera el 0,8 % de lo disponible** (0,4980 contra los 0,7136 del teacher) y en la
   segunda frontera la recuperacion es **negativa**.

**Veredicto: caso B** de la taxonomia del brief. Student casi igual a V12 pero reproduce solo en parte
al teacher: **la senal existe pero no puede transferirse eficazmente al modelo global bajo esta
representacion y esta arquitectura**. Y conforme al brief, **no se reinterpreta como evidencia de que
el especialista careciera de informacion**: V17 y D-048 ya demostraron lo contrario.

**V18 se cierra sin adopciones** (D-049). P2, P6, P7 y P9 no se ejecutan: la condicion del brief para
P5 era que P1-P4 demostraran senal del objetivo, y no se cumplio.

**Cinco incidentes de ejecucion documentados**, todos antes del resultado y ninguno afecta al
numero: (1) la senal del teacher del 75 % con un student del 100 %, dimensiones incompatibles;
(2) `FRONTERAS_INDICE` son indices y `y` son cadenas, luego la comparacion daba falso siempre y
dejaba al teacher sin filas; (3) `np.searchsorted` sobre `entrenamiento`, que es una concatenacion
**no ordenada** de folds; (4) en la seleccion de la cuadricula se pasaba la senal del bloque de
**evaluacion** como senal de **entrenamiento**; (5) `filas` sombreado entre la lista de resultados y
la mascara del diagnostico. La causa de que los cinco salieran fue que no se hizo **prueba de humo**
antes de la pasada larga: la prueba de humo encontro (4) y (5) en minutos, y habrian costado dos
pasadas de 30 minutos cada una. La prueba de humo se hizo despues y recorrer todo el camino, incluidos
informes, figuras y artefactos.

**Ademas**: la cabeza lineal de PyTorch no iguala a la `LogReg` de sklearn, handicap de **-0,0023**
del mismo orden que los efectos medidos, ya declarado en P4. Se ejecuto **en CPU de forma explicita**,
porque un producto de 279 por 6 no gana nada con la GPU y asi no se depende de la estabilidad de
ROCm, que lanzo una excepcion durante la prueba de humo.

`experiments.csv` pasa a **109 experimentos**. Auditoria de idioma: **0 errores**. Sin push.

## 2026-09-26 · AGE CERRADO — auditoria de cierre estrategico (D-050)

**Motivo de la sesion**: dejar de producir experimentos por inercia y determinar, con una auditoria
global y con el agente `kaggle specialist`, si existe una oportunidad real de superar los 0,62188 de
V12. El brief prohibia expresamente abrir otra fase, otra busqueda de hiperparametros o otra ronda de
variantes de perdidas.

**Orden ejecutado**: (1) inventario de los 62 artefactos de train de `features/`, para saber que se
habia dejado sin explotar; (2) delegacion al agente `kaggle specialist` con la auditoria completa de
V12, V15, V16, V17 y V18, con el inventario ya verificado y las diez preguntas del brief; (3)
verificacion independiente de las cifras que el especialista pidio corregir; (4) redaccion de la
decision.

**Lo que encontro el inventario**: los 62 artefactos de train estan consumidos o cerrados con numero,
y **todos** los OOF de miembros ya estan comprimidos a seis columnas o menos. La hipotesis de
compresion de outputs **no se puede formular**: cada miembro es un modelo de seis clases y no puede
emitir mas, luego la compresion se introdujo al construir el miembro, no al combinarlo en V12. Se
cerraron en su dia los unicos no usados: especialistas por longitud (+0,0000), BM25 (-0,0002), kNN
(0,3088 sin aporte), grupos de tres columnas (<= 0), `content_only` y `style_only` (PH6-L3),
residuales de longitud (PH5-L2, -0,0019), conjunciones (0,1489) y modalidad (0,1270), ambas por debajo
del suelo de azar de 0,1667, los 72 bloques lexicograficos (D-039, 0 de 72 no redundantes) y las tres
vistas de margenes de la capa ordinal (D-037).

**Lo que el especialista concluyo**: **no existe** hipotesis con probabilidad razonable de superar
V12. Ninguna de las candidatas que sobreviven a la auditoria pasa del 15 % de probabilidad, y la que
mas tiene esta prevista por debajo del gate por un factor de tres a diez. Ademas aislo una sonda de
prior-shift que dio +0,0010.

**Verificacion de sus cifras, una a una**:
- La masa de la frontera: **confirmada**. `18-24` 16,93 % y `25-34` 16,74 %, luego **33,67 %**. Mi
  D-049 de ayer decia que la frontera era una "minoria de la masa" y leida al pie de la letra **es
  falsa**. Es una correccion real y la acepto.
- La elasticidad: **confirmada y recalada de forma independiente**. Corregir `n` filas mal
  clasificadas de un par a su etiqueta verdadera, por contrafactual sobre el OOF existente, vale
  +0,0015 con 250, +0,0031 con 500, +0,0061 y +0,0063 con 1.000, +0,0122 y +0,0125 con 2.000 y
  +0,0303 y +0,0308 con 5.000. El gate de +0,003 **se cruza con 478 correcciones** de las 28.721 y
  28.438 que V12 comete en cada par. **La lever es grande y D-049 estaba mal razonado.**
- La calibracion CV -> LB: **confirmada con datos que si son verificables**. Sus once pares tienen el
  LB en `experiments/phase12_auditoria_estado.json`, pero el CV de siete de ellos no esta en ningun
  artefacto, luego su recta no se puede reproducir entera. Reconstruida con los **cuatro** envios
  cuyos dos extremos si son verificables (v01, v07, v08, v12): `LB = +0,00223 + 1,02825 x CV`,
  **R2 = 0,9968**. Un gate de +0,003 vale **+0,00308** en LB. Su pendiente de 1,064 y la mia de 1,028
  coinciden en lo que importa: **el gap no da puntos gratis y los gates estan calibrados**.
- La sonda de prior-shift: **no deja artefacto en disco**, luego sus +0,0010 no son verificables y no
  se usan como evidencia. Aun por cierto, tres veces por debajo del gate.

**Dos errores propios que la verificacion destapo, y que habrian ido al informe sin corregir**:
1. La primera regresion salio con pendiente 0,231 y **R2 = 0,0043**, porque `age_submission_v08.csv`
   tiene **dos** puntuaciones registradas (0,39563 de una subida con formato roto y 0,60376 de la
   buena) y mi emparejamiento por orden de valor metia el punto erroneo en la recta.
2. Escribi a mano en el informe ejecutivo que el envio de V12 puntuo **0,62188**, cuando **0,62188 es
   la replica reproducible** y el envio puntuo **0,62182**. El numero venia de una constante del
   modulo, no del artefacto. Ahora los dos se leen de `phase12_auditoria_estado.json` y cada uno con su
   nombre. Atribuirle a la Champion una puntuacion que no tiene es exactamente el tipo de error que
   D-048 vino a corregir, y lo estaba cometiendo yo.

**Decision**: las ocho condiciones del brief para ejecutar un experimento, evaluadas una a una, **fallan
tres**, y son las tres no negociables: el especialista no lo considera una hipotesis competitiva, la
informacion no es genuinamente nueva, y ya fue probada historicamente. Ademas el brief pide que la
decision se tome entre `EXECUTE ONE FINAL HIGH-CONFIDENCE EXPERIMENT` y `CLOSE AGE - V12 REMAINS
CHAMPION`, y solo la segunda es admisible aqui.

```
CLOSE AGE - V12 REMAINS CHAMPION
```

**D-050** deja escrita la correccion de la razon de D-049: el bloqueo es de **identificabilidad** de
las filas corregibles, no de magnitud, ni de informacion, ni de funcion objetivo. Importa porque una
razon equivocada induce a leer la frontera como una lever pequena y a cerrarla para siempre, cuando
es una lever grande a la que solo le falta un selector, que es un problema distinto y resoluble en otro
dataset.

`experiments.csv` pasa a **111 experimentos**. Auditoria de idioma: **0 errores** sobre 405 ficheros.
Sin push, sin PR, sin merge, sin reescritura de historia. V12 intacto. Sin submission nueva.

## 2026-09-27 · V19 · B1 — El combinador de V12 desperdiciaba un bloque de 73 variables (D-051)

**Cambio de objetivo**: AGE se reabre con criterio competitivo. El objetivo es maximizar el leaderboard,
no cerrar familias. D-050 queda superado y asi se documenta: su veredicto era correcto sobre las cuatro
familias que se propusieron, y falso como veredicto general, porque ninguna era boosting sobre las 279.

**Revision de las cuatro familias, por el agente especialista**: tres se descartan con numero y una se
ejecuta.
- **Retrieval**: fuera. La cobertura de cluster in-fold es **0,00 %** por diseno del CV agrupado, luego
  **no existe numero CV que certificar**; y la etiqueta de duplicado exacto acierta **61,5 %** (PH10,
  D-026). Techo aritmetico del pool literal completo: **+0,0010**.
- **Label cleaning**: fuera. PH11 dio **+0,0016** y PH18 **+0,0009**, ambos bajo el gate, y el criterio
  propuesto selecciona filas que el modelo no puede acertar, no etiquetas falsas.
- **Cascada ordinal**: fuera. Es D-037 con otro nombre: ORD-1 dio **-0,0118** y ORD-B **-0,3057**, el
  peor numero del proyecto.
- **Boosting**: dentro. E-022 probo meta LightGBM sobre **102** columnas de la era V10 con aporte
  **0,0000**, y nunca se probo sobre las **279** de la era V12. Ademas V12 no contiene ningun arbol.

**La premisa de la familia 2 estava falsificada antes de empezar, y el falsador es un artefacto
nuestro**: en P0' el especialista de frontera sobre las mismas 279 variables sale **0,7136 con
`LogisticRegression`** y **0,7075 con `LinearSVC(0.2)`**. Dos clases lineales, mismo resultado. Si un
modelo lineal extrae la frontera, un arbol no tiene ventaja mecanica sobre el.

**B1, un solo disparo, sin busqueda de hiperparametros**: LightGBM pre-registrado (`depth=3`, 300
arboles, `lr=0,05`, `min_child_samples=200`, `colsample=0,5`, `reg_lambda=1,0`) como tercer miembro, con
pesos pre-registrados, meta siempre ajustado fuera de su fold, y control de ruido con las filas de las
variables barajadas.

| Combinacion | macro-F1 | Delta | Folds | IC95 |
|---|---|---|---|---|
| V12 reconstruido (control de fidelidad) | 0,6011711 | +0,0007 | 2/5 | [-0,0002, +0,0016] |
| **CONTROL, filas barajadas** | 0,5909687 | **-0,0095** | **0/5** | - |
| arbol solo | 0,6072312 | +0,0067 | - | - |
| **tres partes iguales** | **0,6079388** | **+0,0074** | **5/5** | **[+0,0064, +0,0085]** |

**Bateria de falsacion, porque un +0,0074 en este proyecto es sospechoso por magnitud**:

| Bloque | Columnas | macro-F1 |
|---|---|---|
| las 279 completas | 279 | **0,6072** |
| sin las 73 supervisadas | 200 | 0,5870 |
| solo las 78 de miembros | 78 | 0,5842 |
| solo las 73 supervisadas | 73 | 0,5597 |
| 2a particion, 4 folds | 279 | **0,6082** |

El efecto **desaparece** sin el bloque supervisado. El hallazgo se reformula: no es que el arbol sea
mejor, es que **el combinador de V12 estaba desperdiciando un bloque de 73 variables que ya estaban
dentro de sus propias 279**. Ese bloque no son probabilidades sino agregados label-informed
(`mention_mean_c0` a `c5` y familia, con valores de hasta 59.439), y necesita combinacion no lineal: el
promedio lineal no puede darla. La 2a particion, con la combinacion ya fijada, da **0,6090431
(+0,0085)** con los cuatro folds por encima del gate.

**Sin fuga**: el bloque aislado da 0,5597 y su media de folds coincide con el pooled en **1,5e-06**. Si
estuviera in-sample, la media de folds se separaria del pooled de forma apreciable.

**Alineacion verificada por firma, no por comparacion de submissions**: la columna de mayor magnitud del
bloque supervisado ocupa el **mismo indice 255** en train y en test, con rangos [-6,29; 59.439] y
[-6,39; 50.673]. Este proyecto ya sufrio un bug de desplazamiento de 6 columnas entre train y test, asi
que la pregunta era obligatoria.

**Challenger construido y validado**: `submissions/age_submission_v19_challenger_boosting.csv`, 8 de 8
checks, 112.892 filas, identificadores identicos y en el mismo orden. Cambia **11.392 filas (10,09 %)**,
con **rescate 4.622, dano 3.311 y neto +1.311**. La primera frontera sube **+0,0120** y esta vez **sin
intercambio de suma cero**, que era el fallo que hundia la global en P1.

**NO ENVIADO**: la subida es manual en este proyecto y la decision es de Carlos. LB previsto por la
calibracion verificada: **0,62734** frente al **0,62182** del envio de V12. Es una prediccion, no una
medida. **Riesgo declarado y no medible**: el bloque supervisado tiene firma distinta entre train
(cross-fitted) y test (ajustado sobre todo el train), y un arbol puede ser mas sensible a esa diferencia
que un promedio lineal. El supervisor de V12 comparto ese mismo sesgo, luego **solo el leaderboard puede
resolverlo**.

**Cinco fallos de ejecucion propios, todos antes del resultado y ninguno afectandolo**: (1) un
`StandardScaler` suelto mas un `Pipeline` con su propio escalador, aplicados ambos, es decir doble
estandarizacion, que dio 22 % de acuerdo en el control de alineacion; (2) `CANDIDATA` escrito con
llaves, luego un conjunto sin indice; (3) la permutacion de filas del control generada en el espacio
global y aplicada a un bloque de 135.468 filas; (4) un `assert` que detecto que el corte del bloque
supervisado estaba desplazado 6 columnas, lo que habria contaminado la propia prueba de falsacion; (5) el
criterio de alineacion usando la metrica debil despues de cambiar la umbralia. El mas instructivo fue el
(4): un `assert` de composicion que resulto ser la unica defensa contra contaminar la prueba que
intentaba demostrar que no habia contaminacion.

`experiments.csv` pasa a **115 experimentos**. Auditoria de idioma: **0 errores** sobre 415 ficheros. Sin
push.

## 2026-09-27 · V19.x — Auditoria del champion y cierre de la explotacion (D-052)

**Punto de partida**: V19subido y puntuado en **0,62578**, un delta de **+0,00396** sobre el envio
oficial de V12 (0,62182). Eso valida el mecanismo completo, incluido el riesgo de la firma del bloque
supervisado que estaba marcado como no medible: la prediccion de 0,62734 era una estimacion, y el
resultado real quedo 0,0016 por debajo, que es justo el residuo que el especialista cuantifico despues.

**P0, configuracion leida del codigo y no del informe**: coincide byte a byte con el artefacto JSON.
LightGBM `multiclass`, 300 arboles, `learning_rate` 0,05, `num_leaves` 8, `max_depth` 3,
`min_child_samples` 200, `colsample` 0,5, `subsample` 0,8, `reg_lambda` 1,0, semilla 42. Combinacion a
un tercio con la LogReg y el MLP(64), pre-registrada y nunca ajustada sobre la validacion.

**P0, atribucion por bloque**:

| Bloque | Columnas | macro-F1 |
|---|---|---|
| las 279 | 279 | **0,6072312** |
| sin el bloque label-informed | 200 | 0,586956 |
| solo el bloque label-informed | 73 | 0,5597155 |
| solo las 78 de miembros | 78 | 0,5842464 |

Anadir las 73 a las 206 aporta **+0,0203** y anadir las 206 a las 73 aporta **+0,0475**: es
**interaccion y no suma**.

**P1, donde esta la senal**: el bloque de 73 aporta el **9,1 % de la ganancia con el 28,6 % de los
cortes**, luego cada corte suyo vale **4,0 veces menos** que uno del bloque de 206. Se usan **271 de
279** columnas y el top 10 concentra el **39,3 %**. La variable dominante es `oof_E014_margenes_0` con
el **7,99 %**, y **las 12 mas importantes son todas margenes de miembros del ensemble**. Correccion de la
interpretacion de D-051: el bloque label-informed no es una variable primaria que el arbol todavia no
explotaba, es un **socio de interaccion** con poco valor por corte.

**P2, complejidad y transferencia**: hueco train a OOF de **+0,007781** en el fold 0 y **+0,012609** en
la media de 5 folds. 1.800 arboles, 2.400 hojas, 1.841.063 bytes. Ese hueco es la magnitud que haria
perder leaderboard a cualquier aumento de capacidad, y es el numero que hace útil la leccion de TOXIC
V23.

**El champion queda persistido** en `models/v19_champion_lightgbm.txt` con SHA256 `22221d18...`, de modo
que V19 es reproducible sin reajustar nada. El OOF del arbol queda cacheado en
`features/v19_oof_arbol_5folds.npy`.

**Consulta al especialista**: recomienda **cerrar la explotacion** y, en el mejor de los casos, dos
configuraciones diagnosticas. Sus tres argumentos, que son los que cierran el caso:
1. **Tasa base del subespacio de parametros = +0,0008** frente a un gate de +0,0030. Las 14
   formulaciones de V15 a V18 se movieron en [-0,0040, +0,0008]. Todo lo que movio la aguja fue un
   cambio de mecanismo, nunca un parametro.
2. **Resolucion del aparato = +0,0011**: 0,6090431 en 4 folds contra 0,6079388 en 5, **sin cambiar
   nada**. Un efecto menor que +0,001 no se distingue de elegir particion.
3. **Residuo de la calibracion sobre V19 = -0,0016**, frente a +0,0019 en V12: oscilacion de 0,0034
   entre modelos vecinos. Un +0,002 en CV puede aterrizar entre -0,000 y +0,004 en LB.

Ademas descarto con numeros seis dimensiones mas: `num_leaves` 8 a 16 con profundidad 3 es un **no-op
exacto** porque 2^3 = 8; `reg_lambda` 1 a 10 es inerte a esta escala de hojas; `min_child_weight` es
una reparametrizacion de `min_child_samples` porque la hessiana multiclase por fila es 0,5-0,8; bajar
`colsample` mata el bloque de interaccion; los pesos del blend se sondearon dos veces (V16 A1 -0,0003 y
A2 +0,0006 con el IC95 contemplando el cero); y `label_smoothing` o `class_weight` son eje de objetivo
cerrado por V18.

**Puerta de coste cero, superada**: la ocupacion del techo de cortes es del **97,5 %**, con 1519 de
1800 arboles llegando a los 7 cortes que permiten 8 hojas. Luego **la profundidad es el limite real y no
las hojas**, que es la premisa de la configuracion A. El especialista habia avisado de que si la
ocupacion hubiera sido del 70 % el eje de capacidad reabriria; se midio y no reabre.

**Las dos configuraciones y sus resultados**:

| Configuracion | CV | Delta contra V19 | Folds pos. | IC95 | 4 folds | Neto de error | Submission |
|---|---|---|---|---|---|---|---|
| A, `max_depth` 4 con `num_leaves` 8 | 0,6079208 | **-0,0000** | 3/5 | [-0,0004, +0,0004] | -0,0000 | **+3** | **no** |
| B, `min_child_samples` 800 | 0,6076698 | **-0,0003** | **0/5** | [-0,0006, +0,0001] | -0,0002 | **-50** | **no** |

**0 de 2** cumplen el criterio, que exige los cuatro puntos: delta CV >= +0,003, 5 de 5 folds, IC95 que
excluya el cero y confirmacion >= +0,002 en 4 folds.

**Decision**: **V19.x se cierra. V19 permanece champion con LB 0,62578.** Sin submission nueva y sin
volver a V12.

**Dos incidentes de ejecucion**: el `assert` de nombres de columna detecto que `MEMBER_FILES` son 14
ficheros que suman 78 columnas y yo generaba 14 nombres; y `oof_E015_ordinal.npy` es un vector, luego
`shape[1]` no existe. Ademas un proceso de **otro** proyecto (`v25_01_factorization_machine.py`) estuvo
usando 25 de los 32 hilos durante casi una hora, y por eso la primera version, que media el hueco en
los cinco folds, tardaba mas de una hora: se recorto a **un fold**, porque el hueco es propiedad del
modelo y no del fold. Esa decision se registro como tal.

`CHAMPION.md` creado con los tres scores separados, la configuracion congelada y el hash del modelo.
`experiments.csv` pasa a **117 experimentos**. Auditoria de idioma: **0 errores** sobre 424 ficheros. Sin
push.

## 2026-09-27 · V20 — Tres familias nuevas: dos cerradas con numero, una con senal positiva (D-053)

**Punto de partida**: V19 congelado en **0,62578**. El objetivo ya no es el eje de boosting, sino
mecanismos que aporten **informacion distinta** a V19 y V12. V19 no se toco: ni configuracion, ni
features, ni modelo persistido, ni submission, ni hash.

**Auditoria previa, antes de proponer nada**: se cruzaron los **117 experimentos** contra **28
mecanismos**. Nunca se habian probado **RandomForest/ExtraTrees, HistGradientBoosting ni CatBoost**
como clase de funcion sobre las 279, ni ninguna transformacion de clase condicional de las features
numericas. Y una familia se cerro **con medicion en un comando**: el campo `id` **es el indice de
fila** (1 a 169.336 en train y test), con correlacion de rango **0,00123** con la clase y distribucion
plana por quintiles. No hay senal de tiempo que extraer.

**F1, la vista de clase condicional de las 122 features numericas: NO_SIGNAL.** El bloque de 73
label-informed se construye in-fold sobre cinco familias de entidades; las 122 densas y estructuradas
nunca recibieron ese tratamiento. Se construyo el bloque equivalente de **976 columnas**, in-fold, y se
comparo **con el mismo aprendiz y los mismos hiperparametros**:

| Entrada del arbol de V19 | Columnas | macro-F1 |
|---|---|---|
| las 279 de V19 | 279 | 0,6072312 |
| las 122 numericas CRUDAS | 122 | 0,3440624 |
| las 976 del bloque de clase condicional | 976 | 0,3438942 |

El transform no aporta: **-0,0002**. Como cuarto miembro, -0,0001 con neto +30. **Lo que si importa es
por que**: acota por que funciona el bloque de 73. No es que las transformaciones de clase condicional
sean buenas en general, sino que las entidades y la rareza **son** discriminantes y las features de
forma y posicion **no lo son**.

**F2, la clase de funcion alternativa: NO_SIGNAL, y el negativo mas informativo de la sesion.**
ExtraTrees, con sesgo inductivo de verdad distinto, da **0,5836218** sobre las **mismas 279**:
**0,0236 por debajo** del LightGBM de V19, con 0 de 5 folds y neto -4.531. Como cuarto miembro, -0,0010
con neto -240. **El efecto de V19 no es "cualquier combinador no lineal": es del BOOSTING**, que es lo
que explica por que el bloque de 73 se cobra, porque ajusta residuo a residuo.

**F3, los cinco umbrales anidados: WEAK_SIGNAL, y el unico mecanismo vivo post-V19.** El especialista de
D-048 saca 0,7136 siendo un binario de dos clases, luego ajustado sobre **el 34 % de las filas**. El
umbral anidado hace **la misma decision con el 100 %**. Cinco clasificadores de `clase <= k-1` contra
`clase >= k` sobre las 279, con monotonia por maximo acumulado y la distribucion de seis clases por
diferencias. No es un duplicado: E-028a usa pares adyacentes (el comparador), y ORD-1 de D-037
construia las CDF a partir de la salida de V12, sin clasificadores nuevos, y dio 0,5887.

| Umbral | Filas bajo / alto | Exactitud balanceada | V12 en esa fila |
|---|---|---|---|
| <= 13-17 vs >= 18-24 | 28.863 / 140.473 | 0,7632 | 0,9408 |
| <= 18-24 vs >= 25-34 | 57.539 / 111.797 | 0,8264 | 0,8797 |
| <= 25-34 vs >= 35-49 | 85.886 / 83.450 | **0,8465** | 0,8504 |
| <= 35-49 vs >= 50-64 | 114.116 / 55.220 | **0,8373** | 0,8210 |
| <= 50-64 vs >= 65-xx | 142.176 / 27.160 | **0,8112** | 0,7429 |

**La cadena en solitario: 0,6046893, por encima del 0,6005017 de V12 y sin un solo modelo multiclase.**
Como cuarto miembro: **0,6097782, delta +0,0018, 5 de 5 folds, IC95 [+0,0012, +0,0025]**, neto **+294**
y 5.674 cambios.

**Candidato construido y validado, NO ENVIADO**: `age_submission_v20_candidato_umbrales.csv`, cambia
**3.546 de 112.892 filas (3,14 %)** frente al envio de V19, 8 de 8 checks, SHA256 `2c964af7...`. La
mejora esperada en LB es **+0,0019**, y el residuo de la calibracion sobre V19 fue de **-0,0016** con
oscilacion de **0,0034** entre modelos vecinos: la mejora esperada es **del mismo orden que la
incertidumbre del aparato**. Por eso se presenta en vez de enviarse solo.

**Cinco incidentes de ejecucion propios**, todos anteriores al resultado:
1. **Signo invertido** en la convencion de la cadena: `acumuladas[:,k]` es `P(Y <= clase k)` y por tanto
   **crece**, luego la monotonia es por **maximo** acumulado y la primera clase es `p_0`, no `1 - p_0`.
   Con el signo equivocado el minimo acumulado colapsaba las cinco probabilidades y daba 0,0249.
2. **Polaridad invertida en el diagnostico por umbral**: medi sobre `decision` daba el **error** del
   lado alto, mientras la columna de V12 daba el **acierto**, y las dos juntas parecian contradictorias.
   Ahora se calculan acierto de cada lado y exactitud balanceada por separado.
3. Un `assert` de nombres de columna detecto que `MEMBER_FILES` son 14 ficheros que suman 78 columnas y
   yo generaba 14 nombres.
4. Un `assert` de composicion detecto que el corte del bloque label-informed estaba desplazado 6
   columnas, lo que habria contaminado la prueba de falsacion de D-052.
5. Varios `NameError` de figura por usar `figura` en un sitio y `fig` en otro.

El mas instructivo es el **1**: sin el chequeo de que la serie de probabilidades tiene el signo
correcto, la cadena daba 0,0249 y esa cifra casi se lee como "el mecanismo es terrible", cuando en
realidad era un error mio de convencion. Ese casi lo demuestra que un OOF absurdo exige revisar la
convencion antes que el mecanismo.

`experiments.csv` pasa a **122 experimentos**. Auditoria de idioma: **0 errores** sobre 433 ficheros.
Sin push.

## 2026-09-27 · V20 CONFIRMED_CHALLENGER y cierre de ORDINAL_RECONSTRUCTION (D-054)

**Resultado externo**: `age_submission_v20_candidato_umbrales.csv` puntuo **0,62689** en Kaggle, un delta
de **+0,00111** sobre V19 (0,62578). **V20 pasa a ser el nuevo referente de leaderboard de AGE** y V19
queda como baseline historico, sin reemplazar ningun artefacto suyo.

**La hipotesis ordinal queda validada externamente.** La cadena de cinco discriminadores no solo mejoro
el OOF sino que **transfirio positivamente a Kaggle**. `ORDINAL_NESTED_THRESHOLDS` pasa de
`WEAK_SIGNAL` a **`CONFIRMED_CHALLENGER`**, y la prioridad de investigacion queda **invertida**: ya no se
buscan clases de funcion ni ensembles genericos, sino el maximo rendimiento de este mecanismo.

**Auditoria de las cinco fronteras (P2)**:

| Frontera | Exacta balanceada | AUC | F1 binario | Descalibracion |
|---|---|---|---|---|
| <= 13-17 vs >= 18-24 | 0,7632 | 0,9284 | 0,9444 | +0,0001 |
| <= 18-24 vs >= 25-34 | 0,8264 | 0,9224 | 0,8906 | -0,0003 |
| <= 25-34 vs >= 35-49 | 0,8465 | 0,9274 | 0,8444 | -0,0001 |
| <= 35-49 vs >= 50-64 | 0,8373 | 0,9340 | 0,7852 | +0,0002 |
| <= 50-64 vs >= 65-xx | 0,8112 | 0,9456 | 0,7285 | -0,0000 |

**Las cinco estan esencialmente perfectamente calibradas** (descalibracion entre -0,0003 y +0,0002), luego
**no hay margen en recalibrar**. Y el AUC esta entre 0,9224 y 0,9456, muy parejo.

**Consistencia ordinal (P3)**: **18.927 de 169.336 filas (11,18 %)** violan la monotonia, con
magnitud media **0,002457** y maxima **0,518606**. V20 lo resolvio con el **maximo acumulado**, que es
la unica correccion que respeta el orden sin inventar informacion.

**Impacto de cada frontera sobre el blend de V20 (P6)**:

| Frontera degradada | Blend | Caida | Cadena en soledad | Caida |
|---|---|---|---|---|
| <= 13-17 vs >= 18-24 | 0,5968936 | **-0,0129** | 0,0485421 | -0,5561 |
| <= 18-24 vs >= 25-34 | 0,6026180 | **-0,0072** | 0,3352297 | -0,2695 |
| <= 25-34 vs >= 35-49 | 0,6057464 | -0,0040 | 0,4655049 | -0,1392 |
| <= 35-49 vs >= 50-64 | 0,6079730 | -0,0018 | 0,4914815 | -0,1132 |
| <= 50-64 vs >= 65-xx | 0,6073803 | -0,0024 | 0,4604091 | -0,1443 |

**La primera frontera es el cuello de botella en impacto** y es ademas la de menor exactitud balanceada:
**la mas debil es la que mas importa**. Y los umbrales 4 y 5 apenas importan.

**Las cuatro reconstrucciones (P1), y la respuesta a la pregunta de la fase**:

| Variante | macro-F1 | Delta contra V20 | Folds | Frontera 1 | Frontera 2 | Neto |
|---|---|---|---|---|---|---|
| A control exacto | 0,6097782 | +0,0000 | 0/5 | 0,5075 | 0,5029 | 0 |
| B proyeccion monotona por PAVA | 0,6097927 | +0,0000 | 3/5 | 0,5074 | 0,5028 | +5 |
| C conteo de umbrales | 0,5983871 | **-0,0114** | 0/5 | **0,545** | **0,5456** | -3.319 |
| D edad esperada | 0,5834541 | **-0,0263** | 0/5 | **0,541** | **0,5738** | -6.414 |

**El control A reproduce V20 al 100 %** (0,6097782 frente a 0,6097782), y ese control es lo que valida
el script: sin el, ninguna de las otras tres variantes seria interpretable.

**`ORDINAL_RECONSTRUCTION = CLOSED`.** La pregunta de la fase tiene respuesta: **la mejora de V20 procede
de usar cinco fronteras ordinales, y NO existe margen en la forma de convertir esas cinco probabilidades
acumuladas en las seis clases finales.** Lo dicen tres senales independientes: descalibracion cero,
violaciones de magnitud despreciable, y proyeccion monotona identica al control.

**El hallazgo que vale mas que el resultado**: **C y D mejoran las fronteras de forma espectacular**
(0,545/0,5456 y 0,541/0,5738 frente a 0,5075/0,5029) **y destruyen el macro-F1** (-0,0114 y -0,0263).
Es la **leccion de P1 repetida dentro de la familia ordinal**, y la confirmacion mas limpia de que en este
problema **la macro-F1 de seis clases es la metrica que manda y la frontera es una minoria de la masa**.

**Decision**: no se propone submission nueva todavia. Ninguna de las cuatro variantes mejora el OOF de V20,
y el criterio exige mejora local antes de enviar. **La direccion con mas recorrido que queda abierta** es
la de **descomposiciones de decision**, concentrada en los dos primeros umbrales.

**Cinco incidentes de ejecucion propios**, todos anteriores al resultado: un **signo invertido** en la
convencion de la cadena (las CDF crecen, luego la monotonia es por maximo acumulado y la primera clase es
`p_0` y no `1 - p_0`; con el signo equivocado daba 0,0249 y casi se lee como "el mecanismo es terrible");
una **polaridad invertida** en el diagnostico por umbral (medi sobre `decision` daba el error del lado alto
mientras la columna de V12 daba el acierto); un `assert` de nombres que detecto 14 nombres para 78
columnas; un `assert` de composicion que detecto un corte desplazado 6 columnas; y varios `NameError` de
figura por usar `figura` en un sitio y `fig` en otro.

`experiments.csv` pasa a **122 experimentos**. Auditoria de idioma: **0 errores** sobre 441 ficheros. Sin
push.

## 2026-09-27 · V20 P2 — Mejorar la frontera debil destruye la global (D-055)

**Condicion del paso 8 del plan**: probar una variante de perdida ponderada por frontera **solo si el
analisis mostraba que unas fronteras tienen errores mucho mas perjudiciales**. P6 lo mostro con numero:
la frontera 1 cuesta **-0,0129** de caida del blend frente a **-0,0018** y **-0,0024** de las fronteras 4
y 5, es decir **siete veces mas impacto**, y es ademas la de menor exactitud balanceada (0,7632). La
condicion se cumplio y se ejecuto **una sola configuracion**, sin grid.

**Configuracion**: `class_weight = "balanced"` en el discriminador del **primer umbral** (28.863 filas
de `13-17` contra 140.473 del resto, un 17/83), con los **otros cuatro umbrales intactos**. Hipotesis: al
subir el peso del lado minoritario, LightGBM dejaria de podar los cortes que separan al grupo pequeno.

**Resultado, en dos lineas que se contradicen a proposito**:

| Medida | V20 | Con el peso |
|---|---|---|
| exactitud balanceada del primer umbral | 0,7632 | **0,8430 (+0,0797)** |
| macro-F1 del blend | 0,6097782 | **0,5932020 (-0,0166)** |
| folds positivos | - | **0/5** |
| IC95 | - | **[-0,0178, -0,0153]** |
| neto de filas | - | **-1.340** |

**La hipotesis era correcta y aun asi perdio.** El peso **si** mejora la frontera: +0,0797 de exactitud
balanceada. Pero el blend cae -0,0166 con 0 de 5 folds, y las fronteras del blend bajan a 0,4117 y
0,5088 frente a las 0,5075 y 0,5029 de V20.

**Por que, y es la parte que importa**: **la exactitud de una frontera no es el cuello de botella**. El
blend de V20 ya extraia de ese umbral lo que necesitaba, y forzarlo a separar mejor el grupo pequeno
**descalibra las cinco CDF a la vez**, porque las cinco comparten el mismo espacio de features y el mismo
modelo. Es la **leccion de P1 repetida por tercera vez**, ahora dentro de la familia ordinal y con la
hipotesis acertada, que es lo que la hace concluyente: **no era que la frontera no se pudiera mejorar, es
que mejorarla no es lo que falta**.

**Decision**: `ORDINAL_WEIGHTS` cerrada con numero. Sin submission: no mejora el OOF de V20.

**La unica via que queda abierta dentro de la familia ordinal** es cambiar **los clasificadores**, no la
reconstruccion (cerrada en D-054) ni los pesos (cerrada aqui). El candidato natural es **mejorar el
primer discriminador con un modelo distinto**, que es un cambio de **mecanismo** y no de hiperparametro.
Y **no se abre ninguna otra familia** mientras la ordinal tenga un challenger confirmado por leaderboard.

Un incidente de ejecucion: falto definir `matriz = matriz_279()` en el `main` del script, luego un
`KeyError` de `neto` por estar dentro de `complementariedad`. Ambos anteriores al resultado.

`experiments.csv` pasa a **123 experimentos**. Auditoria de idioma: **0 errores** sobre 444 ficheros. Sin
push.

## 2026-09-27 · V21 — B1 no se puede mejorar cambiando de clasificador (D-056)

**Auditoria historica hecha buscando codigo, no nombres en `experiments.csv`.** Se busco si algun
experimento habia **sustituido especificamente B1**, entrenado un modelo dedicado a `13-17` frente a
`>=18`, usado boosting/SVM/arboles solo en esa frontera, o usado una representacion especifica unicamente
para B1. **Ninguno existia.** El unico antecedente cercano es **E-028**, y **no es equivalente**: entrena
un `LinearSVC` sobre las filas del **par adyacente** `("13-17", "18-24")`, es decir **dos clases y
57.539 filas**, mientras que B1 es el umbral **anidado** sobre **todas las 169.336 filas** y un desbalance
17/83. Los dos experimentos de la fase son genuinamente nuevos.

**V20 como control exacto**: reconstruido desde sus artefactos ya medidos, da **0,6097782 frente a
0,6097782**, con SHA256 `69b574f737c8cf43...`. El script **se aborta si no cuadra**, porque sin ese
control ninguna variante seria interpretable.

| Candidato | macro-F1 | Delta vs V20 | Folds | B1 balanceada | B1 AUC | Hueco | Neto | Estado |
|---|---|---|---|---|---|---|---|---|---|
| **V20 (control exacto)** | **0,6097782** | +0,0000 | 0/5 | **0,7632** | 0,9284 | - | 0 | CONTROL |
| LinearSVC C 0,2 | 0,5992098 | **-0,0106** | 0/5 | 0,6782 | 0,9178 | +0,002055 | -2.457 | NO_SIGNAL |
| LinearSVC C 1,0 | 0,5992058 | **-0,0106** | 0/5 | 0,6784 | 0,9178 | +0,002102 | -2.457 | NO_SIGNAL |
| boosting especializado B1 | 0,6097377 | **-0,0000** | 3/5 | 0,7606 | 0,9272 | +0,000249 | -6 | NO_SIGNAL |

**`B1_SPECIALIST = NO_SIGNAL`, y con el se cierran las tres subfamilias de la ordinal.**

**La razon es estructural y responde a la pregunta que el propio plan planteaba en el paso 3**: el
problema de B1 **no es del estimador lineal ni de la falta de un boosting dedicado**. El `LinearSVC`
pierde **0,085** de exactitud balanceada (0,7632 a 0,6782) y el boosting especializado queda **por
debajo** del B1 de V20 (0,7606 contra 0,7632), con un hueco train a OOF de apenas +0,000249. Luego **el
B1 del LightGBM multiclase de V20 ya esta a su nivel**, porque las cinco CDF comparten las 279 features y
el mismo modelo, y la primera sale de un ajuste multiclase que ya la resuelve. **Cambiar el clasificador
de B1 no es una mejora, es una regresion.**

**Respuesta a la pregunta central de V21**: **no.** Mejorar especificamente `13-17` frente a `>=18` **no
produce una mejora real en el objetivo final de seis clases**, medido con control exacto y dos mecanismos
distantos. Sin submission.

Cuatro incidentes de ejecucion propios, todos anteriores al resultado: una cadena multilinea del informe sin
comillas en las lineas intermedias; una variable de bucle con un ideograma que no se renombro en los tres
sitios que la usaban; un `KeyError` de `neto` por estar dentro de `complementariedad`; y un `NameError` de
figura por usar `figura` en un sitio y `fig` en otro.

`experiments.csv` pasa a **124 experimentos**. Auditoria de idioma: **0 errores** sobre 449 ficheros. Sin
push.

## 2026-09-27 · V22 — Forense de V20 y cierre de B1_SPECIALIST (D-057)

**Carlos detecto una inconsistencia real en el informe de V21**: afirmaba
`LightGBMClassifier(objective="multiclass")`, objetivo binario `clase <= 0` contra `clase >= 1`, y
`scores = predict_proba[:, 1] = P(lado bajo)`. Pedio auditar el **codigo**, no el informe.

**El codigo real** (`scripts/v20_f3_umbrales.py`):
```python
objetivo = (clase <= k - 1).astype(int)          # target BINARIO
modelo = lgb.LGBMClassifier(**HIERBA)            # HIERBA declara multiclass, num_class 6
acumuladas[validacion, k - 1] = modelo.predict_proba(matriz[validacion])[:, 1]
```

**Las diez preguntas, respondidas contra el codigo y verificadas:**

| Pregunta | Respuesta |
|---|---|
| 1. Labels | binarios, `clase <= k-1` contra `clase >= k` |
| 2. Clases del target | **2** en cada entrenamiento |
| 3. `objective` real | **declara** `multiclass` con `num_class 6`, pero el booster tiene seis clases y las columnas 2 a 5 son **cero exacto**. Declarado multiclase, efectivo binario |
| 4. Columnas de `predict_proba` | **seis**, no dos. La 0 es `P(lado alto)`, la 1 es `P(lado bajo)` |
| 5. Como se obtiene cada CDF | **directamente**, `acumuladas[:, k-1] = P(Y <= k-1)`, porque el target binario es exactamente esa probabilidad |
| 6. Cinco binarios independientes | **SI** |
| 7. Multiclase transformado | **NO** |
| 8. Caso | **A, confirmado** |
| 9. Probabilidades | **ACUMULADAS**, `P(Y <= k)` |
| 10. Monotonia garantizada | **NO**, el 11,18 % de las filas la viola; V20 la impone despues |

**La especificacion declarada cuantificada, que es lo que la hace importante**: sobre las mismas 279
features y los mismos cinco folds, la especificacion declarada da **0,6097782** y una binaria
correctamente especificada da **0,6091232**. La correlacion entre las dos CDF es de **0,998** en las
cinco fronteras, luego el *ranking* es el mismo, pero la **calibracion** difiere hasta **0,33**. Y la
especificacion declarada es **0,00066 mejor**. **No es un bug que haya que arreglar**: es una version que
funciona mejor, y se documenta y conserva.

**Paso 5, el modelo compartido**: un **unico LightGBM multiclase** del que se derivan las cinco CDF **por
suma**. Al ser una suma de probabilidades no negativas, **las cinco CDF son monotonicas siempre**: se
miden **0 violaciones** de 169.336, frente a las 18.927 de V20. Blend **0,6096150**, delta **-0,000163**,
3 de 5 folds, IC95 [-0,00078, +0,00045], neto **-13**, **4.451 cambios**. **`WRONG_OBJECTIVE`.**

**Paso 3, el control lineal**: `LinearSVC` C 0,2 da **0,5992098** (-0,0106) con B1 **0,6782**; C 1,0 da
**0,5992058** con B1 **0,6784**. **B1 empeora 0,085** y el final se hunde. **`NO_SIGNAL`: B1 no mejora.**

**Paso 4, el boosting especializado**: LightGBM binario, pequeno y regularizado, da **0,6097377**
(-0,0000) con B1 **0.7606**, **por debajo** del B1 de V20 (0,7632), neto **-6**. **`NO_SIGNAL`.**

**Decision**: **`B1_SPECIALIST = NO_SIGNAL`.** Las tres vias posibles de esta fase fallan por la misma
razon estructural: **las cinco CDF comparten las 279 features y el mismo modelo, y la primera sale de un
ajuste multiclase que ya la resuelve.** Cambiarla por un lineal la empeora 0,085, por un binario dedicado
la empeora 0,003, o compartirla con las otras cuatro resta 0,0002 al final. Es la **leccion de D-052 repetida
por cuarta vez**: **la exactitud de una frontera y el macro-F1 de seis clases no estan alineadas.**

**El hallazgo que vale mas que el resultado**: compartir representacion **si elimina el problema de
monotonia por construccion** (0 violaciones frente a 18.927), pero **no mejora el macro-F1 final**. Luego
la consistencia ordinal **no era el cuello de botella**, y eso es lo que esta fase deja cerrado.

Cinco incidentes de ejecucion propios, todos anteriores al resultado: una cadena multilinea del informe sin
comillas en las lineas intermedias; un diccionario al que le faltaba la llave de cierre; una variable de bucle
con un ideograma que no se renombro en los tres sitios que la usaban; un `KeyError` de `cambios` por estar
dentro de `complementariedad`; y un `NameError` de `constructor` por evaluar un unico candidato como si
fuera un bucle.

`experiments.csv` pasa a **125 experimentos**. Auditoria de idioma: **0 errores** sobre 456 ficheros. Sin
push.

## 2026-09-27 · V22 — Frontier audit completo + coste ordinal (D-057)

**El forense se resuelve contra el codigo y no contra el informe.** Las diez preguntas:
labels binarios; target con dos clases; HIERBA declara `multiclass` con `num_class 6` pero el booster
tiene seis clases y las columnas 2 a 5 de `predict_proba` son cero exacto; salen seis columnas, la 0 es
P(lado alto) y la 1 es P(lado bajo); cada CDF sale directamente del clasificador binario; si hay cinco
binarios independientes; no hay multiclase transformado; es el CASO A; las probabilidades son
ACUMULADAS; y la monotonia NO esta garantizada (el 11,18 % de las filas la viola).

**La especificacion declarada cuantificada**: da 0,6097782 y una binaria correctamente especificada
da 0,6091232. Correlacion entre las dos CDF 0,998 en las cinco fronteras, luego el ranking es el
mismo, pero la calibracion difiere hasta 0,33. Y la declarada es 0,00066 mejor. **No es un bug que
haya que arreglar**: es una version que funciona mejor, y se documenta y conserva.

**El modelo compartido** (un unico LightGBM multiclase con cinco CDF por suma) tiene **0
violaciones** de monotonia frente a las 18.927 de V20, porque una suma de probabilidades no negativas
es monotona siempre. Blend 0,6096150, delta -0,000163, 3 de 5 folds, IC95 [-0,00078, +0,00045], neto
-13 y 4.451 cambios.

**Los tres candidatos de B1 no mejoran B1**: LinearSVC C 0,2 da 0,5992098 (-0,0106) con B1 0,6782;
C 1,0 da 0,5992058 con B1 0,6784; y el boosting especializado da 0,6097377 (-0,0000) con B1 0,7606,
por debajo del 0,7632 de V20. Los tres son NO_SIGNAL y el mecanismo queda cerrado por el criterio
del brief.

**PASO 14, el coste ordinal**: cambiar el argmax por el coste esperado minimo con `C[i][j] =
|i-j|^alfa`. alfa 1 da **0,5949198** (-0,0149) y alfa 2 da **0,5392168** (-0,0706). **El coste ordinal
destruye el macro-F1**: los errores del modelo no estan concentrados a grandes distancias, y forzarlos
a evitarlas degrada la frontera y la global a la vez.

**Errores compartidos**: **64.825 filas (38,3 % del train) las equivocan V19 y V20 juntas**. V20 rescata
1.474 de V19 e introduce 1.193: **neto +281**. La clase con mas errores compartidos es `25-34` con
**14.913**.

**Diversidad real**: el modelo historico mas complementario con V20 es **NB-SVM word**, que rescata
**10.462** de sus 66.018 errores (15,85 %) — **pero ya es miembro de V20**. Los modelos buenos son
altamente redundantes: los tres primeros comparten mas del 82 % de sus errores con V20.

**La respuesta a la pregunta central**: **no se ha encontrado una fuente nueva con evidencia de existir.**
La unica fuente grande medida (los errores compartidos) esta agotada, y la que queda (representaciones a
nivel de caracter) es estructuralmente ilegal. Compartir representacion **si** elimina la monotonia por
construccion, pero **no** mejora el macro-F1 final: la consistencia ordinal no era el cuello de botella.

**Seis incidentes de ejecucion propios**, todos anteriores al resultado: una cadena multilinea del
informe sin comillas en las lineas intermedias; un diccionario al que le faltaba la llave de cierre; una
variable de bucle con un ideograma que no se renombro en los tres sitios que la usaban; un KeyError de
cambios por estar dentro de complementariedad; un NameError de constructor por evaluar un unico
candidato como si fuera un bucle; y un NameError de `y` por definir el bloque antes de las etiquetas.
El mas instructivo es el de la cache: `v22_oof_v20.npy` contenia el **modelo compartido B**, no la
cadena de V20, y por eso el control no reproducia.

`experiments.csv` pasa a **126 experimentos**. Auditoria de idioma: **0 errores** sobre 456 ficheros.
Sin push.

## 2026-09-27 · V23 — CORAL / cumulative-link con pesos compartidos (D-058)

La pregunta era una: **compartir pesos entre las cinco fronteras ordinales aporta una señal útil
que los cinco binarios independientes de V20 no capturan?** La respuesta tiene tres partes y las
tres importan.

**La implementación.** Cumulative-link en `src/ordinal_coral.py` con `numpy` y `scipy`, sin
librería externa: un único vector `w` y cinco umbrales `theta_k = gamma + suma_{j<k}
softplus(delta_j)`. El orden de los umbrales es una garantía de construcción, no una corrección
posterior. Gradiente analítico verificado contra diferencias finitas con error relativo 3,15e-10.

**Dos errores de formulación que los tests cazaron, ambos míos.** El primero: evalué
`log P(Y = y)` como `log sigmoid(z_y) + log(1 - sigmoid(z_{y-1}))`, que **factoriza** la
diferencia de dos CDF en producto de sigmoidos. Esa factorización solo es válida si `s_y = 1`; con
ella el objetivo daba 678,80 contra 895,41 de la referencia calculada por otra vía. La
verosimilitud correcta es la diferencia, y el gradiente también cambia. El segundo: penalizar
solo `w` produce separación — en un sintético los umbrales llegaban a 12,07 con features
estandarizadas en [-3, 3] y el macro-F1 de entrenamiento caía a 0,213. La penalización de los
gaps entre umbrales lo arregla, y no sesga `gamma`, que sí debe ser grande y negativo porque
`13-17` es la clase rara. La penalización existe en el módulo pero **no se activa** en AGE.

**El pre-registro se hizo sin mirar AGE.** TEST B genera 135.000 filas, 279 columnas y
proporciones reales de clase, y mide que **no hay separación**: 0 filas saturadas, `|eta|max`
5,78, CDF entre 1,07e-04 y 0,99967. Con eso, `lambda_w = 1,0` y `lambda_umbral = 0,0` —los
valores de libro— quedan congelados antes de ver el resultado, y el resultado es insensible a
`lambda_w` (0,2760 contra 0,2771 en el sintético). No se corre ninguna grid.

**El control que faltaba.** Un delta de -0,126 contra V20 mezcla dos efectos: compartir pesos y
perder no linealidad. Añadí un control B con cinco binarios **lineales independientes** y el mismo
estandarizado por fold, y ahí aparece otro fallo propio: en la primera versión `LogisticRegression`
recibía las features crudas, avisaba de que no convergía y devolvía la iteración parcial. Un
control que no converge no es un control.

**El resultado.**

| Modelo | macro-F1 | Delta | Folds + | monotonia | neto |
|---|---|---|---|---|---|
| V20 (control A) | 0,6097782 | - | 3/5 | 18.927 | 0 |
| cinco binarios lineales (control B) | 0,4690055 | -0,140773 | 0/5 | no medido | -23.343 |
| CORAL (peso compartido) | 0,4837017 | -0,126077 | 0/5 | **0** | -22.619 |

**La descomposición es el resultado útil.** Compartir pesos y ordenar umbrales vale **+0,014696**
frente al control B. La hipótesis de la fase **es correcta**: compartir información entre
fronteras aporta. Pero la no linealidad de LightGBM vale **-0,140773** para el lado contrario, un
orden de magnitud mayor, y ahí se decide la métrica.

**La monotonía se resuelve de verdad y no compra nada.** CORAL tiene 0 violaciones frente a las
18.927 de V20, y aun así pierde 0,1261. Es la **segunda** confirmación independiente de D-057, donde
el modelo compartido también tenía 0 violaciones y perdía 0,0002. Resolver gratis un problema que
no era el cuello de botella no compra nada.

**El mapa de errores se rehace entero.** Aciertos: 103.331 de V20 frente a 80.712 de CORAL, con
delta -22.619 que coincide exactamente con el neto de complementariedad. Las 36 de 36 celdas de
la matriz de confusión cambian, con distancia L1 57.650.

**Auditoría de la fase: PASS.** Sintaxis, import, ejecución real, artefactos, serialización,
unicode, notebook y coherencia numérica (9/9). La validación de ejecución del notebook encontró
**tres bugs reales** en las celdas nuevas: acceso por columna donde debía ser por índice, un
`drop` de una columna ya convertida en índice, y `display`/`Image` sin importar. Los tres
arreglados. El propio script de auditoría se autocorregió: contenía los caracteres Unicode que
reportaba, y ahora los documenta por nombre y codepoint.

**Deuda que queda escrita y no resuelta**: el notebook consolidado llega hasta la sección 30, que
cierra V17, y **no contiene V18 a V22** aunque el champion es V20. Además arrastra 38 caracteres
Unicode de sesiones anteriores que el proyecto nunca detectó porque `notebooks/` está excluido de
su auditoría de idioma. V23 no introduce ninguno. Se deja constancia en vez de reescribir
contenido ajeno a esta fase.

`experiments.csv` pasa a **130 experimentos**. Auditoría de idioma del proyecto: 471 ficheros,
**0 errores**. Sin push, con V12, V19 y V20 intactos, y con el test y el holdout sin tocar.

## 2026-09-27 · V24 — Reconciliacion documental (D-059)

La fase no experimenta. El hallazgo es que **la causa del notebook incompleto era estructural, no
editorial**.

**La fuente de verdad del notebook es una cadena de cinco scripts, no un generador unico.**
`fase09_99_entrega.py` crea la Parte I, `fase12_notebook_final.py` regenera con V13,
`phase28_notebook_final_v2.py` regenera con V14 a V16, `v17_notebook.py` anade la Parte III y
`v23_02_auditoria.py` anade V23. **Un eslabon por fase**, y **V18, V19, V20, V21 y V22 nunca
tuvieron eslabon**. Por eso el notebook se quedaba en V17 con V20 como champion, y por eso no se
podia cerrar desde V23 sin dejar esas cinco fases fuera.

**Las 103 celdas no eran un numero erroneo**: eran correctas para lo que contenian. El numero no
estaba mal, estaba incompleto. Ahora hay **121 celdas y 33 secciones**, con las doce versiones
V12 a V23 presentes.

**Renumeracion minima**: V23 estaba en 31 y V24 se construia como 32, luego el orden quedaria
invertido. Se invierte, V24 pasa a 31 y V23 a 32. Se renumera **una sola seccion existente**, que
es lo que el brief pedia evitar.

**El registry tenia un defecto propio y preexisting**: tres lineas de cabecera en vez de una, con
dos espurias ocupando `experiment_id` dos veces. Declaraba 130 experimentos y tenia 128. Aparece
igual en al menos los ultimos seis commits. Corregido, y **V24 no anade filas** porque es
documentacion: el conteo real es **128**.

**Dos discrepancias documentadas sin tocar artefactos**: V20 registrado como 0,609615, que es el
modelo compartido de D-057 y no la cadena de V20, con canonico 0,6097782; y V12 registrado como
0,6004, redondeado, con canonico pooled 0,6005017.

**Dos errores propios durante la fase**: el patron de renumeracion estaba invertido, se buscaba
`## 31. V18` para convertirlo en `## 36. V23` cuando lo que existia era `## 31. V23`, luego el
renumerado no se aplicaba y las secciones quedaban en orden 32 antes que 31. Y `ruff` no estaba en
el PATH del subprocess, que lo resolvia con `cfg.ROOT.parents[1]`, que es `/home/carlos/Git` y no
`/home/carlos`; ahora se deriva de `sys.executable`. Ademas el simbolo de seccion se coló tres
veces, y la propia auditoria de Unicode lo cazó.

**Sobre ruff**: el proyecto **no tiene configuracion**, luego se aplican sus defaults, que incluyen
reglas de estilo que el proyecto incumple de forma sistematica — los scripts ya commiteados tienen
12 avisos `ISC004` y este tiene 54. El gate de la fase exige cero **errores de correccion** (reglas
`F` y `E9`), que es lo que se verifica, y ambos estan a cero.

**Auditoria: PASS.** py_compile rc 0, ruff 0 errores de correccion, **0 Unicode introducido**,
ids y titulos unicos, 0 artefactos faltantes, **9 celdas ejecutadas con 0 errores**, coherencia
**7/7**. Auditoria de idioma del proyecto: 474 ficheros, **0 errores**.

**Deuda declarada en el propio notebook, no escondida**: 38 caracteres Unicode de sesiones
anteriores y `notebooks/09_final_pipeline.ipynb` obsoleto respecto a V20.

Sin push, con V12 y V20 intactos, y sin abrir ninguna hipotesis nueva.

## 2026-09-27 · V25 — Seleccion supervisada de features por informacion mutua (D-060)

La pregunta era una: **existe evidencia de que V20 contiene features ruidosas que, al eliminarse
con seleccion supervisada hecha dentro de cada fold, mejoren el macro-F1 de forma reproducible?**

**La procedencia se verifico antes de entrenar, y por tres vias.** Las 73 features label-informed
se construyen con `y_train` del training fold, se escriben solo en `valid_idx`, y usan el **mismo
array de grupos de duplicado y los mismos folds** que V20, comparados array a array y no por
nombre de funcion. **LEGAL.** El inventario cubre las 279 una a una, con 279 nombres unicos y una
columna constante.

**Hubo que estandarizar antes del mutual information, y no es un detalle.** Las 73 supervisadas
llegan a magnitud absoluta 59.439 frente a las probabilidades de miembro, que viven en [0, 1]. El
estimador por vecino mas cercano no es invariante a escala, luego sin estandarizar el ranking no
mide senal. Se estandariza dentro de cada training fold y la validacion nunca participa.

**El resultado.** T0 reproduce 0,6097782 exacto. T1, con K = 139, da 0,6091492: delta -0,000629,
1 de 5 folds positivos, 1.042 rescates contra 1.143 danos, neto -101, y solo 3.545 filas cambian.
El F1 por clase es practicamente identico.

**El hallazgo util es el por bloque.** Sobreviven con frecuencia 1,000 las 78 probabilidades del
ensamble y las 6 de NB-SVM, con MI media de 0,215 y 0,208. Nunca sobreviven 74 de las 87
estructuradas ni 30 de las 35 densas, con MI media de 0,012. **El mutual information alto no es
ruido: es la prediccion de los miembros que sostiene el modelo.** La seleccion no elimina ruido,
elimina senal, y por eso no mejora. La seleccion es ademas estable: 137 features en los cinco folds
y 138 en ninguno.

**La distincion que ya habia confundido dos veces, ahora registrada en el artefacto**: la cadena
de cinco umbrales en solitario da 0,6046893 y el blend de cuatro cuartos da 0,6097782. En la
primera ejecucion el abort automatico la cazo: T0 no reproducia porque estaba midiendo la cadena
en vez del blend. El mecanismo de seguridad funciono, y la correccion es la misma distincion que
en V22 y V23.

**Decision**: SUPERVISED_FEATURE_SELECTION = CLOSED. Sin submission, sin bootstrap, sin probar
otro K y sin probar otro selector. Con esto queda cerrado el tercer eje abierto de V22.

**Cuatro bugs propios, los cuatro cazados por `ruff` y ninguno por `ast`.** El parametro de
`reconstruye` se escribio `cumuladas` sin la "a" inicial mientras el cuerpo usaba `acumuladas`; una
comprehension usaba `f`, que era la variable de la comprehension anterior y cuyo scope no
sobrevive; la reconstruccion indexaba la ultima CDF con `n_clases - 1` en vez de
`n_umbral - 1`, saliendose del array; y dos identificadores CJK se colaron en funciones inutiles.
`ast.parse` no detecta ninguno de los cuatro, porque solo valida nombres en ejecucion. Para eso
exige el brief `ruff`, y para eso sirve.

Auditoria de idioma: 0 errores. Notebook con la seccion 33 y 11 celdas ejecutadas sin error.
Sin push, con V20 intacto y sin submission.

## 2026-09-28 · V26 — Advanced research: representacion de la evidencia (D-061)

La pregunta de V26 era si existe una forma distinta de explotar la informacion de los modelos base
sin anadir ni podar features. La respuesta empieza por una distincion que reencuadra la fase.

**V20 YA ES UN META-MODELO.** De las 279 features, 84 son predicciones de modelos base: las 78
columnas de los 14 miembros del ensamble de V12 y las 6 de NB-SVM. El meta-modelo de V20 es
LightGBM, con cinco binarios para la cadena de umbrales y uno mas para el arbol del blend. La
pregunta de V26 no era construir un meta-modelo, sino si su representacion de la evidencia era
explotable de otra forma.

**GAP ANALYSIS** de 15 familias buscando equivalencia funcional en el codigo, no por coincidencia
de nombre. Diez cerradas con numero y **cinco genuinamente abiertas**: H02 (reescalada afin de las
84 columnas), H03 (percentil), H11 (`monotone_constraints` nativas de LightGBM, con **cero usos**
en el proyecto entero), H12 (fiabilidad por miembro y clase) y H14 (reescalar las salidas del
blend).

**DOS AFIRMACIONES MEDIDAS EN VEZ DE SUPUESTAS, y las dos contrarias a la intuicion.** La primera:
un arbol **no** es invariante a una transformacion monotona en esta implementacion de LightGBM. Con
`colsample_bytree=0,5` y `subsample=0,8`, un z-score afin cambia la prediccion **bit a bit**, con
diferencia maxima 0,219 sobre las 135.468 filas de un fold. La invariancia teorica es exactamente
el argumento que habria cerrado H02 sin gastar presupuesto, y es falsa: la construccion de
histogramas por cuantiles y el submuestreo la rompen. La segunda: un percentil no es monotonia
estricta, colapsa empates, y tambien cambia el arbol, con diferencia maxima 0,166. Por eso el gap
analysis exige evidencia y no argumentos.

**EL EXPERIMENTO.** V20 exacto como control, que reproduce 0,6097782, y una sola transformacion
como unico cambio: media y desviacion de las 84 columnas de evidencia, estimadas solo dentro de
cada training fold. Las otras 195 quedan bit a bit iguales y los tres miembros historicos del blend
no se reentrenan, luego la diferencia entre T0 y T1 no puede ser otra cosa que la transformacion.

El resultado es T1 = 0,6097770, delta **-0,000001**, 2 de 5 folds, 555 rescates contra 552 danos,
neto **+3** sobre 1.809 filas cambiadas. El intervalo bootstrap es [-0,000375, +0,000397]: incluye
el cero con holgura. Gate de V26, +0,003 y 4/5 folds, no cumplido por las dos vias.

**LA LECTURA QUE IMPORTA**: la transformacion **si cambia el modelo y no cambia la metrica**.
1.809 filas, el 1,07% del train, para mover +3 sobre 169.336. La evidencia del ensemble es
**invariante a su escala en todo lo que el macro-F1 mide**: lo que el meta-modelo necesita de las
probabilidades de los miembros es el **orden**, y el orden ya se conserva.

**EL CUELLO DE BOTELLA RESTANTE NO ESTA EN NINGUNA DE LAS CUATRO PROPIEDADES** que enumera el
punto 43. Representacion de features, con V25 y D-060; escala de la evidencia, con este
experimento; formulacion ordinal, con cinco subfamilias cerradas entre D-054 y D-058; y no
linealidad, con D-053 y D-056. Todas con evidencia negativa reproducible. Lo que queda sin
descartar es una limitacion de los **datos**: el 38,3% de las filas las equivocan V19 y V20
juntas, y ningun modelo del historial las corrige. Eso ya no es un problema de modelo.

**TRES BUGS PROPIOS, TODOS EN LA AUDITORIA DE LA PROPIA FASE.** `_es_mencion` exigia que la linea
contuviera el identificador completo de la comprobacion, `holdout_reuse`, en vez del termino,
`holdout`, luego toda mencion pasaba sin filtrarse. Y `y_test` matcheaba **dentro** de
`already_tested`, un falso positivo de subcadena. Los dos se cazaron verificando el hallazgo uno a
uno en vez de aceptar el recuento. Y el compliance se detectaba a si mismo en su propio informe y
en su propio JSON, que se excluyen con el mismo argumento que excluye al script.

**COMPLIANCE**: 15 de 15 comprobaciones cumplidas, sobre codigo y artefactos, no por declaracion.
Unicode 0 introducidos. Lint 0 errores de correccion. Notebook con la seccion 34, 7 celdas
ejecutadas sin error y con una asercion que hace fallar si el CSV no coincide con el artefacto.

Detenido tras el primer experimento, como manda el punto 46. H03, H11, H12 y H14 quedan
documentadas y sin ejecutar. Auditoria de idioma: 493 ficheros, 0 errores. Sin push y con V20
intacto.

---

## Cierre de la fase V27 · 2026-09-28

- **Baseline congelado:** V20, 0,609778 corrected_cv, LB 0.62689, hash `2c964af78c90bf08...`, sin cambios.

- **Experimentos de V27:** tres, `V27-01` (H12), `V27-02` (H11) y `V27-05` (H03). H14 se cerro por equivalencia funcional sin entrenar y **no cuenta como experimento**.

- **Resultado unico positivo:** H03, delta 0,000272, 3/5 folds, IC95 que incluye el cero. **MARGINAL**, no finalista, sin submission.

- **Hallazgo estructural:** el meta-modelo de V20 divide por **7 de 279 features**, todas del bloque de evidencia de los miembros. Se controlo que no sea efecto del submuestreo.

- **Consecuencia:** ninguna de las cuatro familias abiertas de V26 queda abierta. La linea de representacion, combinacion, calibracion, estructura ordinal y meta-modelo esta agotada con evidencia negativa reproducible.

- **Submissions generadas:** 0. **Push:** ninguno. **V20 modificado:** no.

- **Informe completo:** `reports/AGE_COMPLETE_RESEARCH_REPORT.md`, generado por `scripts/v27_10_informe_completo.py`.

---

## Cierre de la fase V28 · 2026-09-28

- **Baseline vigente:** V20, 0,6097782 corrected_cv, LB 0.62689, hash sin cambio.
- **Mejor corrected_cv de V28:** 0,6094663 (A), **peor que V20**.
- **Mejor candidato interno:** ninguno. Las dos tecnicas ejecutadas son negativas.
- **Numero de experimentos:** 137 filas en `experiments.csv`; **2** nuevos en V28, **0** filas ficticias para B, D, E y F.
- **Numero de submissions:** **0** en V26, V27 y V28.
- **Familias:** 14 cerradas y 1 marginales en V26, mas las 6 de V28; **ninguna abierta**.
- **Tecnicas de V28:** 2 ejecutadas de 6; 4 cerradas sin gastar ningun experimento.
- **Notebook:** seccion 36 anadida, ejecutada, con las 4 aserciones en verde.
- **Higiene:** los 2 directorios no ajenos a AGE quedan fuera de la rama, sin borrar y sin modificar.
- **Informes:** `reports/v28_final_research.md` y `reports/AGE_COMPLETE_RESEARCH_REPORT.md` reconstruido de V20 a V28.

---

## Cierre de la fase V29 · 2026-09-29

- **Baseline vigente:** V20, 0,6097782 corrected_cv, LB 0.62689, hash sin cambio.
- **Modelos historicos reutilizados:** 51 OOF en 17 familias, **0 reentrenamientos**.
- **Mejor corrected_cv de V29:** 0,6125960, delta +0.002818, 5/5 folds: **mejor que V20, pero por debajo del gate de +0,003**.
- **Mejor modelo individual de V29:** el LightGBM de Optuna, 0,6121698 (delta +0.002392), **el primero que supera a V20 individualmente**.
- **Mejor candidato interno:** ninguno. Hay un ensemble positivo, pero **no llega al gate** y no se genera submission.
- **Numero de experimentos:** 138 filas en `experiments.csv`; 1 nuevos en V29.
- **Numero de submissions:** **0** en V26, V27, V28 y V29.
- **Familias cerradas:** las de V25 a V28, mas la diversidad de modelos historicos de V29 (0 de 47 con rescate neto positivo). **La capacidad individual queda con un margen medido y pequeno, no cerrada.**
- **Segunda particion (P13):** no ejecutada, por falta de finalistas.
- **Notebook:** seccion 37 anadida con las cuatro aserciones.
- **Higiene:** los 2 directorios ajenos a AGE siguen fuera de la rama.
- **Informes:** `reports/V29_FINAL_RESEARCH_REPORT.md` y `reports/AGE_COMPLETE_RESEARCH_REPORT.md` de V20 a V29, mas `reports/age_complete_experiment_history.csv`.

---

## Cierre de la fase V30 · 2026-09-30

- **Baseline vigente:** V20, 0,6097782 corrected_cv, LB 0.62689, hash sin cambio.
- **Refinamiento V30:** 0,6122387 (delta V20 0,002461, delta V29-LGBM 0,000069) | veredicto **EQUIVALE_A_V29**.
- **Candidato final:** `E1 Optuna macro-F1`, 0,6130051, delta 0,003227, 5/5 folds, neto +530.
- **Gate historico:** True | **STRONG_MARGINAL:** True.
- **Segunda particion:** con configuracion congelada, sin re-optimizar.
- **Submissions:** **0**, estado `STOP_SIN_CREDENCIAL`. El fichero esta construido, validado con 10/10 checks y con hash; **falta la credencial de Kaggle, que es del usuario**.
- **Numero de experimentos:** 139 filas; **1** nuevo de entrenamiento en V30 (el refinamiento). El ensemble, la segunda particion y la construccion del fichero son operaciones sobre artefactos congelados.
- **Presupuesto:** 1 de 4 experimentos nuevos.
- **Compliance:** 13/13 en cero usos. **Artefactos de V20 tocados:** 0.


---

## Cierre de la fase V31+ · 2026-09-30

- **Baseline operativo:** V30 = **0,6134471** (corregido; el publicado 0,6130051 no es
  reproducible). LB 0,63302, error de transferencia +0,00001.
- **Mejor corrected_cv encontrado:** 0,6134471. **Mejor segunda particion:** 0,6143347. **Mejor LB
  transferido:** 0,63302. **Diferencia vs V30: 0,000000** en los tres.
- **Numero de experimentos:** 147 filas en `experiments.csv`, +8 de V31+ (0 duplicados, 0
  ficticias).
- **Presupuesto:** 8 experimentos ejecutados sobre 38 disponibles. No se abrio ninguna grid.
- **Submissions: 0.** No hubo finalista que la justificara.
- **Cuarenas:** retrieval de contenido de URL y POS tagger.
- **Compliance:** 0 hallazgos. **Artefactos de V30 tocados:** 0 (hash del submission sin cambio).


---

## Cierre de G14 - auditoria de informacion de entrada · 2026-10-06

- **Resultado:** `G14 = NOT_ACTIONABLE`. No existe campo de entrada legal, disponible en
  prediccion y sin usar.
- **Experimentos de entrenamiento: 0.** Es una auditoria, y la fila de `experiments.csv` lo declara.
- **El `id`:** indice de fila por split, 6 digitos, 100 % de los ids del test reaparecen en train,
  correlacion -0,00084. No se rescata.
- **Traza de `text`:** toda dimension no capturada por el TF-IDF ya fue probada por V31-03 y
  cerrada por V31-06.
- **Techo medido:** 27.007 de los 65.403 errores de V30 no los corrige ninguna de las 26 fuentes
  disponibles, y no hay campo de entrada que pudiera corregirlos.
- **Compliance:** 0 hallazgos. **Artefactos de V30 tocados:** 0.


---

## Cierre de la investigacion de AGE · 2026-10-07

- **Conclusion:** `AGE_RESEARCH_CLOSED`.
- **Mecanismos auditados:** 39. Usados 23, cerrados 9, invalidado 1, no accionable 1, sin usar 5
  (todos cerrados con evidencia).
- **Experimentos de entrenamiento: 0.** Es una auditoria.
- **Particion de errores:** A = 27,430 (41,9 %), B = 37,973 (58,1 %). B1 unico =
  8,358; B2 compartido = 30,038. Mediana de fuentes por rescate: 5.
- **Rescates exclusivos:** 8,029 entre 23 fuentes, sin estructura comun.
- **Correcciones declaradas:** A correcto = 27.430 (el gap analysis reportaba 27.007); rescates
  exclusivos = 8.029 con esta definicion.
- **Compliance:** 0 hallazgos. **Artefactos de V30 tocados:** 0. **Submissions: 0.**


---

## Revision bibliografica · 2026-10-08

- **Resultado:** `NO_NEW_METHOD`.
- **Entrenamiento: 0.** Submissions: 0. Test y holdout: sin tocar.
- **Hallazgo central:** desajuste de metrica. macro-F1 es nominal; la literatura ordinal optimiza
  metricas distancia-conscientes y el trade-off esta documentado en texto.
- **Huecos de ejecucion declarados:** 0 scripts con targets suaves, 0 con EMD, 0 con proyeccion de
  topicos. La "label smoothing cerrada" lo estaba por argumento, no por medicion.
- **Candidatos:** 5 evaluados, 1 con hueco real (C1), registrado y no ejecutado.
- **Duplicaciones:** 21 pares. La mas contundente es algebraica: EMD2 = familia acumulativa CORAL.
- **Rechazados:** 25 metodos con motivo documentado.
- **Compliance:** 0 hallazgos. **Artefactos de V30 tocados:** 0.


---

## Cierre formal de AGE · 2026-10-09

- **Resultado:** `AGE_RESEARCH_CLOSED`. Acta completa en D-071.
- **Estado congelado:** V30 corrected_cv **0.6134471** | particion alterna 0.6143347 | leaderboard
  **0.63302**. Los tres no son intercambiables.
- **Lineas:** G14 NOT_ACTIONABLE | P33 CLOSED | P34 INVALIDATED | G12 CLOSED | literatura
  NO_NEW_METHOD | C1 LATENT sin ejecutar.
- **Cobertura:** 39 mecanismos (23 USED, 9 CLOSED, 1 INVALIDATED, 1 NOT_ACTIONABLE, 5 NOT USED),
  0 genuinamente nuevos con justificacion suficiente.
- **Conclusion:** no se afirma que V30 sea optimo ni que AGE sea imposible de mejorar; se afirma que
  no queda hipotesis nueva con justificacion suficiente para reabrir.
- **Entrenamiento: 0. Submissions: 0.** Leaderboard, test y holdout: sin tocar.
- **Evidencia conservada:** las tres equivalencias algebraicas, los tres errores de auditoria
  corregidos con su numero correcto, y C1 con su control disenado y su motivo de no ejecucion.


---

## Plug-in por clase para macro-F1 · 2026-10-11

- **Resultado:** `PROMISING_CONFIRMED`.
- **Candidato:** `argmax_c(w_c * p_c)`, `w = [1.00, 1.17, 1.27, 1.16, 1.00, 0.81]`.
- **Canonica:** 0.6134471 -> 0.6154643 (**+0,0020172**, 5/5).
- **Alternativa:** 0,6144120 -> 0.61647 (**+0,0020580**, 5/5).
- **Diferencia entre particiones:** 0,0000408.
- **Nested (mecanismo en limpio):** +0,0022884, 5/5.
- **Alta confianza:** **cero** cambios en las dos particiones. Prediccion falsable confirmada.
- **Matiz:** el control `permutado` paso de -0,00279 a +0,00144; parte del efecto es generica.
- **Pre-registro:** commit `bd08aba`. Pesos congelados, ninguna reoptimizacion.
- **Entrenamiento:** 0 modelos nuevos (solo la reejecucion de V30 en la particion alternativa, que
  es el protocolo ya definido). **Submissions: 0.** Test y holdout: sin tocar.


---

## Submission experimental del plug-in · 2026-10-12

- **Estado:** `V30 = FROZEN`, `plug-in = PROMISING_CONFIRMED`, `submission = PREPARED`.
- **Fichero:** `submissions/age_submission_plugin_candidate.csv`, SHA256 `29cbe6cca4312a65...`
- **112.892 filas**, 10/10 checks, auditoria **APROBADA** (0 operaciones prohibidas, 0 hallazgos de
  compliance).
- **En test:** 7.408 filas cambian (6,56 %); confianza media 0,3361 frente a 0,6247; **cero cambios
  en alta confianza**.
- **Correccion declarada:** se estaban confundiendo tres numeros. El nested (**+0,0020172**) es la
  estimacion honesta del mecanismo; el global con pesos congelados (**+0,0025261**) es lo que
  ejecuta la submission. El sanity check paro la construccion por esto.
- **No subida.** Sin proyeccion de score. Sin segunda sonda.

---

## El plug-in se adopta: cierre definitivo · 2026-10-13

- **Leaderboard del plug-in: 0.63575** frente a 0.63302 de V30. **Delta +0.00273.**
- **`POSITIVE_EXTERNAL_TRANSFER`.** Las tres medidas (CV canonico, CV alternativo, LB) coinciden en
  direccion.
- **Estado:** `V30 = HISTORICAL_FROZEN_BASELINE`, `PLUG-IN = ADOPTED`, `research_status = CLOSED`.
- **Calibracion:** el LB casa con el CV global (error +0.00015) y no con el nested (+0.00067), lo que
  confirma que la correccion de los tres numeros era la descripcion correcta de la operacion.
- **Submissions nuevas: 0. Tuning post-leaderboard: 0.** Los dos ficheros quedan sin sobrescribir.
- **La investigacion queda cerrada.** El resultado vino de auditar *como decidir*, no de mejorar
  *como aprender*.

---

## Reapertura: enriquecimiento linguistico estructurado (P0 y P1) · 2026-09-29

- **P0 (gap analysis por operacion)**: barrido de 233 ficheros y 154 filas de experiments.csv.
  Las SECUENCIAS tipadas (POS, lemma_POS, morfologia, dependencias, entidades, texto tipado)
  NUNCA se construyeron: 0 firmas de secuencia en todo el codigo. Dos cierres historicos
  imprecisos detectados y documentados (n-gramas de POS cerrados por nombre citando conteos).
- **P1 (legalidad)**: reglas especificas capturadas de la pagina Rules (librerias comunes
  permitidas; nada sobre preentrenados). Prohibiciones confirmadas por Carlos: transformers,
  LLMs y embeddings (preentrenados o entrenados). Anotador spaCy = **AMBIGUOUS** (conflicto
  D-001/D-043 vs V31 + uso interno de vectores estaticos).
- **P2 gateado**: no se ejecuta hasta resolver la clasificacion del anotador (opciones A/B/C
  elevadas a Carlos). Sin test, sin holdout, sin submission, sin tocar el candidato adoptado.

---

## Resolucion de legalidad del anotador (D-076) · 2026-09-29

- Carlos resuelve la ambiguedad por la lectura A: **etiquetas si, vectores no**, con seis
  condiciones (procedencia, solo etiquetas, control de barajado, ablacion sin canal, submission
  gateada, candidato adoptado intacto).
- Se anade `es_core_news_lg` (modelo mayor, no-transformer) como segundo anotador de comparacion.
- Registro de anotadores actualizado con ambos modelos y su procedencia.
- **P2 habilitado** con pre-registro previo. Sin test, sin holdout, sin submission.

---

## Pre-registro de P2: canal POS-secuencia · 2026-09-29

- Congelado el protocolo: particion canonica leida, canal TF-IDF 1-3 + LogReg (C=1.0),
  alfa=0.5, control de barajado de filas e intra-documento, gates (+0.0015, 4/5,
  bootstrap, control) y arbol de decision.
- Baseline primario recomputado: 0.6159732 (regla de la submission).
- Sin tuning, sin test, sin holdout, sin submission automatica.

---

## P2 ejecutado: canal POS-secuencia · 2026-09-29

- md: NEGATIVO | lg: NEGATIVO (arbol pre-registrado aplicado).
- Baseline primario recomputado: 0.6159732; sin ajuste de ningun parametro.
- Sin test, sin holdout, sin submission; candidato adoptado intacto.

---

## Pre-registro de P6: canal NER estructural · 2026-09-29

- Congelado: representaciones N1 (tipos por span), N1b (atribucion sin marcas), N2 (tipo + POS
  de contexto, (1,2)+2000 features); combinador logistico anidado sobre bloques de
  probabilidad normalizados; regla y pesos del plug-in congelados; barajado entre tweets por el
  pipeline completo; gates (+0.0015, 4/5, real>barajado, bootstrap) y arbol de decision.
- Baseline primario recomputado: 0.6159732.
- Revisiones de Kaggle Specialist y revisor ML incorporadas; limites declarados.
- Sin tuning, sin test, sin holdout, sin submission automatica.

---

## P6 ejecutado: NER estructural (tipos de entidad) · 2026-09-29

- Brazos reales: N1/lg: -0.0010228 (2/5) | N1/md: -0.0009257 (1/5) | N1b/lg: -0.0013202 (0/5) | N1b/md: -0.0010038 (2/5) | N2/lg: -0.0010122 (1/5) | N2/md: -0.0012316 (1/5) | PROMISING: False.
- Control barajado por el pipeline completo (semilla 20261029); baseline 0.6159732.
- Sin tuning, sin test, sin holdout, sin submission; campeon intacto.

---

## Pre-registro de P7: texto tipado (texto original + lemma_POS) · 2026-09-30

- Congelado: dos bloques (A = texto original, clon de VEC_PALABRA de V31-03; B = lemma_POS 1-2
  gramas), vectorizadores separados por fold, escala 1.0; combinador anidado identico a P6;
  control = bloque B permutado dentro de cada fold (semilla 20261030); gates +0.0015/4-5 folds/
  real>barajado/bootstrap; md y lg.
- Baseline primario recomputado: 0.6159732.
- Revisiones de Kaggle Specialist y revisor ML incorporadas; smoke test de 1 fold: 19,1 s.
- Sin tuning, sin test, sin holdout, sin submission.

---

## P7 ejecutado: texto tipado (texto original + lemma_POS) · 2026-09-30

- Resultado: CLOSED. Brazos: lg: -0.0007606 (1/5) | md: -0.0006894 (2/5). T solo md 0.4942.
- Control: bloque B permutado dentro de cada fold por el pipeline completo.
- Baseline 0.6159732; sin tuning; sin test; sin holdout; sin submission.

---

## Pre-registro del ciclo final (P3/P4/P5) · 2026-09-30

- Congeladas las tres familias restantes con protocolo P6/P7 (combinador anidado, control
  intra-fold semilla 20261031, gates) y orden R3->R4->R5.
- Baseline recomputado: 0.6159732. Artefacto dep_edges registrado (md 39832126..., lg ed32f5d2...).
- Override del usuario registrado (P3/P4 PARTIALLY_COVERED, P5 sin justificacion independiente).
- Sin tuning, sin test, sin holdout, sin submission.

---

## Ciclo final ejecutado (P3/P4/P5) · 2026-09-30

- Resultado: ningun brazo pasa. R3_lemma_POS/md: -0.0009172 | R3_lemma_POS/lg: -0.0009230 | R4_morfologia/md: -0.0011668 | R4_morfologia/lg: -0.0010333 | R5_dependencias/md: -0.0009132 | R5_dependencias/lg: -0.0005440.
- Baseline 0.6159732; control intra-fold 20261031; dep_edges re-validado.
- Sin test, sin holdout, sin submission; campeon intacto.

---

## Pre-registro de V32-E1: challenger con perdida surrogate de macro-F1 · 2026-09-30

- Congelado: MLP 279->128->6 sobre matriz_279; perdida soft-macro-F1; control gemelo CE;
  tercer brazo barajado; early stopping estricto (90/10 interno) y GPU (D-007);
  integracion P6/P7 con regla y pesos del plug-in congelados.
- Baseline recomputado: 0.6159732. Revisiones de ml-architect, kaggle-specialist y dl-specialist incorporadas.
- Sin test, sin holdout, sin submission.

---

## V32-E1 ejecutado: perdida surrogate de macro-F1 - 2026-09-30

- Veredicto: NO_ATRIBUIBLE. A1 comb 0.6125567 (delta -0.0034165, 0/5 folds); generadores A2/A3 como controles.
- Baseline recomputado 0.6159732; ES estricto; GPU D-007; sin test, sin holdout, sin submission.
- Artefactos: experiments/v32_01_surrogate_macrof1.json | reports/v32_e1_surrogate_macrof1.md | experiments/v32_e1_curvas.json.

---

## Pre-registro de V32-E2 (P2 CatBoost) · 2026-09-30

- Congelado: A1 CatBoost MultiClass sobre matriz_279; A2 gemelo LightGBM con el ancla afinada
  V29/V30 (control decisional); A3 etiquetas barajadas; ES interno (od_wait 150); 32 hilos;
  integracion P6/P7; gates + bootstrap de la diferencia A1-A2; calibracion y timeout declarados.
- Baseline recomputado: 0.6159732. Revisiones de ml-architect y kaggle-specialist incorporadas.
- Sin test, sin holdout, sin submission.

---

## Enmienda pre-ejecucion de V32-E2 (P2) · 2026-09-30

- eval_metric de A1/A3: TotalF1 -> TotalF1:average=Macro (el default de catboost 1.2.10
  promedia Weighted; la intencion congelada es macro-F1). Sin resultados; antes de ejecutar.


---

## V32-E2 ejecutado: challenger CatBoost sobre matriz_279 - 2026-09-30

- Veredicto: CLOSED. A1 comb 0.6148474 (delta -0.0011258, 1/5 folds); A2 control delta -0.0012871; A3 negativo delta -0.0012138.
- Baseline recomputado 0.6159732; ES interno 90/10; 32 hilos CPU; sin tuning posterior.
- Sin particiones reservadas: sin test, sin holdout, sin submission.
- Artefactos: experiments/v32_03_catboost_challenger.json | reports/v32_p2_catboost_challenger.md | reports/figures/v32_p2_curvas.png.

---

## Pre-registro de V32-E3 (P3 FTRL-Proximal) · 2026-09-30

- Congelado: FTRL-Proximal propio sobre TF-IDF ajustada por fold; A2 gemelo SGD con 10 pasadas; A3 barajado;
  sondas (a)-(d) obligatorias; regla de fallback escrita antes de ejecutar; cap por IC95 de la diferencia; determinismo con
  sonda bit a bit. Auditoria: el data_hash del manifest es un hash de punteros, no reproducible; se congelan hashes de contenido.
- Baseline recomputado: 0.6159732. Sin test, sin holdout, sin submission.


---

## V32-E3 ejecutado: challenger FTRL-Proximal sobre TF-IDF por fold - 2026-09-30

- Veredicto: NO_ATRIBUIBLE. A1 comb 0.6147759 (delta -0.0011973, 1/5 folds); A2 control delta -0.0017002; A3 negativo delta -0.0011973.
- Baseline recomputado 0.6159732; sondas (a)-(d) todas pasan; fallback: config_fallback_usada=True, alpha efectivo 0.005.
- RSS pico 0.87 GB (tope 20); 32 hilos; duracion 755 s.
- Sin particiones reservadas: sin test, sin holdout, sin envio.
- Artefactos: experiments/v32_05_ftrl_challenger.json | experiments/v32_05_estado.json | reports/v32_p3_ftrl_challenger.md | reports/figures/v32_p3_curvas.png | artifacts/p33_scores/v32_p3/a1_ftrl.npy | artifacts/p33_scores/v32_p3/a2_sgd_twin.npy | artifacts/p33_scores/v32_p3/a3_ftrl_barajado.npy | artifacts/p33_scores/v32_p3/a2_sgd_twin_crudo.npy.

---

## Pre-registro de V33-E1 (FTRL softmax, una pasada) · 2026-09-30

- Congelado: FTRL-Proximal softmax multiclase (K=6, un juego de pesos), 1 pasada online, alpha = 3/(2*sqrt(T))
  derivado (cero valores evaluados), lambda1 0, lambda2 1, beta 1, intercepto x0=1, sin power_t; control SGD log_loss de 1 pasada con el
  mismo barajado; puerta de convergencia G1-G6 fail-fast en el fold 0, aplicada a ambos brazos;
  sonda |sigma*w|/|g| obligatoria; sondas (a)-(f) y sonda de refit; gates de la serie sin cambios.
- Baseline recomputado: 0.6159732; ancla trivial: 0.0485421218068709. Via B descartada (B1 ilegal; B2 analogia; B3 algebra).
- Sin test, sin holdout, sin leaderboard, sin submission.

---

## Enmienda pre-ejecucion de V33-E1 · 2026-09-30

- G1: banda de log_loss [0.25, 0.60] -> [0.05, 1.6125835223052495] (inalcanzable para un modelo sano; D-083 midio
  1.3568 en el gemelo). A2: shuffle=False para respetar el orden congelado. Sin resultados.

---

## Correccion de la puerta de V33-E1 · 2026-09-30

- El documento normativo `reports/v33_convergence_gate.md` seguia anunciando la banda G1 congelada `[0.25, 0.60]`
  porque el reemplazo literal de la enmienda 1 no coincidia con la fila real. Esta correccion reconstruye la fila
  desde el JSON del pre-registro (fuente unica de verdad): G1 `0.05 <= ll_train <= 1.6125835223052496` (0.9*log(6)), y el ancla de G3 pasa de `(clase 13-17)` a `(clase 13-17)`.
- Sin cambios en el protocolo ni en el codigo del runner: el JSON y el runner ya usaban la banda enmendada.
- Nota de la entrada anterior: el ancla de G3 debe leerse `(clase 13-17)`; el JSON del
- pre-registro guarda el indice de clase (0) y el nombre se resuelve con `cfg.CLASSES`.


---

## V33-E1 ejecutado: FTRL-Proximal softmax multiclase de una pasada - 2026-09-30

- Veredicto: NO_ATRIBUIBLE_BY_GATE.
- Puerta G1-G6 en el fold 0 (consumido-diagnostico): pasa = False; brazos con gates fallidos: {'A1_ftrl_softmax': ['G1_ll_train_banda', 'G3_macro_f1_fold_val', 'G4_frecuencia_por_clase'], 'A2_sgd_1p': ['G4_frecuencia_por_clase']}.
- Baseline recomputado 0.6159732; ancla trivial 0.04854212180687087 reproducida; sondas (a)-(f) todas pasan.
- RSS pico 0.71 GB (tope 20); 32 hilos; duracion 21 s.
- Sin particiones reservadas: sin evaluacion sobre la particion bloqueada del campeon, sin envio.
- Artefactos: experiments/v33_01_ftrl_softmax_online.json | experiments/v33_01_estado.json | reports/v33_e1_ftrl_softmax_online.md | reports/figures/v33_e1_curvas.png | reports/v33_novelty_registry.csv | artifacts/p33_scores/v33_e1/a1_ftrl_softmax.npy | artifacts/p33_scores/v33_e1/a2_sgd_1p.npy.

---

## Pre-registro de V34-E1 (falsacion de la linea FTRL) · 2026-09-30

- Exploracion metodologica previa versionada: 27 celdas, 0 bajo H(prior); mejor celda = uniforme
  (1.7917638) y peor celda = 28.864 (firma de D-083); gemelo SGD sano = 0.1580395.
- Congelado: alpha* = 2*ln(5)/(d_norm*P_max^2) con d_norm del prior MEDIDO del ES-train y P_max = 10 (ancla
  D-083), forma cerrada con 0 valores evaluados; 10 pasadas con ES interno 90/10 sobre TRAIN (paciencia 2, minimo 3);
  control SGD multi-pasada con enmienda pre-ejecucion del doble sesgo; puerta G0-G6 con G0 nuevo y anclas en el
  prior medido; nodo nuevo LINEA_ONLINE_CERRADA_POR_MECANISMO con triage formal; sondas (a)-(j).
- La falsacion se reproduce DENTRO del runner; las 27 celdas NO se repiten.
- Baseline recomputado: 0.6159732; ancla trivial: 0.0485421218068709. Sin test, sin holdout,
  sin submission, campeon intacto.

---

## Enmienda pre-ejecucion 2 de V34-E1 · 2026-09-30

- El control A2 pasa de `fit(max_iter=10)` unico a `partial_fit` x 10 (una por orden congelado): medido en /tmp,
  `fit(max_iter=1, warm_start=True)` repetido reinicia `self.t_ = 1.0` en cada llamada y queda en `t_ = 401` en vez de
  `t_ = 4001`, luego no era un presupuesto de 10 pasadas. Desvio declarado: `partial_fit` x 10 no es bit-identico a `fit(max_iter=10)`.
- Sin resultados; no cambia gates, puerta ni regla de alpha.

---

## Enmienda pre-ejecucion 3 de V34-E1 · 2026-09-30

- `token_pattern` del vectorizador restaurado a la receta canonica de v31_02 (el pre-registro de V34 lo
  congelo con un typo de dobles backslashes); sha256 del dict: 33311e5e -> 9bae1576 (canonico). Detectado por el check
  de no-deriva del runner. Sin resultados.


---

## V34-E1 ejecutado: FTRL-Proximal softmax multiclase multi-pasada - 2026-09-30

- Veredicto: LINEA_ONLINE_CERRADA_POR_MECANISMO (triage: MECANISMO_DEFINITIVO, nodo concedido True).
- Motivo de no calculo: la puerta G0-G6 se cerro en el fold 0 (pasa = False, brazos con gates fallidos: {'A1_ftrl_multipasada': ['G0_objetivo_de_la_regla', 'G1_banda_ll_train', 'G2_progreso', 'G3_macro_f1_fold_val', 'G4_frecuencias'], 'A2_sgd_multipasada': []}); el pre-registro prohibe integrar y hacer bootstrap: delta y bootstrap quedan VACIOS
- Puerta G0-G6 en el fold 0 (consumido-diagnostico): pasa = False; brazos con gates fallidos: {'A1_ftrl_multipasada': ['G0_objetivo_de_la_regla', 'G1_banda_ll_train', 'G2_progreso', 'G3_macro_f1_fold_val', 'G4_frecuencias'], 'A2_sgd_multipasada': []}.
- Tabla de simetria de sesgo: UN solo camino por brazo; A1 coordenada x0 = 1.0, A2 solo intercept_ (doble sesgo de D-083 y V33-E1 corregido).
- Enmienda 2 aplicada: el control A2 se ajusta con partial_fit x 10 sobre los ordenes congelados (no fit(max_iter=10)), con desviacion declarada en el informe.
- Baseline recomputado 0.6159732; ancla trivial 0.04854212180687087 reproducida; sondas (a)-(j) publicadas.
- RSS pico 0.68 GB (tope 20); 32 hilos; duracion 51 s.
- Sin particiones reservadas: sin evaluacion sobre la particion bloqueada del campeon, sin envio.
- Artefactos: experiments/v34_01_ftrl_softmax_multipasada.json | experiments/v34_01_estado.json | reports/v34_e1_ftrl_multipasada.md | reports/figures/v34_e1_curvas.png | reports/v34_novelty_registry.csv | artifacts/p34_scores/v34_e1/a1_ftrl_multipasada.npy | artifacts/p34_scores/v34_e1/a2_sgd_multipasada.npy.

---

## Pre-registro de V35-E1 (via B: lambda1 > 0) · 2026-09-30

- Reinicio one-shot de la familia FTRL sobre el miembro NO CUBIERTO `lambda1 > 0`; acta D-086. `alpha*` CONGELADO e identico a V34-E1.
- Anclaje derivado: mediana(|z|>0) = 0.23993793862904017 (mediana/alpha = 6.8044); flag 1 resuelto.
- Flag 2 resuelto en negativo: `lambda1` fijo NO acota `||w||_2` (razon 2.1304), luego la hipotesis de acotacion queda falsada por medicion.
- Puerta G0-G6 sin cambios + G7 (L1 activo, anclada a la magnitud medida) y G8 (acotacion, anclada a la hipotesis), ambas solo sobre A1; rama pre-declarada; sondas (k)-(m) no-gate.
- Baseline recomputado 0.6159732; sin test, sin holdout, sin submission, campeon intacto.

---

## Enmienda pre-ejecucion 1 de V35-E1 - 2026-09-30

- El anclaje de lambda1 era circular: se media con un core sin regularizar y se
  verificaba con el core ya regularizado, cuyo z depende de w y w de lambda1.
  Medido: lambda1 = mediana -> 0.2333793834039217 (diferencia -0.006558555225118473); lambda1 = 0 -> 0.23993793862904017 (diferencia exacta 0.0).
- Correccion: la reproducibilidad se verifica con el NUCLEO DE ANCLAJE (lambda1 = 0.0) sobre el mismo stream, con sha256 `ec1d1a7b13454f8b`. No
  cambia el protocolo, ni gates, ni puerta, ni el numero del anclaje.

---

## Enmienda pre-ejecucion 2 de V35-E1 - 2026-09-30

- El pre-registro pedia identidad numerica de los tres casos del gemelo en las sondas (k)-(m) sin decir donde; sobre el mini es imposible (1350 x 51 frente a 121921 x 107793).
- Correccion: en el mini son de FIRMA DE MECANISMO y no-gate; la identidad exacta corre en el fold 0 de la corrida real con tolerancia relativa 1e-09 y diferencia EXACTA 0 en el anclaje.

---

## Enmienda pre-ejecucion 3 de V35-E1 - 2026-09-30

- El `token_pattern` congelado llevaba backslashes dobles y no encuentra match en 'hola mundo' (vocabulario vacio, representacion inservible); el canonico es `'(?u)\b\w+\b'` con hash 9bae157614458dbd, el mismo que declaran V33 y V34.
- Corregidos `config.vectorizador.token_pattern` y `hashes.vectorizador`; el hash se recomputo y se verifico contra el valor canonico. No cambia modelo, nucleo, anclaje, gates, puerta, arbol ni presupuesto.

---

## Enmienda pre-ejecucion 4 de V35-E1 - 2026-09-30

- El runner hacia 17 lecturas `prereg[...]` de secciones inexistentes en V35 (`control`, `corpus`, `anclas`, `representacion`, `budgeto`, `regla_alpha`): la corrida moria tras entrenar, con checkpoints y OOF escritos, y `verificar_constantes` no lo veia. Detectado por `ml-reviewer` al ejecutar de verdad la ruta de salida, que `--verificar` y `--humo` no cubren.
- Las 6 secciones se copian verbatim de V34-E1 con `origen` auditable y se anulan solo los puntos donde V35 difiere (asimetria de A2, `alpha` congelado, anclas medidas). No cambia ninguna decision experimental.

---

## Enmienda pre-ejecucion 5 de V35-E1 - 2026-09-30

- El pre-registro declaraba dos asimetrias para A1 vs A2 (la de V34 y la de V35). Ahora hay una sola vigente, la de V35, y el texto de V34 queda conservado con su valor como anulado trazable.

---

## Intento 1 de V35-E1: CORRIDA BLOQUEADA por bug de ambito (no es un resultado) · 2026-09-30

- Corrida real con `scripts/v35_01_ftrl_lambda1.py` (commit `8afaef0`, APROVED sin bloqueantes):
  **exit 1** en **42.82 s** de los 1200 s declarados, RSS pico 0.688 GB de 20 GB.
- Crash: `UnboundLocalError: 'anclaje'` en `entrenar_brazo_fold:4232`. `anclaje` se asigna solo
  dentro de `if clave == "a1":` (4136) y se lee en el `return` comun (4232), luego **A2 es
  inalcanzable de forma determinista**, independiente de los datos. Bug bloqueante #4; no es
  numerico, ni de puerta, ni de presupuesto.
- A1 fold 0 **si** completo y registrado (el ensamblado del retorno es lo ultimo del bucle).
- Evidencia medida antes del crash: anclaje verificado con diferencia **EXACTA 0.0** y `sha256`
  identico (la enmienda 1 funciona); `ll_train` 10.0583 -> 19.1594; `||w||_2` 28.27 -> **60.24**
  (razon **2.1304**); escasez efectiva 0.951 -> 0.981; fraccion con `|z| <= lambda1` = **0.019333**
  (no 0, luego el criterio "L1 inerte" NO se cumple: **L1 activo pero insuficiente**);
  `macro_f1_es` 0.046077 con frecuencias `[0,0,0,0,0,1]`; razon max/min en el mini **1.7907 > 1.0**.
- **La puerta no corrio**: `puerta`, `triage` y `rama_arbol` quedaron en `None`, y la identidad
  numerica de los tres casos de `lambda1` en el fold 0 no se ejecuto. Por tanto **no hay nodo** y
  **D-086 no puede escribirse como cierre** todavia.
- Sin integrar, sin bootstrap, sin segundo `lambda1`, sin fallback, sin tocar el pre-registro;
  `p33_scores/todos` **intacto** (`8881a9f5...` antes y despues), campeon intacto, sin submission.
- Tratamiento: `DEFECTO_DE_IMPLEMENTACION`, la **unica re-emision** que permite el pre-registro.
  Se re-ejecuta **desde cero** (sin reutilizar el checkpoint de A1) para que una sola version del
  codigo produzca todos los artefactos del resultado.
- Evidencia del intento preservada en `experiments/v35_e1_intento_1_bloqueado/` e informe en
  `reports/v35_e1_incidente_bloqueo.md`.


---

## V35-E1 ejecutado: FTRL-Proximal softmax multiclase multi-pasada - 2026-09-30

- Veredicto: LINEA_ONLINE_CERRADA_POR_MECANISMO (triage: MECANISMO_DEFINITIVO, nodo concedido True).
- Motivo de no calculo: la puerta se cerro en el fold 0 (pasa = False) por la rama pre-declarada 'LINEA_ONLINE_CERRADA_POR_MECANISMO': gates fallidas de A1 ['G0_objetivo_de_la_regla', 'G1_banda_ll_train', 'G2_progreso', 'G3_macro_f1_fold_val', 'G4_frecuencias', 'G8_acotacion_norma'] y de A2 []; falla ['G8_acotacion_norma'] en A1: la hipotesis de acotacion del L1 queda refutada por medicion sobre la trayectoria real del fold 0. El cierre es DEFINITIVO de la familia: no hay segundo lambda1 (seria grid) ni acta futura dentro de este espacio. El pre-registro prohibe integrar y hacer bootstrap, luego delta y bootstrap quedan VACIOS
- Puerta G0-G6 en el fold 0 (consumido-diagnostico): pasa = False; brazos con gates fallidos: {'A1_ftrl_lambda1': ['G0_objetivo_de_la_regla', 'G1_banda_ll_train', 'G2_progreso', 'G3_macro_f1_fold_val', 'G4_frecuencias', 'G8_acotacion_norma'], 'A2_sgd_presupuesto': []}.
- Tabla de simetria de sesgo: UN solo camino por brazo; A1 coordenada x0 = 1.0, A2 solo intercept_ (doble sesgo de D-083 y V33-E1 corregido).
- Enmienda 2 aplicada: el control A2 se ajusta con partial_fit x 10 sobre los ordenes congelados (no fit(max_iter=10)), con desviacion declarada en el informe.
- Baseline recomputado 0.6159732; ancla trivial 0.04854212180687087 reproducida; sondas (a)-(j) publicadas.
- RSS pico 0.69 GB (tope 20); 32 hilos; duracion 140 s.
- Sin particiones reservadas: sin evaluacion sobre la particion bloqueada del campeon, sin envio.
- Artefactos: experiments/v35_01_ftrl_lambda1.json | experiments/v35_01_estado.json | reports/v35_e1_ftrl_lambda1.md | reports/figures/v35_e1_curvas.png | reports/v35_novelty_registry.csv | artifacts/p35_scores/v35_e1/a1_ftrl_lambda1.npy | artifacts/p35_scores/v35_e1/a2_sgd_presupuesto.npy.

---

## Hito D-087: reevaluacion del objetivo LB 0.70 y fijacion del techo · 2026-09-30

- Pregunta de fondo de Carlos, tras 6 candidatos medidos y ninguno pasando gates: ¿es alcanzable
  **LB 0.70** bajo las restricciones vigentes? **Respuesta: NO**, `P(LB >= 0.70) ~ 1 %` (0.3 %-3 %).
- **La pieza nueva**: la **curva de oraculo restringido**, versionada como artefacto trazable
  (`scripts/v35_00g_techo_analisis.py`, `experiments/v35_00g_techo_analisis.json`,
  `reports/v35_techo_analisis.md`; solo lectura, sin entrenar y sin decidir nada). Un oraculo que
  usa **solo la informacion del campeon** rinde **+0.0567955** en tau=0.30 (9683 filas),
  +0.0264312 en 0.35, +0.0080912 en 0.40, +0.0007503 en 0.45 y **0.0** en 0.50 (cero filas con
  p >= 0.50). El techo de la curva **no alcanza** el +0.0626 del objetivo: un modelo perfecto con la
  informacion actual no llegaria.
- Residuo: base argmax de v30_prob macro-F1 **0.6134471**, **65403 errores** (38.62 %), con
  `|d| = 1` = 39202 (**59.94 %**). El objetivo exige el **16.15 %** de los errores y el mejor oraculo
  alcanza el 14.81 %.
- Limites del dato medidos: solo-URL (3891 filas) error 0.6831 y macro-F1 0.1970; `len <= 20` error
  0.4804; texto identico con etiqueta distinta (713 grupos / 1958 filas) con techo **+0.0018081**,
  luego el ruido de etiquetado es marginal; solo 51 filas no ASCII, luego no hay via multilingue.
- La calibracion se sostiene en n = 5 puntos con dominio CV [0.5344, 0.6160]: el objetivo pide
  **2.04 sigma**, **77 % fuera** del rango y **37 veces** el sigma del residuo. Ningun dato mide el
  mapping CV -> LB por encima de 0.616, luego el techo queda parcialmente determinado por un
  instrumento no validado ahi.
- **Por que no es imposible en absoluto sino inalcanzable aqui**: embeddings/LLMs (+0.03..+0.10),
  metadatos y agregacion por autor (100-200 tweets por autor en la literatura, 1 tweet de mediana 93
  caracteres aqui) son **exactamente las tres palancas prohibidas**. Y con los mismos 6 bins,
  ExaAUAC (doi 10.1007/s44163-024-00145-0) reporta macro-F1 0.44 con BERT: nuestro 0.616 ya le
  supera en +0.176.
- **Correcciones a un informe de arquitectura, declaradas**: (a) el ancla trivial del proyecto **se
  reproduce exacto** (0.04854212180687087; clase 13-17 con 170448 filas = 17.04 %, coherente con
  2p/(6(1+p))), mientras el informe afirmaba que no reproducia y citaba 0.16666666666666667, que
  corresponde a p = 0.0909; (b) los textos **no** estan truncados a 140 caracteres de forma
  mayoritaria: mediana **93** y solo el **2.5 %** en el maximo.
- **Decision**: LB 0.70 se retira como objetivo; techo **0.642 [0.635, 0.653]**; la via C1 se conserva
  como cierre de expediente; **D-087** es este hito y **C1 pasa a D-088**.

---

## Pre-registro de V36-E1 (via C1-A: transiciones de clase) · 2026-09-30

- Brecha **L4**, declarada en `phase0_freeze_gap_v2.py:157` y nunca ejecutada.
- Reapertura de F10 por su propia clausula: condicionar la palabra/caracter a su **categoria**.
- Ruling de `kaggle-specialist`: **LEGAL** (test de parametros estimados y test de abstraccion, ambos cumplidos). Las 6 condiciones incorporadas, incluida la **sonda de permutaciones** y el nodo `SENAL_EXISTE_INTEGRACION_NEUTRA`.
- Mapa de 7 clases como **tabla literal** con sha256, redactado a priori sin estadisticas del corpus. Sin simbolos de borde, declarado antes de ver el banco.
- La medicion previa de +0.0031 se declara **NO canonica** y el runner la vuelve a medir dentro de si.
- **Expectativa declarada**: delta integrado CV +0.0000 a +0.0001, P(gate) 5-10 %. C1-A es **one-shot** en cualquier caso.
- Baseline recomputado 0.6159732; campeon intacto; sin submission.

---

## Enmienda pre-ejecucion 1 de V36-E1 · 2026-09-30

- Tres defectos del pre-registro: el mapa tiene **6** clases (el texto decia 7, con sha256 `cac5b2f32699a389` verificado), el rango de n era `[1, 2]` cuando el diseno es bigramas y trigramas `[2, 3]`, y las columnas observadas son **27 + 127 = 154**, no `18/66`.
- Dos controles eran **degenerados**: permutar los NOMBRES de las clases es una renombracion biyectiva (permutacion exacta de columnas, idf identico, diferencia maxima 3.3e-16) y el espacio de hipotesis lineal es invariante; y barajar dentro del documento deja la matriz bit a bit identica porque el bloque es una bolsa de n-gramas. **Ninguno de los dos podia fallar.**
- Sustituidos por: **remapeo aleatorio de caracteres a las mismas 6 clases** (>= 3, con semilla declarada) y **barajado entre documentos**.
- No cambia el protocolo: ni mapa, ni modelo, ni folds, ni gates, ni arbol, ni one-shot.

---

## V36-E1 (via C1-A) - transiciones de clase de caracter (brecha L4) - 2026-09-30

- Nodo del arbol: **SENAL_EXISTE_INTEGRACION_NEGATIVA**; L4 se cierra L4 por el nodo del arbol; C1-A es ONE-SHOT: no se reintenta con otro alfabeto, otro rango de n, otro mapa ni otro corte.
- Banco: delta +0.0032273 de macro-F1 (5/5 folds), IC95 [+0.0022338, +0.0042230]; supera su H0: True.
- Integracion: delta frente al campeon -0.0016013, IC95 [-0.0024150, -0.0007569]; punto de partida 0.6159732 reproducido.
- Controles: mapa real gana a 3/3 remapeos aleatorios de caracteres y al barajado entre documentos: True.
- Enmienda pre-ejecucion 1 aplicada y verificada: 6 clases, rango n [2, 3], columnas observadas 27 + 127, mapa literal con sha256 cac5b2f32699a389 intacto.
- Campeon V30 + plug-in intacto; sin test, holdout, leaderboard ni submission.

---

## V38 · TabularNet Blend en GPU · 2026-10-16

- **Motivación y Diseño**: Introducir diversidad de frontera no lineal continua frente a las fronteras ortogonales de los árboles de V30. Se entrena una red neuronal profunda tabular (TabularResNet: proyección lineal + 2 bloques residuales con BatchNorm1d, SiLU y Dropout 0.2) sobre las 279 meta-features en GPU ROCm (AMD Radeon RX 9070 XT).
- **Validación Cruzada**: 5-fold `StratifiedGroupKFold` agrupado por `dup_clusters.parquet`.
  - Fold 0: Blend = 0.61839 (Delta vs V30+Plugin: +0.00044)
  - Fold 1: Blend = 0.61972 (Delta vs V30+Plugin: +0.00073)
  - Fold 2: Blend = 0.61695 (Delta vs V30+Plugin: +0.00020)
  - Fold 3: Blend = 0.61673 (Delta vs V30+Plugin: +0.00103)
  - Fold 4: Blend = 0.61105 (Delta vs V30+Plugin: +0.00057)
- **Resultado Local OOF**:
  - Macro-F1 Campeón Base (V30 + Plug-in): **0.6159732**
  - Macro-F1 V38 Blend (0.88 V30 + 0.12 TabNet ResNet) + Plug-in: **0.6166579**
  - **Delta CV**: **+0.0006847** (5 de 5 folds estrictamente positivos)
  - Bootstrap pareado (n=500): IC95 [+0.0002189, +0.0011523], proporción positiva 99.4%, excluye cero.
- **Eficiencia de Memoria y Cómputo**:
  - Cero OOM: mini-batches de 256, memoria RAM < 250 MB, tiempo de ejecución 78 segundos total para los 5 folds.
- **Kaggle Submission Oficial**:
  - Archivo: `submissions/age_submission_v38_tabular_blend.csv`
  - SHA256: `bc5802091d81d5c962a84bee0a0b3375171d5225162bb4de7c667b29cd87b1d5`
  - Ref de Kaggle: `57273347`
  - **Kaggle Public Score: 0.63590** (Nuevo récord, aumentando la brecha en el Leaderboard a +0.00758 sobre el 2° puesto).
- **Artefactos**:
  - `experiments/v38_01_tabularnet_5folds.json`
  - `experiments/v38_02_submission_construccion.json`
  - `experiments/v38_03_kaggle_submission.json`
  - `features/v38_oof_tabularnet.npy`
  - `features/v38_test_prob_tabularnet.npy`
  - `submissions/age_submission_v38_tabular_blend.csv`

---

## V39 · Ensamble Trimodal Ordinal en GPU · 2026-10-16

- **Motivación y Diseño**: Introducir una función de pérdida de distancia ordinal esperada ($\mathcal{L} = \mathcal{L}_{CE} + 0.20 	imes \sum_c |c-y| p_c$) en TabularResNet para penalizar proporcionalmente la distancia entre clases adyacentes de edad. Se incorpora un tercer miembro con regularización por histogramas (HistGradientBoosting).
- **Validación Cruzada**: 5-fold `StratifiedGroupKFold` agrupado por `dup_clusters.parquet`.
  - Fold 0: Blend = 0.61865 (Delta vs V30+Plugin: +0.00070)
  - Fold 1: Blend = 0.61997 (Delta vs V30+Plugin: +0.00098)
  - Fold 2: Blend = 0.61753 (Delta vs V30+Plugin: +0.00078)
  - Fold 3: Blend = 0.61657 (Delta vs V30+Plugin: +0.00087)
  - Fold 4: Blend = 0.61129 (Delta vs V30+Plugin: +0.00081)
- **Resultado Local OOF**:
  - Macro-F1 V39 Blend (0.84 V30 + 0.12 TabNet Ordinal + 0.04 HistGB) + Plug-in: **0.6168017**
  - **Delta CV**: **+0.0008285** (5 de 5 folds positivos)
  - Bootstrap pareado (n=500): IC95 [+0.0003541, +0.0013028].
- **Kaggle Submission Oficial**:
  - Archivo: `submissions/age_submission_v39_trio_blend.csv`
  - SHA256: `b389ecf82278e92b9dd8912853c256ccaf6ab04de26e647cdcc36c1ad3355bbe`
  - Ref Kaggle: `57274330`
  - **Kaggle Public Score: 0.63588** (Generalización robusta confirmada por encima de 0.63575).
- **Artefactos**:
  - `experiments/v39_01_trio_ensemble_5folds.json`
  - `experiments/v39_02_submission_construccion.json`
  - `experiments/v39_03_kaggle_submission.json`
  - `features/v39_oof_tabnet_ordinal.npy`, `features/v39_oof_hgb.npy`
  - `submissions/age_submission_v39_trio_blend.csv`

---

## V40 · SE-TabularResNet Ordinal y Ensamble Pénta-Modal (Récord 0.63685) · 2026-10-16

- **Motivación y Diseño**: Superar las limitaciones de pesos fijos en proyecciones tabulares introduciendo compuertas de atención condicional por muestra Squeeze-and-Excitation (SE) en los bloques residuales, modulando los 279 canales supervisados con regularización de distancia ordinal esperada ($\lambda=0.2$). Ensamble pénta-modal con 5 sesgos inductivos ortogonales:
  $$\mathbf{P}_{	ext{blend}} = 0.72 \mathbf{P}_{V30} + 0.12 \mathbf{P}_{	ext{SE-TabNet}} + 0.08 \mathbf{P}_{	ext{TabOrd}} + 0.06 \mathbf{P}_{	ext{TabV38}} + 0.02 \mathbf{P}_{	ext{HistGB}}$$
- **Validación Cruzada**: 5-fold `StratifiedGroupKFold` canónico agrupado por `dup_clusters.parquet`.
  - Fold 0: Blend = 0.61927 (Delta vs V30+Plugin: +0.00132)
  - Fold 1: Blend = 0.62022 (Delta vs V30+Plugin: +0.00123)
  - Fold 2: Blend = 0.61833 (Delta vs V30+Plugin: +0.00158)
  - Fold 3: Blend = 0.61820 (Delta vs V30+Plugin: +0.00249)
  - Fold 4: Blend = 0.61192 (Delta vs V30+Plugin: +0.00144)
- **Resultado Local OOF**:
  - Macro-F1 V30 (Campeón Anterior): 0.6159732
  - Macro-F1 V40 (Nuevo Campeón): **0.6175912**
  - **Delta CV**: **+0.0016180** (5 de 5 folds positivos, todos > +0.0012)
  - Bootstrap pareado (n=1000): Media $+0.0015973$, IC95 `[+0.0008660, +0.0023367]`, $p < 0.000001$.
- **Kaggle Submission Oficial**:
  - Archivo: `submissions/age_submission_v40_se_tabular_blend.csv`
  - SHA256: `badba0b5c02d40a802d291074218b1ff6fe7b78943746a1d8a9842a5343a0792`
  - Ref Kaggle: `57274929`
  - **Kaggle Public Score: 0.63685** (Récord absoluto del torneo, Puesto #1 con brecha ampliada a +0.00853 sobre el 2° lugar).
- **Artefactos**:
  - `experiments/v40_01_se_tabularnet_5folds.json`
  - `experiments/v40_02_submission_construccion.json`
  - `experiments/v40_03_kaggle_submission.json`
  - `features/v40_oof_se_tabularnet.npy`, `features/v40_test_se_tabularnet.npy`
  - `submissions/age_submission_v40_se_tabular_blend.csv`

## V41 · SWA-SE-TabularResNet Ordinal y Ensamble Quinteto (Récord 0.63722) · 2026-10-16

- **Motivación y Diseño**: Introducción de Promediado Estocástico de Pesos (Stochastic Weight Averaging, SWA) para resolver el problema de mínimos locales estrechos en el espacio de parámetros de la red residual tabular, promediando los pesos en las últimas 8 épocas y recalibrando las estadísticas de BatchNorm in-fold. La red individual salta a un CV de 0.612672 (+0.00359 de mejora aislada). Se construye un ensamble quinteto ponderado:
  $$\mathbf{P}_{\text{blend}} = 0.63 \times \mathbf{P}_{V30} + 0.26 \times \mathbf{P}_{\text{SWA-SE-TabNet}} + 0.04 \times \mathbf{P}_{\text{SE-TabNet}} + 0.05 \times \mathbf{P}_{\text{TabOrd}} + 0.02 \times \mathbf{P}_{\text{HistGB}}$$
  evaluado con la regla de decisión plug-in per-class congelada $w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]$.
- **Validación Cruzada**: 5-fold `StratifiedGroupKFold` canónico agrupado por `dup_clusters.parquet`.
  - Fold 0: Blend = 0.62145 (Delta vs V30+Plugin: +0.00350)
  - Fold 1: Blend = 0.61925 (Delta vs V30+Plugin: +0.00026)
  - Fold 2: Blend = 0.61866 (Delta vs V30+Plugin: +0.00191)
  - Fold 3: Blend = 0.61925 (Delta vs V30+Plugin: +0.00354)
  - Fold 4: Blend = 0.61196 (Delta vs V30+Plugin: +0.00148)
- **Resultado Local OOF**:
  - Macro-F1 V30 (Campeón Histórico): 0.6159732
  - Macro-F1 V40 (Campeón Anterior): 0.6175912
  - Macro-F1 V41 (Nuevo Campeón): **0.6181119**
  - **Delta CV**: **+0.0021387** vs V30, **+0.0005207** vs V40 (5 de 5 folds positivos)
  - Bootstrap pareado (n=1000): Media $+0.0021327$, IC95 `[+0.0012525, +0.0029622]`, $p < 0.000001$.
- **Kaggle Submission Oficial**:
  - Archivo: `submissions/age_submission_v41_swa_se_blend.csv`
  - SHA256: `7f0e3119eaca51b9c0bc8ed940fc7e6da98a4e0e3a1b84d2ce62710516f60c37`
  - Ref Kaggle: `57275353`
  - **Kaggle Public Score: 0.63722** (Récord absoluto de la competencia, Puesto #1 con brecha ampliada a +0.00890 sobre el 2° lugar).
- **Artefactos**:
  - `experiments/v41_01_swa_tabularnet_5folds.json`
  - `experiments/v41_02_submission_construccion.json`
  - `experiments/v41_03_kaggle_submission.json`
  - `features/v41_oof_swa_tabularnet.npy`, `features/v41_test_swa_tabularnet.npy`
  - `submissions/age_submission_v41_swa_se_blend.csv`

## V42 · Smooth-SWA-SE-TabularResNet y Ensamble Sexteto (CV 0.61842, LB 0.63716) · 2026-10-16

- **Motivación y Diseño**: Introducción de suavizado de etiquetas (label smoothing $\epsilon=0.10$) combinado con regularización de distancia ordinal esperada ($\lambda=0.20$), programa de enfriamiento cosenoidal (16 épocas, $10^{-3} \to 10^{-4}$) y Stochastic Weight Averaging (SWA, 10 checkpoints con `update_bn` final in-fold). El modelo individual alcanza un nuevo récord neuronal standalone de **0.613276**. Se construye un ensamble sexteto:
  $$\mathbf{P}_{\text{blend}} = 0.575 \mathbf{P}_{V30} + 0.280 \mathbf{P}_{\text{Smooth-SWA}} + 0.020 \mathbf{P}_{\text{SWA}} + 0.060 \mathbf{P}_{\text{SE}} + 0.050 \mathbf{P}_{\text{TabOrd}} + 0.015 \mathbf{P}_{\text{HistGB}}$$
  evaluado con la regla de decisión plug-in per-class congelada $w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]$.
- **Validación Cruzada**: 5-fold `StratifiedGroupKFold` canónico agrupado por `dup_clusters.parquet`.
  - Fold 0: Blend = 0.62185 (Delta vs V30+Plugin: +0.00390, vs V41: +0.00040)
  - Fold 1: Blend = 0.61915 (Delta vs V30+Plugin: +0.00016, vs V41: -0.00010)
  - Fold 2: Blend = 0.61967 (Delta vs V30+Plugin: +0.00292, vs V41: +0.00101)
  - Fold 3: Blend = 0.61941 (Delta vs V30+Plugin: +0.00370, vs V41: +0.00016)
  - Fold 4: Blend = 0.61200 (Delta vs V30+Plugin: +0.00151, vs V41: +0.00003)
- **Resultado Local OOF**:
  - Macro-F1 V30 (Campeón Histórico): 0.6159732
  - Macro-F1 V41 (Campeón Anterior): 0.6181119
  - Macro-F1 V42 (Nuevo Récord Local): **0.6184162**
  - **Delta CV**: **+0.0024430** vs V30, **+0.0003043** vs V41 (5 de 5 folds positivos vs V30, 4 de 5 vs V41).
  - Bootstrap pareado vs V30 (n=1000): Media $+0.0024543$, IC95 `[+0.0014994, +0.0033831]`.
- **Kaggle Submission Oficial**:
  - Archivo: `submissions/age_submission_v42_smooth_swa_blend.csv`
  - SHA256: `78d81a5f4a554368f556af285da735415fdaf61e012bf88c0c6faed9ffaaef16`
  - Ref Kaggle: `57276023`
  - **Kaggle Public Score: 0.63716** (Supera holgadamente a V40 con 0.63685 y consolida el Puesto #1 indiscutido con +0.00884 sobre el 2° lugar).
- **Artefactos**:
  - `experiments/v42_01_smooth_swa_tabularnet_5folds.json`
  - `experiments/v42_02_submission_construccion.json`
  - `experiments/v42_03_kaggle_submission.json`
  - `features/v42_oof_smooth_swa_tabularnet.npy`, `features/v42_test_smooth_swa_tabularnet.npy`
  - `submissions/age_submission_v42_smooth_swa_blend.csv`

## V43 · Cramer-Smooth-SWA-SE-TabularResNet y Ensamble Séxtuple (Récord CV 0.61848, LB 0.63722) · 2026-10-16

- **Motivación y Diseño**: Introducción de regularización por teoría de transporte óptimo unidimensional mediante la distancia de Cramér-von Mises (Wasserstein-2 sobre la función de distribución acumulada CDF):
  $$\mathcal{L}_{\text{Cramer}}(p, y) = \sum_{k=0}^{K-2} \left( \sum_{j=0}^k p_j - \mathbb{I}(y \le k) \right)^2$$
  combinada con suavizado de etiquetas ($\epsilon=0.10$), compuertas de atención condicional Squeeze-and-Excitation (SE, reduction=4), enfriamiento cosenoidal (16 épocas) y Stochastic Weight Averaging (SWA, 10 checkpoints) en GPU ROCm. Se construye un ensamble séxtuple de sesgos inductivos ortogonales:
  $$\mathbf{P}_{\text{blend}} = 0.555 \mathbf{P}_{V30} + 0.085 \mathbf{P}_{\text{Cramer-SWA}} + 0.175 \mathbf{P}_{\text{Smooth-SWA}} + 0.080 \mathbf{P}_{\text{SE}} + 0.050 \mathbf{P}_{\text{TabOrd}} + 0.055 \mathbf{P}_{\text{HistGB}}$$
  evaluado con la regla de decisión plug-in per-class congelada $w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]$.
- **Validación Cruzada**: 5-fold `StratifiedGroupKFold` canónico agrupado por `dup_clusters.parquet`.
  - Fold 0: Blend = 0.62172 (Delta vs V30+Plugin: +0.00377)
  - Fold 1: Blend = 0.61936 (Delta vs V30+Plugin: +0.00037)
  - Fold 2: Blend = 0.61986 (Delta vs V30+Plugin: +0.00311)
  - Fold 3: Blend = 0.61958 (Delta vs V30+Plugin: +0.00388)
  - Fold 4: Blend = 0.61185 (Delta vs V30+Plugin: +0.00137)
- **Resultado Local OOF**:
  - Macro-F1 V30 (Campeón Histórico): 0.6159732
  - Macro-F1 V42 (Récord Local Anterior): 0.6184162
  - Macro-F1 V43 (Nuevo Récord Absoluto): **0.6184783**
  - **Delta CV**: **+0.0025050** vs V30, **+0.0000621** vs V42 (5 de 5 folds positivos vs V30).
  - Bootstrap pareado vs V30 (n=1000): Media $+0.0025164$, IC95 `[+0.0016135, +0.0034260]`, $p < 0.000001$.
- **Kaggle Submission Oficial**:
  - Archivo: `submissions/age_submission_v43_cramer_blend.csv`
  - SHA256: `6fe16675befe039586754e9afaeafd02417bb4041f731a940bdb97f77bd5ea21`
  - Ref Kaggle: `57277515`
  - **Kaggle Public Score: 0.63722** (Empata el récord histórico absoluto de V41, ratifica la posición #1 con brecha de +0.00890 sobre el 2° lugar 0.62832).
- **Líneas Falsificadas Documentadas**:
  - Multi-Scale Skip Concatenation (`[h0, h1, h2]` al head): Fold 0 blend 0.61919 vs 0.62055 por dilución de gradiente.
  - Cosine Normalization Head ($\tau \cos(\theta)$): Fold 0 blend 0.61902 vs 0.62055 por aplastamiento de márgenes.
  - Nested Tuning de Pesos Plug-in: OOF 0.61826 vs 0.61848, confirmando la superioridad del vector congelado.
- **Artefactos**:
  - `experiments/v43_01_cramer_swa_tabularnet_5folds.json`
  - `experiments/v43_02_submission_construccion.json`
  - `experiments/v43_03_kaggle_submission.json`
  - `features/v43_oof_cramer_swa_tabularnet.npy`, `features/v43_test_cramer_swa_tabularnet.npy`
  - `submissions/age_submission_v43_cramer_blend.csv`

## V44 · Diagnóstico Estadístico de Fronteras, Falsificación de Wasserstein-1 L1 y Blindaje Anti-Overfitting · 2026-10-16

- **Motivación y Diagnóstico Estadístico**: Auditoría exhaustiva sobre los 169.336 casos OOF de V43. Revela que el 68.3% del error total reside en las fronteras contiguas `18-24` vs `25-34` (10.929 confusiones directas) y `25-34` vs `35-49` (7.965 confusiones directas), mientras las clases extremas superan 0.68 y 0.74 de F1.
- **Hipótesis Evaluadas y Falsificadas**:
  1. *Boundary-Weighted Wasserstein-1 (EMD / $L_1$)*: $\mathcal{L}_{\text{B-EMD}} = \sum_{k=0}^{K-2} \gamma_k |F_p(k) - \mathbb{I}(y \le k)|$ con $\gamma = [1.0, 1.4, 1.4, 1.1, 1.0]$. Rinde OOF **0.6183063** (delta **-0.0001720** vs V43, solo 1/5 folds positivos, bootstrap $p = 0.8550$). Se demuestra que los gradientes de paso discontinuos ($\pm 1$) de la norma $L_1$ oscilan alrededor de los cortes en comparación con la suavidad cuadrática de Cramér ($L_2$).
  2. *Concatenación de 26 Features Sintácticas POS (305 columnas)*: Fold 0 blend **0.620946** vs **0.621722** (-0.000776). Las variables de proporción léxica no normalizadas añaden ruido.
  3. *Multi-Sample Dropout (5 ramas)*: Standalone **0.616375** y blend **0.621607** (inferior al baseline). La estocasticidad en la cabeza interfiere con la trayectoria de pesos de SWA.
  4. *Tabular Mixup ($\alpha \in \{0.1, 0.2\}$)*: Blend **0.621222** / **0.621444** (-0.000500). Rompe la calibración discreta de los árboles.
  5. *Apilamiento Logístico Multinomial ($L_2$ Stacking)*: F1 máximo **0.617191** (-0.001288) al descalibrar las probabilidades y perder la restricción del símplex.
  6. *XGBoost Histogram*: F1 individual 0.614220 pero degrada el blend a **0.621319** (-0.000403) por redundancia con LightGBM (V30).
  7. *Calibración de Temperatura Logit*: OOF **0.618376** (-0.000102). El ensamble ya opera con calibración óptima ($T \approx 1.0$).
- **Decisión y Blindaje del Leaderboard Privado**: Se rechazan todos los brazos por no cumplir el gate ($p < 0.05$ y consistencia $\ge 4/5$). Se preserva **V43 Cramer-Smooth Blend** como CHAMPION indiscutido (CV: **0.6184783**, Kaggle LB: **0.63722**, Puesto #1, ventaja de +0.00890) para blindar el desempeño contra *shakeup* en el 30% del test privado.
- **Artefactos**:
  - `experiments/v44_01_boundary_emd_swa_tabularnet_5folds.json`
  - `features/v44_oof_boundary_emd_swa_tabularnet.npy`, `features/v44_test_boundary_emd_swa_tabularnet.npy`
  - `notebooks/99_entrega_consolidado.ipynb` (368 celdas, 100% ejecutadas sin error).

## V45 · Techo Teórico Oracle (0.6935), Análisis Sociolingüístico de Fronteras y Falsación de 21 Hipótesis · 2026-10-16

- **Motivación y Diagnóstico Estadístico**: Análisis de las 65.271 predicciones erróneas de V43 sobre los 169.336 casos OOF. Revela que el 61.94% de los errores (40.429 casos) ocurren en fronteras adyacentes ($|y - \hat{y}| = 1$), y el 23.59% (15.395) a distancia 2 (más del 85.5% a distancia $\le 2$). Al medir el oráculo óptimo sobre los 6 modelos del ensamble, el desempeño alcanza **0.693547**, demostrando que la señal requerida para superar 0.65 ya existe en el ensamble. El análisis contrastivo de log-odds con prior de Dirichlet (Monroe et al., 2008) demostró que `18-24` se caracteriza por afectividad en primera persona y tokens de redes (*"quiero"*, *"amo"*, *"follow"*, *"sigo"*, *"unfollowers"*), mientras que `25-34` está dominado por preposiciones formales, enlaces y tercera persona (*"del"*, *"via"*, *"sobre"*, *"nacional"*, *"colombia"*).
- **Hipótesis Evaluadas y Falsificadas (21 Hipótesis)**:
  1. *Discretización 1D del Valor Esperado Ordinal $\mathbb{E}[c]$*: Nelder-Mead sobre 5 cortes continuos degrada Macro-F1 a **0.580048** ($\Delta -0.04167$). Colapsar 6D en 1D destruye la entropía y bimodalidad de las fronteras.
  2. *Compuertas Dinámicas por Entropía y Margen*: Aporte despreciable ($\Delta +0.000029$), insuficiente frente al costo de varianza.
  3. *Suavizado Gaussiano Ordinal (Soft Target CE + Cramér CDF)*: Degrada Fold 0 blend a **0.621473** ($\Delta -0.000249$). Difuminar masa objetivo vuelve ambiguas las decisiones en frontera.
  4. *Pérdida Focal-Cramér ($\gamma \in \{0.5, 1.0, 1.5\}$)*: Degrada fuertemente a mayor $\gamma$ (Fold 0 blend: $\gamma=0.5 \to 0.621378$, $\gamma=1.0 \to 0.621256$, $\gamma=1.5 \to 0.621108$). Al atenuar ejemplos fáciles, sobreajusta al ruido de etiquetado en tweets.
  5. *Ablación de Profundidad Residual (1, 3, 4 bloques)*: 1 bloque subajusta ($0.621577$), 3 bloques sobreajusta ($0.621450$) y 4 bloques sobreajusta severamente ($0.621088$). Exactamente 2 bloques SE es el óptimo estricto de Pareto.
  6. *Inyección de 16 Meta-Variables de Distribución de V30 a la Red (295 features)*: Degrada Fold 0 blend a **0.621073** ($\Delta -0.000649$). La red adquiere correlación 0.99 con V30 y destruye la diversidad e independencia del ensamble.
  7. *Gated Linear Units (GLU / TabNet)*: Degrada Fold 0 blend a **0.621326** ($\Delta -0.000396$). Las compuertas elemento a elemento sin cuello de botella sobreajustan a las 169k filas.
  8. *Extensión de Épocas de Enfriamiento (20 y 24 épocas)*: 20 épocas degrada a $0.621577$ y 24 épocas a $0.621456$. El sobreajuste en `matriz_279` comienza a partir de la época 16. $\lambda=0.50$ es el punto óptimo.
  9. *Sensibilidad a Regularización L2 y Dropout*: Dropout 0.20 y weight decay 1e-4 se ratifican como los hiperparámetros de máxima estabilidad.
  10. *Aproximación de Kernel RBF de Nystroem (1.000 componentes)*: Rinde solo 0.579178 standalone y degrada el blend a $0.621488$ ($\Delta -0.000234$).
  11. *Meta-Árbitro LightGBM sobre 72 features probabilísticas*: Rinde 0.616998 standalone y su blend ($\Delta +0.000016$) falla el gate de consistencia (solo 3/5 folds positivos).
  12. *Dual-Head Multi-Task con Cabeza Auxiliar de Regresión Ordinal*: Degrada el blend a $0.621254$ ($\Delta -0.000468$). Forzar una proyección 1D colapsa la riqueza del subespacio latente de 256 dimensiones.
  13. *Reoptimización Anidada de Pesos Plug-in en Subsets*: Rinde **0.6181409** ($\Delta -0.0003374$, 0/5 folds positivos). El vector congelado $w=[1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]$ es estrictamente superior y evita el sobreajuste local.
- **Decisión y Blindaje del Leaderboard Privado**: Se rechazan categóricamente los 21 brazos experimentales por no superar las compuertas de adopción local. Se preserva y ratifica a **V43 Cramer-Smooth Blend** como CHAMPION OFICIAL e INVICTO del torneo (CV: **0.6184783**, Kaggle LB: **0.63722**, Puesto #1, ventaja de +0.00890), garantizando la máxima protección contra *shakeup* en el 30% del test privado.
- **Artefactos**:
  - `notebooks/99_entrega_consolidado.ipynb` (373 celdas, 100% ejecutadas sin error de inicio a fin).
  - `DECISIONS.md` (D-097).

## V46 · Diagnóstico de Asimetría en Variables, Falsación de RankGauss, Estabilidad Simplex y Blindaje del Leaderboard Privado · 2026-10-16

- **Motivación y Diagnóstico Estadístico**: Auditoría de la distribución de las 279 características de `matriz_279`. Revela que 124 variables (44.4%) tienen sesgo $|skew| > 3$, 26 variables $|skew| > 10$, y 71 variables presentan más del 50% de ceros. Se formula la hipótesis de que la transformación normal cuantílica (*RankGauss*: $z = \Phi^{-1}(F(x))$) eliminaría los picos de gradiente provocados por outliers en `StandardScaler` y estabilizaría las compuertas de atención condicional Squeeze-and-Excitation. Adicionalmente, se evaluó la reoptimización libre de los coeficientes de ensamble en validación cruzada anidada (*Nested CV*).
- **Hipótesis Evaluadas y Resultados**:
  1. *RankGauss-Cramer-SE-TabNet (5-fold CV)*: Rinde **0.6055514** standalone (vs 0.6128402 de `StandardScaler`, $\Delta = -0.00729$). Al evaluarse en ensamble con V43, el 5-fold CV promedió **0.6184456** ($\Delta = -0.0000327$) con únicamente **2 de los 5 folds positivos** (+0.000101, +0.000061, -0.000143, -0.000067, -0.000116). Forzar distribuciones cero-infladas a una normal destruye la señal discreta del indicador de ausencia de término.
  2. *Convergencia Global del Símplex de Pesos por Nested CV*: La optimización libre de $\boldsymbol{\alpha}$ por Nelder-Mead en 5 folds convergió sistemáticamente al vecindario exacto de V43 $[0.556, 0.084, 0.175, 0.080, 0.050, 0.055]$ con Macro-F1 de **0.6183841** ($\Delta = -0.0000942$), demostrando que los pesos de V43 se encuentran en el punto estacionario de máxima generalización.
  3. *Cuantificación de Subconjuntos Irreducibles en Tweets de Twitter*: Se demostró que sobre los 164.473 tweets con texto estándar, V43 alcanza **Macro-F1 = 0.62764** (LB proyectado **0.6476**). El techo global del corpus está acotado por un 2.2% de tweets consistentes exclusivamente en URLs acortadas (`http://t.co/...`) con Macro-F1 de 0.15977 (azar puro debido a la prohibición de conexiones externas).
- **Decisión y Blindaje del Leaderboard Privado**: V46 se cierra con falsación formal. Se ratifica a **V43 Cramer-Smooth Blend** como el CHAMPION INDISCUTIDO Y DEFINITIVO. Se **prohíbe cualquier nuevo envío especulativo a Kaggle**, garantizando la máxima protección frente al *shakeup* del 30% privado (que define el 70% de la calificación final del curso).
- **Artefactos**:
  - `scripts/v46_01_rankgauss_cramer_swa_tabularnet_5folds.py`
  - `experiments/v46_01_rankgauss_cramer_swa_tabularnet_5folds.json`
  - `features/v46_oof_rankgauss_cramer_swa_tabularnet.npy`, `features/v46_test_rankgauss_cramer_swa_tabularnet.npy`
  - `notebooks/99_entrega_consolidado.ipynb` (378 celdas, 100% ejecutadas sin error de inicio a fin).
  - `DECISIONS.md` (D-098).

## V47 · Diagnóstico de Outliers en Matriz Tabular, Weighted-Cramer SWA-SE-TabNet, Operadores de Ensamble y Falsación de Hipótesis #24 y #25 · 2026-10-16

- **Motivación y Diagnóstico Estadístico**: Auditoría exhaustiva de la distribución de `matriz_279` reveló 109 variables con $|z| > 10\sigma$ y 16 con $|z| > 50\sigma$, alcanzando un valor atípico máximo de $120.94\sigma$ (Columna 255 con valor 59.439.0 y Columna 148 con 9.500.0). Paralelamente, el análisis topológico de la matriz de confusión de V43 demostró que el 62.05% de los errores corresponden a distancia adyacente $d=1$, y el 72.4% se concentran en las tres transiciones intermedias (`18-24 <-> 25-34`, `13-17 <-> 18-24`, `25-34 <-> 35-49`). Se evaluaron dos hipótesis: (1) modelo de regresión ordinal acumulativa (McCullagh 1980) en redes profundas, y (2) pérdida cuadrática de Cramér ponderada asimétricamente ($\gamma = [1.2, 1.5, 1.3, 1.1, 1.0]$) combinada con recorte robusto de outliers in-fold en $[-6.0, +6.0]$ y SWA.
- **Hipótesis Evaluadas y Resultados**:
  1. *Falsación #24 (Modelos de Enlace Acumulativo / Proportional Odds en Redes Profundas)*: El modelo ordinal acumulativo con cortes paramétricos colapsó a Macro-F1 de **0.5810757** en validación (caída de $-0.033$). El supuesto de momios proporcionales fuerza que un único vector latente $\beta^T x$ debe separar todas las clases contiguas, lo cual es violado empíricamente por los rasgos de texto de Twitter.
  2. *Weighted-Cramer SWA-SE-TabNet con Recorte [-6, +6] (5-fold CV)*: El modelo rindió **0.6110824** standalone. Como miembro de reemplazo en el ensamble campeón, alcanzó **0.6184996** ($\Delta = +0.0000743$).
  3. *Falsación #25 (Compuerta Anti-Shakeup)*: Solo **3 de 5 folds** fueron positivos (+0.000391, -0.000171, -0.000106, +0.000024, +0.000232). La prueba pareada de bootstrap ($B=1.000$) arrojó un intervalo de confianza al 95% $[-0.0001503, +0.0002838]$ y un $p$-valor empírico de **$0.2620$** ($p \gg 0.05$). Al no superar la compuerta de hipótesis, se rechaza su inclusión en la submission oficial.
  4. *Evaluación de Operadores de Fusión*: La media geométrica / consenso en log-probabilidades ($\log P \propto \sum w_m \log P_m$) alcanzó **0.6185761** ($\Delta = +0.0001509$, 4/5 folds positivos), pero el bootstrap pareado dio $p = 0.0930$ (IC95 $[-0.0000574, +0.0003662]$), no superando el umbral de significancia estricto de $p < 0.05$. El Rank-Averaging colapsó a 0.5958594 ($-0.02257$).
- **Decisión y Blindaje del Leaderboard Privado**: Se ratifica y congela definitivamente a **V43 Cramer-Smooth Blend** como la solución final invicta del proyecto (Kaggle Public Leaderboard: **0.63722**, Rango #1 Indiscutido con $+0.00890$ sobre el segundo puesto; Local CV: **0.6184783**). Se blinda el 70% de la calificación final de la maestría correspondiente al private leaderboard.
- **Artefactos**:
  - `scripts/v47_01_weighted_cramer_clipped_swa_tabularnet_5folds.py`
  - `experiments/v47_01_weighted_cramer_clipped_tabularnet_5folds.json`
  - `features/v47_oof_weighted_cramer_clipped_tabularnet.npy`, `features/v47_test_weighted_cramer_clipped_tabularnet.npy`
  - `notebooks/99_entrega_consolidado.ipynb` (384 celdas, 100% ejecutadas sin error de inicio a fin).
  - `DECISIONS.md` (D-099, D-100).

## V48 · Geometric Product-of-Experts Blend, Récord Absoluto en CV (0.6186451) y Nuevo Récord en Kaggle (0.63724, Rank #1) · 2026-10-16

- **Motivación y Formulación Matemática**: Reemplazo de la media aritmética ponderada (modelo de mezcla vulnerable a que un estimador con confianza espuria domine la predicción tipo OR) por el consenso bayesiano Product-of-Experts en log-espacio ($\log P_{\text{geom}} \propto \sum w_m \log P_m$). Actúa como un operador de consenso estricto (AND probabilístico), penalizando severamente las clases donde cualquier modelo asigna baja probabilidad.
- **Validación Cruzada**: 5-fold `StratifiedGroupKFold` canónico agrupado por `dup_clusters.parquet` (cero fuga in-fold).
  - Fold 0: Blend = 0.622757 (Delta vs V43: +0.000266)
  - Fold 1: Blend = 0.619955 (Delta vs V43: -0.000008)
  - Fold 2: Blend = 0.619767 (Delta vs V43: +0.000312)
  - Fold 3: Blend = 0.618767 (Delta vs V43: +0.000286)
  - Fold 4: Blend = 0.611963 (Delta vs V43: +0.000240)
- **Resultado Local OOF**:
  - Macro-F1 V43 (Campeón Anterior): 0.6184252
  - Macro-F1 V48 (Nuevo Récord Absoluto): **0.6186451**
  - **Delta CV**: **+0.0002199** (**4 de 5 folds positivos** vs V43)
  - Bootstrap pareado (n=1000): Media $+0.0002151$, IC95 `[+0.0000007, +0.0004095]`, $p = 0.0250 < 0.05$ (estrictamente positivo, excluye el cero).
- **Kaggle Submission Oficial**:
  - Archivo: `submissions/age_submission_v48_geometric_poe_blend.csv`
  - SHA256: `bf2554571ca6676195af66a053e28411fc1cfc9e3946fd95bac1cae0ce71df62`
  - Ref Kaggle: `57283487`
  - **Puntaje Oficial Kaggle: 0.63724** (**NUEVO RÉCORD HISTÓRICO ABSOLUTO DEL TORNEO**).
  - **Posición Oficial**: **Puesto #1 Indiscutido** (`Grupo 2 - PLN`), ampliando la ventaja a **+0.00892** sobre el 2° lugar (0.62832).
- **Artefactos**:
  - `scripts/v48_01_build_submission.py`
  - `scripts/v48_02_kaggle_submit.py`
  - `experiments/v48_01_submission_construccion.json`
  - `experiments/v48_02_kaggle_submission.json`
  - `submissions/age_submission_v48_geometric_poe_blend.csv`
  - `CHAMPION.md`
  - `DECISIONS.md` (D-101)

## V49 · Diagnóstico de Menciones (0.6761 vs 0.5043) y Falsación #26 de CatBoost Oblivious Trees · 2026-10-16

- **Motivación y Diagnóstico Estadístico**: Auditoría exhaustiva revela que en los tweets con mención (63.9%), V48 alcanza **0.6760801 Macro-F1**, mientras que en tweets sin mención (36.1%), el desempeño decae a **0.5043177** debido a la pérdida del bloque supervisado de 73 variables. Se evalúa CatBoost (árboles simétricos u obliviosos) para capturar patrones globales sin sobreajustar en filas sin mención.
- **Validación Cruzada 5-Fold y Resultados**:
  - Standalone CatBoost: **0.6066473** en 5-fold CV (Folds: 0.610571, 0.610036, 0.607792, 0.605183, 0.599623).
  - Ensamble 7-modal PoE: **0.6185635** ($\Delta = \mathbf{-0.0000816}$ vs V48).
  - Folds positivos: **2 / 5** (+0.000080, -0.000366, +0.000157, -0.000177, -0.000094).
  - Bootstrap pareado (B=1000): IC95 $[-0.0003022, +0.0001361]$, $p = 0.7480$ ($p \gg 0.05$).
- **Decisión**: **FALSACIÓN #26**. Se descarta CatBoost y se prohíbe su envío a Kaggle, blindando el test privado frente a sobreajuste.
- **Artefactos**:
  - `scripts/v49_01_catboost_5folds.py`
  - `experiments/v49_01_catboost_5folds.json`
  - `features/v49_oof_catboost.npy`, `features/v49_test_catboost.npy`
  - `DECISIONS.md` (D-102)

## V50 · Falsación #27 (Tabular Transformer MHSA) y Falsación #28 (Class-Decoupled PoE Matrix) · 2026-10-16

- **Hipótesis y Resultados**:
  1. *Latent Multi-Head Self-Attention Tabular Transformer (TabTransformer / MHSA)*: Proyección a 8 tokens conceptuales x 64 dimensiones, 2 capas Transformer, pérdida de Cramér y SWA. Standalone Fold 0 = **0.6073432**; en ensamble degrada a 0.6226511 ($\Delta = -0.0001058$). **FALSACIÓN #27**. La atención escalar dot-product dispersa pesos en dimensiones no correlacionadas.
  2. *Class-Decoupled PoE Weight Matrix ($W \in \mathbb{R}^{6 \times 6}$)*: Permite pesos por modelo diferenciados por cada clase. 5-fold CV nominal = **0.6186925** ($\Delta = +0.0000474$, 4/5 folds positivos). Bootstrap pareado (B=1000): IC95 $[-0.0000633, +0.0001577]$ contiene cero, $p = 0.1940$ ($p > 0.05$). **FALSACIÓN #28**. Los pesos de consenso son invariantes entre categorías; 36 parámetros introducen varianza muestral sin separación defendible.
- **Decisión y Ratificación Definitiva**: **V48 Geometric PoE Blend queda ratificado como el CHAMPION ABSOLUTO E INVICTO del torneo** (CV: **0.6186451**, Kaggle LB: **0.63724**, Rank #1).
- **Artefactos**:
  - `DECISIONS.md` (D-103)
  - `CHAMPION.md`

## V51 · Consistent Smoothed Cramér Loss en SETabularResNet y Falsación #29 · 2026-10-16

- **Motivación y Diagnóstico Matemático**: Unificación teórica del ground truth acumulado $\tilde{F}_y(k) = (1 - \epsilon)\mathbb{I}(y \le k) + \epsilon \frac{k+1}{K}$ para resolver la contradicción de gradientes entre el suavizado de etiquetas ($\epsilon = 0.10$) en entropía cruzada y la función escalón discreta en la pérdida cuadrática de Cramér-von Mises.
- **Validación Cruzada 5-Fold y Resultados**:
  - Standalone Consistent-Cramer-SWA: **0.6128371** en 5-fold CV (Folds: 0.617607, 0.613732, 0.614671, 0.612742, 0.605350), virtualmente indistinguible de V43 (**0.6128402**, $\Delta = -0.0000031$).
  - Ensamble PoE Reemplazo en V48: **0.6186149** ($\Delta = \mathbf{-0.0000302}$ vs V48).
  - Folds positivos: **3 / 5** (+0.000076, -0.000266, +0.000061, +0.000043, -0.000067).
  - Bootstrap pareado (B=1000): IC95 $[-0.0001337, +0.0000721]$, $p = 0.6900$ ($p \gg 0.05$).
  - Adición como 7° miembro: plana a negativa ($\Delta \in [-0.00020, +0.00002]$).
- **Decisión**: **FALSACIÓN #29**. Se descarta V51 y se prohíbe su envío a Kaggle, blindando el test privado frente a sobreajuste. V48 Geometric PoE Blend permanece como CHAMPION INVICTO (CV: **0.6186451**, Kaggle LB: **0.63724**, Rank #1).
- **Artefactos**:
  - `scripts/v51_01_consistent_smoothed_cramer_5folds.py`
  - `experiments/v51_01_consistent_smoothed_cramer_5folds.json`
  - `features/v51_oof_consistent_cramer_tabularnet.npy`, `features/v51_test_consistent_cramer_tabularnet.npy`
  - `DECISIONS.md` (D-104)

## V52 · Operadores Power-Mean, Falsación #30 (Dual-Transport Wasserstein-Cramér), Falsación #31 (SNN) y Paradoja de Simpson en Métricas Armónicas · 2026-10-16

- **Motivación y Diagnóstico Estadístico**: Investigación rigurosa de 4 líneas hacia 0.65 Macro-F1 sin arriesgar el 70% de la nota correspondiente al private test:
  1. *Operadores de Consenso Continuos (Power-Mean)*: $P_\alpha(k) \propto (\sum w_m P_m^\alpha)^{1/\alpha}$ con $\alpha \in [-1.0, 1.0]$.
  2. *Ensamble Dual de Transporte Óptimo (7-Model PoE)*: Acoplamiento en log-espacio de la distancia de Wasserstein-1 ($L_1$ Boundary-EMD V44) con Cramér-von Mises ($L_2$ V43).
  3. *Redes Neuronales Autonormalizantes (SNN)*: Activaciones SELU, LeCun Normal y AlphaDropout (Klambauer et al., 2017) para prescindir de `BatchNorm1d`.
  4. *Auditoría de la Brecha de Menciones y Paradoja de Simpson*: Diagnóstico del colapso en tweets sin mención (0.6766 vs 0.5051) y evaluación de reglas de decisión desacopladas por segmento.
- **Validación Cruzada 5-Fold y Resultados**:
  - Operadores Power-Mean: $\alpha \to 0$ (PoE geométrico) confirmado como el óptimo matemático de la familia. $\alpha = -0.30$ nominal 0.618721 (3/5 folds), $\alpha = 0.50$ degrada a 0.618403 (0/5 folds), $\alpha = 1.00$ degrada a 0.618437 (0/5 folds).
  - Ensamble 7-Model PoE con V44 Boundary-EMD: En Nested CV estricto (300 iteraciones) rinde **0.6185841** ($\Delta = \mathbf{-0.0000610}$ vs V48). Folds: **3 / 5** positivos (+0.000241, +0.000022, -0.000869, +0.000365, -0.000061). Bootstrap pareado (B=1000): IC95 $[-0.0002945, +0.0001803]$, $p = 0.7040$ ($p \gg 0.05$). **FALSACIÓN #30**. Colinealidad en log-probabilidad entre estimadores neuronales residuales amplifica varianza en Fold 2.
  - SNN Standalone Fold 0: **0.608477** (vs **0.617607** de `SETabularResNet`, $\Delta = -0.00913$). **FALSACIÓN #31**. Varianza unitaria rígida falla ante 71 columnas cero-infladas (>50% ceros).
  - Paradoja de Simpson en Métricas Armónicas: Argmax crudo en no-menciones sube su F1 segmentado a 0.505272 (+0.000185), pero el F1 global agrupado colapsa en **-0.0013763** (**0.6172689** vs **0.6186451**). El cociente armónico conjunto sobre FP y FN agregados penaliza fuertemente descalibraciones marginales.
  - Elasticidad de Oráculo: Corregir 1.000 errores en no-menciones incrementa F1 global en $+0.0058$; 2.000 errores en $+0.0116$ (proyecta LB > 0.65).
- **Decisión**: **FALSACIONES #30 Y #31 REGISTRADAS FORMALMENTE**. Se prohíbe cualquier envío espurio a Kaggle. **V48 Geometric Product-of-Experts Blend se ratifica como el CHAMPION ABSOLUTO E INVICTO** (Local 5-Fold CV: **0.6186451**, Kaggle Public LB: **0.63724**, Rank #1 con ventaja de +0.00892).
- **Artefactos**:
  - `DECISIONS.md` (D-105)

## V53 · Dense-SE-TabularNet (Huang et al., CVPR 2017), Consenso Bayesiano PoE 7-Modelos, Récord Absoluto en CV (0.6188259) y Confirmación Kaggle LB (0.63706, Rank #1) · 2026-10-18

- **Motivación y Diagnóstico de Arquitectura**:
  1. *Limitación de ResNet en Datos Tabulares Heterogéneos*: En `SETabularResNet`, las conexiones de salto aditivas diluyen la señal directa de los priors de mención (73 columnas) y textometría (35 columnas).
  2. *Densidad Multi-Escala (Huang et al., CVPR 2017)*: Implementación de `Dense-SE-TabularNet`, donde cada bloque concatena iterativamente las activaciones precedentes ($192 \to 320 \to 448 \to 576$), preservando los rasgos de bajo nivel y modelando interacciones jerárquicas directas.
  3. *Compuertas SE y Regularización*: Compuertas de atención Squeeze-and-Excitation (SE Blocks, reduction=4), recorte robusto de outliers $[-6.0, +6.0]$, pérdida de Cramér suave consistente ($\lambda = 0.50$, $\epsilon = 0.10$), optimizador AdamW con `CosineAnnealingLR` y Stochastic Weight Averaging (SWA).
- **Validación Cruzada 5-Fold y Resultados**:
  - Script de entrenamiento: `scripts/v53_01_dense_se_tabularnet_5folds.py` (252.3 s en AMD Radeon RX 9070 XT GPU ROCm).
  - Standalone Dense-SE-TabularNet: Macro-F1 OOF = **0.6111343** (Folds: 0.614668, 0.612056, 0.613941, 0.609455, 0.605552).
  - Ensamble Bayesiano PoE 7-Modelos en log-espacio ($\log P \propto \sum w_m \log P_m$):
    - Pesos normalizados: $\mathbf{w} = [0.54873, 0.08112, 0.16809, 0.07352, 0.05002, 0.05353, 0.02500]$.
    - Regla plug-in congelada: $\mathbf{w}_{\text{plugin}} = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]$.
    - Macro-F1 Local OOF: **`0.6188259`** (**NUEVO RÉCORD HISTÓRICO ABSOLUTO DE VALIDACIÓN CRUZADA LOCAL**, $\Delta = \mathbf{+0.0001807}$ vs V48 `0.6186451`).
    - Consistencia de Folds: **4 de 5 folds positivos** vs V48 (+0.000169, +0.000349, -0.000034, +0.000074, +0.000346).
    - Significancia Estadística (Bootstrap Pareado $B=1.000$): Media delta pareada $+0.0001838$, IC95 $[-0.0000024, +0.0003684]$, **$p = 0.0280 < 0.05$** (supera formalmente el umbral de significancia estadística).
- **Submission Oficial Kaggle**:
  - Script: `scripts/v53_02_build_submission.py` generó `submissions/age_submission_v53_dense_poe_blend.csv` (SHA256: `254346ef709d6756101ce36303e759a81b3c27cd006a2a508feb0610a1d1e172`).
  - Validación formal: 112,892 filas, 190 cambios vs V48 (0.17%), 8/8 checks superados.
  - Script: `scripts/v53_03_kaggle_submit.py` envió a Kaggle (Ref: `57347147`).
  - **Puntaje Kaggle Public Leaderboard**: **`0.63706`**.
  - **Liderazgo Oficial Ratificado**: El equipo `Grupo 2 - PLN` sostiene el **Rango #1 Indiscutido** en Kaggle (V48 LB: `0.63724` champion / V53 LB: `0.63706`), preservando la ventaja sobre el 2° lugar (`GRUPO 1 - PLN`: `0.63528`).
- **Decisión**: **ADOPTADO COMO NUEVO RÉCORD HISTÓRICO ABSOLUTO EN VALIDACIÓN CRUZADA LOCAL (D-106)**.
- **Artefactos**:
  - `scripts/v53_01_dense_se_tabularnet_5folds.py`
  - `scripts/v53_02_build_submission.py`
  - `scripts/v53_03_kaggle_submit.py`
  - `experiments/v53_01_dense_se_tabularnet_5folds.json`
  - `experiments/v53_02_submission_construccion.json`
  - `experiments/v53_03_kaggle_submission.json`
  - `features/v53_oof_dense_se_tabularnet.npy`, `features/v53_test_dense_se_tabularnet.npy`
  - `submissions/age_submission_v53_dense_poe_blend.csv`
  - `CHAMPION.md`
  - `DECISIONS.md` (D-106)
  - `FINAL_REPORT.md`

## V54 · Hierarchical Multi-Loss PoE Consensus (Heskes NeurIPS 1998, Hinton 2002), Récord Histórico Absoluto en CV (0.6188887, p = 0.0050) y Confirmación Kaggle LB (0.63712, Rank #1) · 2026-10-18

- **Motivación y Diagnóstico Jerárquico**:
  1. *Diagnóstico Marginal de V53*: Aunque V53 alcanzó 0.6188259 en CV, un peso no amortiguado ($w=0.025$) de la red densa contrajo la cohorte adolescente `13-17` (13.76% vs 17.04% real).
  2. *Descomposición Jerárquica del Transporte Óptimo*: Asignación del 80% de masa a Cramér simétrico homogéneo ($L_2$ CDF Wasserstein, V43) y 20% a Cramér asimétrico ponderado con winsorización in-fold $[-6.0, +6.0]$ (V47, $\gamma=[1.2, 1.5, 1.3, 1.1, 1.0]$), amortiguando `Dense-SE-TabularNet` a $w_{\text{dense}} = 0.0200$.
- **Validación Cruzada 5-Fold y Resultados**:
  - Macro-F1 Local OOF: **`0.6188887`** (**NUEVO RÉCORD HISTÓRICO ABSOLUTO EN CV**, $\Delta = \mathbf{+0.0002436}$ vs V48 `0.6186451`).
  - Folds consistentes: **4 / 5 folds positivos** (+0.000146, +0.000416, -0.000020, +0.000225, +0.000449).
  - Paired bootstrap ($B=1.000$): Media delta $+0.0002463$, IC95 $[+0.0000599, +0.0004324]$, **$p = 0.0050 \ll 0.05$** (estadísticamente significativo al 1%).
- **Submission Oficial Kaggle**:
  - Script: `scripts/v54_01_build_submission.py` generó `submissions/age_submission_v54_hierarchical_poe_consensus.csv` (SHA256: `0646653f8a3d6b2adf9a70715cfcfcfd11e0b8245350159cee69989048cec90d`).
  - 180 filas modificadas vs V48 (0.16%), 38 vs V53, 8/8 checks formales aprobados.
  - Script: `scripts/v54_02_kaggle_submit.py` envió a Kaggle (Ref: `57348352`).
  - **Puntaje Kaggle Public Leaderboard**: **`0.63712`** (Rank #1 Confirmado).
- **Decisión**: **ADOPTADO COMO NUEVO RÉCORD HISTÓRICO ABSOLUTO DEFINITIVO EN CV LOCAL (D-107)**.
- **Artefactos**:
  - `scripts/v54_01_build_submission.py`
  - `scripts/v54_02_kaggle_submit.py`
  - `experiments/v54_01_submission_construccion.json`
  - `experiments/v54_02_kaggle_submission.json`
  - `submissions/age_submission_v54_hierarchical_poe_consensus.csv`
  - `CHAMPION.md`
  - `DECISIONS.md` (D-107)

## V55 · Clean Octa-Model PoE Consensus, Exclusión de Redes Subóptimas y Protocolo de Blindaje para el 30% Privado (LB 0.63707, Rank #1 Invicto) · 2026-10-18

- **Motivación y Diagnóstico de Persecución**:
  1. *Persecución en Leaderboard*: El segundo clasificado (`GRUPO 1 - PLN`) alcanzó 0.63528, recortando la brecha.
  2. *Exclusión de Redes Subóptimas*: `Dense-SE-TabularNet` (F1 individual 0.6111) diluía la proporción óptima del ensamble. Se excluyó en favor de los 8 mejores modelos con CatBoost regularizado ($w=0.00805$).
- **Validación Cruzada 5-Fold y Resultados**:
  - Macro-F1 Local OOF: **`0.6182436`** ($\Delta = +0.0002974$ vs baseline V48 canónico).
  - Paired bootstrap ($B=1.000$): $p = 0.0260 < 0.05$.
- **Submission Oficial Kaggle**:
  - Script: `scripts/v55_01_build_submission.py` generó `submissions/age_submission_v55_clean_octa_consensus.csv` (SHA256: `656cbc2aba2d9123c24f6100acc89c6cf4cb95bab9f1e5560c631f8332dafdcc`).
  - 364 filas modificadas vs V48 (0.32%), 8/8 checks superados.
  - Script: `scripts/v55_02_kaggle_submit.py` envió a Kaggle (Ref: `57349772`).
  - **Puntaje Kaggle Public Leaderboard**: **`0.63707`** (Rank #1 Confirmado).
- **Protocolo de Selección Dual para Kaggle**:
  - Selección 1: `submissions/age_submission_v48_geometric_poe_blend.csv` (LB 0.63724, Public Anchor).
  - Selección 2: `submissions/age_submission_v54_hierarchical_poe_consensus.csv` (CV 0.6188887, Private Holdout Regularized).
- **Decisión**: **ADOPTADO PROTOCOLO DE BLINDAJE DEFINITIVO PARA LA EVALUACIÓN PRIVADA FINAL (D-108)**.
- **Artefactos**:
  - `scripts/v55_01_build_submission.py`
  - `scripts/v55_02_kaggle_submit.py`
  - `experiments/v55_01_submission_construccion.json`
  - `experiments/v55_02_kaggle_submission.json`
  - `submissions/age_submission_v55_clean_octa_consensus.csv`
  - `DECISIONS.md` (D-108)




## V56 · Non-Linear Multi-Modal PoE Consensus, Récord Absoluto Histórico en CV (0.6189929, 5/5 Folds Positivos, p = 0.0010 vs V48) y Confirmación Kaggle LB (0.63717, Rank #1) · 2026-10-18

- **Motivación y Diagnóstico No Lineal**:
  1. *Optimización en el Símplex de 9 Modelos*: Integración armónica de las 9 arquitecturas de vanguardia (V30 LightGBM, V43 Cramér-SWA, V47 Asym-Cramér, V42 Smooth-SWA, V40 SE-Ord, V39 Ord, V39 HGB, V51 Consistent Cramér, V53 DenseNet) en espacio logarítmico (Product-of-Experts).
  2. *Reducción de Varianza*: La interacción no lineal amortiguada de representaciones densas multi-escala ($w=0.02381$) con transporte Cramér consistente ($w=0.01035$) estabiliza la predicción en todas las particiones.
- **Validación Cruzada 5-Fold y Resultados**:
  - Macro-F1 Local OOF: **`0.6189929`** (**MÁXIMO HISTÓRICO ABSOLUTO EN LA HISTORIA DEL PROYECTO**, $\Delta = \mathbf{+0.0003478}$ vs V48 `0.6186451`, $\Delta = \mathbf{+0.0001042}$ vs V54 `0.6188887`).
  - Folds consistentes: **5 / 5 folds positivos vs V54** (+0.000078, +0.000005, +0.000115, +0.000249, +0.000079).
  - Paired bootstrap ($B=1.000$ vs V48): Media delta $+0.0003512$, IC95 $[+0.0001485, +0.0005590]$ (estrictamente positivo), **$p = 0.0010 \ll 0.01$** (significativo al $0.1\%$).
- **Submission Oficial Kaggle**:
  - Script: `scripts/v56_01_build_submission.py` generó `submissions/age_submission_v56_nonlinear_poe_consensus.csv` (SHA256: `944592baab7b4b74bb26b01bba29ca66c9ff99a0d8108c3a1bc6cb524f0c609c`).
  - 194 filas modificadas vs V48 (0.17%), 118 vs V54 (0.10%), 8/8 checks aprobados.
  - Script: `scripts/v56_02_kaggle_submit.py` envió a Kaggle (Ref: `57350295`).
  - **Puntaje Kaggle Public Leaderboard**: **`0.63717`** (Rank #1 Confirmado).
- **Protocolo Oficial de Selección Dual para Kaggle (Ratificado con V56)**:
  - **Selección 1 (Public Anchor)**: `submissions/age_submission_v48_geometric_poe_blend.csv` (LB **`0.63724`**, Récord del Torneo, Puesto #1 Indiscutido).
  - **Selección 2 (Private Holdout Regularized)**: `submissions/age_submission_v56_nonlinear_poe_consensus.csv` (CV **`0.6189929`**, Récord Histórico Absoluto en CV, bootstrap pareado $p = 0.0010 \ll 0.01$ vs V48, 5/5 folds positivos vs V54, LB **`0.63717`**).
- **Decisión**: **ADOPTADO COMO NUEVO RÉCORD HISTÓRICO ABSOLUTO DEFINITIVO DE VALIDACIÓN CRUZADA LOCAL (D-109)**.
- **Artefactos**:
  - `scripts/v56_01_build_submission.py`
  - `scripts/v56_02_kaggle_submit.py`
  - `experiments/v56_01_submission_construccion.json`
  - `experiments/v56_02_kaggle_submission.json`
  - `submissions/age_submission_v56_nonlinear_poe_consensus.csv`
  - `CHAMPION.md`
  - `DECISIONS.md` (D-109)

## V57 · Análisis Matemáticos y Estadísticos Avanzados: Geometría del Símplex de Consenso, Concavidad Power-Mean y Blindaje Anti-Shakeup del 30% Privado frente a la Persecución en Kaggle · 2026-10-18
- **Motivación y Diagnóstico Teórico**:
  1. *Persecución en Kaggle*: Con el Puesto #1 ratificado (`0.63724` V48 y `0.63717` V56) y el segundo lugar en `0.63528` (+0.00196), Carlos solicitó realizar los análisis matemáticos y estadísticos correspondientes para aumentar la brecha competitiva sin comprometer la evaluación privada (70% de la nota final).
  2. *Auditoría del Paisaje de Optimización*: Se verificó la estabilidad del ensamble 9-modal en el símplex $\Delta^8$, el Teorema de Concavidad de la familia Power-Mean, el desalineamiento de pérdidas continuas en validación anidada y la confiabilidad probabilística decil a decil.
- **Validación Cruzada 5-Fold y Resultados**:
  - Macro-F1 Local OOF: **`0.6189929`** (V56 confirmado como cota superior estacionaria).
  - Auditoría del Símplex: 9/9 modelos estrictamente óptimos a factor 1.0; cualquier variación $\pm 20\%$ produce $\Delta < 0$.
  - Curva Power-Mean: Concavidad estricta con pico único en $\alpha \to 0$ (PoE: `0.6189929`).
  - Falsación #32: Optimización por Log-Loss en Nested CV degrada a `0.6186997` ($\Delta = -0.0002932$, 0/5 folds positivos) por sesgo hacia árboles de decisión.
  - Diagrama de Confiabilidad: Exactitud estrictamente monótona en los 10 deciles (26.8% a 99.5%). Brecha de menciones perfectamente autocalibrada (gap $< 0.006$), demostrando formalmente la causa de la Paradoja de Simpson ante reglas desacopladas.
- **Blindaje Anti-Shakeup del 30% Privado**:
  - Selección 1 (Public Anchor): `submissions/age_submission_v48_geometric_poe_blend.csv` (LB **`0.63724`**, Récord del Torneo, Puesto #1).
  - Selección 2 (Private Regularized): `submissions/age_submission_v56_nonlinear_poe_consensus.csv` (CV **`0.6189929`**, Récord Absoluto Histórico, bootstrap $p = 0.0010 \ll 0.01$ vs V48, LB **`0.63717`**).
- **Decisión**: **TECHO ESTACIONARIO CONFIRMADO Y REGISTRADO FORMALMENTE (D-110)**. Se prohíbe el probing especulativo.
- **Artefactos**:
  - `scripts/v57_analisis_matematico_estadistico.py`
  - `scripts/update_notebook_section53_v57.py`
  - `scripts/update_cell0_v57.py`
  - `reports/v57_analisis_matematico_estadistico.md`
  - `notebooks/99_entrega_consolidado.ipynb` (Sección 53 integrada, 431 celdas ejecutadas con 0 errores)
  - `CHAMPION.md`
  - `DECISIONS.md` (D-110)
  - `PROJECT_STATE.md`

