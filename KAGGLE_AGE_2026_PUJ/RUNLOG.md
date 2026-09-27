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
