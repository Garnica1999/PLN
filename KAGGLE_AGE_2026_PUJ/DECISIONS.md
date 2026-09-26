# DECISIONS — KAGGLE_AGE_2026_PUJ

Registro de decisiones de arquitectura y modelado (ADR). Formato:
`Status · Date · Context · Decision · Alternatives Considered · Consequences`.

---

## D-001 — Interpretación de las restricciones de recursos externos
- **Status**: Aceptada (2026-09-22) · **Riesgo asumido, documentado**
- **Context**: Las reglas dicen: "no se pueden usar modelos transformer ni modelos LLM [...]
  no se pueden usar modelos de vectorizacion, ni entrenados desde cero por nosotros ni por
  terceros ya preentrenados para usar".
- **Decision**: interpretación amplia aprobada por Carlos (sin consulta al profesor):
  - **Prohibido**: transformers/LLMs (BERT, RoBERTa, DeBERTa, T5, GPT, ELECTRA, ALBERT) y
    todo modelo de vectorización/embedding, preentrenado o entrenado desde cero
    (Word2Vec, FastText, GloVe, Doc2Vec, SBERT, embeddings de palabra/carácter aprendidos).
  - **Permitido**: TF-IDF/CountVectorizer/HashingVectorizer, features artesanales,
    selección estadística (χ², MI), SVD/LSA, POS tagger no-transformer de spaCy, lexicones,
    clasificadores clásicos (lineales, NB, SVM, árboles, boosting), MLP sobre features
    densas y ensembles/stacking.
  - **Datos externos etiquetados**: NO se usan (aunque las reglas no los prohíben), para
    preservar la integridad académica y evitar ventaja no atribuible al modelado.
- **Alternatives**: (a) consultar al profesor y esperar respuesta; (b) interpretación ultra
  conservadora sin TF-IDF. Descartadas por decisión explícita de Carlos (opción "amplia").
- **Consequences**: la Fase de DL con embeddings queda cancelada (`07_deep_learning.ipynb` ->
  `07_modelos_densos.ipynb`). La tabla de ablación incluirá variantes **sin POS tagger** y
  **sin lexicones** de modo que, si el profesor objeta alguna zona gris, exista un modelo
  compliant ya medido y desplegable sin rehacer el proyecto.

---

## D-002 — Métrica de desarrollo
- **Status**: Provisional (2026-09-22)
- **Context**: Kaggle reporta "F1 Score"; el kernel métrico oficial (`metric/f1-score`)
  expone `average` con default `binary`, inválido para 6 clases, por lo que el organizador
  fija macro/weighted/micro (no legible sin permisos de administración).
- **Decision**: **macro-F1** como métrica primaria de desarrollo (reportando también
  weighted-F1 y micro). El envío v01 (calibración) se usa para identificar empíricamente la
  métrica oficial comparando el LB público contra macro/weighted calculados localmente
  (precisión de 5 decimales). Clases casi uniformes => la diferencia esperada es ~0,1-0,2 pp.
- **Alternatives**: asumir weighted; esperar respuesta del organizador.
- **Consequences**: toda comparación de modelos se hace con macro-F1 y test pareado;
  `experiments.csv` guarda macro y weighted para recalcular el ranking si la métrica oficial
  resultara ser weighted.

---

## D-003 — Esquema de validación (anti-leakage por duplicados)
- **Status**: Aceptada (2026-09-22, se congela tras `01_data_audit`)
- **Context**: el dataset no tiene identificador de autor (imposible `GroupKFold` por autor),
  pero **1,26% de las filas de test aparecen literalmente en train** y **4,24% comparten
  firma de 60 caracteres** (99,2% de consistencia de etiqueta dentro del grupo). Además hay
  0,8% de duplicados exactos intra-train y 203 textos con etiquetas contradictorias.
- **Decision**: validación primaria `StratifiedGroupKFold` (5 folds × 3 semillas) con
  **grupos = clústeres de near-duplicados** (texto normalizado exacto ∪ firma de prefijo de
  60 chars sobre texto normalizado sin URLs/menciones/RT). Se corre además `StratifiedKFold`
  como **control** cuya diferencia con el primario **cuantifica el leakage por duplicados**.
- **Alternatives**: `StratifiedKFold` ingenuo (optimista); agrupar por autor (imposible).
- **Consequences**: las métricas reportadas serán conservadoras y comparables entre modelos;
  la brecha entre ambos esquemas se documenta como hallazgo metodológico en el informe.

---

## D-004 — Política de envíos a Kaggle
- **Status**: Aceptada (2026-09-22)
- **Context**: ~10 submissions puntuadas; la nota depende del LB privado (70%).
- **Decision**: v01 = calibración (mejor baseline de la Etapa B: verifica métrica oficial y
  brecha CV<->LB). v02..v0N = candidatos justificados por CV. Se reservan ≥3 envíos para el
  final. **El LB no se usa como conjunto de validación** ni para selección de modelo.
- **Consequences**: cada envío se registra en `RUNLOG.md` con CV previa, fecha y score.

---

## D-005 — Tratamiento del padding/truncado a 140 caracteres
- **Status**: Aceptada (2026-09-22)
- **Context**: 99,9% de los textos de train miden exactamente 140 chars en bruto, pero la
  mediana tras `strip` es 93; los tweets fueron almacenados en ancho fijo y truncados al
  límite histórico de Twitter.
- **Decision**: el padding de espacios finales se elimina (`rstrip`) por ser **artefacto de
  exportación**, no señal. La pérdida de contexto por truncado a 140 se documenta como
  limitación del corpus (y como hipótesis de error analysis: tweets largos truncados).
- **Consequences**: `src/data.py` aplica `rstrip` en la carga; el EDA compara longitudes
  con y sin strip para dejar evidencia.

---

## D-006 — Prohibición de embeddings => redefinición de la fase de deep learning
- **Status**: Aceptada (2026-09-22)
- **Context**: D-001 prohíbe todo modelo de vectorización, incluidos los entrenados desde
  cero por nosotros.
- **Decision**: se cancela CNN/LSTM/GRU con embeddings aprendidos. `07_deep_learning.ipynb`
  se reemplaza por `07_modelos_densos.ipynb` (MLP sobre features densas, boosting sobre
  densas+SVD, sin representaciones aprendidas de texto).
- **Consequences**: la GPU deja de ser necesaria para el camino crítico; solo el MLP denso
  puede usarla opcionalmente (ver D-007).

---

## D-007 — Protocolo de uso de la GPU (guardián de recurso compartido)
- **Status**: Aceptada (2026-09-22) · Orden explícita de Carlos
- **Context**: la máquina tiene una GPU dedicada (AMD RX 9070 XT, ROCm 7.2) que Carlos usa
  para otros trabajos. Medido el 2026-09-22: sin PIDs de cómputo KFD activos, con línea base
  de escritorio ≈7% de uso y ≈13% de VRAM retenida; varios procesos gráficos mantienen
  handles en `/dev/kfd`, por lo que la detección por handles da falsos positivos.
- **Decision**: antes de **cualquier** paso que use GPU se ejecuta `src/gpu.require_gpu()`:
  1. Señal primaria: `rocm-smi --showpids` -> si hay un PID de cómputo ajeno, la GPU está
     **ocupada**.
  2. Señal secundaria: `rocm-smi --showuse --showmemuse` con umbrales calibrados sobre la
     línea base (ocupada si uso ≥ 25% o VRAM ≥ 40% en 3 lecturas separadas 5 s).
  3. Complemento: `torch.cuda.mem_get_info()`.
  4. Si está **libre**: adquiere lock propio no bloqueante (`~/.cache/kaggle_age_2026_puj_gpu.lock`)
     y registra el uso en `RUNLOG.md`.
  5. Si está **ocupada**: **no inicia nada y devuelve el control al usuario** avisando que la
     GPU está ocupada y entrando en **modo espera**; nunca mata procesos ni fuerza el recurso.
     La espera es acotada (`--wait-gpu N` reintentos cada 30 s como máximo) y nunca indefinida.
- **Alternatives**: esperar indefinidamente; detener procesos ajenos (prohibido).
- **Consequences**: el pipeline es CPU-first; si la GPU está ocupada, el paso GPU se difiere,
  se ejecuta en CPU o se pospone, sin bloquear el camino crítico. Cada evento de espera queda
  en `RUNLOG.md`.

---

## D-008 — Decisiones derivadas del estado del arte (Fase 1)
- **Status**: Aceptada (2026-09-22)
- **Context**: revisión de 30 trabajos (`reports/state_of_art.csv`, 29 verificados con DOI/URL)
  y análisis de procedencia del corpus (`reports/tables/dataset_provenance.md`).
- **Decision**:
  1. **Representación principal**: n-gramas de carácter (`char_wb` 3-5) con TF-IDF, en unión
     con n-gramas de palabra (1-2/1-3). Evidencia: en PAN 2019 (ES) los char n-grams + Random
     Forest superaron a los embeddings (0,8972 vs 0,8444); el corpus es ASCII puro, lo que
     hace del n-grama de carácter la representación más informativa disponible.
  2. **No usar argmax crudo**: optimizar macro-F1 con probabilidades calibradas y ajuste de
     umbral/prior por clase one-vs-rest (Li et al. 2010; Parambath et al. 2014; Narasimhan
     et al.; Lipton/Elkan). Entra en la Fase 19, no antes.
  3. **Lexicones internos**: construir "léxicos de edad" por regresión L2 sobre n-gramas
     (Sap et al. 2014) en lugar de depender de lexicones externos con tildes (que este corpus
     ASCII no puede aprovechar).
  4. **Regresión ordinal auxiliar** (Nguyen et al. 2011): probar como modelo complementario
     usando el punto medio del rango (18/25/35/50/65) si el tiempo lo permite.
  5. **Expectativas**: macro-F1 realista 0,30-0,45; un resultado > 0,50 debe activar la
     revisión anti-leakage antes de celebrarse.
  6. **No colapsar etiquetas** (p. ej. fusionar `50-64` y `65-xx`) aunque la literatura lo
     haga: la submission exige las 6 clases; la decisión se revisará con la matriz de confusión.
- **Alternatives**: apostar por features densas + boosting como vía principal (descartada como
  vía principal, se mantiene en la ablación).
- **Consequences**: el orden de experimentos cambia (char n-grams primero, luego unión), y la
  Fase 19 incorpora ajuste de umbrales por clase como componente esperado de la mejora.

---

## D-009 — Corpus ASCII puro: se descartan las features de emoji
- **Status**: Aceptada (2026-09-22)
- **Context**: el corpus contiene 96 caracteres distintos y un único carácter no-ASCII (`¿`,
  88 apariciones): sin tildes, sin `ñ`, sin emojis, sin puntuación tipográfica. El conteo
  inicial de emoticonos estaba contaminado por `http://` (74.904 falsos positivos), corregido
  contando sobre el texto sin URLs.
- **Decision**: (a) excluir del modelado las features de emoji (`n_emoji`, `emoji_diversity`,
  `has_emoji`) por varianza nula; (b) documentar las variantes de normalización de
  tildes/emojis como no-ops en el experimento de la Fase 4; (c) mantener el conteo de
  emoticonos textuales (ya corregido) porque sí existe en el corpus.
- **Alternatives**: conservar las features constantes (inútiles y ruidosas para los tests).
- **Consequences**: el catálogo de features queda en 35 features densas útiles; el alfabeto
  mínimo refuerza la apuesta por n-gramas de carácter.

---

## D-010 — Representación adoptada: char n-grams + canal dedicado de menciones
- **Status**: Aceptada (2026-09-22) · Decidida por evidencia experimental (Etapa B)
- **Context**: matriz de 14 configuraciones evaluadas con modelo fijo (LinearSVC C=1) y
  validación agrupada por clúster (`experiments/representaciones_resultados.json`).
- **Evidencia**:
  - `char_wb 3-6` (0,5413) y `char_wb 3-5` (0,5408) superan a todos los esquemas de palabra
    (`word 1-gram` 0,5025; `word 1-2` 0,4971; `word 1-3` 0,4941): +0,0442 en 5/5 folds.
  - La unión clásica word+char (E-001, 0,5344) queda **por debajo** del char solo (−0,0069,
    0/5 folds): los word n-grams diluyen sin aportar.
  - **Quitar menciones cuesta −0,1251** (0/5 folds) y sustituirlas por `<user>` cuesta −0,1237:
    la identidad de las cuentas mencionadas es una de las señales más fuertes del corpus.
  - Un **canal TF-IDF dedicado a los tokens de mención** añade +0,0144 sobre char solo (5/5
    folds) -> **0,5557 ± 0,0052**, el mejor resultado de la Etapa B.
  - Quitar URLs (−0,0067) o puntuación (−0,0080) también degrada: la forma del tweet es señal.
  - Tras normalización completa, 7.240 tweets quedan vacíos (vs 104) y 13.137 con ≤1 token.
- **Decision**: representación principal =
  `TF-IDF char_wb(3,6) sobre texto crudo` ∪ `TF-IDF word(1,1) sobre tokens de mención`,
  con `LinearSVC(C=1)`. El texto **no se normaliza**.
- **Alternatives**: unión word+char (descartada por evidencia); texto normalizado (descartada,
  −0,1442); features densas solas (0,3567, muy inferior).
- **Consequences**: los notebooks 04 y 05 documentan la decisión; la submission v02 usa esta
  representación. Las variantes de normalización quedan registradas como **experimentos
  negativos** con su cuantificación (evidencia académica de que "limpiar" no siempre ayuda).

---

## D-011 — El ajuste de decisión por clase NO aporta (resultado negativo)
- **Status**: Aceptada (2026-09-22) · Fase 19
- **Context**: la literatura de macro-F1 multiclase (Li et al. 2010; Parambath et al. 2014;
  Narasimhan et al., ver D-008) recomienda ajustar umbrales/prior por clase. Se implementó
  ascenso coordenado sobre pesos por clase aplicados a las probabilidades OOF, con protocolo
  **leave-one-fold-out** para medir la ganancia sin optimismo.
- **Evidencia**: LogReg 0,5603 -> **0,5609** (+0,0006, 4/5 folds); SGD 0,5579 -> 0,5578 (−0,0002);
  cota optimista (ajustar y evaluar en el mismo OOF) apenas 0,5617.
- **Decision**: **no** se adopta ajuste de decisión; se mantiene el argmax.
- **Interpretación**: la técnica explota asimetrías de prior/errores entre clases, y aquí las
  clases están balanceadas **por diseño** (16,0-17,0% cada una), por lo que no hay nada que
  corregir. Queda documentado como experimento negativo con su explicación.
- **Consequences**: la Fase 19 se cierra sin cambios en el modelo; el esfuerzo se redirige a
  modelos y ensambles.

---

## D-012 — Ensamble por stacking adoptado (Fase 20)
- **Status**: Aceptada (2026-09-22)
- **Context**: 4 modelos clásicos sobre la representación ganadora, con matrices OOF de las
  mismas particiones agrupadas: LinearSVC(C=0.2) 0,5628 · LogReg 0,5603 · SGD 0,5580 ·
  ComplementNB 0,5370.
- **Evidencia**: diversidad real (desacuerdo ComplementNB vs SVM 25,2%; SVM vs LogReg 6,2%);
  voto suave 0,5628 (sin ganancia); **stacking con meta-LogReg y evaluación LOFO = 0,5656,
  superando al mejor individual en 5/5 folds**.
- **Decision**: adoptar el stacking de 4 modelos como modelo candidato (submission v03).
- **Alternatives**: voto suave (descartado: no aporta); modelo individual (queda como respaldo
  por su simplicidad: 0,5628 con un único modelo).
- **Consequences**: el pipeline final debe ajustar 4 modelos base + meta-modelo (≈5,5 min en
  CPU); el costo es aceptable y el riesgo de sobreajuste del meta-modelo está controlado por la
  evaluación LOFO.

---

## D-013 — Ensamble extendido de 9 fuentes adoptado (Fase 20b)
- **Status**: Aceptada (2026-09-23)
- **Context**: el análisis de errores (Fase 17) mostró confusión casi exclusivamente adyacente
  (clases medias 18-24 y 25-34 con F1 0,476/0,449) y fuerte dependencia de la longitud. Se
  probaron cuatro hipótesis derivadas de esa evidencia.
- **Evidencia (todo con LOFO sobre las mismas particiones agrupadas)**:
  - Vocabulario de caracteres a 700k features (E-016): 0,5658 individual, mejor que el tope de
    300k (0,5628) -> el límite estaba truncando vocabulario útil.
  - Canal de hashtags (E-014): 0,5644 individual; +0,0019 en el ensamble.
  - Regresión ordinal (E-015): 0,3973 como clasificador directo (débil), pero aporta estructura
    ordinal al meta-modelo (+0,0003 con costo casi nulo).
  - Modelo de palabra como miembro **diverso** (E-017): +0,0023 en el ensamble.
  - Combinación final: **0,5748 ± 0,0048** vs 0,5656 de la Fase 20 (+0,0092).
- **Decision**: adoptar el ensamble de 9 fuentes con meta-modelo estandarizado (submission v04).
- **Alternatives**: mantener la v03 (más simple, 0,5656); voto suave (descartado).
- **Consequences**: el pipeline final ajusta 9 modelos base (≈7 min en CPU); el respaldo simple
  (un solo LinearSVC C=0.2) queda documentado con 0,5658 para quien prefiera reproducibilidad
  mínima. El ajuste de decisión por clase sigue descartado (D-011) y no se re-explora.

---

## D-014 — Meta-modelo enriquecido con features densas (condicionamiento por longitud)
- **Status**: Aceptada (2026-09-23) · E-018
- **Context**: la Fase 17 mostró que la longitud domina el error (52,5% en el quintil más corto
  vs 37,2% en el más largo), es decir, el modelo se equivoca de forma dependiente del tipo de
  tweet.
- **Decision**: el meta-modelo del ensamble recibe, además de los 49 scores de los 9 modelos
  base, las **35 features densas** (longitud, retuit, menciones, URLs, textometría y estilo),
  estandarizadas.
- **Evidencia**: LOFO 0,5784 ± 0,0044 vs 0,5748 sin densas (+0,0036, con las 5 particiones);
  añadir interacciones explícitas longitud×ordinal no mejora más (0,5785), lo que indica que el
  meta-modelo ya aprende esa estructura.
- **Alternatives**: mantener el meta-modelo solo con scores (más simple, −0,0036).
- **Consequences**: la submission v05 usa esta configuración; el pipeline final debe disponer de
  las features densas de test, que ya se calculan de forma determinista y sin leakage (D-009).

---

## D-015 — El MLP denso entra al ensamble como miembro diverso (no como modelo principal)
- **Status**: Aceptada (2026-09-23) · E-019, E-020
- **Context**: la Fase 14 exigía evaluar modelos densos. La GPU estaba ocupada por una carga
  ajena, así que —por el protocolo D-007— los MLP se entrenaron en **CPU** sin interrumpir nada.
- **Evidencia**:
  - MLP sobre las 35 features densas: 0,3381 ± 0,0040 (peor que LightGBM sobre densas, 0,3567).
  - MLP sobre densas + SVD(64) del espacio de caracteres: 0,4071 ± 0,0060. La SVD solo explica
    el **7,8%** de la varianza: la reducción lineal destruye la señal dispersa.
  - **Pero** al añadirlo al meta-modelo del ensamble: 0,5784 -> **0,5851 ± 0,0042 (+0,0066)**,
    el mayor aporte individual desde el canal de menciones.
- **Decision**: el deep learning **no** se usa como modelo principal (queda muy por debajo de los
  lineales dispersos), pero **sí** como miembro del ensamble por su diversidad de errores.
- **Interpretación**: confirma la lección central del proyecto: en texto corto, informal y sin
  recursos preentrenados, la representación dispersa + modelos lineales domina; el valor del
  modelo no lineal no está en su exactitud sino en la **decorrelación** de sus errores.
- **Consequences**: la submission v06 incorpora el MLP; el notebook consolidado documenta ambos
  resultados (el negativo como clasificador y el positivo como miembro del ensamble).

---

## D-016 — Modelo final (12 miembros) y declaración de saturación
- **Status**: Aceptada (2026-09-24) · E-021 a E-027
- **Context**: con la mejor combinación en 0,5851 (E-020) se exploraron cuatro vías nuevas
  guiadas por el análisis de errores y por familias de modelo no probadas.
- **Evidencia (todas con LOFO sobre las mismas particiones)**:
  - Modelo **ordinal acumulativo** (5 clasificadores binarios `P(y>k)`, Frank & Hall): 0,5527 como
    miembro individual, casi a la par del mejor lineal pero estructuralmente distinto.
  - Modelo de **centroides (Rocchio)**: 0,4017 individual, aporta diversidad.
  - Añadidos al ensamble: 0,5851 -> **0,5860** (+0,0009).
  - **Meta-modelo LightGBM**: 0,5858, no supera a la regresión logística.
  - **Especialistas por longitud** (SVM entrenados solo con tweets cortos/largos): **+0,0000**.
  - **BM25** (k1=1.2, b=0.75) sin normalización L2: 0,4688 individual y **−0,0002** en el ensamble
    (la normalización por documento de `TfidfVectorizer` es esencial para el SVM lineal).
  - **kNN (k=25) sobre SVD(64)**: 0,3088 individual y sin aporte.
- **Decision**: el **modelo final** es el ensamble de **12 fuentes** (los 10 anteriores + ordinal
  acumulativo + centroides) con meta-modelo enriquecido: **CV LOFO 0,5860 ± 0,0045**
  (submission v07). Se **declara la saturación** de esta línea de trabajo.
- **Interpretación**: las cuatro últimas rondas aportaron +0,0003, +0,0009, +0,0000 y −0,0002.
  Dentro del espacio permitido (sin embeddings ni transformers), la representación dispersa sobre
  texto crudo con canales de mención/hashtag y un ensamble diverso parece haber alcanzado su techo;
  las mejoras restantes exigirían representaciones aprendidas, que las reglas prohíben.
- **Consequences**: se conserva v07 (0,60372) como mejor envío registrado y se envía v08 como
  candidato final.

### Addendum 2026-09-24 — Supervisión ordinal fina (E-028)
Tras declarar la saturación se probó la vía indicada como trabajo futuro:
- **Pares adyacentes** (5 SVM entrenados solo con cada par de rangos contiguos): precisión media de
  **0,7428** en las fronteras, ordenándolas por dificultad (`18-24|25-34` 0,6856 es la más dura;
  `50-64|65-xx` 0,8099 la más fácil), coherente con la matriz de confusión. Aporte al ensamble:
  **+0,0006** (0,5860 -> 0,5866).
- **Jerarquía gruesa->fina** (3 grupos jóvenes/medios/mayores, precisión 0,7128): aporte nulo por
  encima de los pares (+0,0003 acumulado frente a +0,0006 solo con pares).
- **Barrido del C del meta-modelo**: C=1.0 ya era óptimo (0,5862 / 0,5866 / 0,5865 para 0,3 / 1 / 3).
- **Conclusión**: la saturación se confirma; el techo de esta línea está en ~0,587 de CV y las
  mejoras restantes son menores que la desviación entre folds (±0,004).

---

## D-017 — Features de segunda generación y candidato AGE_ADVANCED_V01
- **Observation**: el sistema anterior explotaba forma superficial y bolsa de palabras, pero el gap
  analysis (`reports/advanced_feature_gap_analysis.md`) mostró cinco brechas: estadística
  supervisada de entidades, distribuciones internas, rareza/novedad léxica, posición y léxico por
  clase.
- **Hypothesis**: incorporar esa señal al meta-modelo mejora macro-F1 con mejora consistente y
  confirmada fuera del CV de desarrollo.
- **Experiment**: ADV-02 (160 features nuevas, 87 locales + 73 supervisadas in-fold, evaluadas con
  LOFO) y ADV-03 (holdout congelado del 15% de clústeres, semilla 2026).
- **Result**: CV **0,5866 -> 0,5897 (+0,0031)**, positiva en **5/5 folds**; holdout congelado
  **0,5914 -> 0,5966 (+0,0052)**, confirmado. Aportes: menciones supervisadas +0,0016 · fuente de
  RT +0,0007 · rareza +0,0003 · léxico por clase +0,0001 · hashtags −0,0001.
- **Decision**: adoptar el candidato `AGE_ADVANCED_V01` (267 features del meta) y generar la
  submission v09. Las features supervisadas se calculan **siempre in-fold** (regla §74).
- **Reason**: la mejora es estable entre folds y se confirma en un conjunto que el meta-modelo no
  vio durante el desarrollo; el aporte proviene de información que el TF-IDF no expresa
  (identidad supervisada de las menciones), no de más complejidad.
- **Alternatives**: quedarse en el baseline congelado v08 (descartado por pérdida de señal
  medible).
- **Consequences**: el catálogo de features pasa a 160 filas en 11 familias
  (`reports/advanced_feature_catalog.csv`); el retrieval train->test queda **sin implementar** y
  marcado REQUIERE_VERIFICACIÓN por riesgo de reglas.

---

## D-018 — Cierre de las líneas avanzadas y política del último envío
- **Observation**: tras el candidato `AGE_ADVANCED_V01` (v09, LB 0,60807) se probaron las tres
  líneas restantes del brief con mayor prioridad: incertidumbre/interacciones (§38/§70), selección
  de features (§47) y NB-SVM (§22/§P).
- **Hypothesis**: alguna de ellas aportaría señal no capturada por el meta-modelo actual.
- **Experiment**: ADV-04 (LOFO con 51 features de incertidumbre y 5 interacciones), ADV-05
  (NB-SVM uno-vs-rest y selección por ANOVA F), ADV-06 (candidato v10 = v09 + NB-SVM).
- **Result**:
  - Incertidumbre e interacciones: **+0,0002** (dentro del ruido); la desviación entre folds baja
    de 0,0043 a 0,0036.
  - Selección de features: top-50 0,5731 · top-100 0,5867 · **top-150 0,5896** · top-200 0,5890
    frente a **0,5897** del conjunto completo -> no aporta.
  - NB-SVM: **0,5641** como modelo (por debajo del SVM TF-IDF 0,5658) pero **+0,0007** como miembro
    del ensamble (4/5 folds).
  - Candidato v10: **0,5904 ± 0,0047** (273 features del meta).
- **Decision**: (1) cerrar las líneas de incertidumbre, interacciones y selección de features;
  (2) incorporar NB-SVM como miembro en el candidato v10; (3) **generar la submission v10 y no
  enviarla**: con 9 de ~10 envíos puntuados usados, gastar el último en +0,0007 (dentro del ruido
  entre folds) no es una decisión defendible.
- **Reason**: el valor esperado de la mejora (+0,0007 ± 0,0013 en el LB) es indistinguible de cero;
  reservar el envío para una mejora mayor o para el cierre de la competencia maximiza el resultado
  esperado.
- **Consequences**: la mejor marca registrada sigue siendo **0,60807** (v09). El archivo
  `submissions/age_submission_v10.csv` queda listo para subir si Carlos decide lo contrario.

---

## D-019 — La frontera 18-24|25-34 es un techo de información, no de modelado
- **Observation**: es la frontera más difícil del problema (confusión 15,3% + 12,5%; F1 de `25-34`
  0,449 y de `18-24` 0,476). El ensamble acierta solo **0,486** en las filas cuya etiqueta real cae
  en ese par, mientras un SVM binario entrenado **solo** con el par alcanza **0,6856**.
- **Hypothesis**: ¿hay señal complementaria sin explotar en esa frontera? ¿Un especialista con
  compuerta (brief §73) puede resolverla?
- **Experiment**: ADV-07. Diagnóstico de perfil de error + tres arquitecturas de enrutamiento:
  (a) especialista LogReg sobre las features del meta con compuerta por top-2; (b) enrutamiento por
  el margen del par-SVM con umbrales τ ∈ {0; 0,5; 1,0; 1,5}.
- **Result**:
  - (a) especialista con compuerta: **−0,0036** (2.587 errores corregidos, 3.217 introducidos).
  - (b) por margen del par-SVM: τ=0 -> −0,0095; τ=0,5 -> −0,0000 (41 corregidas / 43 introducidas);
    **τ=1,0 -> 3.571 filas activadas con 0 correcciones y 0 daños**; τ=1,5 -> 301 activaciones, 0 cambios.
  - Es decir: cuando el par-SVM es lo bastante confiable para creerle, **coincide con el ensamble**;
    cuando discrepa, acierta tanto como falla.
- **Perfil del error (n=57.023 en la frontera)**: los aciertos tienen más menciones (0,836 vs 0,651)
  y más hashtags (0,383 vs 0,238); los errores tienen más URLs (0,409 vs 0,312) y algo más longitud.
  Las señales de **interacción social** discriminan; las de **difusión de enlaces**, no.
- **Análisis de términos**: `18-24` se marca por lenguaje de *métricas sociales* y emoción
  (*perfil, follow, sigo, amo, odio*); `25-34` por **temas** informativos y deportivos
  (*nacional, partido, equipo, mundial, repost*). Ambos cohortes retuitean el **mismo** contenido
  genérico (humor, deportes), y ahí el texto no contiene la edad del autor.
- **Decision**: **cerrar la línea**. No se incorpora ningún especialista de frontera y no se gasta
  un envío en ella.
- **Reason**: la evidencia muestra un **techo de información**: el error es irreducible con las
  señales disponibles porque las clases medias contiguas comparten tema y difieren poco en registro.
  Mejorar esto exigiría señales que las reglas prohíben (embeddings/transformers) o metadatos que el
  corpus no tiene (autor, fecha, perfil).
- **Consequences**: el ensamble de la fase avanzada (v09 enviada, v10 lista sin enviar) queda como
  **modelo final**. `submissions/age_submission_v10.csv` permanece disponible si se decide usar el
  último envío puntuado.

---

## D-020 — El combinador: segmentación dura falla, ponderación suave funciona
- **Observation**: el diagnóstico de intersección mostró que en el **25,94%** de los tweets algún
  modelo base acierta mientras el ensamble falla, y que los rescatadores son sistemáticamente los
  modelos diversos (softmax ordinal 36%, centroides 31%, MLP 31%). El error depende además del tipo
  de tweet (`solo_url` 61,1%; `mencion_alta` 19,2%) y de la longitud (54,5% -> 35,1%).
- **Hypothesis**: el meta-modelo lineal aplica pesos globales y no puede condicionar su confianza al
  contexto del tweet; un combinador que sí pueda hacerlo recuperará parte de esos casos.
- **Experiment**: PH2-L6 — cuatro combinadores con LOFO: LogReg global (referencia), MLP global,
  segmentado por tipo de tweet, segmentado por banda de longitud.
- **Result**: **MLP 0,5967 ± 0,0040 vs LogReg 0,5906 ± 0,0047 (+0,0061, 5/5 folds)**; segmentado
  por tipo −0,0007; por longitud −0,0015. Holdout congelado: **0,5981 -> 0,6024 (+0,0043)**.
- **Decision**: adoptar el MLP superficial como meta-modelo (submission v11). Cerrar la
  segmentación dura — es el segundo fracaso de esa estrategia (antes con la compuerta de frontera,
  −0,0036).
- **Reason**: particionar el espacio reduce los datos de cada submodelo y con ello la calidad de la
  ponderación; la red aprende la misma estructura de forma **suave y compartida entre segmentos**,
  que es lo que la evidencia favorece.
- **Evidence**: `experiments/phase2_ponderacion.json`, `phase3_holdout_v11.json`,
  `reports/error_intersection_analysis.csv`.

---

## D-021 — Dos defectos corregidos: alineación del meta y features de forma degeneradas
- **Observation**: la primera submission v11 resultó **degenerada** (78.576 predicciones en `13-17`,
  una sola en `25-34`) pese a pasar todos los checks de formato.
- **Diagnosis**: (1) el bloque NB-SVM se anexaba al final del lado test pero no del lado train,
  desplazando 6 columnas del resto de bloques; (2) cuatro features de forma de palabra eran
  constantes porque el tokenizador solo aceptaba minúsculas, y `shape_alphanumeric_ratio` era cero
  por definición al separar el tokenizador dígitos de letras.
- **Decision**: corregir el orden de concatenación en ambos lados, arreglar el tokenizador y
  recalcular alfanuméricos con una expresión propia; **validar siempre la distribución de etiquetas
  predichas** además del formato.
- **Reason**: la validación estructural es necesaria pero no suficiente; un bug de alineación
  produce archivos formalmente válidos y semánticamente inútiles.
- **Evidence**: `experiments/phase3b_recorregido.json` (0 columnas con varianza nula tras el
  recálculo; MLP 0,5967 confirmado).

---

## D-022 — La palanca es el combinador: promedio de combinadores como modelo final
- **Observation**: en esta fase, ninguna línea de features nuevas superó el umbral de ruido
  (residuales de longitud **−0,0019**; NB-SVM de palabra +0,0009; estilo solo 0,2793 frente a
  contenido solo 0,5537). En cambio, dos intervenciones sobre el **combinador** dieron las únicas
  ganancias reales: el MLP como meta-modelo (+0,0061, D-020) y ahora el **promedio de dos
  combinadores diversos** (+0,0027).
- **Hypothesis**: la diversidad de sesgos —que sostiene el ensamble de primer nivel— también paga en
  el segundo nivel si se promedian combinadores de sesgo distinto (no lineal vs lineal).
- **Experiment**: PH7 — promedio 0,5/0,5 de las probabilidades out-of-fold del meta MLP y del meta
  LogReg, evaluado con LOFO y confirmado en el holdout congelado.
- **Result**: CV **0,6005** (MLP solo 0,5978; LogReg 0,5915); holdout **0,6077** (MLP 0,6034;
  LogReg 0,5983) -> **Δ +0,0043** sobre el MLP, por encima del criterio de +0,003.
- **Decision**: adoptar el promedio de combinadores como modelo final (submission v12).
- **Reason**: es un promedio de ponderación **fija** (sin nada ajustado a los datos), luego no añade
  riesgo de sobreajuste, y su ganancia es consistente en ambas particiones (CV y holdout).
- **Evidence**: `experiments/phase7_nbsvm_canal_meta.json`, `phase8_holdout_v12.json`.

---

## D-023 — El estilo no es la señal ausente: el contenido manda (hipótesis refutada)
- **Observation**: las fronteras centrales resisten (18-24|25-34: 0,486) y ambas cohortes retuitean
  el mismo contenido genérico. Hipótesis natural: el estilo conservaría la señal demográfica.
- **Experiment**: PH6 — dos canales disjuntos: **estilo puro** (palabras funcionales, forma de las
  palabras de contenido enmascarada preservando longitud y mayúsculas, puntuación, marcadores de
  entidad sin identidad) y **contenido puro** (palabras de contenido e identidades de entidad).
- **Result**: estilo solo **0,2793** (casi azar) · contenido solo **0,5537** (frente a 0,5658 del
  texto completo) · aporte conjunto al ensamble **+0,0008** en 3/5 folds.
- **Decision**: cerrar la línea y registrar el hallazgo interpretativo.
- **Reason**: en este corpus el **contenido** transporta casi toda la señal (quitar palabras
  funcionales cuesta solo 0,012) y el estilo aislado es insuficiente. Eso explica de forma
  cuantitativa por qué las fronteras centrales resisten: comparten tema y el estilo no basta para
  separarlas.
- **Evidence**: `experiments/phase6_contenido_estilo.json`, `features/oof_style_only.npy`.

---

## D-024 — Nota sobre la brecha CV<->LB en las últimas dos versiones
- **Observation**: las brechas CV<->LB de v11 (+0,0208) y v12 (+0,0213) quedan por encima del rango
  de las ocho mediciones anteriores (+0,0145 a +0,0184).
- **Hypothesis**: el combinador no lineal (y su promedio) explota mejor la **memorización de
  near-duplicados** del test, precisamente el componente que el CV agrupado por clústeres bloquea
  por diseño; a mayor capacidad del combinador, mayor brecha.
- **Evidence**: la brecha creció al adoptar el MLP como meta-modelo (v11) y se mantuvo al promediar
  combinadores (v12), mientras el Δ CV y el Δ holdout fueron coherentes entre sí (+0,0038 vs
  +0,0043).
- **Decision**: **no** cambiar el protocolo de validación. El CV agrupado sigue siendo el estimador
  honesto para comparar modelos; el holdout congelado se mantiene como confirmación y predijo bien
  el leaderboard en las dos últimas rondas.
- **Reason**: interpretar la brecha como "el CV está mal" llevaría a optimizar contra el leaderboard,
  que es exactamente lo que el brief prohíbe. La brecha es estructural y explicada.

---

## D-025 — El promedio de dos combinadores es el óptimo (cierre de la línea)
- **Observation**: si la diversidad de sesgos sostiene las ganancias del combinador (D-020, D-022),
  cabría esperar que un tercer combinador diverso mejore el promedio.
- **Experiment**: PH9 — cuatro combinadores evaluados con LOFO sobre las 279 features: LogReg
  (0,5915), RidgeClassifier (0,5836), SGD modified_huber (0,4774) y MLP (0,5978); y sus promedios.
- **Result**: **MLP + LogReg = 0,6005** (óptimo) · MLP+LogReg+Ridge 0,5999 · +SGD 0,5676 · los
  cuatro 0,5702.
- **Decision**: cerrar la línea. El modelo final es el **promedio de dos combinadores** (v12).
- **Reason**: la diversidad solo paga cuando los combinadores tienen **calidad comparable**. Ridge
  es peor y correlaciona con LogReg (aporta poco y diluye), y SGD es demasiado débil (arrastra el
  promedio hacia abajo). Es la misma lección del ensamble de primer nivel aplicada al segundo: la
  diversidad sin calidad no compone.
- **Evidence**: `experiments/phase9_combinadores.json`.

---

## D-026 — Las dos vías autorizadas no pagan: retrieval y POS (cierre con evidencia)
- **Observation**: quedaban dos líneas bloqueadas por reglas: retrieval train->test y POS tagger de
  spaCy. Ambas fueron autorizadas explícitamente por Carlos.
- **Experiment**: PH10 — (A) features de clúster de near-duplicados con etiquetas de train, más la
  etiqueta recuperada por duplicado exacto, en dos usos (features del combinador y sobrescritura
  dura); (B) 26 features de POS con `es_core_news_md` (modelo no-transformer, permitido por D-001).
- **Result**:
  - **A**: la cobertura de clúster **in-fold es 0,00%** porque el CV agrupado elimina por diseño los
    duplicados entre train y validación; el único canal medible son los duplicados exactos (1,36% de
    las filas) y **su etiqueta recuperada solo acierta 61,5%** — por debajo de la precisión del
    modelo en esas filas. La sobrescritura dura **empeora** (−0,0009) y las features de clúster son
    constantes (ruido del combinador: +0,0010).
  - **B**: POS **−0,0032**; ambas vías juntas −0,0013.
- **Decision**: cerrar ambas líneas. El retrieval queda **permitido pero descartado por evidencia
  propia**: su techo no lo fija la cobertura (4,26%) sino la **consistencia de etiqueta de los
  duplicados**, que es 61,5% — insuficiente para superar al modelo.
- **Reason**: memorizar contenido idéntico con etiquetas contradictorias introduce más error del que
  corrige. Es un resultado sobre el **corpus**, no sobre la técnica: en un corpus con etiquetado más
  consistente entre duplicados, el mismo mecanismo sí aportaría.
- **Evidence**: `experiments/phase10_dos_vias.json`, `reports/error_intersection_analysis.csv`.

---

## D-027 — Limpieza de filas ruidosas: mejora real pero insuficiente para cambiar el modelo
- **Observation**: el corpus tiene etiquetas contradictorias para un mismo texto y textos duplicados
  que sobre-representan contenido concreto; ambos podrían sesgar el combinador en las fronteras.
- **Experiment**: PH11 — exclusión de filas **solo en el lado de entrenamiento** (la validación se
  mantiene, porque el test contendrá casos igualmente ruidosos), evaluada con LOFO sobre el
  combinador promediado del modelo final.
- **Result**: excluir las contradictorias (643 filas, 0,38%) -> **+0,0001** (3/5 folds); excluir los
  textos duplicados (2.123 filas, 1,25%) -> **+0,0016** (3/5 folds, con std algo mayor); baseline
  0,6004 ± 0,0042.
- **Decision**: **no** se adopta como cambio del modelo final (por debajo del umbral de +0,003), pero
  se documenta como limpieza opcional con efecto positivo pequeño y consistente en la dirección.
- **Reason**: el efecto está acotado por el tamaño del conjunto limpiado (1,25% de las filas) y no
  supera el ruido entre folds. Excluir los duplicados ayuda algo más que excluir las contradictorias,
  lo que sugiere que el sesgo relevante es la **sobre-representación de contenido**, más que la
  contradicción explícita de etiqueta.
- **Consequences**: si en el futuro se reentrena el modelo final por otro motivo, conviene aplicar
  esta limpieza (coste nulo, dirección positiva). No justifica por sí sola un reentrenamiento.
- **Evidencia metodológica adicional**: la primera ejecución del experimento usó la normalización
  agresiva de `normalize_for_dedup`, que hace colapsar miles de tweets cortos distintos al mismo
  texto y **infló los conflictos de 643 a 12.989 filas** (7,67%). Se detectó, se corrigió la
  definición y se repitió; el resultado inflado (−0,0042) era un artefacto. Queda registrado como
  recordatorio de que la elección de la normalización cambia el diagnóstico del ruido.

---

## D-028 — El consenso y el desacuerdo por familia no superan el umbral (cierre con evidencia)
- **Observation**: la fase B midió que en **26,70%** de las filas el ensamble falla y algún miembro
  acierta, y que el consenso de votos tiene un gradiente fuerte (error 60,2% en el quintil más bajo
  frente a 18,1% en el más alto). ADV-04 había medido el bloque **genérico** de incertidumbre por
  miembro (confianza, margen, entropía, acuerdo) sobre el baseline v09 con meta LogReg: +0,0002.
- **Experiment**: PH14 — 46 features nuevas en tres bloques: C1 acuerdo/entropía/margen **por familia**
  (7 familias), C2 divergencias Jensen-Shannon entre familias, C3 gating contextual
  (reparto de votos x banda de longitud y entropía x flags). LOFO con las mismas particiones y el
  combinador de producción (promedio MLP + LogReg).
- **Result**: +C1 **0,5986** (−0,0018, 1/5 folds); +C1+C2 **0,6002** (−0,0002, 3/5); +C1+C2+C3
  **0,5998** (−0,0006, 3/5). Referencia 0,6004 ± 0,0042.
- **Decision**: cerrar la familia. Ninguna variante se adopta.
- **Reason**: el consenso es una **función determinista** de los scores que el combinador ya recibe y
  el MLP los aproxima sin ayuda explícita. Es coherente con ADV-04 (+0,0002) y con D-020: las
  ganancias del segundo nivel vinieron de **diversidad de sesgos** (MLP + LogReg, +0,0027) y de una
  **vista nueva** (NB-SVM de palabra, +0,0009), no de re-expresar la misma información.
- **Evidence**: `experiments/phase14_consenso_familias.json`.

---

## D-029 — Calibración de miembros, blending en logits y diversidad de subconjuntos: sin mejora
- **Observation**: los miembros entran al meta en representaciones heterogéneas (márgenes SVM sin
  normalizar, log-probabilidades, un score ordinal continuo) y el segundo nivel es un promedio de dos
  combinadores. PH9 probó diversidad de **algoritmo** (Ridge/SGD) sobre las mismas features, pero nunca
  se probó la calibración de los scores ni la diversidad de **vista de features**.
- **Experiment**: PH15 — unificación de los bloques de 6 columnas a log-softmax (T=1) y temperatura por
  miembro ajustada in-fold (rejilla 0,25-4,0 por NLL); promedio de combinadores en probabilidades y en
  logits. PH15b — promedios de 2, 3 y 4 combinadores incluyendo uno entrenado solo con las 84 columnas
  de *stacking puro* (78 scores + 6 NB-SVM de palabras).
- **Result**:
  - D1a (log-prob T=1): promedio **0,6005** (+0,0001, 3/5); logits **0,6007** (+0,0003, 3/5).
  - D1b (temperatura in-fold): promedio 0,6003 (−0,0001, 2/5); logits 0,6002 (−0,0002, 2/5). La
    rejilla eligió T=0,5 para la mayoría de los miembros (agudizar) y 2,5 para uno.
  - D5: producción **0,6004 idéntico** (control de determinismo); MLP puro 0,5976; LogReg pura 0,5915;
    MLP sobre 84 = 0,5851; LogReg sobre 84 = 0,5879; 3 vías 0,5987 (−0,0017); 4 vías 0,5976 (−0,0028).
- **Decision**: cerrar ambas líneas. Ninguna se adopta.
- **Reason**: (a) con estandarización por columna, la temperatura es una transformación afín por bloque
  que el modelo lineal absorbe y que el MLP tampoco necesita; (b) el combinador de la vista reducida es
  más débil (0,585) y muy correlacionado con el de la vista completa: la **diversidad sin calidad
  comparable no compone** (coherente con D-025); (c) el máximo observado (+0,0003) está dentro del
  error estándar pareado.
- **Evidence**: `experiments/phase15_calibracion_scores.json`,
  `experiments/phase15b_diversidad_subsets.json`.

---

## D-031 — Limpieza selectiva del entrenamiento: positiva pero acotada (sin adopción)
- **Observation**: D-027 midió la limpieza global a nivel de combinador (+0,0001 contradictorias,
  +0,0016 eliminando textos duplicados). Quedaban sin probar dos variantes: conservar **un
  representante** por grupo repetido (en vez de eliminar todas las copias) y aplicar la limpieza solo
  a las **clases centrales**, donde se concentra el 84,5% del error.
- **Experiment**: PH18 — F2 (un representante por clúster de near-duplicados) y F3 (clúster selectivo
  en 18-24, 25-34 y 35-49), aplicadas **solo al lado de entrenamiento** de cada fold.
- **Result**: F2 = 0,6003 (−0,0001, 2/5 folds); F3 = **0,6013** (+0,0009, 3/5). Referencia
  0,6004 ± 0,0042.
- **Decision**: cerrar sin adopción. La familia completa (D-027 + PH18) queda en el rango
  [+0,0001, +0,0016]: efecto real, consistente en dirección, pero por debajo del umbral.
- **Reason**: el efecto está acotado por la fracción de filas afectada. Eliminar copias redundantes
  ayuda algo más que conservar un representante, lo que confirma que el sesgo relevante es la
  **sobre-representación de contenido**, no la falta de diversidad de copias.
- **Consequences**: si en el futuro se reentrena el modelo final por otro motivo, la limpieza de
  duplicados (+0,0016) sigue siendo la variante preferida; ninguna justifica un reentrenamiento por sí
  sola.
- **Evidence**: `experiments/phase18_limpieza_selectiva.json`.

---

## D-030 — Señal ordinal auxiliar y selección greedy de miembros: nulas
- **Observation**: la familia ordinal es la que más rescata errores del ensamble (19,96% cuando el
  ensamble falla) y el meta recibe el score ordinal continuo, pero no una **edad implícita agregada**
  ni medidas de consistencia; la selección greedy con reemplazo (Caruana) es el estándar de *ensemble
  selection* y nunca se había probado en este proyecto.
- **Experiment**: PH16 — G1: 15 features de edad implícita por familia, dispersión y consistencia
  ordinal (signo de la fuente continua decidido **in-fold**); G2: media greedy de miembros
  seleccionados con reemplazo sobre las filas de entrenamiento (10 pasos; paso óptimo elegido en train,
  nunca en validación).
- **Result**: G1 = **0,6001** (−0,0003, 1/5 folds; reproducido idéntico en dos corridas). G2 = **0,5770**
  (−0,0234): la media greedy queda 2,3 puntos por debajo del meta-modelo.
- **Decision**: cerrar ambas.
- **Reason**: (G1) la información ordinal ya está en las probabilidades de los miembros ordinales y el
  combinador no lineal la explota; re-expresarla como esperanza no añade nada. (G2) la selección greedy
  optimiza la media simple de probabilidades, muy por debajo del combinador que pondera según el
  contexto (0,6004): **el combinador supera a la selección**, evidencia adicional de D-020.
- **Evidence**: `experiments/phase16_ordinal_greedy.json`,
  `experiments/phase16_ordinal_greedy_g2.log`.

---

## D-032 — Multivista: la vista estructural es la única señal nueva (marginal, no adoptada)
- **Observation**: todas las representaciones vigentes comparten el **contenido** del texto. La fase 6
  mostró que el estilo aislado es casi azar (0,2793), pero eso se midió enmascarando el contenido, no
  **estructuralizando** el tweet.
- **Experiment**: PH21 — dos vistas sin contenido: (a) **estructura**: secuencia de tipos de token
  (palabra, mención, hashtag, URL, número, puntuación, mayúsculas); (b) **extremos**: primeras y últimas
  palabras. Protocolo de dos pasos del brief §13: primero **diversidad de error** (puerta: rescate
  ≥ 15% de los errores del v12), después valor en el ensemble (columnas añadidas al final del meta,
  igual que el NB-SVM de palabras). PH22 — refuerzo: alfabeto ampliado (cubetas de longitud,
  mayúsculas, clases de puntuación, marcadores de frontera) y combinaciones de vistas.
- **Result**:
  - **Puerta de diversidad**: estructura, F1 individual 0,2285 con **rescate 18,17%** y acuerdo 27,96%
    (pasa); extremos, F1 0,4578 con rescate 18,75% y acuerdo 54,92% (pasa).
  - **Ensemble (LOFO, referencia 0,6004)**: **estructura +0,0018 (4/5 folds)**; extremos +0,0000 (2/5);
    ampliada +0,0005 (4/5); estructura+extremos −0,0003; ampliada+extremos +0,0006.
  - **Holdout congelado** (25.204 filas, semilla 2026): base **0,6077** (reproducido exacto) ->
    estructura **0,6108 (+0,0031)**.
- **Decision**: **no se adopta**. El gate primario pre-registrado es el CV (Δ ≥ +0,003) y la vista se
  queda en +0,0018. Se documenta como la única señal nueva con dirección positiva y se deja un archivo
  candidato **local** (`submissions/age_submission_v13_candidato_estructura_NO_ADOPTADO.csv`) para
  decisión de Carlos. **No se envía nada a Kaggle.**
- **Reason**: (a) rigor de protocolo: el criterio se declaró antes de medir y no se mueve después;
  (b) el efecto es del orden de +0,002-0,003 (t pareado ≈ 2,7 en CV; ≈1 error estándar en el holdout):
  real pero pequeño; (c) mecanísticamente plausible —la vista abstrae la *forma* (longitudes,
  mayúsculas, tipos de entidad) que los n-gramas de carácter ponderan de otro modo—, pero el sistema ya
  extraía casi toda esa señal (13,87% de las filas no tienen ningún miembro correcto).
- **Consequences**: si Carlos decide gastar un envío, la justificación es la confirmación en holdout
  (no el CV) y debe quedar registrada como decisión suya. El refuerzo (alfabeto ampliado) **no** mejora
  la vista simple, así que la vía de "hacer la vista más rica" queda cerrada.
- **Evidence**: `experiments/phase21_vista_estructural.json`,
  `experiments/phase22_estructura_ampliada.json`, `experiments/phase19_holdout_VistaEstructura.json`,
  `experiments/phase23_submission_candidato.json`.

---

## D-033 — Cierre de la búsqueda de señal residual: línea saturada
- **Decision**: la línea se declara **saturada** según el criterio pre-registrado. Catorce variantes
  evaluadas sobre el modelo v12 en esta fase (consenso familiar, gating contextual, calibración de
  miembros, blending en logits, diversidad de subconjuntos, estructura v2, limpieza selectiva, edad
  implícita, selección greedy y cuatro variantes multivista): **ninguna cumple Δ ≥ +0,003 con ≥4/5
  folds en el CV agrupado**. El modelo final **no cambia** (v12: CV 0,6004; holdout 0,6077; leaderboard
  0,62182).
- **Reason**: el sustrato quedó verificado (fase A: reconstrucción exacta, acuerdo 100% con la
  submission v11), la validación es honesta (agrupada por clústeres, con control de determinismo
  reproducido bajo concurrencia) y el análisis de errores acota el margen: **13,87% de las filas no
  tienen ningún miembro correcto** y el 84,5% del error se concentra en las dos fronteras centrales.
  Las únicas direcciones positivas son la limpieza de duplicados (D-027/D-031, +0,0016 máximo) y la
  vista estructural (D-032, +0,0018 CV), ambas por debajo del umbral.
- **Consequences**: (1) no se genera ni se envía submission nueva salvo decisión explícita de Carlos;
  (2) cualquier trabajo futuro con reglas más laxas debe atacar primero la frontera `18-24|25-34`
  (42,5% del error) y el segmento `solo_url` (59,9% de error); (3) la limpieza de duplicados sigue
  siendo la mejora de coste nulo recomendada si alguna vez se reentrena el modelo por otro motivo.
- **Evidence**: `reports/fase_residual_informe.md` (consolidado), `experiments/phase20_resumen.json`.

---

## D-034 — Hallazgo de integridad: la submission v12 enviada no es reconstruible al 100% (93,88%)
- **Observation**: al preparar el candidato marginal se reconstruyó el modelo v12
  (`scripts/phase8_holdout_v12.py`) y el acuerdo con `age_submission_v12.csv` fue del **93,88%**, cuando
  la reconstrucción del v11 con la misma metodología dio **100,0000%** (fase A).
- **Diagnóstico**: PH24-PH26, con evidencia convergente:
  1. La **matriz de test cacheada** y una reconstrucción **fresca completa** (13 fuentes del pipeline +
     NB-SVM + supervisadas) coinciden en todos los bloques (max|Δ| = 1,5e-05; densas/estructuradas 0,0)
     y producen **predicciones idénticas (100,0000%)**. Los 13 sub-bloques de fuentes están en su
     posición correcta (correlación 1,0): **no hay desalineación**.
  2. El **ajuste final es determinista**: dos ajustes idénticos en el mismo proceso dan **0 filas
     distintas** (LogReg 418 iteraciones y MLP 30 épocas en ambos).
  3. El **protocolo del holdout sí se reproduce exacto**: la fase 19 re-ajustó la base v12 y obtuvo
     **0,6077**, el mismo número registrado por la fase 8.
- **Decision**: documentar la discrepancia y sus límites. **No afecta a ninguna conclusión de la fase**:
  todas las comparaciones se hicieron pareadas en el mismo proceso (CV con las mismas particiones y
  semillas; holdout con la base re-ajustada en la misma corrida). Afecta solo a la **reconstrucción
  absoluta** del artefacto subido.
- **Reason**: matrices equivalentes + ajuste determinista + receta idéntica => el archivo subido (o el
  estado del código en el momento de generarlo) difiere del estado committeado que hoy se puede
  reproducir. Es un problema de **trazabilidad del artefacto**, no de datos ni de metodología.
- **Consequences**: (1) queda una matriz de test fresca verificada
  (`features/meta_features_test_v12_fresca.npy`) y el candidato se regeneró con ella; (2) cualquier
  comparación futura contra "v12" debe re-ajustar el modelo base en el mismo proceso (como hace la fase
  19), no comparar contra el archivo subido; (3) se recomienda re-ejecutar `phase8_holdout_v12.py` si se
  quiere dejar un artefacto v12 reproducible (~35 min, no cambia ningún resultado de la fase).
- **Evidence**: `experiments/phase24_diagnostico_matriz_test.json`,
  `experiments/phase25_submission_fresca.json`, `experiments/phase26_estabilidad_ajuste.json`.

---

## D-035 — Veredicto final de la fase: se mantiene la saturación
- **Decision**: con el hallazgo de D-034 acotado (no afecta a las comparaciones pareadas), el veredicto de
  D-033 se mantiene: **ninguna variante cumple el criterio pre-registrado** (Δ ≥ +0,003 en CV con ≥4/5
  folds). La única señal positiva es la vista estructural (CV +0,0018 con 4/5 folds; holdout +0,0031) y
  **no se adopta**. El modelo final sigue siendo el v12.
- **Consequences**: el archivo `submissions/age_submission_v13_candidato_estructura_NO_ADOPTADO.csv`
  queda disponible para decisión de Carlos, con la comparación hecha contra la v12 re-ajustada en el
  mismo proceso (cambia 10,47% de las filas). **No se envía nada a Kaggle.**
- **Evidence**: `reports/fase_residual_informe.md`, `experiments/phase20_resumen.json`.

---

## D-036 — D-034 queda cerrada funcionalmente: el v12 reproducible puntúa 0,62188
- **Observation**: el artefacto `age_submission_v12_reproducible.csv`, que solo coincide en el 93,88%
  de las filas con el archivo enviado, se subió a Kaggle y obtuvo **0,62188**, ligeramente **por
  encima** del 0,62182 del archivo histórico.
- **Decision**: cerrar D-034. La discrepancia de reconstrucción existe pero **no tiene impacto en el
  desempeño**: los dos artefactos son equivalentes a efectos de leaderboard. Ambos se conservan
  intactos como artefactos distintos (el enviado 0,62182 y el reproducible 0,62188, ahora el mejor
  registro del proyecto).
- **Consequences**: el leaderboard de referencia del proyecto pasa a **0,62188**. V12 sigue congelado.

---

## D-037 — La capa de decisión ordinal no mejora a V12: cierre de las nueve hipótesis (A-I)
- **Observation**: quedaba abierta la línea de **decisión** (no de representación): convertir las
  señales ya disponibles en una decisión ordinal de seis clases. Los modelos ordinales (E-015, E-021a)
  y los márgenes por pares (E-028a) ya existían como miembros, pero **nunca** se usó una capa de
  decisión sobre probabilidades congeladas. La matriz de cobertura
  (`reports/age_hypothesis_coverage.md`) dejó 9 líneas genuinamente abiertas.
- **Experiment**: 33 capas ajustadas con **corrected CV** (LOFO sobre las OOF de V12; el holdout está
  agotado y no se reutiliza), agrupadas en: A umbrales acumulados (2), B combinación Luce de márgenes
  adyacentes (3), C corrección residual de fronteras (4), D decodificación por matriz de transición
  (5), E proyección ordinal (6), F coste ordinal (3), G desempate por edad esperada (3),
  H calibración por banda de confianza (2), I corrector mínimo (5).
- **Result**: **ninguna alcanza el gate** (Δ ≥ +0,003 con ≥4/5 folds). La mejor es
  **G desempate por edad esperada (tau=0,05): 0,6015 (+0,0011, 4/5 folds)** con rescate 2.705 y daño
  2.822 (**neto −117**). En todas las capas el rescate es igual o menor que el daño; la corrección
  residual más suave (C4) llega a **neto exactamente 0** con 1.397 filas modificadas.
- **Decision**: **cerrar las nueve líneas**. V12 permanece congelado; no se genera submission.
- **Reason**: tres hallazgos. (1) **El argmax de V12 ya es la mejor decisión** de esa distribución de
  probabilidades: cualquier redistribución (proyección, calibración por banda, decodificación por
  confusión) degrada de forma monótona. (2) **Minimizar el error ordinal no es mejorar macro-F1**:
  el desempate por E[Y] mejora el MAE ordinal (0,6552 -> 0,6454) y las fronteras extremos, pero
  empeora las dos fronteras centrales (+462 y +534 errores). (3) **La evidencia de fronteras ya está
  dentro del meta-modelo**: un modelo de umbrales acumulados sobre las features del meta alcanza
  0,5887 por sí solo y su fusión con V12 no aporta; los márgenes adyacentes de E-028a no forman una
  escala de Bradley-Terry válida (0,2947 al decodificarlos).
- **Consequences**: (1) la línea de decisión queda cerrada con 33 capas medidas; (2) el diagnóstico
  (§36 del brief) queda establecido: el error restante se reparte en un **techo de información**
  (13,87% de filas sin ningún miembro correcto, 35% del error) y un **65% de agregación** (26,70% de
  filas rescatables) que ninguna regla de decisión sobre las probabilidades convierte en aciertos;
  (3) cualquier trabajo futuro con reglas más laxas debe atacar la frontera, no la decisión.
- **Evidence**: `experiments/phase31_decision_ordinal.json`,
  `experiments/phase32_correccion_probabilistica.json`,
  `experiments/phase33_corrector_minimo.json`, `reports/age_residual_decision_research.md`,
  `reports/age_residual_audit.md`, `reports/age_boundary_models.csv`.

---

## D-038 — Auditoría de paquetes: el entorno contiene transformers y torch, el proyecto no los usa
- **Observation**: `transformers`, `sentence-transformers` y `torch` están instalados en
  `/home/carlos/venv` y listados en `requirements.txt`. El barrido de imports en `src/` y
  `scripts/` encuentra un único `import torch`, dentro de una comprobación de capacidad de GPU
  (`src/gpu.py`).
- **Decision**: documentarlo explícitamente en lugar de ocultarlo. La restricción de AGE es sobre
  el **uso** (sin embeddings, transformers ni modelos de lenguaje en el camino de modelado), y ese
  uso es 0. Se registra que el entorno no está limpio de paquetes para que la auditoría sea
  auditable en ambos sentidos.
- **Consequences**: cualquier revisión futura debe distinguir "paquete instalado" de "paquete
  usado"; el informe de auditoría incluye las dos columnas.

---

## D-039 — La línea de lexicones se cierra: el bloque es nuevo pero no aporta sobre V12
- **Observation**: 24 subcategorías léxicas (registro, etapas de vida, dominios temáticos),
  1.369 entradas efectivas, 72 features agregadas. El cribado de redundancia muestra que el bloque
  **no** es una reformulación de lo que V12 ya tiene (correlación máxima contra las 279 features de
  V12 por debajo del umbral). El control *lexicon-only* da 0,152 (registro), 0,073 (vida) y 0,203
  (temas) frente a un suelo de azar de 0,1667: solo **temas** supera el azar por sí solo, y por poco.
- **Experiment**: fusión con V12 bajo corrected CV pareado, con control de permutación de varias
  semillas (DD-6): ver `experiments/phase39_fusion_v12.json` y `reports/age_lexicon_results.csv`.
- **Result**: ningún bloque alcanza el gate (Δ ≥ +0,003 con ≥4/5 folds) y ninguno supera a su
  control permutado. La complementariedad es nula o negativa (rescate ≤ daño).
- **Decision**: **cerrar la línea**. V12 permanece congelado; no se genera submission.
- **Reason**: es la segunda línea consecutiva (tras la capa de decisión) en la que la señal es real
  pero insuficiente. La explicación coherente con el diagnóstico previo: el error residual de V12
  no es de representación sino de **observabilidad de la edad** (35% del error es techo de
  información). Un léxico general añade contexto temático y de registro, que es exactamente lo que
  los n-gramas de V12 ya representan bien; lo que falta es información que el corpus no contiene.
- **Consequences**: (1) la línea de features está agotada bajo las restricciones de AGE; (2) el
  criterio que queda abierto es la **frontera** (18-24|25-34 y 25-34|35-49), no más señal léxica;
  (3) el artefacto `lexicons/` y su catálogo se conservan como recurso reutilizable.
- **Evidence**: `reports/age_lexicon_audit.md`, `reports/age_lexicon_results.csv`,
  `reports/age_lexicon_ablation.csv`, `reports/age_lexicon_redundancy.csv`,
  `reports/age_lexicon_complementarity.md`, `reports/age_lexicon_error_analysis.md`.

---

## D-040 — Fallo del producto matricial en `X @ X.T`: documentado, contenido y con guard
- **Observation**: la correlación de Spearman vectorizada daba `+0,0001` donde `scipy` daba
  `+0,9985`. La causa no era el código: en este entorno (`numpy 1.26.4` sobre `Python 3.14.7`,
  combinación no soportada, con numpy enlazado al BLAS de referencia del sistema) el producto
  matricial devuelve **valores erróneos cuando los dos operandos son vistas de la misma
  matriz**, es decir en el patrón `X @ X.T`: la diagonal de un Gram 2000x5 sale 0,7392 en lugar de
  1.999,0, y en 5000x279 sale 298,64 en lugar de 4.962,39. Falla con 1 hilo y con 32, con matriz
  C-contigua y F-contigua, y con datos enteros pequeños. `np.corrcoef` y `np.cov` con
  `rowvar=True` caen en el mismo fallo (devuelven `(5000, 5000)` en lugar de `(6, 6)`) porque
  internamente hacen la auto-transpuesta.
- **Experiment**: auditoría de los 11 sitios del repositorio que usan producto matricial o
  `corrcoef`, más prueba de que `MLPClassifier` da predicciones idénticas con la ruta parcheada.
  Reproducción mínima en `reports/v13_entorno_numerico.md`.
- **Result**: **ningún resultado del proyecto está afectado**. Los 10 sitios del proyecto usan
  rutas sanas (`A @ B.T` con matrices distintas, `A.T @ z` matriz-vector, `corrcoef` con 1D), y el
  único script con la ruta rota era el nuevo `v13_p0b_colinealidad.py`, ya corregido. V12
  (CV 0,6004 / LB 0,62188) y todas las decisiones anteriores siguen vigentes. La primera ejecución
  de P0b dio "0 pares correlacionados", que era un artefacto del fallo y se descartó.
- **Decision**: **documentar sin maquillar** y añadir defensa permanente: `src/verificacion_numerica.py`
  con `autotest_numerico()` (comprueba seis rutas en las que el proyecto se apoya y detiene el
  experimento si alguna falla; enumera las tres rutas prohibidas) y `grama_exacta()` (productos
  cruzados por sumas explícitas, que coincide con `spearmanr` con 1,2e-16). Todo experimento de
  V13 llama al autotest en su primera línea.
- **Reason**: un fallo silencioso de BLAS que devuelve ceros en vez de correlaciones es
  especialmente peligroso en una investigación cuyo criterio de adopción es un Δ pequeño: habría
  producido "no hay colinealidad" y, con ello, la conclusión opuesta.
- **Consequences**: (1) ninguna métrica previa se revisa; (2) la colinealidad de V12 se vuelve a
  medir con la ruta verificada; (3) el cribado de redundancia de todo bloque nuevo (P1, P17, P19,
  P20, P21) debe pasar por `grama_exacta`, nunca por BLAS.
- **Evidence**: `reports/v13_entorno_numerico.md`, `src/verificacion_numerica.py`.

---

## D-041 — V13 se cierra sin adopciones: tres mecanismos medidos, no conjeturas
- **Observation**: seis pistas de variables, dos diagnósticos de estructura de decisión y cuatro
  diagnósticos de causa, 16 experimentos en total, con 8 hipótesis pre-registradas. Ninguna se
  confirma en forma accionable. El control *block-only* más alto fue el de P1 (evidencia por token,
  0,3997 frente a un suelo de 0,1667, el doble del mejor bloque léxico cerrado) y aun así su fusión
  con V12 da −0,0002 con 3/5 folds, con controles permutados (0,6000/0,6003/0,6000) indistinguibles del
  bloque real. P17 (pronombres) dio 0,1918 con rescate neto −64.075; P20 (modalidad) 0,1270, por
  debajo del azar, porque solo el 1,6 % del corpus tiene un marcador de duda; P19 (árboles de
  conjunciones) 0,1539, también bajo el azar, por la misma razón de cobertura. P3a, el especialista
  binario de frontera entrenado con las mismas 279 variables, **empeora** en ambos pares
  (0,4963 -> 0,4944 y 0,4974 -> 0,4942) y deja el global en 0,5993.
- **Experiment**: `reports/v13_final_research.md` y las tres tablas agregadas
  (`v13_master_results.csv`, `v13_hypothesis_registry.csv`, `v13_candidate_comparison.csv`).
- **Result**: tres mecanismos, todos medidos. (1) **Saturación del espacio de variables**: P0b
  medió un VIF mediano de 18,87 con 170 variables por encima de 10 y 87 de 278 direcciones
  explicando el 95 % de la varianza, así que añadir señal no es una apuesta neutral; y no es
  redundancia, porque la criba da 0 de 36, 0 de 26 y 0 de 23 variables redundantes. (2)
  **Información nueva inaccesible al combinador**: la regresión desde V12 reproduce la varianza del
  bloque de P1 (R² mediano 0,522, máximo 0,959) pero la información de clase no (control de la
  reconstrucción 0,0485); y no hay puerta que la alcance, porque el máximo de 90 combinaciones de
  umbrales es 0,6005 con 0 filas desviadas. (3) **El modelo está en el límite de su espacio**: un
  modelo nuevo y dedicado a la frontera es peor que V12 restringido al par.
- **Decision**: **cerrar V13 con V12 congelado**. No abrir P2, P6, P18, P21, P24 (más variables del
  mismo tipo), ni P4 y P5 (más estructura de decisión, ya refutada por P3a). V12 conserva la
  referencia congelada de 0,6004 en CV y 0,62188 en leaderboard. Ninguna submission nueva.
- **Reason**: el §28 del plan prohíbe ejecutar por cantidad, y aquí la evidencia acumulada dice que el
  espacio está agotado, no que falten ideas. Cerrar con un diagnóstico causal es más útil que
  acumular más filas en la tabla de experimentos.
- **Consequences**: (1) la única vía que queda abierta es una **decisión de alcance**, no un
  experimento: relaxing la restricción §3 para permitir un recurso lingüístico externo, o cambiar la
  métrica y redefinir el gate §24; (2) se conservan como infraestructura reutilizable la criba de
  redundancia contra la matriz completa, `src/verificacion_numerica.py` y el protocolo de
  pre-chequeo de rescate, que evitó pagar ~70 minutos de CPU en dos pistas; (3) la predicción
  pre-registrada de que un bloque lingüístico sube la rama LogReg y se diluye en la MLP **no se
  confirma ni se refuta en su alternativa**: ninguna de las dos cabezas mejora.
- **Evidence**: `reports/v13_final_research.md`, `reports/v13_reproducibility.md`,
  `reports/v13_hypothesis_registry.csv`, `reports/v16c_reconstruccion.md`,
  `reports/v16d_puerta.md`, `reports/v3_especialistas.md`, `experiments.csv` (89 experimentos).

---

## D-042 — Gate de adopción revisado: el efecto baja a +0,001, la significación no
- **Observation**: el cierre de V13 (D-041) dejó abierta una vía de alcance, y Carlos la asume
  explícitamente: el gate §24 exigía un Δ ≥ +0,003, muy por encima del ruido de las
  variaciones medidas (los controles permutados de P1b se movían entre 0,6000 y 0,6003, es decir
  ±0,0003, y el bootstrap de una diferencia de −0,0002 dio un IC95 de [−0,0015; +0,0010]). Con un
  listón de +0,003, V13 cerró sin adopciones, pero también sin descartar que un efecto real de
  +0,0015 esté pasando desapercibido.
- **Decision**: **nuevo gate §24**, que sustituye al anterior a partir de ahora:
  1. **Efecto**: Δ de macro-F1 **≥ +0,001**.
  2. **Folds**: **≥ 3 de 5** folds positivos (antes 4 de 5). Con efectos más pequeños, exigir 4/5
     penaliza la varianza del estimador, no la calidad del efecto.
  3. **Significación**: el IC95 del bootstrap **pareado excluye el cero**. **Este punto no se toca**,
     y es el que hace el trabajo: con 169.336 filas, un efecto de +0,001 tiene un intervalo de
     confianza del orden de ±0,0015, de modo que un Δ de +0,001 **no puede aprobar el criterio 3** y
     el gate sigue rechazando el ruido. Bajar el listón de efecto sin tocar el de significación
     sería exactamente lo contrario de ser menos riguroso.
  4. **Segunda partición**: el efecto debe reproducirse al menos en **3 de 4** espacios de
     validación disjuntos (partición por grupos en cuatro partes, sin usar el OOF completo para
     elegir nada). Esto es nuevo y es la consecuencia directa de bajar el efecto: la
     re-ejecución sobre otra partición es la que distingue un efecto de +0,0015 de un máximo de la
     rejilla de hiperparámetros.
  5. Se mantiene la exigencia del control de permutación (DD-6) y la doble métrica (§36), y se
     añade: **ningún candidato puede mejorar la global a costa de empeorar una frontera central**.
- **Reason**: el gate antiguo estaba calibrado para un mundo donde el ruido de las medidas era de
  ±0,0003 y el listón de +0,003 era defendible. Con V13 ya medido, se sabe que el listón de efecto
  era el que excluía candidatos, no la significación. Bajarlo **y** añadir la segunda partición
  mantiene el rigor y amplía la sensibilidad.
- **Consequences**: (1) P1b **sigue sin adoptarse** (Δ = −0,0002); (2) cualquier candidato futuro se
  juzga con este gate; (3) el cambio se aplica a V14 y no se reabre V13: los experimentos cerrados
  con el gate viejo mantienen su veredicto, que sigue siendo el correcto con el criterio más estricto.
- **Evidence**: `experiments/v13_p1b_fusion_tokens.json` (ruido de los controles), plan §24,
  `reports/v13_final_research.md` §8.

---

## D-043 — Se relaja §3: se permite un recurso lingüístico externo, con procedencia documentada
- **Observation**: la mayor limitación declarada en el cierre de V13 es que las tres listas de
  vocabulario eran **propias** (R-1) y, por tanto, exhaustivas por construcción: 1.359 entradas en los
  lexicones, 26 columnas de pronombres, 23 de modalidad, 36 de conjunciones. Carlos asume
  explícitamente esta vía de alcance. El entorno ya tiene el recurso: **spaCy 3.8.13 con el modelo
  español `es_core_news_md`**, que es un modelo de análisis lingüístico (**no** transformer, **no**
  LLM, **no** embeddings preentrenados para clasificación), y que además de POS entrega
  **análisis morfológico, lemas, entidades nombradas y dependencias**.
- **Qué se mantiene prohibido**: embeddings, Word2Vec, FastText, GloVe, sentence embeddings,
  BERT/RoBERTa, transformers, LLM, modelos externos de clasificación, corpus o datasets externos
  con etiquetas, y **pseudo-etiquetas o labels de test**. Se mantiene R-5 (el test solo se
  transforma), R-6 (gate completo) y R-7 (doble métrica).
- **Qué se permite ahora**: un **analizador morfosintáctico** como fuente de variables
  morfológicas, léxicas y sintácticas, y una lista léxica externa de**una sola función** (por
  ejemplo sentimiento) si se documenta su procedencia y su SHA256, sin usarla para entrenar nada
  externo. Se mantiene la prohibición de que el recurso externo aporte *etiquetas*: solo forma.
- **Por qué no es la línea ya cerrada por D-026**: D-026 midió **26 recuentos de etiqueta POS** y
  obtuvo −0,0032. Eso no es lo mismo que la información morfológica que el mismo modelo entrega
  (tiempo, aspecto, persona, género, número, modo) ni que la estructura de dependencias. Son tipos de
  variable distintos, y el gap audit ya habia marcado la morfología como cerrada solo en su versión
  de rasgos por sufijo, que sí es redundante con `char_wb(3-5)`.
- **Procedencia**: `es_core_news_md`, instalado en el venv, SHA256 del modelo registrado en el
  informe de la fase. Ningún download: el recurso ya está en la máquina.
- **Consequences**: (1) V14 es una fase nueva con su propio pre-registro de hipótesis; (2) V13 no se
  reabre; (3) la auditoría de D-038 se actualiza: `spacy` pasa de "no usado" a "usado para
  análisis lingüístico", que es una vía autorizada y no un modelo de clasificación.
- **Evidence**: `python -c "import spacy; spacy.util.get_installed_models()"` ->
  `['es_core_news_md']`; D-026 en este mismo archivo; `plan/V13_ADVANCED_RESEARCH_PLAN.md`.

---

## D-044 — V14 se cierra: relajar §3 no cambia el mecanismo de V13
- **Observation**: con D-043 se permitió un analizador morfosintáctico español externo y con D-042
  se bajó el listón del gate a +0,001. V14 construyó 37 variables nuevas (morfología verbal y
  nominal, lemas, entidades nombradas, sintaxis de dependencias) y las midió con el criterio más
  favorable posible para ellas.
- **Result**:
  - **V14a (control aislado)**: LogReg **0,2409** y LinearSVC 0,2270 frente al suelo de azar 0,1667.
    Es **el mejor bloque aislado de todo el proyecto** por encima de los lexicones (0,2033), los
    pronombres (0,1918) y la modalidad (0,1270). El cribado marca solo 4 de 37 como redundantes, y
    son las de longitud y diversidad, que es exactamente lo que P0b predijo para la textometría.
  - **V14b (fusión)**: macro-F1 **0,5990**, delta **−0,0015**, **1 de 5 folds** positivos. Falla
    cinco de los seis criterios del gate. Los tres controles de permutado (0,5962, 0,5975, 0,5986)
    son **peores** que la candidata, o sea que la señal es real; y aun así **la segunda partición da
    0 de 4 folds positivos con una media de −0,0031**, y el IC95 pareado es [−0,0027; −0,0002] con
    solo un 0,9 % de resmuestras positivas. La frontera `18-24|25-34` empeora de 0,4963 a 0,4919.
- **Decision**: **NO ADOPTAR**. Cerrar V14 y con él la vía de relajar §3. V12 sigue congelado en
  0,6004 (CV) y 0,62188 (leaderboard).
- **Reason**: es la conclusión de alcance más importante del proyecto, y va contra la intuición.
  La restricción de recursos **no era el cuello de botella**. V14 tiene las mejores variables
  disponibles: más fuertes en aislamiento que ninguna línea anterior, no redundantes, y procedente de una
  fuente genuinamente externa. Y a pesar de eso degradan V12 de forma **reproducible** en dos
  particiones independientes. El mecanismo que V13 identificó —el meta-modelo no puede explotar señal
  nueva dentro de este espacio de variables— queda confirmado con un tratamiento experimental más
  fuerte, no refutado. Bajar el gate tampoco lo cambia: con D-042, un Δ de +0,001 tendría que pasar
  además el IC95 y la segunda partición, y −0,0015 falla los tres.
- **Consequences**: (1) la conclusión de V13 se generaliza: **el cuello de botella es
  arquitectónico, no de datos ni de restricciones**; (2) el gate revisado de D-042 queda validado
  por su primer uso real: aceptó el caso difícil de distinguir señal de ruido (el bloque **supera**
  a sus controles permutados) y aun así rechazó el candidato, que es exactamente el comportamiento
  que se le pedía; (3) la segunda partición como criterio resultó ser la medida más informativa de
  las seis, y conviene exigirla en cualquier trabajo futuro del proyecto; (4) no queda abierta la
  lista léxica externa de una sola función (por ejemplo sentimiento): el precedente de V14 dice que
  otro bloque del mismo tipo daría el mismo resultado, y abrirlo sería ejecutar por cantidad.
- **Evidence**: `reports/v14_linguistico.md`, `reports/v14_fusion_linguistico.md`,
  `experiments/v14a_linguistico.json`, `experiments/v14b_fusion_linguistico.json`,
  `reports/v13_final_research.md` §4.

## D-045 — V15: la redundancia de entrada es funcional; podar empeora y no se adopta

- **Observation**: P0b midió colinealidad sobre la matriz de 279 variables y lo dejó como
  diagnóstico, sin evaluación. Carlos pidió el paso completo: medir qué variables de entrada están
  correlacionadas entre sí, cuáles se asocian con la salida, y **eliminar las redundantes**.
- **Result**:
  - **V15a (estructura, in-fold)**: sobre las 312 variables (279 de V12 y 33 lingüísticas de V14),
    2 columnas constantes, 402 pares con |rho| >= 0,80, 245 con |rho| >= 0,90, 15 duplicados
    exactos, **58 clusters que cubren 171 de 310 variables**. Asociación con la salida por eta
    cuadrado: mediana 0,0306, máximo 0,5599, y **112 variables con eta2 < 0,01**. El **Jaccard de
    la estructura de pares entre folds es 1,0000** (mínimo 1,0000): la redundancia es
    perfectamente reproducible y la poda no depende de la partición.
  - **V15b (poda evaluada)**: un representante por cluster, el de mayor eta2 **dentro del fold** →
    197 variables. Macro-F1 **0,5965** frente a 0,6005 de V12: **delta -0,0040 con 0 de 5 folds
    positivos**; segunda partición **0 de 4** con media -0,0065; IC95 pareado [-0,0054; -0,0027].
- **Decision**: **NO ADOPTAR la poda**. V12 se queda con sus 279 variables. La poda queda documentada
  como medición, no como técnica.
- **Reason**: es un resultado con contenido y va contra la intuición de partida. La intuición dice
  que las variables casi idénticas sobran; la medición dice que **estaban cargando con algo**. La
  explicación coherente con todo lo demás es la del conjunto de estimadores débiles: variables
  |rho| >= 0,90 entre sí no son la misma variable, son lecturas ligeramente distintas del mismo
  phenomenon, y promediarlas reduce la varianza igual que reduce el ruido en cualquier promedio. Quitar 15 duplicados *exactos* sí debería ser inofensivo, y el resultado muestra que
  el coste no está en los duplicados sino en el conjunto de las casi idénticas.
  La lección metodológica también es real: **la colinealidad afecta a la interpretación de
  coeficientes, no a la predicción**, que es lo que P0b insinuaba y V15 confirma con evaluación
  completa.
- **Consequences**: (1) no se vuelve a intentar poda de variables como vía de mejora; (2) queda
  registrado que la estructura de redundancia es estable (Jaccard 1,0000), lo que la hace un buen
  material para **interpretabilidad**, no para rendimiento; (3) el trabajo de variables continues
  en la línea de V16, que ataca la arquitectura, que es donde siguen apuntando todos los
  diagnósticos de V13, V14 y V15.
- **Evidence**: `reports/v15_correlacion.md`, `reports/v15_poda.md`,
  `reports/v15_poda_comparativa.csv`, `reports/v15_clusters.csv`,
  `reports/v15_asociacion_salida.csv`, `experiments/v15a_correlacion.json`,
  `experiments/v15b_poda.json`, `features/v15b_oof.npy`.
