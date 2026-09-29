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
    cuadrado: mediana 0,0306, máximo 0,5599, y **112 variables con eta2 < 0,01 (110 de las 310 analizables; el recuento de 112 sale sobre las 312 columnas de la matriz, e incluye las 2 constantes, que son 0 por definicion)**. El **Jaccard de
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

## D-046 — V16: la arquitectura aporta en las fronteras, y no alcanza para la metrica global

- **Observation**: con D-045 cerrado, la unica palanca que quedaba era la forma de combinar las dos
  cabezas. V12 arrastra desde la primera version un promedio fijo 0,5 LogReg + 0,5 MLP que nunca se
  optimizo, y todos los diagnosticos de V13, V14 y V15 senalaban al meta-modelo como cuello de
  botella. Carlos pidio anadir este paso.
- **Result** (mismas 279 variables, mismo protocolo, sin variables nuevas):
  - **A0, promedio fijo (produccion)**: macro-F1 0,6005.
  - **A1, peso aprendido por fold** (el peso se elige sobre una particion interior agrupada del
    train exterior, sin mirar la validacion): macro-F1 **0,6002**, delta **-0,0003**, **0 de 5
    folds**. Los pesos elegidos son **0,1 · 0,5 · 0,5 · 0,4 · 0,5**: el 0,5 de produccion se elige
    solo en tres de los cinco folds.
  - **A2, combinador logistico** sobre las doce probabilidades de las dos cabezas (apilamiento en
    dos niveles): macro-F1 **0,6011**, delta **+0,0006**, **3 de 5 folds**. Segunda particion
    **3 de 4**. IC95 pareado **[-0,0003; +0,0015]**, que **contiene el cero**.
  - **Doble metrica (§36)**: la frontera `18-24|25-34` pasa de 0,4963 a **0,5065** y la
    `25-34|35-49` de 0,4974 a **0,5131**. Son **+0,0102 y +0,0157**, tres a cinco veces el efecto
    global, y en el mismo sentido.
- **Decision**: **NO ADOPTAR**. Falla dos de los cinco criterios (efecto minimo y significacion) y el
  gate manda. V12 sigue congelado en 0,6005 (CV) y 0,62188 (leaderboard), sin submission nueva.
- **Reason**: (1) A1 descarta de golpe la version mas barata de la hipotesis arquitectonica: el 0,5
  de produccion no era una decision desafortunada, era la correcta; (2) A2 es el primer delta
  positivo del proyecto y aparece en la direccion que la hipotesis predecia, pero su magnitud esta
  **por debajo del ruido de la estimacion**, asi que no puede sostenerse; (3) la mejora de las
  fronteras es un hallazgo real, no un espejismo, y encaja con todo lo demas: **el problema de este
  conjunto de datos esta en las fronteras de edad**, y ahi el meta-modelo tiene algo que aportar. Lo
  que no tiene es magnitud suficiente para que la media ponderada por clases la registre.
- **Consequences**: (1) **leccion operativa, la mas util de la fase**: una mejora de +0,0157 en la
  frontera `25-34|35-49` es **invisible** en la macro-F1 global. Cualquier trabajo futuro que se
  dirija a las fronteras tiene que medir sobre ellas desde el principio, no como metrica
  secundaria; (2) el gate D-042 queda validado por un segundo caso difficile, y este es el
  instructivo: la **intuicion** (las fronteras mejoran, luego el modelo mejora) y el **gate**
  discrepan, y el gate gana porque el intervalo de confianza no excluye el cero; (3) V16 no cierra
  del todo el problema arquitectonico: cierra el combinador de dos niveles sobre las dos cabezas.
  Quedan sin medir la perdida de la propia combinacion (un objetivo distinto de un promedio) y las
  arquitecturas de mas de dos niveles.
- **Evidence**: `reports/v16_arquitectura.md`, `reports/v16_arquitectura_comparativa.csv`,
  `experiments/v16_arquitectura.json`, `features/v16_oof_peso_aprendido.npy`,
  `features/v16_oof_combinador_logistico.npy`, `plan/V13_ADVANCED_RESEARCH_PLAN.md`.

## D-047 — V17 se cierra: la senal local es grande, y no es identificable

- **Observation**: V16 mostraba que un combinador logistico mejora las dos fronteras centrales
  (+0,0102 y +0,0157) sin mejorar la global. La pregunta de V17 era si esa senal se puede convertir
  en ganancia global con especialistas de frontera y una compuerta. La referencia del proyecto
  arrastraba ademas una discrepancia sin resolver entre 0,6004 y 0,6005.
- **Result**:
  - **Referencia reconciliada**: 0,6004 y 0,6005 son **dos estimadores** del mismo macro-F1, no el
    mismo numero redondeado: pooled 0,6005017473 frente a media de folds 0,6004103057, con una
    diferencia de 9,14416e-05 (efecto Jensen). **Canonico 0,6005 pooled.**
  - **A2 de V16 se reproduce** desde sus artefactos, con las fronteras en 0,5065 y 0,5131.
  - **La complementariedad local es real y grande**: el especialista de pares E-028a, que ya existia,
    saca 0,6864 y 0,7218 en las filas de cada frontera frente a 0,4963 y 0,4974 de V12; resuelve
    14.356 y 15.571 filas y rompe 3.519 y 2.872. Techo con enrutamiento perfecto 0,7481 y 0,7726.
  - **El mecanismo, medido**: los errores de V12 en la frontera son **errores confiados**
    (separacion 0,809 cuando el especialista acierta frente a 0,539 cuando acierta V12), y solo el
    12,7 % y el 8,0 % de las victorias del especialista caen dentro de la region donde V12 tiene el
    par arriba. Escuchar al especialista ahi sin mas vale **-0,0080**.
  - **Ninguna formulacion funciona**: compuerta de ambiguedad +0,0002; regla conjunta +0,0000;
    **P9** arbitro aprendido con protocolo anidado y 47 columnas existentes, **-0,0001 y 0 de 5
    folds**; **P5** combinador local que redistribuye la masa del par, **-0,0002, 1 de 5 folds, y las
    fronteras empeoran**.
  - El techo con un **arbitro perfecto** seria +0,0107 y +0,0077, que **supera el gate de adopcion**.
    El problema nunca fue el techo, fue la identificacion.
- **Decision**: **V17 se cierra sin adopciones.** V12 sigue congelado en 0,6005 (pooled) y 0,62188
  (leaderboard), sin submission nueva. Ninguna pista queda PROMISING ni FINALIST.
- **Reason**: la respuesta a la pregunta cientifica de la fase es precisa y no es la que se
  sospechaba. La senal local existe y es grande (opcion C cierta), el problema no es solo de
  decision en el sentido de umbral (opciones D y E falsas), y no es un problema de metrica porque el
  techo superaria el gate si fuera alcanzable. **Lo que no existe es una forma de identificar las
  filas**, y eso no se arregla con mas compilaciones de la misma senal. Importa notar que las dos
  formulaciones razonables (seleccion y combinacion) se midieron y fallaron, y que P9 falso de forma
  explicita una prediccion registrada antes de mirar el resultado.
- **Consequences**: (1) **leccion principal, y es la accionable**: los margenes de pares ya son
  entradas de V12 (V15a lo midio al listar la matriz) y V12 no los explota, cuando por si solos valen
  0,6864 y 0,7218 en las filas de su frontera. Hay informacion accesible en el espacio de entradas
  que el combinador no usa, y eso **no se resuelve con una compuerta mas: exige cambiar la funcion
  objetivo**, que hoy es macro-F1 sobre seis clases y no sobre las fronteras; (2) la compuerta por
  ambiguedad queda cerrada con tres evidencia independientes (ADV-07 en 2025, P1b y P9 en V17);
  (3) el perfil de confianza (P8) queda como contribution aunque no sea accionable: es la primera
  medicion de que la confianza de V12 no identifica sus propios errores de frontera; (4) la
  tercera frontera se diagnostico sin gastar CPU y muestra el mismo mecanismo, asi que no se abre.
- **Evidence**: `reports/v17_final_research.md`, `reports/v17_reproducibility.md`,
  `reports/v17_compliance.md`, `reports/v17_candidate_comparison.csv`,
  `reports/v17_master_results.csv`, `reports/v17_hypothesis_registry.csv`,
  `reports/v17_gap_audit.md`, `reports/v17_p0_reproduccion.md`, `reports/v17_p1_fronteras.md`,
  `reports/v17_p1b_curva_compuerta.md`, `reports/v17_p1c_regla_conjunta.md`,
  `reports/v17_p8_perfil_confianza.md`, `reports/v17_p9_arbitro.md`,
  `reports/v17_p5_combinador_local.md`, 10 figuras, `experiments.csv` con 102 experimentos.


## D-048 — Correccion: las 279 variables de V12 si contienen la senal de frontera

- **Observation**: V13 P3a publico que un especialista binario entrenado sobre las 279 variables de V12
  acertaba 0,4944 y 0,4942 en las filas de las dos fronteras centrales, frente a 0,4963 y 0,4974 de
  V12. Sobre ese numero se construyo el diagnostico de que **la representacion** era la causa de la
  ventaja del especialista disperso, y la auditoria de P0 de V18 lo confirmo y lo uso para reenfocar la
  fase entera.
- **Result**: el numero publicado **no es la exactitud del especialista**, es una mezcla. El campo
  `filas_actuables` de P3a vale 57.019 y es el recuento sobre **todo** el dataset (el 33,7 % de
  169.336 filas), no dentro del par. Dentro de las 57.023 filas de `18-24|25-34` solo **38.424** son
  actuables y **18.599 quedan congeladas** con la prediccion de V12, donde V12 acierta el 4,2 %. La
  mezcla da exactamente 0,4944. Ejecutando **el codigo y las variables exactos de P3a** con sus dos
  codificaciones de etiqueta (indice de clase y binaria), la exactitud del especialista es:

  | especialista binario en las filas del par | 18-24\|25-34 | 25-34\|35-49 |
  |---|---|---|
  | **279 variables de V12 (denso)** | **0,7136** | **0,7493** |
  | TF-IDF disperso (E-028a) | 0,6864 | 0,7218 |
  | V12 global de seis clases | 0,4963 | 0,4974 |
  | *publicado por P3a (mezcla con la compuerta)* | *0,4944* | *0,4942* |

- **No hay fuga**: las 72 columnas supervisadas se construyen en `advanced_02_features.py:191` con
  exactamente el mismo `make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")`,
  asi que cada fila procede de modelos que no la vieron. El 0,7136 es una medida out of fold legitima.
- **Decision**: **se corrige el diagnostico**. Las 279 variables de V12 contienen **mas** informacion de
  frontera que la representacion dispersa, y el combinador de V12 no extrae casi nada de ella. El
  resultado de sistema de P3a (delta -0,0012 con compuerta) **se mantiene**, porque la compuerta era
  mala; lo que estaba mal era la lectura de por que fallaba.
- **Reason**: el error es una **lectura de recuento**: comparar un numero global con uno del par y
  concluir que la compuerta congelaba cuatro filas. Sobrevivio porque nadie midio al especialista sin
  compuerta; la cifra que se volvio a medir durante cuatro fases fue la que P3a publico, que era
  derivada y no medida.
- **Consequences**: (1) **la premisa de V18 se invierte**: el objetivo de V18 deja de estar sin donde
  tomar senal, porque la senal esta en el espacio de caracteristicas del propio estudiante. El hueco
  real es 0,4963 extraido contra 0,7136 disponible; (2) la asimetria que D-047 dejo abierta es mayor y
  mejor caracterizada de lo que se penso; (3) la hipotesis que V18 tiene que poner a prueba ya no es
  "cambiar el objetivo" en abstracto, sino **capturar en un modelo de seis clases un 0,22 que ya esta
  disponible en sus propias variables**; (4) una consecuencia futura que este hallazgo deja abierta y
  que no es de V18: V12 comprime la salida de cada miembro a seis numeros, y esa compresion es
  precisamente la que preserva o destruye la informacion de frontera, porque el especialista binario
  la recupera desde esas mismas seis columnas por miembro.
- **Evidence**: `reports/v18_p0_reproduccion.md`, `reports/v18_representacion_confound.csv`,
  `experiments/v18_p0_reproduccion.json`, `reports/v18_gap_audit.md` (seccion 5.b, cuya conclusion se
  corrige aqui), `experiments/v3_p3a_especialistas.json` (el campo mal interpretado).

## D-049 — V18 se cierra: la senal esta disponible y ninguna funcion objetivo la cobra

- **Observation**: la auditoria de P0 de V18 concluyo que la ventaja del especialista de frontera
  venia de la **representacion**, y la premisa de la fase era: cambiar el objetivo de
  entrenamiento en vez del decision layer. La evidencia que sostenia esa conclusion era el 0,4944
  que V13 P3a publico.
- **Result**:
  - **D-048 corrige la premisa**: ese 0,4944 no era la exactitud del especialista sino una mezcla con
    la prediccion de V12 en las 18.599 filas del par que la compuerta dejaba congeladas. El
    especialista sobre las **279 variables de V12** saca **0,7136** y **0,7493**, mas que el
    **0,6864** y **0,7218** del TF-IDF disperso, sin fuga. **La senal estaba en el propio espacio de
    variables del modelo todo el tiempo**, y V12 extrae 0,4963 de 0,7136 disponibles.
  - **P1, re-peso de clases** (tres niveles pre-registrados, fidelidad del control 0,000000): el
    nivel moderado deja 0,6001 (-0,0004, 3 de 5 folds) y el nivel fuerte sube la frontera **+0,082**
    hasta 0,5786 mientras la global **baja** a 0,5841 (-0,0164). El intercambio es de suma
    cero con localizacion exacta: `18-24` gana +0,0170 y `13-17` paga -0,0166.
  - **P3, margen de adyacencia** (cinco brazos, PyTorch, todo pre-registrado): el margen central sube
    `18-24|25-34` **+0,0392** (0,4865 a 0,5257) y baja `25-34|35-49` **-0,0552** (0,4833 a 0,4281)
    con lambda 0,1. Estas dos cifras son de la **cabeza lineal sola**, que es donde actua la perdida
    de margen; en el sistema combinado el mismo efecto se amortigua a +0,0209 y -0,0268, y con
    lambda 0,5 la segunda frontera se desploma hasta 0,4013. Las dos fronteras **estan acopladas por
    la clase `25-34`**. Se declaran las dos medidas porque responder a preguntas distintas y
    confundirlas seria repetir el error que D-048 corrige.
  - **P4, multi-objetivo** (cuatro brazos que separan arquitectura de objetivo): el termino auxiliar
    **aditivo, con cero parametros extra**, mueve la frontera **+0,0038** como maximo, y esa cifra es de
    la **cabeza global** (0,4858 a 0,4896 en el brazo C2); en el sistema combinado el movimiento es de
    +0,0017. El mejor brazo queda en -0,0012. Handicap de montaje declarado: la cabeza lineal de PyTorch no iguala a la
    `LogReg` de sklearn y el control queda 0,0022-0,0023 bajo V12, handicap del mismo orden que los
    efectos medidos.
  - **P5, destilacion** (teacher cross-fitted en cinco inner folds agrupados, cuadricula de lambda y
    temperatura seleccionada dentro del train exterior): **todos** los brazos destilados superan a su
    propio control, luego **la senal del teacher no es inerte**, pero la mejora maxima es de
    **+0,0008** con **0 de 5 folds** positivos, y la frontera recupera el **0,8 %** de lo que el
    teacher tiene.
  - **Ninguna adopcion** en toda la fase. V12 sigue en 0,6005017473 pooled y 0,62188 en el
    leaderboard, sin submission nueva.
- **Decision**: **V18 se cierra sin adopciones.** P1, P3, P4 y P5 quedan cerradas. P2, P6, P7 y P9 no se
  ejecutan: la condicion que el brief puso para P5 era que P1-P4 demostraran senal del objetivo, y no
  se cumplio; y el brief ordeno cerrar tras P5 salvo razon experimental extraordinaria.
- **Reason**: la explicacion comun de los cinco negativos es precisa y medible: **la macro-F1 de seis
  clases es una media, y la frontera es una minororia de la masa**. Todo mecanismo que la mejore pagando
  fuera de ella pierde (P1 lo paga en `13-17`, P3 lo paga en la otra frontera), y todo mecanismo que
  la mejore sin pagar fuera no consigue moverla (P4 y P5). No es un problema de informacion, que D-048
  descarto, ni de arquitectura, que P4 descarto al no anadir parametros y tampoco rendir. Es un
  problema de la magnitud que la funcion objetivo da a la region donde vive el error.
- **Consequences**: (1) **la accion que este trabajo recomienda no es un experimento**: cambiar la
  funcion objetivo **a la que se optimise**, no la que se informa, o trenear y seleccionar sobre las
  fronteras directamente; (2) una consecuencia que queda a mano y tiene nombre propio: **V12 comprime
  la salida de cada miembro a seis numeros, y el especialista binario recupera 0,7136 de esas mismas
  seis columnas por miembro**, luego la compresion es en si misma un objeto de estudio; (3) el
  precedente de D-048 se generaliza como metodo: **un numero derivado puede pasar cuatro fases sin que
  nadie lo mida**, y la unica forma de detectinglo es releer todos los artefactos y recalcular, no
  confiar en el informe que los resumen; (4) la destilacion se midio con el protocolo que el brief exigio
  y dio un resultado informativo y no nulo: la senal se transfiere, con una eficiencia del 0,8 %.
- **Evidence**: `reports/v18_final_research.md`, `reports/v18_candidate_comparison.csv`,
  `reports/v18_master_results.csv`, `reports/v18_hypothesis_registry.csv`,
  `reports/v18_p0_reproduccion.md`, `reports/v18_p1_peso_clases.md`, `reports/v18_p3_margen.md`,
  `reports/v18_p4_multiobjetivo.md`, `reports/v18_p5_destilacion.md`, `reports/v18_distillation.csv`,
  `reports/v18_reproducibility.md`, `reports/v18_compliance.md`, 7 figuras de V18.

## D-050 — Correccion de la razon de D-049, y cierre de AGE

- **Observation**: la auditoria de cierre estrategico (`scripts/age_cierre_auditoria.py`, que no
  entrena nada y solo verifica cifras sobre artefactos existentes) midio dos cosas que D-049
  afirmaba o implicaba sin haberlas medido.
- **Result**:
  1. **D-049 estaba mal razonado, aunque acertara en la conclusion.** Escribio que "la macro-F1 de seis
     clases es una media, y la frontera es una minoria de la masa". Leida al pie de la letra, **es
     falsa**: `18-24` son 16,93 % de las filas y `25-34` el 16,74 %, luego juntas son **33,67 %**, un
     tercio y no una minoria.
  2. **La lever no es pequena, y ahora esta medida.** Corregir `n` filas mal clasificadas de un par a
     su etiqueta verdadera, que es lo que haria un selector perfecto, vale por contrafactual sobre el
     OOF existente: **+0,0015 con 500 filas, +0,0031 con 1.000, +0,0063 con 2.000**. El gate de
     adopcion de +0,003 **se cruza con unas 478 correcciones netas**, de las 28.721 y 28.438 que V12
     comete en cada par. Es decir, la magnitud de la oportunidad es de sobra.
  3. **La calibracion CV -> LB confirma que el gate no engaña.** Regresando el leaderboard sobre el CV
     de los cuatro envios cuyos dos extremos son verificables en disco (v01, v07, v08, v12):
     **LB = 0,00223 + 1,02825 x CV**, **R2 = 0,9968**. Un gate de +0,003 en CV vale **+0,00308** en LB
     y llevaria a **0,62278**. El gap es un offset aditivo estable, de +0,01716 a +0,02132, y **no da
     puntos gratis**: los gates locales estan calibrados, ni mas cerrados ni mas abiertos de lo que
     parecen.
  4. **La hipotesis de compresion de outputs muere por construccion.** Los 62 artefactos de train de
     `features/` estan consumidos o cerrados con numero, y **todos** los OOF de miembros estan ya
     comprimidos a seis columnas o menos, porque cada miembro es un modelo de seis clases y no puede
     emitir mas. La compresion no la introduce V12: se introdujo al construir cada miembro.
- **Decision**: **AGE se cierra con V12 como champion.** No se ejecuta ningun experimento adicional y
  no se genera ninguna submission. Se **reescribe la razon de D-049** con el diagnostico correcto.
- **Reason**: el bloqueo **no es de magnitud, ni de informacion, ni de funcion objetivo**. Es de
  **identificabilidad**: existen 28.721 filas mal clasificadas en la primera frontera, corregirlas
  darian +0,012 de macro-F1, y solo ~478 bastan para el gate, pero **no existe un selector que
  identifique cuales**. Eso es lo que midieron los cinco selectores de V17 (compuerta +0,0002, regla
  conjunta +0,0000, arbitro aprendido -0,0001 con 0 de 5 folds, combinador local -0,0002) y la pagina
  de confianza de V16, que con 90 puertas no desvia ni una fila. Los errores de V12 en frontera son
  ademas **errores confiados**: margen 0,809 cuando el especialista acierta contra 0,539 cuando V12
  acierta, luego la confianza no discrimina. Y solo el 12,7 % y el 8,0 % de las victorias del
  especialista caen en la region de frontera. La razon correcta importa: una razon equivocada induce a
  leer la frontera como una lever pequena y a cerrarla para siempre, cuando es una lever grande a la que
  solo le falta un selector, que es un problema distinto y resoluble en otro dataset.
- **Consequences**: (1) **V12 queda congelado** en 0,6005017473 pooled, 0,62182 en el envio y 0,62188
  en la replica; (2) AGE **no se reabre** salvo que aparezca una fuente de informacion nueva, una regla
  nueva del concurso o evidencia externa que cambie el espacio de busqueda. En particular, y de forma
  permanente: **no mas fases, ni busqueda de hiperparametros, ni variantes de perdidas, ni de
  features, ni de ensembles, ni de destilacion, ni de especialistas**; (3) la accion que el trabajo
  deja recomendada no es un experimento: **optimizar o seleccionar sobre las fronteras directamente**,
  lo que exige cambiar la metrica efectiva del entrenamiento, y queda como recommendation para quien
  retome el problema con un dataset que permita agrupar por autor (D-003) o con etiquetas mas
  informativas; (4) una correccion de trazabilidad: la sonda de prior-shift que menciono el agente
  especialista **no dejo artefacto en disco**, luego su +0,0010 no es verificable y no se usa como
  evidencia; aun tomandolo por cierto, queda tres veces por debajo del gate.
- **Evidence**: `reports/age_cierre_auditoria.md`, `reports/age_cierre_auditoria.csv`,
  `reports/age_lb_calibracion.csv`, `figures/age_elasticidad_frontera.png`,
  `experiments/age_cierre_auditoria.json`, `scripts/age_cierre_auditoria.py`.

## D-051 — V19: el combinador de V12 desperdiciaba un bloque de 73 variables (ADOPTAR, sin enviar)

- **Observation**: Carlos reabrio AGE con objetivo competitivo y cuatro familias por revisar. La
  revision del agente `kaggle specialist` descarto **tres** con numero y dejo **una**: retrieval tiene
  cobertura de cluster in-fold **0,00 %** por diseno del CV agrupado y su techo aritmetico es
  **+0,0010**; label cleaning ya se ejecuto en PH11 (+0,0016) y PH18 (+0,0009); y la cascada ordinal
  probabilistica ya se midio en ORD-1 (-0,0118) y ORD-B (-0,3057). La premisa de la familia 2
  ("las combinaciones relevantes son no lineales") **esta falsificada** por P0': el especialista de
  frontera sobre las mismas 279 variables sale **0,7136 con `LogisticRegression`** y **0,7075 con
  `LinearSVC(0.2)`**, dos clases lineales con el mismo resultado.
- **Experiment**: un boosted tree **pequeno y pre-registrado** (`depth=3`, `n_estimators=300`,
  `lr=0,05`, `min_child_samples=200`, `colsample_bytree=0,5`, `reg_lambda=1,0`) como **tercer miembro**
  de V12, con los pesos de combinacion pre-registrados, LOFO agrupado de 5 folds, meta siempre
  ajustado fuera de su fold, y **control de ruido** con el mismo arbol y las filas de las variables
  barajadas.
- **Result**:
  - El control de ruido queda **0,5909687 (delta -0,0095, 0 de 5 folds)**, por debajo de V12, luego se
    comporta como debe. La ventaja del arbol real sobre su gemelo barajado es **+0,4495**.
  - **El arbol aislado da 0,6072312** y la combinacion de tres partes iguales **0,6079388 (delta
    +0,0074, 5 de 5 folds, IC95 [+0,0064, +0,0085])**, con IC95 que excluye el cero. El MLP(64) solo
    da 0,5989481 y la LogReg sola 0,5915679, luego **el combinador de V12 promedia algo peor que su
    mejor parte**.
  - **La bateria de falsacion acoto el origen**: las 279 dan 0,6072, **sin las 73 columnas
    supervisadas bajan a 0,5870** (por debajo de V12), solo las 78 de miembros dan 0,5842 y solo las 73
    supervisadas dan 0,5597.
  - **La segunda particion de 4 folds, con la combinacion ya fijada, da 0,6090431 (+0,0085)** con los
    cuatro folds por encima del gate (0,6077 / 0,6060 / 0,6102 / 0,6122).
  - Sin fuga: el bloque supervisado aislado da 0,5597 y su media de folds coincide con el pooled en
    **1,5e-06**. Si estuviera in-sample, la media de folds se separaria del pooled.
  - Alineacion verificada por **firma**: la columna de mayor magnitud del bloque supervisado ocupa el
    **mismo indice 255** en train y en test, con rangos [-6,29; 59.439] y [-6,39; 50.673]. El 0,9057 de
    acuerdo al reconstruir la submission de V12 esta junto al **0,9388** que registro la fase 25 con
    esta misma matriz fresca, y corresponde a que el re-ajuste es otro modelo.
  - **Sin leakage, sin colision de plataformas, sin busqueda de hiperparametros**: una sola
    configuracion, sin tuning.
- **Decision**: **ADOPTAR como challenger**. Submission construida y validada con 8 de 8 checks en
  `submissions/age_submission_v19_challenger_boosting.csv`. **NO ENVIADA**: la subida de este proyecto
  es manual y la decision es de Carlos.
- **Reason**: el hallazgo **no** es que los arboles sean mejores, y la formulacion importa. Es que **el
  combinador de V12 estaba desperdiciando un bloque de 73 variables que ya estaban dentro de sus
  propias 279**. Ese bloque no son probabilidades sino agregados label-informed (`mention_mean_c0` a
  `c5` y familia, con valores de hasta 59.439), y necesita combinacion no lineal: el promedio lineal
  no puede darla, y por eso el arbol gana donde las dos cabezas lineales pierden. La primera frontera
  sube **+0,0120** y esta vez **sin intercambio de suma cero**, que era el fallo que hundia la global
  en P1. El cambio respecto a V12 es de **11.392 filas (10,09 %)** con **rescate 4.622, dano 3.311 y
  neto +1.311**.
- **Consequences**: (1) **D-050 queda superado**: su veredicto de cierre era correcto **sobre las cuatro
  familias que el brief propuso**, y era falso como veredicto general, porque no contenia boosting sobre las 279 variables; (2) la accion recomendada por D-050 ("optimizar sobre las fronteras") **no era la
  unica via**: aqui la frontera mejora como **consecuencia** de arreglar el combinador, no como
  objetivo; (3) **riesgo declarado y no medible**: el bloque supervisado tiene firma distinta entre
  train (cross-fitted) y test (ajustado sobre todo el train), un combinador lineal promedia esa
  diferencia sin quejarse y un arbol puede ser mas sensible. El supervisor de V12 se ajusto tambien
  sobre el train completo, luego comparte el sesgo y **el riesgo solo lo resuelve el leaderboard**;
  (4) el LB previsto por la calibracion verificada es **0,62734** frente al **0,62182** del envio, y es
  una **prediccion, no una medida**; (5) si el envio confirma, la linea correcta no es buscar mas
  arboles sino **rehacer el combinador** con el bloque supervisado como variable de primer orden.
- **Evidence**: `experiments/v19_b1_boosting.json`, `experiments/v19_b1_verificacion.json`,
  `experiments/v19_b1_confirmacion.json`, `experiments/v19_b1_challenger.json`,
  `reports/v19_b1_boosting.md`, `reports/v19_b1_verificacion.md`, `reports/v19_b1_confirmacion.md`,
  `reports/v19_b1_challenger.md`, `reports/v19_b1_challenger.csv`,
  `figures/v19_b1_boosting.png`, `figures/v19_b1_control.png`,
  `submissions/age_submission_v19_challenger_boosting.csv`.

## D-052 — V19.x se cierra: el eje de parametros del champion es plano (V19 PERMANECE CHAMPION)

- **Observation**: con V19 en **0,62578** de leaderboard, el objetivo paso a maximizar el score sobre
  un mecanismo que ya funciona, y no a abrir familias nuevas. Antes de tocar un hiperparametro se
  auditio el champion y se consulto al agente `kaggle specialist`, que **recomendo cerrar la
  explotacion** y, en el mejor de los casos, gastar dos configuraciones diagnosticas.
- **Audit P0-P1-P2** (sin entrenar ninguna configuracion nueva):
  - **P0**: la configuracion leida de la constante del modulo coincide **byte a byte** con el artefacto
    JSON. La matriz es la de V12, sin seleccion. Semilla del arbol 42.
  - **P0, atribucion**: las 279 dan **0,6072312**, sin las 73 quedan en **0,586956** y las 73 solas dan
    **0,5597155**. Anadir las 73 a las 206 aporta **+0,0203** y anadir las 206 a las 73 aporta **+0,0475**:
    es **interaccion y no suma**.
  - **P1**: el bloque de 73 aporta solo el **9,1 % de la ganancia** con el **28,6 % de los cortes**,
    luego cada corte suyo vale **4,0 veces menos** que uno del bloque de 206. Se usan **271 de 279**
    columnas, el top 10 concentra el **39,3 %**, y la variable dominante es `oof_E014_margenes_0` con
    el **7,99 %**. **Las 12 variables mas importantes son todas margenes de miembros del ensemble.** El
    bloque label-informed es un **socio de interaccion**, no una variable primaria.
  - **P2**: hueco train a OOF de **+0,007781** en el fold 0 (+0,012609 en la media de 5 folds).
    1.800 arboles, 2.400 hojas, 1.841.063 bytes.
- **Los tres argumentos del especialista, que son los que cierran el caso**:
  1. **Tasa base del subespacio de parametros = +0,0008** frente a un gate de +0,0030. Las 14
     formulaciones cerradas de V15 a V18 se movieron en [-0,0040, +0,0008]. Todo lo que alguna vez movio
     la aguja fue un **cambio de mecanismo**, nunca un parametro.
  2. **Resolucion del aparato = +0,0011**: 0,6090431 en 4 folds contra 0,6079388 en 5, **sin cambiar
     nada**. Un efecto real menor que +0,001 no es distinguible de elegir particion.
  3. **Residuo de la calibracion sobre V19 = -0,0016**: el CV 0,6079388 predecía 0,62735 y el LB real
     es 0,62578. En V12 el residuo fue +0,0019, luego la oscilacion entre modelos vecinos es de
     0,0034. **Un +0,002 en CV puede aterrizar en cualquier punto entre -0,000 y +0,004 en LB.**
- **Experiment**: las dos unicas configuraciones autorizadas, con la delta medida **contra V19** y no
  contra V12, y con confirmacion en la particion de 4 folds.
- **Result**:
  - **Puerta de coste cero, superada**: la ocupacion del techo de cortes es del **97,5 %** (1519 de
    1800 arboles llegan a los 7 cortes que permiten 8 hojas), luego **la profundidad es el limite real y
    no las hojas**, que es la premisa de la configuracion A. Si hubiera sido del 70 %, el eje de
    capacidad reabriria; no lo esta.
  - **A, `max_depth` 4 con `num_leaves` 8**: CV 0,6079208, **delta -0,0000**, 3 de 5 folds, IC95
    [-0,0004, +0,0004], 4 folds -0,0000, neto de error **+3** filas. **No merece submission.**
  - **B, `min_child_samples` 800**: CV 0,6076698, **delta -0,0003**, **0 de 5 folds**, IC95
    [-0,0006, +0,0001], 4 folds -0,0002, neto de error **-50** filas. **No merece submission.**
  - **0 de 2 configuraciones** cumplen el criterio de submission, que exige los cuatro puntos: delta
    CV >= +0,003, 5 de 5 folds, IC95 que excluya el cero y confirmacion >= +0,002 en 4 folds.
- **Decision**: **V19.x se cierra. V19 permanece champion con LB 0,62578.** No se genera ninguna
  submission y **no se vuelve a V12**, porque cualquier `LB > 0,62578` es el unico criterio valido y
  V19 ya lo supera.
- **Reason**: el eje de parametros alrededor del champion es **plano**, y la razon de que sea plano
  esta medida: la profundidad ya esta al 97,5 % de su techo, luego no queda un tercer nivel que
  capturar, y subir el suelo de filas solo poda los cortes de peor ratio ganancia por corte. Y el ruido
  del aparato de medicion es del mismo orden que cualquier efecto plausible, de modo que **una
  diferencia de CV pequeña aqui no es evidencia de nada**, sino la distancia entre dos particiones de
  lo mismo.
- **Consequences**: (1) **`CHAMPION.md` creado**, con los tres scores separados y sin volver a llamar
  0,62188 "LB oficial"; (2) el modelo champion queda **persistido con SHA256** en
  `models/v19_champion_lightgbm.txt`, de modo que V19 es reproducible sin reajustar; (3) el OOF del
  arbol queda cacheado en `features/v19_oof_arbol_5folds.npy` para cualquier trabajo futuro; (4) **AGE
  queda cerrado tambien en su segunda vertiente**: no queda ni un parametro que tocar; (5) lo que
  reabre AGE queda anotado y es solo tres cosas: fuente de informacion nueva, regla nueva del concurso
  o evidencia externa. La unica linea que el registro dejaba abierta, **E-022 aplicado a las 279 de la
  era V12 con regularizacion explicita**, es exactamente lo que V19 ya cobro, asi que esta cerrada.
- **Evidence**: `experiments/v19_p0_auditoria.json`, `experiments/v19_x1_explotacion.json`,
  `reports/v19_p0_auditoria.md`, `reports/v19_importancia.csv`, `reports/v19_complejidad.csv`,
  `reports/v19_x1_explotacion.md`, `reports/v19_x1_comparativa.csv`,
  `reports/v19_x1_complementariedad.csv`, `figures/v19_importancia.png`, `figures/v19_complejidad.png`,
  `figures/v19_x1_explotacion.png`, `CHAMPION.md`.

## D-053 — V20: dos familias cerradas con numero y una con senal positiva post-V19

- **Observation**: con V19 en **0,62578** y su eje de parametros agotado, Carlos reabre AGE con el
  objetivo de encontrar **mecanismos que aporten informacion distinta** a V19 y V12, y con la regla de
  clasificar cada familia como `NEW_SIGNAL`, `WEAK_SIGNAL`, `NO_SIGNAL`, `OVERFIT`, `DUPLICATE` o
  `ILLEGAL`. V19 queda **congelado**: no se toco su configuracion, sus features, su modelo persistido,
  su submission ni su hash.
- **Auditoria previa, sin proponer nada**: se cruzaron los **117 experimentos** contra **28
  mecanismos**. Tres familias del brief **nunca** se habian probado: **RandomForest/ExtraTrees,
  HistGradientBoosting y CatBoost** como clase de funcion sobre las 279, y **ninguna transformacion de
  clase condicional** aplicada a las features numericas. Y una familia quedo **cerrada con medicion en
  un comando**: el campo `id` **es el indice de fila**, 1 a 169.336 en train y en test, con correlacion
  de rango **0,00123** con la clase y distribucion plana por quintiles. **No hay senal de tiempo que
  extraer**, y la familia no consume un hueco.
- **F1, la vista de clase condicional de las 122 features numericas: `NO_SIGNAL`.** El bloque de 73
  label-informed se construyo in-fold sobre **cinco** familias de entidades (mencion, hashtag, fuente de
  retweet, rareza, lexico) y las **122 features densas y estructuradas nunca recibieron ese
  tratamiento**. Se construyo el bloque equivalente, **976 columnas**, in-fold. Comparacion
  **emparejada con el mismo aprendiz y los mismos hiperparametros**: arbol sobre las 279 da
  **0,6072312**, sobre las **122 crudas 0,3440624**, sobre las **976 del bloque 0,3438942**. El
  transform **no aporta nada** (delta -0,0002). Como cuarto miembro: **-0,0001**, neto **+30**.
  **Consecuencia que importa mas que el resultado**: acota por que funciona el bloque de 73. No es que
  las transformaciones de clase condicional sean buenas en general, sino que las familias de entidades
  y la rareza **son** discriminantes y las features de forma, posicion y entropia **no lo son**.
- **F2, la clase de funcion alternativa: `NO_SIGNAL`, y el negativo mas informativo de la sesion.**
  ExtraTrees, con sesgo inductivo de verdad distinto (cortes aleatorios, arboles profundos, bagging, sin
  boosting ni learning rate), da **0,5836218** sobre las **mismas 279**: **0,0236 por debajo** del
  LightGBM de V19, con 0 de 5 folds. Como cuarto miembro: **-0,0010**, neto **-240**. **El efecto de
  V19 no es "cualquier combinador no lineal": es del BOOSTING**, y eso explica el mecanismo por el que
  el bloque de 73 se cobra, que es el ajuste secuencial por residuo y no la no linealidad en general.
- **F3, los cinco umbrales anidados: `WEAK_SIGNAL`, y el unico mecanismo vivo post-V19.** El
  especialista de D-048 saca 0,7136 siendo un binario de **dos clases**, luego ajustado sobre **el 34 %
  de las filas**. El umbral anidado hace **la misma decision con el 100 %**. Se ajustan cinco
  clasificadores de `clase <= k-1` contra `clase >= k` sobre las 279, con monotonia por maximo acumulado
  y la distribucion de seis clases por diferencias. **No es un duplicado**: E-028a usa pares
  **adyacentes**, que es el comparador; y ORD-1 de D-037 construia las CDF **a partir de la salida de
  V12**, sin clasificadores nuevos, y dio 0,5887.
- **Result**:
  - **Exactitud balanceada de los cinco discriminadores**: 0,7632 / 0,8264 / 0,8465 / 0,8373 / 0,8112,
    es decir **+0,26 a +0,35 sobre el azar**, y **superan a V12 medido en la misma fila en tres de los
    cinco umbrales** (0,8465 contra 0,8504; 0,8373 contra 0,8210; 0,8112 contra 0,7429).
  - **La cadena en solitario: 0,6046893, por encima del 0,6005017 de V12 y sin un solo modelo
    multiclase.** Ese es el resultado de la sesion: una geometria distinta llega a la misma region.
  - **Como cuarto miembro: 0,6097782, delta +0,0018, 5 de 5 folds, IC95 [+0,0012, +0,0025],** neto
    **+294** filas y **5.674** cambios de prediccion en el train.
- **Decision**: **F1 y F2 se cierran con numero. F3 sigue abierta como `WEAK_SIGNAL`, y su submission
  queda construida y validada pero NO ENVIADA.** V19 sigue siendo champion con 0,62578 y no se toca.
- **Reason**: F3 es el unico mecanismo con senal positiva y estable, y su justificacion es **estructural
  y no estadistica**: la cadena de cinco discriminadores binarios alcanza por si sola 0,6046893, por
  encima de V12, lo que demuestra que hay una geometria de decision distinta disponible. Pero la
  mejora esperada en LB es de **+0,0019**, y el residuo de la calibracion sobre V19 fue de **-0,0016**
  con oscilacion de **0,0034** entre modelos vecinos: **la mejora esperada es del mismo orden que la
  incertidumbre del aparato**. Por eso se presenta y no se envia solo, que es justo lo que distingue a
  este candidato de los cuatro de V19.x descartados, donde no habia cambio de mecanismo.
- **Consequences**: (1) `age_submission_v20_candidato_umbrales.csv` queda lista, con **3.546 de 112.892
  filas cambiadas (3,14 %)** frente al envio de V19, 8 de 8 checks y SHA256 `2c964af7...`; (2) el bloque
  de 1,3 GB queda **fuera de git** y se reconstruye con `scripts/v20_f1_f2.py`; (3) **la comprension de
  por que funciona V19 queda mas aguda**: el bloque de 73 importa por ser un socio de interaccion con
  entidades discriminantes, y el boosting lo cobra porque ajusta residuo a residuo; (4) la direccion
  con mas recorrido que queda abierta es la de **descomposiciones de decision**, no la de clases de
  funcion ni la de transformaciones de features.
- **Evidence**: `experiments/v20_f1_f2.json`, `experiments/v20_f3_umbrales.json`,
  `experiments/v20_f3c_candidato.json`, `reports/v20_f1_f2.md`, `reports/v20_f1_f2_comparativa.csv`,
  `reports/v20_f1_f2_complementariedad.csv`, `reports/v20_f3_umbrales.md`,
  `reports/v20_f3_umbrales.csv`, `reports/v20_f3c_candidato.md`, `figures/v20_f1_f2.png`,
  `figures/v20_f3_umbrales.png`, `submissions/age_submission_v20_candidato_umbrales.csv`.

## D-054 — V20 CONFIRMED_CHALLENGER (0,62689) y ORDINAL_RECONSTRUCTION cerrada

- **Observation**: la submission `age_submission_v20_candidato_umbrales.csv` puntuo **0,62689** en
  Kaggle, un delta de **+0,00111** sobre V19 (0,62578). V20 pasa a ser el **nuevo referente de
  leaderboard de AGE** y V19 queda como **baseline historico**. No se reemplaza ningun artefacto de
  V19.
- **Result**: el **A control exacto** de la fase de reconstruccion reproduce V20 al 100 %:
  **0,6097782 frente a 0,6097782**, con 0 de 5 folds y neto 0. Ese control es lo que valida el
  script de reconstruccion, y sin el ninguna de las otras tres variantes seria interpretable.
- **La hipotesis ordinal queda validada externamente**: la cadena de cinco discriminadores no solo mejoro
  el OOF (`V19 0,60794-0,60904` frente a `V20 0,6097782`) sino que **transfirio positivamente a
  Kaggle** (0,62689 > 0,62578). La familia `ORDINAL_NESTED_THRESHOLDS` pasa de `WEAK_SIGNAL` a
  **`CONFIRMED_CHALLENGER`**, y queda **invertida la prioridad**: ya no se buscan clases de funcion ni
  modelos genericos, sino el maximo rendimiento de este mecanismo.
- **Auditoria de las cinco fronteras (P2)**: exactitud balanceada **0,7632 / 0,8264 / 0,8465 / 0,8373 /
  0,8112**, AUC **0,9284 / 0,9224 / 0,9274 / 0,9340 / 0,9456**, y **descalibracion entre -0,0003 y
  +0,0002**, es decir **las cinco estan esencialmente perfectamente calibradas**. No hay margen en
  recalibrar.
- **Consistencia ordinal (P3)**: **18.927 de 169.336 filas (11,18 %)** violan la monotonia, con
  magnitud media **0,002457** y maxima **0,518606**. **V20 resolvio eso con el maximo acumulado**, que
  es la unica correccion que respeta el orden sin inventar informacion.
- **Impacto de cada frontera sobre el blend (P6)**: degradar la frontera 1 cuesta **-0,0129**, la 2
  **-0,0072**, la 3 **-0,0040**, la 4 **-0,0018** y la 5 **-0,0024**. **La primera frontera es el cuello
  de botella en impacto**, y es ademas la de menor exactitud balanceada (0,7632). Eso separa "frontera
  debil" de "frontera cuyos errores importan poco": **aqui la mas debil es la que mas importa**.
- **Las cuatro reconstrucciones (P1), y la respuesta a la pregunta de la fase**:
  - **A control**: 0,6097782, delta **+0,0000**, 0/5, neto 0. Reproduce V20 exacto.
  - **B proyeccion monotona por PAVA**: 0,6097927, delta **+0,0000**, 3/5, IC95 [-0,0000, +0,0001],
    neto **+5**. **Identica a A**, y eso era lo esperable: con una magnitud media de violacion de
    0,002457, **una proyeccion monotona no tiene nada que corregir**.
  - **C decision por conteo de umbrales**: 0,5983871, delta **-0,0114**, 0/5, neto **-3.319**, con
    fronteras **0,545 y 0,5456** frente a las 0,5075 y 0,5029 de V20.
  - **D decision por edad esperada**: 0.5834541, delta **-0,0263**, 0/5, neto **-6.414**, con fronteras
    **0,541 y 0,5738**.
- **Decision**: **`ORDINAL_RECONSTRUCTION = CLOSED`.** La pregunta de la fase tiene respuesta: **la mejora
  de V20 procede de usar cinco fronteras ordinales, y NO existe margen en la forma de convertir esas cinco
  probabilidades acumuladas en las seis clases finales.** Los datos lo dicen en tres direcciones
  independientes: la descalibracion es cero, el 11,18 % de violaciones tienen magnitud despreciable, y la
  proyeccion monotona es identica al control.
- **Un hallazgo que vale mas que el resultado**: **C y D mejoran las fronteras de forma espectacular**
  (0,545/0,5456 y 0,541/0,5738 frente a 0,5075/0,5029) **y destruyen el macro-F1** (-0,0114 y -0,0263).
  Es la **leccion de P1 repetida dentro de la familia ordinal**, y es la confirmacion mas limpia de que en
  este problema **la macro-F1 de seis clases es la metrica que manda y la frontera es una minoria de la
  masa**. Un mecanismo que mejora la frontera y empeora la global no es una mejora.
- **Consequences**: (1) **la prioridad de investigacion queda invertida** y es explotar la familia ordinal,
  no buscar clases de funcion ni ensembles genericos; (2) **la direccion con mas recorrido que queda
  abierta** es la de **descomposiciones de decision**, porque la cadena sola ya alcanza 0,6046893 por
  encima de V12 y el primer umbral concentra el mayor impacto; (3) **los umbrales 4 y 5 apenas importan**
  (-0,0018 y -0,0024), luego cualquier trabajo futuro debe concentrarse en los dos primeros; (4) **no se
  propone submission nueva todavia**, porque ninguna de las cuatro variantes mejora el OOF de V20 y el
  criterio exige mejora local antes de enviar.
- **Evidence**: `experiments/v20_p0_fronteras.json`, `experiments/v20_p1_reconstruccion.json`,
  `reports/v20_p0_fronteras.md`, `reports/v20_p0_fronteras.csv`, `reports/v20_p0_consistencia.csv`,
  `reports/v20_p1_reconstruccion.md`, `reports/v20_p1_comparativa.csv`,
  `reports/v20_p1_complementariedad.csv`, `figures/v20_p0_fronteras.png`,
  `figures/v20_p0_consistencia.png`, `figures/v20_p1_reconstruccion.png`.

## D-055 — V20 P2: mejorar la frontera mas debil destruye la global (ORDINAL_WEIGHTS cerrada)

- **Observation**: el paso 8 del plan pedia probar una variante de perdida ponderada por frontera **solo si el
  analisis mostraba que unas fronteras tienen errores mucho mas perjudiciales**. P6 lo mostro con numero:
  la frontera 1 cuesta **-0,0129** de caida del blend frente a **-0,0018** y **-0,0024** de las fronteras 4
  y 5, es decir **siete veces mas impacto**, y es ademas la de menor exactitud balanceada (0,7632). La
  condicion se cumplio y se ejecuto **una sola configuracion**, sin grid.
- **Experiment**: `class_weight = "balanced"` en el discriminador del **primer umbral** (28.863 filas de
  `13-17` contra 140.473 del resto, un 17/83), con los **otros cuatro umbrales intactos**. Hipotesis: al
  subir el peso del lado minoritario, LightGBM dejaria de podar los cortes que separan al grupo pequeno y
  mejoraria la exactitud balanceada de ese umbral.
- **Result**:
  - **Exactitud balanceada del primer umbral: 0,7632 -> 0,8430, es decir +0,0797.** La hipotesis era
    correcta y el mecanismo funciona: el peso **si** mejora la frontera.
  - **Pero el blend de V20 cae a 0,5932020, un delta de -0,0166, con 0 de 5 folds** y IC95
    [-0,0178, -0,0153], con rescate 5.234, dano 6.574 y **neto -1.340**.
  - Las fronteras del blend bajan a **0,4117 y 0,5088**, frente a las 0,5075 y 0,5029 de V20.
- **Decision**: **`ORDINAL_WEIGHTS` se cierra con numero.** No se propone submission: la variante no mejora
  el OOF de V20 y el criterio exige mejora local antes de enviar.
- **Reason**: este es el **tercer negativo limpio y del mismo tipo** en el proyecto, y el mas instructivo
  porque aqui **la hipotesis era correcta**. Mejorar la frontera mas debil en **+0,0797** de exactitud
  balanceada **destruye el macro-F1 en -0,0166**. Luego **la exactitud de una frontera no es el cuello de
  botella**: el blend de V20 ya extraia de ese umbral lo que necesitaba, y forzarlo a separar mejor el
  grupo pequeno descalibra las cinco CDF a la vez, porque **las cinco comparten el mismo espacio de
  features y el mismo modelo**. Es la leccion de P1 repetida por tercera vez, ahora dentro de la familia
  ordinal y con la hipotesis acertada.
- **Consequences**: (1) **la direccion con recorrido no es reponderar fronteras**, porque la exactitud de
  una frontera y el macro-F1 global no estan alineadas; (2) **el primer umbral no era el cuello de botella
  en impacto** a pesar de ser el mas debil en metrica, y eso queda medido; (3) **la unica via que queda
  abierta dentro de la familia ordinal** es cambiar **los clasificadores**, no la reconstruccion (cerrada en
  D-054) ni los pesos (cerrada aqui), y el candidato natural es **mejorar el primer discriminador con un
  modelo distinto**, que es un cambio de mecanismo y no de hiperparametro; (4) **no se abre ninguna otra
  familia** mientras la ordinal tenga un challenger confirmado por leaderboard.
- **Evidence**: `experiments/v20_p2_pesos.json`, `reports/v20_p2_pesos.md`, `reports/v20_p2_pesos.csv`,
  `figures/v20_p2_pesos.png`.

## D-056 — V21: B1 no se puede mejorar cambiando de clasificador (NO_SIGNAL, y la razon es estructural)

- **Observation**: la familia ordinal seguia abierta **unicamente** en este eje: sustituir el discriminador
  de B1 (`13-17` frente a `>=18`) manteniendo las otras cuatro fronteras y la reconstruccion de V20
  **completamente congeladas**. V20 sigue en LB **0,62689** y OOF **0,6097782**.
- **Auditoria historica, hecha buscando codigo y no nombres.** Se busco si algun experimento habia
  sustituido especificamente B1, entrenado un modelo dedicado a `13-17` frente a `>=18`, usado
  boosting/SVM/arboles solo en esa frontera, o usado una representacion especifica unicamente para B1.
  **Ninguno existia.** El unico antecedente cercano es **E-028**, y **no es equivalente**: entrena un
  `LinearSVC` sobre las filas del **par adyacente** `("13-17", "18-24")`, es decir **dos clases y 57.539
  filas**, mientras que B1 es el umbral **anidado** sobre **todas las 169.336 filas** y un desbalance
  17/83. Ambos experimentos de esta fase son genuinamente nuevos.
- **V20 como control exacto**: reconstruido desde sus artefactos ya medidos, da **0,6097782 frente a
  0,6097782**, con SHA256 `69b574f737c8cf43...`. **Sin ese control ninguna variante seria
  interpretable**, y el script se aborta si no cuadra.
- **Result**:

  | Candidato | macro-F1 | Delta vs V20 | Folds | B1 balanceada | B1 AUC | Hueco train->OOF | Neto | Estado |
  |---|---|---|---|---|---|---|---|---|
  | **V20 (control exacto)** | **0,6097782** | +0,0000 | 0/5 | **0,7632** | 0,9284 | - | 0 | CONTROL |
  | LinearSVC C 0,2 | 0,5992098 | **-0,0106** | 0/5 | 0,6782 | 0,9178 | +0,002055 | -2.457 | `NO_SIGNAL` |
  | LinearSVC C 1,0 | 0,5992058 | **-0,0106** | 0/5 | 0,6784 | 0,9178 | +0,002102 | -2.457 | `NO_SIGNAL` |
  | boosting especializado B1 | 0,6097377 | **-0,0000** | 3/5 | 0,7606 | 0,9272 | +0,000249 | -6 | `NO_SIGNAL` |

- **Decision**: **`B1_SPECIALIST = NO_SIGNAL`.** Los tres candidatos fallan, y los dos mecanismos
  evaluados (control lineal y boosting especializado) **cierran la unica via abierta de la familia
  ordinal**, porque no hay un tercer mecanismo de B1 en el plan.
- **Reason**: la lectura es estructural y no estadistica, y responde a la pregunta que el propio plan
  planteaba en el paso 3: **el problema de B1 no es del estimador lineal ni de la falta de un boosting
  dedicado**. El `LinearSVC` pierde **0,085** de exactitud balanceada (0,7632 a 0,6782) y el boosting
  especializado queda **por debajo** del B1 de V20 (0,7606 contra 0,7632), con un hueco train a OOF de
  apenas +0,000249. Luego **el B1 del LightGBM multiclase de V20 ya esta a su nivel**, y la razon por la
  que un clasificador binario dedicado no lo mejora es la misma que en D-052: **las cinco CDF comparten
  las 279 features y el mismo modelo, y la primera sale de un ajuste multiclase que ya la resuelve.**
  Cambiar el clasificador de B1 no es una mejora, es una regresion.
- **Consequences**: (1) **la familia ordinal queda cerrada en sus tres subfamilias**: reconstruccion
  (D-054), pesos (D-055) y ahora **clasificadores de B1**; (2) **la respuesta a la pregunta central de V21
  es negativa**: mejorar especificamente `13-17` frente a `>=18` **no produce una mejora real en el objetivo
  final de seis clases**, y eso esta medido con el control exacto y con dos mecanismos distintos; (3) **no se
  propone submission**, porque ningun candidato mejora el OOF de V20 y el criterio exige mejora local
  antes de enviar; (4) **lo que queda abierto ya no es un eje de la familia ordinal**: si se quiere seguir,
  el siguiente paso seria volver a la comprension de **por que** el primer umbral tiene el menor impacto en
  metrica y el mayor en blend, que es una pregunta de analisis y no de modelo.
- **Evidence**: `experiments/v21_b1.json`, `reports/v21_b1.md`, `reports/v21_b1_comparativa.csv`,
  `reports/v21_b1_complementariedad.csv`, `reports/v21_b1_ordinal.csv`, `figures/v21_b1.png`.

## D-057 — V22: forense de V20 resuelto (Caso A) y B1_SPECIALIST cerrada

- **Observation**: el informe de V21 afirmaba `LightGBMClassifier(objective="multiclass")`, objetivo binario
  `clase <= 0` contra `clase >= 1`, y `scores = predict_proba[:, 1] = P(lado bajo)`. Carlos detecto que
  las tres no pueden aceptarse sin explicacion y pidio auditar el **codigo**, no el informe.
- **Result del forense, verificando y no asumiendo**:
  1. **Labels**: binarios, `(clase <= k-1).astype(int)` contra `clase >= k`. **Dos clases** por entrenamiento.
  2. **Target**: `num_class = 2`, porque el vector de etiquetas solo contiene 0 y 1.
  3. **`objective`**: HIERBA **declara** `"multiclass"` con `num_class = 6`, pero LightGBM construye un
     booster de **seis** clases y las columnas 2 a 5 de `predict_proba` son **cero exacto**. El objetivo
     *declarado* es multiclase y el *efectivo* es binario: ambas cosas son ciertas a la vez.
  4. **Columnas de `predict_proba`**: **seis**, no dos. La 0 es `P(lado alto)` y la 1 es `P(lado bajo)`.
  5. **Como se obtiene cada CDF**: **directamente**, `acumuladas[:, k-1] = P(Y <= k-1)`, porque el target
     binario es exactamente esa probabilidad. No es una derivacion.
  6. **Cinco binarios independientes**: **SI**, el bucle `for k in range(1, 6)` ajusta cinco instancias.
  7. **Multiclase transformado**: **NO**. No existe ningun modelo multiclase del que se deriven.
  8. **V20 es una cadena de cinco independientes**: **CASO A, confirmado**.
  9. **Las probabilidades son ACUMULADAS**, `P(Y <= k)`, no de clase.
  10. **La monotonia NO esta garantizada**: cada modelo es independiente y el **11,18 %** de las filas la
      viola. V20 la impone **despues**, con el maximo acumulado.
- **La especificacion declarada cuantificada, que es lo que la hace importante**: sobre las mismas 279
  features y los mismos cinco folds, la especificacion declarada da **0,6097782** y una binaria
  correctamente especificada da **0,6091232**. La correlacion entre las dos CDF es de **0,998** en las cinco
  fronteras, luego el *ranking* es el mismo, pero la **calibracion** difiere hasta **0,33**. Y la
  especificacion declarada es **0,00066 mejor**. **No es un bug que haya que arreglar**: es una version que
  funciona mejor, y se documenta y conserva.
- **Paso 5, el modelo compartido**: un **unico LightGBM multiclase** del que se derivan las cinco CDF **por
  suma**. Al ser una suma de probabilidades no negativas, **las cinco CDF son monotonicas siempre**: se
  miden **0 violaciones** de 169.336, frente a las 18.927 de V20. Blend **0,6096150**, delta **-0,000163**,
  3 de 5 folds, IC95 [-0,00078, +0,00045], neto **-13**, **4.451 cambios**. **`WRONG_OBJECTIVE`.**
- **Paso 3, el control lineal**: `LinearSVC` C 0,2 da **0,5992098** (-0,0106) con B1 **0,6782**; C 1,0 da
  **0,5992058** con B1 **0,6784**. **B1 empeora 0,085** y el final se hunde. **`NO_SIGNAL`: B1 no mejora.**
- **Paso 4, el boosting especializado**: LightGBM binario, pequeno y regularizado, da **0,6097377**
  (-0,0000) con B1 **0,7606**, **por debajo** del B1 de V20 (0,7632), neto **-6**. **`NO_SIGNAL`.**
- **Decision**: **`B1_SPECIALIST = NO_SIGNAL`, y con el se cierra la sustitucion de B1.** Y el modelo
  compartido **no mejora el macro-F1 final**, luego la respuesta a la tercera pregunta es también negativa.
- **Reason**: **las tres vias posibles de esta fase fallan por la misma razon estructural.** Las cinco CDF
  comparten las 279 features y el mismo modelo; la primera sale de un ajuste multiclase que ya la resuelve, y
  cambiarla por un lineal la empeora 0,085, por un binario dedicado la empeora 0,003, o compartirla con las
  otras cuatro resta 0,0002 al final. **B1 del LightGBM multiclase de V20 ya esta a su nivel**, y la
  leccion de D-052 se confirma por cuarta vez: **la exactitud de una frontera y el macro-F1 de seis clases
  no estan alineadas.**
- **Consequences**: (1) **V20 queda congelado sin tocar**, y su especificacion declarada se documenta en lugar
  de corregirse, porque es 0,00066 mejor; (2) **la familia ordinal queda cerrada en sus cuatro subfamilias**:
  reconstruccion (D-054), pesos (D-055), clasificadores de B1 (D-056) y **ahora representacion compartida**;
  (3) **no se propone submission**: ningun candidato mejora el OOF de V20; (4) **lo que queda abierto es de
  analisis, no de modelo**: por que el primer umbral tiene la menor metrica y el mayor impacto en el blend.
- **Evidence**: `experiments/v22_ordinal.json`, `experiments/v22_b1_specialist.json`,
  `reports/v22_ordinal.md`, `reports/v22_b1_specialist.md`, `reports/v22_b1_specialist.csv`.

## D-058 — V23: CORAL cerrado. Compartir pesos funciona y no alcanza (ORDINAL_SHARED_WEIGHTS = NO_SIGNAL)

- **Observation**: V22 cerro la familia ordinal en cuatro subfamilias y dejo una pregunta con
  evidencia de poder importar: si las cinco fronteras ordinales **compartieran** su vector de
  pesos y sus umbrales estuvieran **ordenados**, la monotonia dejaria de ser una correccion
  posterior para ser una garantia de construccion. V20 ajusta cinco binarios independientes y el
  **11,18 %** de las filas (18.927 de 169.336) viola el orden de sus CDF.
- **Result**: implementada la formulacion cumulative-link en `src/ordinal_coral.py` con `numpy` y
  `scipy`, sin ninguna libreria externa: un unico `w` y cinco umbrales
  `theta_k = gamma + suma_{j<k} softplus(delta_j)`, que crece **por construccion**. Verificada
  contra diferencias finitas (error relativo **3,15e-10**) y contra la densidad de clase
  calculada por otra via (diferencia **0,0**).
- **Un error de formulacion propio, encontrado por los tests**: la primera version evaluaba
  `log P(Y = y)` como `log sigmoid(z_y) + log(1 - sigmoid(z_{y-1}))`, que **factoriza** la
  diferencia de dos CDF en un producto de sigmoidos. Esa factorizacion **solo es valida si
  `s_y = 1`**; con ella el objetivo daba 678,80 frente a 895,41 de la referencia. La
  verosimilitud correcta es la **diferencia** `sigmoid(theta_y - eta) - sigmoid(theta_{y-1} - eta)`,
  evaluada con `logaddexp` y `log1p`. El gradiente tambien cambia: `d/d(eta) = 1 - s_y - s_{y-1}`
  y `d/d(theta_k) = ∓s_k (1 - s_k) / P(Y = y)`.
- **Un segundo error, de optimizacion**: penalizar solo `w` produce **separacion**. En un
  sintetico los umbrales llegaban a **12,07** con features estandarizadas en [-3, 3] y el
  macro-F1 de entrenamiento caia a **0,213**. La solucion esta en penalizar los **gaps** entre
  umbrales, que acota la separacion sin sesgar `gamma` (que si debe ser grande y negativo cuando
  la clase baja es rara, como `13-17`). La penalizacion existe en el modulo pero **no se activa**
  en AGE.
- **Pre-registro sin mirar AGE**: TEST B demuestra que a la escala del proyecto (135.000 filas,
  279 columnas, proporciones reales de clase) **no hay separacion**: 0 filas saturadas,
  `|eta|max = 5,78`, CDF entre 1,07e-04 y 0,99967. Luego `lambda_w = 1,0` y `lambda_umbral = 0,0`
  (los valores de libro) quedan congelados **antes** de ver el resultado, y el resultado es
  insensible a `lambda_w` (0,2760 con 1,0 frente a 0,2771 con 0,1 en el sintetico).
- **Control A, V20**: reproduce **0,6097782** exacto desde artefacto, con las **18.927**
  violaciones de D-057 confirmadas. Si no hubiera reproducido, la fase se detenía.
- **Control B, el que faltaba**: un delta de -0,126 contra V20 mezcla **dos** efectos, compartir
  pesos y perder no linealidad. Sin un segundo control la atribucion es imposible, asi que se
  anadio **cinco binarios lineales independientes** con el mismo estandarizado por fold.
  **Sin el, un control que no converge no es un control**: en la primera version
  `LogisticRegression` recibia las features crudas, avisaba de no converger y devolivia la
  iteracion parcial. Corregido.
- **Medicion**:

  | Modelo | macro-F1 | Delta | Folds + | monotonia | rescate | dano | neto |
  |---|---|---|---|---|---|---|---|
  | V20 (control A) | **0,6097782** | - | 3/5 | **18.927** | - | - | 0 |
  | cinco binarios lineales (control B) | 0,4690055 | -0,140773 | 0/5 | no medido | 10.936 | 34.279 | -23.343 |
  | **CORAL (peso compartido)** | **0,4837017** | **-0,126077** | **0/5** | **0** | 24.468 | 47.087 | -22.619 |

- **Descomposicion, que es el resultado util**: compartir pesos y ordenar umbrales vale
  **+0.014696** frente al control B. La hipotesis de la fase **es correcta**: compartir
  informacion entre fronteras aporta. Pero la no linealidad de LightGBM vale **-0.140773**,
  un orden de magnitud mayor, y ahi se decide la metrica.
- **La monotonia se resuelve de verdad y no compra nada**: CORAL tiene **0** violaciones frente a
  las **18.927** de V20, y aun asi pierde 0,1261. Es la **segunda** confirmacion independiente de
  D-057 (el modelo compartido tambien tenia 0 y perdia 0,0002): **la consistencia ordinal no era
  el cuello de botella**. Resolverlo gratis no compra nada.
- **Error intersection**: aciertos 103.331 de V20 frente a 80.712 de CORAL (delta -22.619, que
  coincide exactamente con el `neto` de complementariedad). Las **36 de 36** celdas de la matriz
  de confusion cambian, con distancia L1 **57.650**: no es que CORAL cambie algunas filas, es que
  rehace el mapa de errores entero.
- **Decision**: **`ORDINAL_SHARED_WEIGHTS = NO_SIGNAL` y `CORAL = CLOSED`.** Delta -0,126077, 0/5
  folds positivos, ningunasubmission. El bootstrap **no se calcula**, porque el brief lo reserva
  al caso en que se supera el gate de delta. No se abre ninguna familia nueva.
- **Reason**: las dos fuentes de senal que la hipotesis ofrecia estan medidas y las dos estan
  agotadas. Compartir pesos aporta +0,0147, y +0,0147 es menor que el ruido de resolucion del
  aparato (V19 a V20 fueron +0,0011 en LB). Ordenar umbrales vale 0 violaciones, y 0 violaciones
  ya era gratis en D-057. La conclusion es uncomfortable pero es la que sostienen los numeros:
  **la formulacion ordinal no es donde queda la senal**, ni compartida ni independiente.
- **Consequences**: (1) **la familia ordinal queda cerrada en sus cinco subfamilias**;
  (2) **no se propone submission**; (3) **no se abre HistGradientBoosting, RBF-SVM, feature
  selection supervisada, prototipos, monotonic ranking ni random projection**, segun el brief;
  (4) queda una sola pregunta de analisis sin responder, que ya no es de modelo.
- **Evidence**: `experiments/v23_00_tests.json`, `experiments/v23_01_coral.json`,
  `experiments/v23_02_auditoria.json`, `reports/v23_coral_analysis.md`,
  `reports/v23_coral_results.csv`, `reports/v23_coral_audit.md`, `reports/v23_00_tests_coral.md`,
  `reports/figures/v23_coral_monotonicidad.png`, `reports/figures/v23_coral_complementariedad.png`,
  `reports/figures/v23_coral_folds.png`, celdas 31.1 a 31.13 de `notebooks/99_entrega_consolidado.ipynb`.

## D-059 — V24: la causa del notebook incompleto era estructural, no editorial

- **Observation**: V23 reporto a la vez `experiments.csv = 130` y `notebook = 103 celdas`, y
  reconocio que el notebook consolidado llegaba hasta V17 mientras el champion es V20. Carlos pidio
  no experimentar hasta reconciliarlo.
- **Result, la causa raiz medida**: la fuente de verdad del notebook **no es un generador unico**.
  Es una **cadena de cinco scripts**, cada uno de una fase, aplicados en orden sobre el mismo
  archivo: `fase09_99_entrega.py` (crea la Parte I), `fase12_notebook_final.py` (V13),
  `phase28_notebook_final_v2.py` (V14 a V16), `v17_notebook.py` (anade la Parte III) y
  `v23_02_auditoria.py` (anade la seccion de V23). La cadena tiene **un eslabon por fase**, y
  **V18, V19, V20, V21 y V22 nunca tuvieron eslabon**. Ese es el motivo exacto de la deuda, y
  explica por que no se podia cerrar desde V23 sin dejar esas cinco fases fuera.
- **Por que 103 celdas no era un numero erroneo**: era correcto **para lo que contenia**, V12 a V17
  mas la seccion de V23 anadida al final. El numero no estaba mal, estaba incompleto. Ahora son
  **121 celdas** y **33 secciones**, con las doce versiones V12 a V23 presentes.
- **Renumeracion minima**: V23 estaba numerada 31 y V24 se construia como 32, luego el orden
  quedaria invertido. Se invierte: **V24 pasa a la seccion 31 y V23 a la 32**. Se renumera **una
  sola seccion existente**, que es lo que el brief pedia evitar.
- **El registry arrastra un defecto propio,Measuredo**: `experiments.csv` tenia **tres lineas de
  cabecera**, no una. Las dos espurias ocupaban el identificador `experiment_id` **dos veces**, luego
  el registry declaraba 130 experimentos y tenia **128**. El defecto es **preexistente**: aparece
  igual en al menos los ultimos seis commits, muy anterior a V23. Se corrige a una sola cabecera.
  **V24 no anade ninguna fila**, porque es documentacion y no un experimento, asi que el conteo
  solo puede bajar.
- **Dos discrepancias documentadas sin sobreescribir artefactos**, como pide el punto 7 del brief:
  1. **V20**: `v22_frontier_audit.json` registra **0,609615**, que es la salida del **modelo
     compartido** de D-057 y no la cadena de V20. El canonico es **0,6097782**, verificado por
     reproduccion exacta en cuatro artefactos.
  2. **V12**: `phase39_fusion_v12.json` registra la referencia **redondeada a cuatro decimales**
     (0,6004). El canonico pooled es **0,6005017**, que usan `v18_cierre.referencia_pooled` y
     `v22_frontier_audit.baselines.V12`.
  En ambos casos se muestra `ORIGINAL` junto a `CORRECTED` con su motivo y **el artefacto
  historico no se modifica**.
- **Auditoria de la fase: PASS.** `py_compile` rc 0, `ruff` con **0 errores de correccion** (el
  proyecto no tiene configuracion de ruff y sus avisos de estilo `ISC004` son sistematicos: los
  scripts ya commiteados tienen 12), **0 caracteres Unicode introducidos**, notebook con ids y
  titulos unicos, **0 artefactos faltantes**, **9 celdas ejecutadas con 0 errores** y coherencia
  numerica **7 de 7**.
- **Decision**: **V24 = reconciliacion documental, sin experimento.** Sin submissions, sin
  consultar Kaggle, sin leer el test ni el holdout, sin tocar V12 ni V20, y **sin abrir ninguna
  hipotesis**: el proximo experimento se decide despues y no automaticamente.
- **Reason**: la deuda no era de contenido sino de proceso. Documentar la cadena de generadores
  hace que la ausencia de un eslabon sea visible la proxima vez, en lugar de descubrirse por el
  recuento de celdas.
- **Consequences**: (1) el notebook representa las doce fases con hipotesis, metodo, control,
  corrected CV, decision y ruta de artefactos; (2) existe una **tabla maestra** y una **figura de
  evolucion** construidas desde artefactos, no desde memoria; (3) el registry es integro y su
  conteo real es **128**; (4) la deuda restante queda **declarada en el propio notebook**, sin
  esconderla.
- **Deuda que queda abierta y no se cierra aqui**: el notebook conserva **38 caracteres Unicode** de
  sesiones anteriores, porque el proyecto excluye `notebooks/` de su auditoria de idioma; y
  `notebooks/09_final_pipeline.ipynb` es un notebook antiguo de 10 celdas, obsoleto respecto a V20.
- **Evidence**: `experiments/v24_reconciliacion.json`, `reports/v24_reconciliacion.md`,
  `reports/v24_tabla_maestra.csv`, `reports/figures/v24_evolucion_corrected_cv.png`,
  `scripts/v24_notebook_reconciliacion.py`, seccion 31 de
  `notebooks/99_entrega_consolidado.ipynb`.

## D-060 — V25: la seleccion supervisada no elimina ruido, elimina senal (CLOSED)

- **Observation**: V22 cerro el frente de representaciones y dejo abierta la seleccion
  supervisada de features. V24 fue solo documental. Carlos pidio evaluar si las 279 features
  contienen ruido que perjudica a V20.
- **Pregunta que se responde, y no otra**: > Existe evidencia de que V20 contiene un subconjunto
  significativo de features ruidosas que, al eliminarse mediante seleccion supervisada realizada
  correctamente dentro de cada fold, mejore de forma reproducible el macro-F1?
- **La procedencia, verificada antes de entrenar** (`v25_00_inventario.py`, tres comprobaciones
  independientes): las **73 features label-informed** se construyen con `y_train` **del training
  fold**, se escriben **solo en `valid_idx`**, y usan **el mismo array de grupos de duplicado y los
  mismos folds** que V20, comparados array a array y no por nombre de funcion. **LEGAL.** El
  inventario cubre las 279 una a una: 78 miembros del ensamble, 35 densas, 87 estructuradas, 73
  label-informed y 6 NB-SVM, con 279 nombres unicos y una columna constante.
- **Por que hay que estandarizar antes de medir el mutual information**: `mutual_info_classif`
  estima con vecino mas cercano y **la metrica de distancia no es invariante a escala**. Las 73
  supervisadas llegan a magnitud absoluta **59.439** frente a las probabilidades de miembro, que
  viven en [0, 1]. Sin estandarizar, el estimador compara distancias en unas columnas y en otras
  no, y el ranking deja de medir senal. Se estandariza **dentro de cada training fold** y la
  validacion nunca participa.
- **Medicion**, con T0 y T1 usando **el mismo codigo de V20**, la misma `HIERBA` y la misma cadena
  de umbrales, luego la diferencia entre ellos **no puede ser otra cosa que las columnas
  seleccionadas**:

  | Modelo | corrected_cv | Delta | Folds + | rescue | damage | net | filas |
  |---|---|---|---|---|---|---|---|
  | T0 V20 sin seleccion | **0,6097782** | - | 0/5 | - | - | 0 | 0 |
  | T1 seleccion MI top 50% (K=139) | 0,6091492 | **-0,000629** | **1/5** | 1.042 | 1.143 | **-101** | 3.545 |

- **El hallazgo util es el por bloque**, no el delta: la seleccion **no elimina ruido, elimina
  senal**. Sobreviven con frecuencia **1,000** las **78 probabilidades del ensamble** y las **6 de
  NB-SVM**, con MI media de 0,215 y 0,208. **Nunca** sobreviven **74 de las 87 estructuradas** ni
  **30 de las 35 densas**, con MI media de 0,012. El mutual information alto **no es ruido: es la
  prediccion de los miembros que sostiene el modelo**, y el boosting la estaba usando.
- **La seleccion es estable**: 137 features en los cinco folds y 138 en ninguno. La mitad estable
  no es casual, y desde luego no es ruido.
- **La perdida es pequena y repartida**: 1.042 rescates contra 1.143 danos, neto **-101**, y solo
  3.545 filas cambian. El F1 por clase es practicamente identico: la caida se reparte y **ninguna
  clase se ve afectada de forma especifica**.
- **Decision**: **`SUPERVISED_FEATURE_SELECTION = CLOSED`.** Delta -0,000629, **1 de 5** folds
  positivos, gate no cumplido por las dos vias. Sin submission, sin bootstrap (el brief lo reserva
  al caso en que se supera el gate de delta), **sin probar otro K y sin probar otro selector**.
- **Reason**: la hipotesis era plausible y la evidencia la refuta con mecanismo, no con magnitud.
  La mitad de las variables que el mutual information considera mas informativas son
  **probabilidades de los miembros del ensamble**, y la metrica macro-F1 depende de ellas. Quitar
  ruido exigiria que el MI alto fuera ruido, y no lo es. El resultado es coherente con V15, que ya
  habia visto que podar empeora, pero por un **mecanismo distinto y mas claro**: V15 podaba por
  redundancia no supervisada, y V25 muestra que la seleccion **supervisada** tampoco mejora porque
  lo que ordena son las probabilidades del ensamble, no la calidad de las variables.
- **Consequences**: (1) **la familia de seleccion de features queda cerrada**, y con ella el tercer
  eje abierto de V22; (2) `experiments.csv` pasa a **130** con `V25-00-INVENTARIO` (auditoria, sin
  entrenar) y `V25-01-SELECCION-MI`; (3) **V20 sigue siendo el champion** con LB **0,62689** intacto;
  (4) no se genera submission y no se abre ninguna hipotesis nueva.
- **Cuatro bugs propios cazados por `ruff` y no por `ast`**, todos en `v25_01_seleccion.py`:
  el parametro de `reconstruye` se escribio `cumuladas` sin la "a" inicial mientras el cuerpo
  usaba `acumuladas`; una comprehension usaba `f` de la comprehension anterior, cuyo scope no
  sobrevive; la reconstruccion indexaba la ultima CDF con `n_clases - 1` en vez de
  `n_umbral - 1`, que se salia del array; y dos identificadores CJK se colaron en funciones
  inutiles. **`ast.parse` no detecta ninguno de los cuatro**: solo valida nombres en ejecucion. Eso
  es exactamente para lo que el brief exige `ruff`.
- **Evidence**: `experiments/v25_00_inventario.json`, `experiments/v25_01_seleccion.json`,
  `experiments/v25_02_auditoria.json`, `reports/v25_feature_inventory.csv`,
  `reports/v25_feature_selection.md`, `reports/v25_feature_selection.csv`,
  `reports/v25_feature_selection_frequency.csv`, `reports/v25_00_inventario.md`,
  `reports/v25_02_auditoria.md`, `reports/figures/v25_frecuencia_seleccion.png`,
  `reports/figures/v25_delta_por_fold.png`, seccion 33 de
  `notebooks/99_entrega_consolidado.ipynb`.

## D-061 — V26: la evidencia del ensemble es invariante a su escala (H02 CLOSED)

- **Observation**: V25 cerro la seleccion supervisada de features y demostro que la senal fuerte
  esta en las **84 columnas** de las 279 que son predicciones de modelos base. Carlos pidio buscar
  una forma distinta de explotar esa informacion sin anadir ni podar features.
- **La distincion que gobierna el analisis, y que el brief no contemplaba**: **V20 ya es un
  meta-modelo.** De las 279 features, 84 son predicciones de los 14 miembros del ensamble de V12 y
  de NB-SVM, y el meta-modelo es **LightGBM**: cinco binarios para la cadena de umbrales y uno para
  el arbol del blend. La pregunta de V26 no es construir un meta-modelo, sino si su
  **representacion de la evidencia** es explotable de otra forma.
- **Gap analysis** (`v26_00_gap_analysis.py`): 15 familias buscadas por **equivalencia funcional**
  en el codigo, no por coincidencia de nombre. **5 genuinamente abiertas**: H02 (reescalada afin),
  H03 (percentil), H11 (`monotone_constraints` nativas de LightGBM, **cero usos** en el proyecto),
  H12 (fiabilidad por miembro y clase) y H14 (reescalar las salidas del blend). Las otras 10 estan
  cerradas con numero.
- **Dos afirmaciones que hubo que MEDIR en vez de suponer**, y las dos son contrarias a la
  intuicion:
  1. **Un arbol NO es invariante a una transformacion monotona en esta implementacion de
     LightGBM.** Con `colsample_bytree=0,5` y `subsample=0,8`, un **z-score afin** cambia la
     prediccion **bit a bit**, con diferencia maxima **0,219** sobre las **135.468** filas de un
     fold. La invariancia teorica es exactamente el argumento que habria cerrado H02 sin gastar
     presupuesto, y es **falso**: la construccion de histogramas por cuantiles y el submuestreo lo
     rompen. Por eso se midio antes de decidir.
  2. **Un percentil no es una transformacion monotonia estricta**: colapsa valores a empates, luego
     pierde orden fino, y el arbol tambien cambia, con diferencia maxima **0,166**.
- **Seleccion**: `V26_SELECTED_TRACK = H02`, por novedad metodologica, evidencia de que V20 no lo
  captura y coste bajo. H03, H11, H12 y H14 quedan documentadas y **no se ejecutan**.
- **Medicion**, con V20 exacto como control y una sola transformacion como unico cambio:

  | Modelo | corrected_cv | Delta | Folds + | rescue | damage | net | filas |
  |---|---|---|---|---|---|---|---|
  | T0 V20 control | **0,6097782** | - | 0/5 | - | - | 0 | 0 |
  | T1 evidencia de miembros reescalada | 0,6097770 | **-0,000001** | **2/5** | 555 | 552 | **+3** | 1.809 |

  El intervalo bootstrap es **[-0,000375, +0,000397]**: incluye el cero con holgura.
- **La lectura que importa**: **la transformacion si cambia el modelo y no cambia la metrica**.
  1.809 filas cambian, el 1,07% del train, y el saldo es **+3 sobre 169.336**. La evidencia del
  ensemble es **invariante a su escala en todo lo que el macro-F1 mide**.
- **Decision**: **`H02 CLOSED`.** Delta -0,000001, 2 de 5 folds, frente a un gate de **+0,003** y
  **4/5** folds. Sin submission, sin bootstrap de promocion, y **sin segundo experimento**: el
  punto 46 del brief obliga a detenerse tras el primero.
- **Reason**: la hipotesis era plausible y la evidencia la refuta con claridad. Parte de la senal
  **esta** en las probabilidades de los miembros, pero su **escala no importa**: lo que el
  meta-modelo necesita de ellas es el **orden**, y el orden ya se conserva. Reescalar cambia la
  prediccion sin cambiar la informacion, y por eso mueve 1.809 filas para mover **+3**.
- **Consequences**: (1) **la escala de la evidencia del ensemble no es el cuello de botella**, y con
  ella quedan descartadas por evidencia reproducible las cuatro candidatas del punto 43;
  (2) **H03, H11, H12 y H14 siguen abiertas y no se ejecutan sin autorizacion**; (3) el cuello de
  botella restante **no esta en la representacion de las features, ni en la combinacion de las
  predicciones, ni en la calibracion de la evidencia**, segun el informe final; (4) **V20 sigue
  siendo el champion** con LB **0,62689** y hash sin cambio, y `experiments.csv` pasa a **132**.
- **Un bug propio en la propia auditoria, y es el tercero de la serie**: `_es_mencion` exigia que
  la linea contuviera el **identificador completo** de la comprobacion (`holdout_reuse`) en vez del
  **termino** (`holdout`), luego toda mencion pasaba sin filtrarse y el compliance se detectaba a si
  mismo. Ademas `y_test` matcheaba **dentro** de `already_tested`, un falso positivo de subcadena.
  Los dos se cazaron verificando el hallazgo uno a uno, no aceptando el recuento.
- **Evidence**: `experiments/v26_00_gap_analysis.json`, `experiments/v26_01_rank_representacion.json`,
  `experiments/v26_02_auditoria.json`, `reports/v26_hypothesis_gap_analysis.md`,
  `reports/v26_hypothesis_gap_analysis.csv`, `reports/v26_master_results.csv`,
  `reports/v26_hypothesis_registry.csv`, `reports/v26_reproducibility.md`,
  `reports/v26_compliance.md`, `reports/v26_final_research.md`, cinco figuras `v26_*`, seccion 34
  de `notebooks/99_entrega_consolidado.ipynb`.

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

## D-062 - V27: el cuello de botella de AGE no esta en el modelo, esta en los datos

- **Observation**: el brief autorizo continuar secuencialmente por las cuatro familias que V26 dejo abiertas (**H12** fiabilidad por modelo y clase, **H11** restricciones monotonas nativas, **H14** reescalar la salida del blend, **H03** percentil), con gate de +0,003 y 4/5 folds, y orden H12, H11, H14, H03.

- **H12 CLOSED, por resultado y por equivalencia**: la tabla fiabilidad(modelo, clase) se construye replicando un valor por par sobre todas las filas, luego es **constante por fila**, y una columna constante no puede generar splits. Delta **exactamente 0,0000000** y **0 filas cambian**. Se verifico que LightGBM con columnas constantes devuelve predicciones identicas bit a bit. Su unica variante razonable, fiabilidad por evidencia, es una interaccion entre miembros, cerrada como **H07**.

- **H11 CLOSED**: 0,6097782 de baseline, delta -0,000355, 2/5 folds, neto -55. Las restricciones se fijaron **antes** de entrenar, sobre **42 features** de los 7 modelos con probabilidad genuina por clase. `E019b_densas_svd` se excluyo pese a que sus componentes sumen 1, porque es un TruncatedSVD de features densas y un SVD no genera probabilidades de clase.

- **H14 CLOSED por equivalencia funcional, sin experimento**: reescalar la salida del blend es H02, H03, H01 o un cambio de combinador, y los cuatro estan cerrados con numero.

- **H03 MARGINAL**: delta 0,000272, 3/5 folds, rescue 564, damage 513, neto +51, 1,764 filas cambian, IC95 que **incluye el cero**. Unico resultado positivo de V26 y V27, y no finalista. No se genera submission y no se reintenta.

- **El hallazgo que explica los tres fracasos**: el meta-modelo de V20 divide por **7 de sus 279 features**, todas del bloque de evidencia de los miembros; **0 de 35 densas, 0 de 87 estructuradas, 0 de 73 label-informed y 0 de 6 NB-SVM**. Se descarto que sea efecto del submuestreo: sin `colsample_bytree` y sin `subsample` el numero no cambia. Eso explica de una sola vez por que la seleccion de V25 no podria mejorar, por que reescalar no movio la metrica y por que las restricciones monotonas no podian actuar.

- **Decision**: **V20 permanece como baseline absoluto** en 0,6097782 (LB 0.62689, hash sin cambio). Las 15 familias del gap analysis quedan cerradas o marginales, ninguna abierta. **No se genera ninguna submission.** La investigacion de esta via se cierra: no queda ninguna hipotesis defendible sin medir, y lo unico abierto es senal nueva, que las reglas de AGE impiden obtener de las fuentes obvias.

- **Consequences**: el cuello de botella restante esta en la **limitacion de los datos**: el 38,3 % de filas que V19 y V20 fallan juntas es la unica masa grande y medida, y esta agotada con la senal actual. Toda intervencion futura sobre las 279 features actua sobre la parte del sistema que el meta-modelo **ya no lee**, salvo que antes se demuestre que aporta senal que el bloque de evidencia de los miembros no aporta.

---

## D-063 - V28: ni la varianza de la estimacion ni la descomposicion de la decision son el cuello de botella

- **Observation**: V25 a V27 habian eliminado el podado de features, la escala de la evidencia, las restricciones monotonas, los percentiles, la fiabilidad constante y las reformulaciones ordinales. V28 se pregunto por mecanismos **estructuralmente distintos** para combinar, estabilizar o descomponer la evidencia: bagging del meta-modelo, geometria del consenso, descomposicion pairwise, fusion de evidencia, incertidumbre explicita y subespacio aleatorio.

- **El gap analysis fue el trabajo mas rentable.** Cuatro de las seis tecnicas se cerraron **sin gastar un experimento**: **B** porque `advanced_04_incertidumbre.py` implementa un superconjunto y esta medido (267 features 0,5897 frente a 318 features 0,5898, delta +0,0001); **D** porque un pool de evidencia es un combinador, y H04 mas H06 estan cerrados con numero, con V20 ya siendo el segundo nivel; **E** por la regla critica del brief, que prohibe duplicar B; y **F** porque `HIERBA` ya tiene `colsample_bytree=0,5`, que es exactamente aleatorizar la exposicion de cada arbol a un subespacio manteniendo todas las columnas.

- **A CLOSED**: bagging de N=5 miembros con bootstrap de Poisson(1), agregado por promedio de probabilidades. Delta **-0,0003120**, 1/5 folds, neto -46, IC95 que incluye el cero. **La varianza existia**: los cinco miembros acordaban solo en el 0.9189 de las decisiones. Y promediarla empeoro. Ademas 1.500 estimadores en vez de 300 no dieron ganancia, lo que descarta la explicacion por capacidad.

- **C CLOSED**: cinco binarios **disjuntos**, uno por par adyacente, contra los cinco **anidados** de `clase <= k-1` de V20. Delta **-0,0675530**, 0/5 folds, neto -10,996, IC95 que **excluye el cero**. Los pares si reducen los errores de su objetivo local, de 39,556 a 32,826, y si son complementarios en la masa dificil (7,788 de las 61,949 filas que V19 y V20 fallan juntas). El daño esta en las clases que no son frontera, donde un clasificador que solo ve dos clases no puede colocar una fila lejos de su par.

- **Decision**: **V20 permanece como baseline absoluto** en 0,6097782 (LB 0.62689). Las seis tecnicas quedan respondidas: cuatro por equivalencia funcional y dos por experimento, ambas negativas. **0 finalistas, 0 submissions, ninguna linea de V20 tocada.** El cuello de botella restante es la **limitacion de los datos**, y V28 no ha reducido la masa que ningun modelo corrige.

- **Consequences**: queda agotado el espacio de mecanismos accesible **sin senal nueva**: representacion, combinacion, calibracion, estructura ordinal, geometria del consenso, varianza de estimacion y descomposicion de la decision tienen todas evidencia negativa reproducible. Cualquier linea futura tiene que demostrar primero que aporta senal que el bloque de evidencia de los miembros no aporta; otra forma de combinar lo que ya existe esta cerrada por V25 a V28.

- **Una correccion que se declara**: el IC95 del bootstrap de **V26-01** quedo a `null` en su artefacto y no es recuperable sin reentrenar. Se marco `no persistido` en la tabla historica en vez de citar una cifra sin fuente. No afecta a ninguna decision: H02 se cerro por delta y folds, y el bootstrap era confirmatorio.

---

## D-064 - V29: el cuello era la capacidad individual, no la diversidad; y el margen es real pero insuficiente

- **Observation**: V25 a V28 habian cerrado representacion, escala, monotonia, percentiles, fiabilidad, ordinales, bagging del meta-modelo y descomposicion pairwise. Quedaban dos ejes sin tocar: la **capacidad de un miembro individual** y la **composicion del ensemble**. La hipotesis central de V29 era que V20 estaba limitado por la falta de diversidad util entre los errores de sus miembros, y no por su capacidad.

- **El paso mas rentable no fue entrenar, fue leer**: el historial tenia 51 matrices OOF reutilizables en 17 familias, y **ninguna se reentreno**. Entre ellas aparecio un duplicado por argmax (`v19_oof_arbol_5folds` y `v22_seis_compartida` toman las MISMAS decisiones), y el cuarto de cadena de V20 hubo que reconstruirse desde `v20_f3_oof_cumuladas.npy` porque usar ese artefacto habria comparado a V20 consigo mismo y habria falseado la medicion de diversidad.

- **La diversidad historica queda cerrada con numero**: de los 47 modelos fuera del circulo de V20, **0** tienen rescate unico mayor que dano unico frente a V20, y con el pool historico los tres metodos de ensemble eran negativos (-0.000682, -0.000361, -0.000608, todas 2 de 5 folds). Ademas, el mayor rescate bruto lo tenia un modelo con CV 0,06, que resucita filas al azar: por eso la metrica de decision son fracciones normalizadas y no numeros absolutos.

- **La capacidad individual SI era un cuello de botella**: el LightGBM de Optuna alcanza 0,6121698 frente a los 0,6097782 de V20, 0,002392 de mejora con la **misma** matriz de 279 features, los mismos folds y la misma estructura anidada de holdout interior. Es **el primer modelo individual que supera a V20 en todo el proyecto**, y lo hace sin tocar la representacion. Ademas es el **unico** con rescate neto positivo de todo el inventario: corrige el 8,6 % de los errores de V20 y rompe el 5,4 % de sus aciertos.

- **Y el ensemble mejora con el, y solo con el**: al incorporar el modelo nuevo al pool, los tres metodos pasan de negativo a positivo, el mejor llega a 0,6125960 con delta +0.002818 y 5/5 folds positivos. El control H04 se invierte: el modelo nuevo recibe entre el 42,7 % y el 61,5 % del peso, V20 baja al 8,2 %-24,7 %, y las familias con peso pasan de 3 a 4. **La composicion deja de ser H04 de nuevo.**

- **Pero no llega al gate, y por dos motivos independientes**: el delta exigido es +0,003 y el mejor es +0.002818, un 94% del gate; y con la calibracion `LB = +0,00223 + 1,02825 x CV` ese margen valdria **+0,0014 en el leaderboard**, frente a una oscilacion de **0,0034** entre dos modelos vecinos del propio proyecto. Seria indistinguible de haber cambiado de modelo.

- **Decision**: **V20 permanece como baseline absoluto** en 0,6097782 (LB 0.62689). **0 finalistas, 0 submissions, ninguna linea de V20 tocada.** P13 no se ejecuta porque no hay finalistas. La hipotesis central queda contestada con matiz: **V20 estaba limitado parcialmente por la capacidad de sus miembros, no por la diversidad complementaria**, y una busqueda controlada de Optuna **si** encuentra un modelo y un ensemble mejores, pero no lo bastante mejores para ser candidatos.

- **Consequences y relacion con D-062**: D-062 (V27) concluyo que el cuello de botella estaba en los datos. V29 **no lo derriba, lo afina**. Lo que confirma: la masa de 61.949 filas que ningun modelo corrige sigue intacta, y ninguna combinacion de 16 miembros la toca. Lo que corrige: **quedaba margen de capacidad individual, y es real**; no era cierto que el eje estuviera agotado. Lo que no cambia: ese margen es un orden de magnitud inferior al necesario para producir un candidato defendible. Un eje agotado no es un eje donde buscar un decimal mas, es un eje donde la busqueda empioja. La hipotesis de diversidad de modelos **historicos** queda cerrada con numero: 0 de 47.

- **Desviaciones de presupuesto que se declaran**: (a) Optuna se ejecuto con **10 trials por fold** en vez de los 30-50 que sugiere el brief, porque medido en esta maquina el fold 0 no termino en 55 minutos con 30 trials, y el proceso fue terminado por el sistema a los 45 minutos con el techo de iteraciones ya reducido; (b) la pista de Optuna sobre MLP **no completo**, y su resultado parcial era muy inferior a V20; (c) se corrigio un defecto del propio criterio de pool, que excluia al mejor modelo individual por ser poco diverso, lo cual habria impedido responder a la pregunta de V29; (d) dos errores propios corregidos antes de commit: `rescate_unico` y `dano_unico` estaban intercambiados en la primera medicion de diversidad, y el nombre del artefacto de la cadena de V20 no coincidia con el que se guardaba.

---

## D-065 - V30: el margen de V29 no se consolida, y la transferencia queda pendiente de una credencial

- **Observation**: V29 demostro que el cuello de botella era la **capacidad individual** del meta-modelo, no la diversidad: un LightGBM sobre la misma matriz de 279 features alcanzo 0,6121698 frente a los 0,6097782 de V20, y el ensemble llego a 0,6125960 con 5 de 5 folds y neto +480, un 94 % del gate. V30 se pregunto si ese margen se consolidaba.

- **El mecanismo de V29, identificado**: `HIERBA` de V20 esta **sub-parametrizado**. Los cinco mejores trials de V29 usan `num_leaves` de 43 a 87 frente a los **8** de V20, y `min_child_samples` de 22 a 94 frente a **200**, con `reg_lambda` casi nulo frente a **1,0**. No era falta de senal: era un meta-modelo por debajo de su propia capacidad.

- **El refinamiento local dio 0,000069** frente al LightGBM de V29, con **3/5 folds** a su favor. Veredicto: **EQUIVALE_A_V29**. Con 200 trials sembrados en el mejor punto de V29, la region estaba ya encontrada: no queda margen por extraer de la capacidad del meta-arbol.

- **El candidato final** es `E1 Optuna macro-F1`: 0,6130051 corrected_cv, delta 0,003227 frente a V20, **5/5 folds** positivos, rescate 3,768 contra dano 3,238, neto **+530**. El IC95 del bootstrap es [0.00229, 0.004219] y **no incluye** el cero.

- **Decision**: el gate historico exige delta >= +0,003 **y** 4 de 5 folds. El candidato cumple la segunda al 100 % y la primera al 108%: **no es finalista formal**, pero se registra como **STRONG_MARGINAL** (5/5 folds con delta entre +0,002 y +0,003), que es la bandera que el propio brief pide separar del gate. **V20 permanece como baseline absoluto.**

- **La autorizacion de submission se ejecuto pero no se pudo completar**: el fichero `age_submission_v30_final_candidate.csv` esta construido, con **10/10 checks** y hash `f543a715a1881c6128a860301b84a7273c3db8f8a709545be1e6267822e713be`, pero `~/.kaggle/kaggle.json` solo contiene la clave `token`, que es el formato antiguo, y el CLI 2.2.4 exige `username` y `key` u OAuth. **STOP de infraestructura, no resultado experimental.** No se pide ni se copia ningun secreto por esa via: la credencial la establece el usuario en su terminal. Y como no hubo score, **no se infiere nada sobre la transferencia**, en un sentido ni en el contrario.

- **Consequences**: queda agotada tambien la capacidad del meta-arbol. Entre V25 y V30, los ejes cerrados son: representacion, escala, fiabilidad, monotonia, percentiles, estructura ordinal, bagging del meta-modelo, descomposicion pairwise, geometria del consenso, diversidad de modelos historicos **y capacidad individual**. El unico margen que queda es un orden de magnitud menor que el gate, y la unica via con recorrido sigue siendo senal nueva en los datos, que las reglas de AGE cierran en las fuentes obvias.

- **Una correccion del propio metodo que se declara**: el pool historico de V29 excluia al mejor modelo individual por ser poco diverso, lo que habria impedido responder a la pregunta de V29. Se corrigio antes de commitear, y el criterio corregido esta en `scripts/v29_05_pool.py`. Tambien se detectaron dos procesos de Optuna escribiendo el mismo artefacto, y el que sobrevive es el de 30 trials por fold, que es el del rango que recomienda el brief y ademas el mejor de los dos.

---

## D-066 - V31+: el baseline correcto era 0,6134471, y la cifra publicada no era reproducible

- **Observation**: la primera accion de V31+ fue reproducir V30 desde sus artefactos, y **no
  reprodujo**. El OOF recomputado dio 0,6134471 frente a los 0,6130051 que V30 habia publicado, con
  una desviacion de 0,000442. Es exactamente la desviacion que produce distinguir dos drivers
  distintos, y por eso lo primero fue auditar en vez de asumir.

- **La causa, y no era un bug de V30**: el CV publicado se compuso con los **pesos por fold** que
  Optuna optimizo en cada fold por separado, y **esos pesos nunca se persistieron**: el artefacto
  `v30_02_ensemble.json` solo guarda la media entre folds. La composicion del OOF con pesos por
  fold y la composicion con pesos medios son dos OOF distintos, y la submission uso **los medios**.

- **El leaderboard resuelve cual de las dos es la real**:

  | Configuracion | corrected_cv | LB predicho | LB real | error |
  |---|---|---|---|---|
  | pesos por fold (**no persistida**) | 0,6130051 | 0,63255 | 0,63302 | +0,00047 |
  | pesos medios (**lo que se envio**) | 0,6134471 | 0,63301 | 0,63302 | **+0,00001** |

  Con la calibracion interna `LB = +0,00223 + 1,02825 x CV`, la cifra operativa acierta el
  leaderboard hasta la quinta decimal y la publicada se queda tres veces mas lejos. El hash del
  fichero enviado confirma que la submission uso los pesos medios.

- **Decision**: el baseline operativo de V31+ y de adelante es **0,6134471**. Se mantiene
  publicada la cifra de V30 por trazabilidad, marcada como **NO REPRODUCIBLE** en
  `reports/v31_candidate_comparison.csv`, en vez de sobrescribirla en silencio. La clase de
  reproducibilidad del baseline pasa de R1 a **R3**.

- **Consequences**: (a) la comparacion de V31+ se mide contra la cifra que **reproduce y que se
  envio**, no contra la que estaba impresa, y eso cambia el estado de V30 de 0,6130051 a
  0,6134471; (b) el error de transferencia de V30 baja de +0,00047 a **+0,00001**, lo que
  convierte a V30 en un **caso A limpio** (CV robusto, segunda particion robusta y LB que
  mejora) y valida la calibracion interna como instrumento fiable para fijar los cortes de esta
  fase; (c) deja una leccion de proceso: una cifra puede estar en el informe, en el artefacto y
  en el notebook y aun asi **no ser reconstruible**, porque lo que falta no es el numero sino el
  dato que lo produce.

---

## D-067 - V31+: las familias linguisticas son senal real y no aportan a V30

- **Observation**: V31+ se abrio buscando senal nueva en representacion: URL, composicion
  lexica, ortografia, discurso, posicion y forma de palabra, con el protocolo corregido
  (`StratifiedGroupKFold` sobre `dup_clusters`, folds identicos a V20, OOF estrictamente
  out-of-fold) y con un banco donde **se varia la representacion y se fijan modelo, folds,
  semilla y vectorizador**, para separar "falta senal" de "falta capacidad".

- **El Track A se reencuadro por los datos**: la prevalencia de URL en train es **monotona en la
  edad**, del 33,4 % en 13-17 al 50,0 % en 65-xx, una distribucion real. Pero el **99 % de las
  URLs son `t.co`** con un token de ruta opaco de 10 caracteres, y hay **0 status URLs**. La
  taxonomia de dominios que propone el brief es casi vacia en este dataset: no hay semantica de
  dominio que extraer. Ademas enmascarar la URL **cuesta** 0,006969, asi que la URL no es ruido, y
  el canal URL explico solo aporta 0,001894. La URL ya se estaba leyendo.

- **La senal linguistica es real y consistente**: el canal `ortografia+composicion+discurso` bate
  al texto solo en **+0,006820 con 5 de 5 folds**. Dos familias salieron negativas y se cierran:
  canal por posicion (-0,002303, 0/5 folds) y forma de palabra (-0,002984, 0/5 folds).

- **Y aun asi no aporta a V30**: como cuarto miembro de la mezcla, **+0,0000642** sobre el control
  reoptimizado; como 48 features en el meta-espacio de 279, **+0,0002948** sobre el control
  barajado. Ambos muy por debajo del gate de +0,0015. El patron es el mismo que V29 identifico
  con `HIERBA`: no le falta senal al canal, le falta sitio donde V30 no la exprese ya.

- **Los dos controles son lo que hace interpretable el resultado**. El brazo "V30 con 3 pesos
  reoptimizados" da +0,0003477, o sea que **la mayor parte del aparente +0,0004119 de la
  integracion es el efecto de tener pesos optimizados en vez de fijos, no el canal**. Y el
  control de barajado, que baraja **solo** el canal, deja a V30 intacto y da 0,6134855: confirma
  que la senal viaja en las filas correctas, y a la vez que su magnitud es pequena.

- **Dos errores propios, corregidos antes de commit y declarados**: (a) `v31_04` leia
  `neto <= 0` como "redundante", que es una tautologia, porque un clasificador mas debil siempre
  tiene rescate neto negativo; el criterio correcto es la **tasa de error compartido**, que si da
  diversidad real (acuerdo 0,752, 11.761 filas que el canal acierta y V30 no). (b) El primer
  control de barajado barajaba los **cuatro** miembros en vez de solo el canal, lo que lo
  reducia a 0,1639 y producia un veredicto de "aporte real" sin fundamento. La version corregida
  baraja unicamente el miembro nuevo.

- **Track J explica por que no queda margen de representacion**: el **90 % de las diez
  confusiones principales de V30 son entre bandas adyacentes**, no saltos. Hay 9.780 errores de
  alta confianza (15,0 % de los errores), pero se reparten en un problema de **resolucion de
  escala**, no de informacion ausente. Y la familia que ataca eso, los modelos ordinales, ya esta
  cerrada por la historia del proyecto (H10, CORAL, D-058), con su propia evidencia.

- **Dos lineas en cuarentena, no cerradas por resultado sino por ambiguedad**: (a) la
  recuperacion del contenido de una URL, porque **las reglas de la compilacion no estan
  disponibles en este entorno** (sin copia local, la API de Kaggle no autentica, y el unico
  fichero de reglas del arbol pertenece a otra compilacion y esta mal titulado). No se hizo
  **ninguna** llamada de red. (b) el POS tagger como predictor, por ser un modelo externo
  preentrenado y las reglas no son explicitas; se descarta en vez de asumir que es legal.

- **Decision**: **0 finalistas, 0 submissions, ninguna linea de V30 tocada.** V30 permanece como
  baseline absoluto con corrected_cv 0,6134471 y LB 0,63302. Las familias se cierran asi:
  URL CLOSED, verbos CLOSED, posicion CLOSED, forma de palabra CLOSED, representaciones sparse
  nuevas CLOSED, integracion en el meta-espacio CLOSED, capacidad de modelos CLOSED;
  ortografia, composicion y discurso **MARGINAL** (senal real, ya absorbida); mineria guiada por
  errores **DIAGNOSTICO**; retrieval de contenido y POS **QUARANTINED**.

- **Consequences y relacion con D-064 y D-065**: D-064 (V29) concluyo que la capacidad si era un
  cuello, y D-065 (V30) que el margen de V29 no se consolidaba. V31+ **afina el diagnostico una
  vez mas**: el cuello no es la capacidad ni la senal lingüistica expresable con representacion
  sparse explicita. Lo que queda es resolucion entre bandas vecinas, que es un eje distinto y ya
  explorado. La formulacion correcta no es que V30 sea el limite teorico, sino que **las familias
  evaluadas no produjeron evidencia suficiente**, y que la unica via con recorrido real
  (**senal nueva en los datos**) sigue cerrada por las reglas de AGE en sus fuentes obvias.

---

## D-068 - G14: no existe informacion de entrada legal sin utilizar, y el `id` no es rescatable

- **Observation**: cerradas las vias de correccion en score-space (P33, P34, G12), la unica linea
  con margen estructural que quedaba era G14: comprobar si existia un campo de entrada legal,
  presente en prediccion y que ningun modelo hubiera consumido. La duda concreta estaba en el
  campo `id`: si fuera un *snowflake* de Twitter, codificaria la fecha de publicacion, que es
  informacion real, legal y disponible en test.

- **El `id` no es un snowflake, y la prueba es geometrica antes que estadistica**: va de **1 a
  169.336** en train y de **1 a 112.892** en test, monotono y sin huecos, con **6 digitos** en el
  maximo. Un id de tweet tiene 18-19. Y hay un dato que cierra el caso por si solo: **los 112.892
  ids del test aparecen tambien en train**, es decir el 100 %, para filas distintas. El `id` 1 es
  un tweet distinto en cada split. Un campo cuyo valor identifica a dos filas con salidas
  distintas no puede ser feature: no aporta informacion sobre la fila.

- **La correlacion con la clase es -0,00084.** Nada, como corresponde a un indice de fila.

- **Auditoria del resto del esquema**: el ZIP de la competicion contiene exactamente tres ficheros
  (`age_train.csv`, `age_test.csv`, `age_submission.csv`), sin campos ocultos. Los tres campos son
  `text` (en uso en las nueve familias de modelo), `age_range` (target, prohibido como entrada) e
  `id` (administrativo). La matriz campo-por-modelo da: `text` USED o DERIVED en 9 de 9, `id`
  NOT_USED en 9 de 9, `age_range` PROHIBITED en 9 de 9.

- **Traza de `text`**: ninguna dimension queda en el limbo. Las que el TF-IDF no captura (case,
  puntuacion, elongaciones, posicion) fueron cubiertas por los diez canales de V31-03 y su
  integracion se cerro sin aporte en V31-06. Eso es el **caso C** del brief, informacion ya
  contenida o ya probada, no un canal nuevo.

- **Decision**: **G14 = NOT_ACTIONABLE**, y se declara por que no es `CLOSED`: `CLOSED` significa
  que habia un canal y estaba cubierto; `NOT_ACTIONABLE` significa que no hay canal, ni cubierto ni
  descubierto. **No se intenta rescatar el `id`** con modulo, hash, posicion, patrones de digitos,
  buckets ni sinonimos, como el brief prohibe expresamente: eso no seria encontrar informacion,
  seria ajustar una correlacion espuria sobre 169.336 filas.

- **Consequences**: la fase de investigacion queda con sus seis lineas cerradas (correccion de
  score global, local, pendiente, canal linguistico, capacidad del meta-modelo, diversidad de
  ensembles) y con la **entrada enumerada por completo**. La conclusion que se puede sostener es
  que **no existe actualmente informacion de entrada legal, observable en prediccion y no
  utilizada por los modelos, que pueda abrir una nueva linea experimental**. No es lo mismo que
  decir que V30 sea optimo: es decir que las vias que el protocolo permite estan recorridas. Y
  queda la cifra que mide el techo: de los 65.403 errores de V30, **27.007 no los corrige ninguna
  de las 26 fuentes disponibles**, y G14 anade que tampoco hay un campo sin explotar que pudiera
  corregirlos.

---

## D-069 - Cierre de la investigacion de AGE: `AGE_RESEARCH_CLOSED`

- **Observation**: G14 cerro el **input gap** (no hay campos de entrada legal sin usar). Quedaba
  abierta la pregunta distinta del **mechanism gap**: si existe alguna forma de APRENDER de los
  mismos inputs que no se haya probado. La auditoria se hizo sobre 39 mecanismos buscados en el
  codigo **por operacion**, no por nombre de libreria.

- **La medicion que fundamenta la decision** refina la particion A/B del gap analysis y anade una
  distincion que faltaba:

  | Grupo | Filas | % |
  |---|---|---|
  | A: nadie corrige | 27.430 | 41,9 % |
  | B: alguien corrige | 37.973 | 58,1 % |
  | - B1 rescate **unico** | 8.358 | 12,8 % |
  | - B2 rescate **compartido** | 30.038 | 45,9 % |

  Mediana de fuentes por error rescatado: **5**, con un maximo de 25. **B2 domina**, y eso importa
  porque un rescate compartido significa que la informacion esta en varias fuentes a la vez y el
  problema es de integracion; solo un rescate unico apuntaria a un problema de mecanismo.

- **Los rescates exclusivos no tienen estructura comun**, y se midio en vez de suponerse: 8.029
  filas repartidas entre **23 fuentes distintas**, con la mas frecuente concentrando solo el
  **20,5 %**. Longitud media 81,6 frente a 84,2 de todos los errores; confianza de V30 0,429 frente
  a 0,442; fraccion con URL 41,5 % frente a 41,3 %. **Indistinguibles en toda dimension medida.**
  Veredicto: heterogeneidad, que es la firma de ruido de modelo y no de un canal sin explotar.

- **De 39 mecanismos, 23 estan usados y 9 cerrados.** Los cinco que no aparecen en el codigo se
  cierran con la medicion delante, no por inercia: **focal loss**, **hard example mining** y
  **label smoothing** apuntan a los ejemplos dificiles y a la frontera, que es la poblacion A
  (27.430 filas sin ninguna fuente que las corrija) y donde P34 midio que el ruido aleatorio gana a
  la senal real. **ranking objective** no mueve la metrica evaluada. **bootstrap de grupos** es
  reduccion de varianza con otra unidad de remuestreo, y el bagging del meta-modelo ya se cerro en
  -0,000312.

- **Las cinco vias de integracion del score estan cerradas con control de atribucion**:

  | Via | Resultado |
  |---|---|
  | feature anadida al meta-espacio (V31-06) | +0,0002948 sobre el control barajado |
  | miembro de la mezcla (V31-05) | +0,0000642 sobre el reoptimizado |
  | residuo global (P33) | -0,0002152; el ruido aleatorio gana |
  | residuo local por pares (P34) | -0,0002039; cero rescates de alta confianza |
  | residuo con pendiente (G12) | +0,0003018 sin superar a la constante, y baja el AUC |

  En las cinco el control gana o empata. **No es que la integracion se intentara poco: se intento
  cinco veces con la misma conclusion.**

- **Decision**: **`AGE_RESEARCH_CLOSED`**. V30 permanece como mejor resultado validado: 0,6134471
  de corrected_cv, 0,6143347 en la particion alterna y **0,63302** en el leaderboard.

- **Dos correcciones que la auditoria encontro, y se declaran**: (a) la particion del gap analysis
  reportaba A = 27.007 y el valor correcto es **27.430**; el defecto estaba en un enmascarado
  enrevesado para excluir a los miembros de V30 de su propio rescate. (b) La definicion de "rescate
  exclusivo" del inventario previo daba 122; con la de aqui da **8.029**. Los dos numeros miden
  cosas distintas y se declara cual se usa.

- **Consequences**: la conclusion es **sobre el espacio auditado**, y el espacio esta enumerado: 39
  mecanismos, 6 familias de representacion, 10 familias de modelo, la entrada completa y las cinco
  vias de integracion. Lo que **no** se afirma: que V30 sea optimo; que no exista ninguna mejora
  posible; que no exista informacion linguistica (P33 midio AUC 0,5411 en el residuo). Lo que si se
  afirma, con la medicion a la vista, es que **no quedan actualmente mecanismos de aprendizaje
  metodologicamente defendibles y no recorridos dentro de las restricciones del proyecto**. Para que
  algo nuevo funcionara haria falta informacion que no esta en los datos: metadatos de cuenta
  (prohibidos por ser una tarea de texto), contenido de URLs (no hay ninguna de status) o modelos
  semanticos (prohibidos por las reglas).

---

## D-070 - Revision bibliografica: `NO_NEW_METHOD`

- **Observation**: cerrada la linea experimental, la pregunta que quedaba era si existe literatura
  que revele un mecanismo no probado. Se reviso el estado del arte de author profiling, ordinal
  classification y edad a partir de texto (ACL Anthology, EMNLP, COLING, IEEE TKDE, PRL,
  Information Fusion, CLEO/PAN, arXiv 2020-2026).

- **El hallazgo central es un desajuste de metrica, y decide la revision**:

  ```text
  macro-F1 es NOMINAL: trata todos los errores igual.
  La literatura ordinal optimiza metricas DISTANCIA-CONSCIENTES: MAE, QWK, RPS, 1-off.
  ```

  Dos fuentes convergentes lo dicen de forma explicita: **Vargas et al., IEEE TKDE 2026** (12
  datasets ordinales, 20 configuraciones) encuentra que el etiquetado suave mejora la
  generalizacion ordinal pero **la cross-entropy categorica rinde mejor en promedio en metricas
  nominales**; y **Kasai et al., arXiv 2405.11775** (texto, comparativa explicita de perdidas)
  afirma que **hay un trade-off: la mejora ordinal se paga con metricas nominales**.

- **La validacion contra el repositorio encontro huecos de ejecucion reales**, y se declaran en
  vez de taparlos: AGE **nunca corrio** un target suave ni `label smoothing` (**0 scripts, 0
  menciones**), ni EMD/Wasserstein (**0 scripts**), ni una proyeccion de features fuera de
  direcciones de topico (**0 scripts**). El proyecto entero tiene **solo dos perdidas custom**:
  margen (V18-P3) y destilacion (V18-P5). Es decir, la `label smoothing` que la auditoria dijo
  "cerrada" lo estaba **por argumento, no por medicion**.

- **Dos recuentos iniciales fueron falsos positivos de la propia busqueda**, y se declaran:
  `complement` matcheaba "complementario" en el texto en espanol (391 veces) y `sord` matcheaba
  `tensordot` (15 veces).

- **Los cinco candidatos, y por que ninguno justifica reabrir**:

  | ID | Mecanismo | Veredicto |
  |---|---|---|
  | C1 | perdidas distancia-conscientes (SORD, triangular/beta, OLL/MLL) | unico con hueco real, pero la evidencia empuja contra macro-F1 y es un **superconjunto de label smoothing** |
  | C2 | ruido ordinal explicito (Ord-NLL) | el eje mas no auditado, y **colapsa a C1 en expectativa**: su propio paper demuestra la equivalencia |
  | C3 | regularizacion unimodal (ORCU) | su ganancia medida es de **calibracion**; no supera a su componente de soft labels |
  | C4 | separacion estilo/contenido (DePro, INLP) | la ganancia es **estrictamente OOD** (DePro pierde in-distribution) y el CV de AGE es IID |
  | C5 | geometria ordinal (CPL, Mahalanobis) | ganancia atada a un extractor preentrenado (VGG-16) |

- **La duplicacion mas contundente es algebraica**: `EMD2(one-hot) = suma_k (F_pred(k) -
  1[y<=k])^2`, es decir el objetivo que la literatura presenta como ordinal distinto **ES** la
  familia acumulativa CORAL/all-threshold con perdida L2, cerrada en V22/V23/H10/D-058.

- **Decision**: **`NO_NEW_METHOD`**. Ningun candidato cumple a la vez las tres condiciones del
  brief: compatible con el reglamento, no equivalente a algo cerrado, y con evidencia de mejora de
  macro-F1 en clasificacion de edad a partir de texto con modelos clasicos. **C1 queda registrado y
  NO ejecutado**, con su control ya disenado, porque su expectativa honesta es no cruzar el gate y
  el brief pide decidir si reabrir, no gastar en una falsacion.

- **Consequences**: el mapa de mecanismos queda auditado **dos veces**: contra el codigo del
  proyecto (39 mecanismos) y contra la literatura 2020-2026. La revision aporta cinco piezas de
  evidencia de cierre que antes no existian: el argumento algebraico de EMD, la distincion entre
  calibracion y forma unimodal, la equivalencia en expectativa Ord-NLL/soft labels, la advertencia
  documentada del trade-off nominal/ordinal, y el eje de ruido ordinal con su contra-argumento de
  mismo-proceso. Lo que **no** se afirma: que V30 sea optimo, ni que no exista ninguna mejora
  posible. Lo que si: la literatura no ofrece un mecanismo que AGE no haya cubierto o que no
  empuje contra su metrica.

---

## D-071 - Acta de cierre de AGE: `AGE_RESEARCH_CLOSED`

- **Decision**: se cierra formalmente la investigacion de AGE. **No se ejecuta C1 ni ninguna otra
  falsacion.** V30 permanece como mejor resultado validado y como baseline congelado.

- **Estado congelado, y es el dato que gobierna todo lo demas**:

  ```text
  V30 corrected_cv      = 0.6134471
  V30 particion alterna = 0.6143347
  V30 leaderboard       = 0.63302
  ```

  Los tres numeros son distintos y **no son intercambiables**: 0,6134471 es la unica configuracion
  reproducible desde los artefactos y la que se envio; 0,6143347 es una **reejecucion con
  reentrenamiento** sobre los mismos grupos con otra semilla, y su diferencia con la primera es
  reentrenamiento en el 99 %, medido; 0,63302 es el leaderboard, evidencia de transferencia que no
  se uso para seleccionar nada.

- **Estado de cada linea, cerrado y con su veredicto**:

  | Linea | Veredicto |
  |---|---|
  | G14 auditoria de entrada | **NOT_ACTIONABLE** |
  | P33 residuo global | **CLOSED** |
  | P34 correccion local por pares | **INVALIDATED** |
  | G12 pendiente sobre el margen | **CLOSED** |
  | Revision bibliografica | **NO_NEW_METHOD** |
  | C1 perdidas distancia-conscientes | **LATENT / NO EJECUTADA** |

- **La conclusion metodologica, con sus tres negaciones explicitas**:

  > **No se afirma que V30 sea optimo.** **No se afirma que AGE sea imposible de mejorar.** Se
  > afirma que, dentro de las restricciones vigentes, los inputs legalmente disponibles y los
  > mecanismos auditados, **no queda actualmente una hipotesis nueva con justificacion suficiente
  > para reabrir la investigacion**.

  La formulacion importa: la primera y la segunda negacion evitan vender un cierre como un optimo, y
  la tercera acota la afirmacion al espacio auditado en vez de al espacio entero. Un cierre honesto
  no puede ser mas amplio que su auditoria.

- **La cobertura auditada, que es lo que sostiene la tercera afirmacion**:

  ```text
  39 mecanismos auditados
  23 USED          9 CLOSED          1 INVALIDATED
   1 NOT_ACTIONABLE 5 NOT USED       0 GENUINELY_NEW con justificacion suficiente para ejecutar
  ```

  Los cinco `NOT USED` no quedan en el limbo: **focal loss**, **hard example mining** y **label
  smoothing** apuntan a los ejemplos dificiles y a la frontera, que es la poblacion A (27.430 filas
  donde ninguna de 26 fuentes acierta) y donde P34 midio que el ruido aleatorio gana a la senal
  real. **ranking objective** no mueve la metrica evaluada. **bootstrap de grupos** es reduccion de
  varianza con otra unidad de remuestreo, y el bagging del meta-modelo ya se cerro en -0,000312.

- **C1 queda LATENT, y se documenta con su control por si se revisa**:

  | Campo | Valor |
  |---|---|
  | Hipotesis | codificar en el objetivo que confundir 13-17 con 18-24 es menos grave que con 65-xx cambia la geometria del ajuste |
  | Evidencia a favor | SORD (CVPR 2019) mejora exactitud en Adience en dos replicas independientes; Vargas et al. (TKDE 2026) encuentra que el etiquetado suave mejora la generalizacion ordinal |
  | Evidencia en contra | **el mismo Vargas et al. dice que la CE categorica rinde mejor en promedio en metricas NOMINALES**, y Kasai et al. (texto) lo confirma: hay un trade-off y la mejora ordinal se paga con metricas nominales |
  | Por que no se ejecuta | macro-F1 es nominal. La expectativa honesta es **no cruzar el gate**, y el brief pide decidir si reabrir, no gastar en una falsacion |
  | Control ya disenado | tres brazos: CE one-hot, label smoothing uniforme (el cierre de la auditoria), y kernel con etiquetas permutadas (regularizacion pura). Gate: delta >= +0,0015 con 4/5 folds **y** delta > 8,9e-04 con signo consistente |
  | Diferencia con lo cerrado | no es CORAL (K-1 salidas acumulativas) ni pairwise (K(K-1)/2 binarios); **si es un superconjunto de label smoothing**: con q_k = exp(-beta|k-y|^2) colapsa a LS cuando beta tiende a 0 |

- **Las equivalencias algebraicas, conservadas porque son anticiclicas**: sin ellas, una iteracion
  futura volveria a proponer como nuevas tres tecnicas que ya estan cubiertas.

  1. `EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2`. El objetivo que la literatura presenta como
     ordinal distinto **es** la familia acumulativa CORAL/all-threshold con perdida L2, cerrada en
     V22/V23/H10/D-058.
  2. El minimizador esperado de **Ord-NLL** es un target suave (Corollary 1 de su propio paper):
     colapsa a C1 en expectativa.
  3. Los **targets suaves** son un superconjunto de **label smoothing**: su gradiente es `p - q =
     (p - e_y) + (e_y - q)`, es decir CE one-hot mas un **sesgo fijo dependiente de la clase**, no
     una superficie de decision distinta.

- **Los errores de auditoria corregidos, conservados con su numero correcto**:

  | Lo que se reporto | Lo correcto | Causa |
  |---|---|---|
  | A = 27.007 | **27.430** | enmascarado enrevesado para excluir a los miembros de V30 de su propio rescate |
  | 122 rescates exclusivos | **8.029** | definiciones distintas: el previo medía otra cosa. Ninguno de los dos es erroneo |
  | OOF de V31-03 como probabilidades | son **`decision_function`** | dividirlos por la suma de su fila **invierte el orden**: F1 0,5048 -> 0,0445 |
  | "soft labels ya cerrado" | cerrado **por argumento, no por medicion** | 0 scripts, 0 menciones; el proyecto entero tiene 2 perdidas custom |

  Y los **falsos positivos de mis propias busquedas por regex**, declarados para que nadie los
  repita como evidencia: `complement` matcheaba "complementario" en espanol (391 veces) y `sord`
  matcheaba `tensordot` (15 veces).

- **Consequences**: la investigacion queda cerrada con el mapa auditado **dos veces**, contra el
  codigo del proyecto y contra la literatura 2020-2026. Una reapertura futura tendria que apoyarse
  en una de tres cosas: **evidencia nueva** que cambie el diagnostico, un **cambio de restriccion**
  (por ejemplo que se permitieran metadatos o modelos preentrenados, que es donde la literatura
  situa la mayor parte de la ganancia), o una **hipotesis cientificamente distinta** que no este en
  las 39 auditadas. Cambiar el `beta` de una familia cerrada no cumple ninguna de las tres.

---

## D-072 - Plug-in por clase: `PROMISING_CONFIRMED`, y una correccion al registro historico

- **Observation**: la auditoria de cobertura habia cerrado la investigacion con 39 mecanismos. Al
  revisar el problema con mirada nueva aparecio un objeto que **no se habia probado nunca**: la
  **regla de decision**. No la representacion, no el modelo, no la correccion del score: como se
  convierte la probabilidad en clase.

- **La medicion que lo motiva**: las seis clases de train son **perfectamente balanceadas** (0,160 a
  0,170) y la prediccion de V30 **no lo esta** (0,146 a 0,187). Su F1 por clase va de **0,4992**
  (25-34) a **0,7351** (65-xx). `argmax` maximiza accuracy, no macro-F1, y cuando el reparto
  predicho se desvia hacia las clases faciles hay margen que `argmax` no aprovecha.

- **El candidato**: `argmax_c(w_c * p_c)` con `w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010,
  0.8128]`, derivados por ascenso por coordenadas sobre validacion interna de la particion canonica.
  Sube 25-34, 18-24 y 35-49; baja 65-xx; deja 13-17 y 50-64 intactas.

- **Las dos particiones**:

  | Particion | V30 | plug-in | delta | folds+ | IC95 | alta conf. |
  |---|---|---|---|---|---|---|
  | canonica | 0,6134471 | 0,6154643 | **+0,0020172** | 5/5 | [+0,00101, +0,00295] | **0** |
  | alternativa | 0,6144120 | 0,6164700 | **+0,0020580** | 5/5 | [+0,00112, +0,00306] | **0** |

  **La diferencia entre las dos particiones es 0,0000408.** Reproduce casi a la cuarta cifra. El
  diagnostico `nested` (pesos re-derivados dentro de la particion alternativa) da **+0,0022884**:
  el mecanismo aparece solo.

- **La prediccion falsable, confirmada dos veces**: P34 establecio que una mejora que solo toca la
  frontera es de tipo calibracion. Aqui la prediccion era falsable —un multiplicador por clase no
  puede alterar el orden cuando el margen es grande— y se cumplio en las dos particiones: **cero**
  cambios en filas de alta confianza. Las 11.225 filas que cambian tienen confianza media 0,3357
  frente a 0,6136 del conjunto.

- **El matiz que se declara**: el control `permutado` cambio de signo entre particiones (−0,00279 en
  la canonica, +0,00144 en la alternativa), asi que `real − permutado` baja de +0,0048 a +0,00062.
  La lectura: parte del efecto es generica —cualquier perturbacion que diversifique el reparto ayuda
  algo a macro-F1— y la asignacion correcta de cada multiplicador a su clase anade unos **+0,0006**
  sobre eso. **El permutado no es un control negativo perfecto**: lo que aisla es la diferencia, no
  el nivel.

- **La fuga que se declaro en el pre-registro, y como queda**: los pesos congelados se derivaron de
  la union de cinco ajustes que cubren las 169.336 filas, asi que no son fuera de muestra para esas
  filas. En la canonica puntuaban **por encima** de los anidados (+0,00254 frente a +0,00202), que
  es la firma de haber visto los datos. **En la particion alternativa eso se invierte**
  (`congelado − nested = −0,0002304`): el efecto no se infla bajo un reparto nuevo.

- **La clasificacion, con las tres respuestas separadas**: **no** es una mejora de representacion,
  **no** es una mejora del modelo, **si** es una mejora de la regla de decision. El candidato sube
  macro-F1 **bajando accuracy** de 0,61464 a 0,61254: sacrifica aciertos para subir el recall de las
  clases debiles, que es lo que macro-F1 premia.

- **Correccion al registro historico**: `V20-F3-UMBRALES` quedo etiquetado en el gap analysis como
  *"correccion de umbral"* y **no lo era**. Fue una **cadena de cinco discriminadores binarios
  anidados con monotonia**, es decir una arquitectura ordinal. La regla de decision post-hoc no se
  habia probado nunca, y ese error mio de clasificacion contribuyo a que la linea pareciera cerrada.
  Los informes historicos no se modifican.

- **Decision**: **`PROMISING_CONFIRMED`**, no `FINALIST` y no `ADOPTED`. El delta esta por encima del
  gate de PROMISING (+0,0015) y **por debajo del de FINALIST** (+0,003). Y la particion alternativa
  usa los **mismos grupos** repartidos de otra forma, asi que es una reejecucion y no una particion
  independiente: ese limite viene de que `dup_clusters` cubre el dataset entero, y esta declarado
  desde P00. **No se ha generado ninguna submission y el leaderboard no se ha consultado.**

- **Consequences**: la conclusion de cierre `AGE_RESEARCH_CLOSED` queda **matizada por este
  hallazgo**. La formulacion correcta pasa a ser: no quedan mecanismos de aprendizaje sin recorrer,
  **pero quedaba uno de decision**, y produce una mejora confirmada de ~+0,0020 en CV que no habia
  sido medida. Eso no invalida la auditoria —los 39 mecanismos siguen auditados— sino que muestra
  que su enumeracion estaba incompleta: auditaba *como aprender* y no *como decidir*.
