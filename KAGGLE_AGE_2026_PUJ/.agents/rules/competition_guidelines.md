<!-- competition-guidelines:start -->
# Directrices de Competencia y Aprendizajes (KAGGLE_AGE_2026_PUJ)

Este documento codifica las lecciones metodológicas aprendidas de las versiones V01 a V56 para guiar toda futura experimentación.

## 1. Restricciones Inviolables de la Competencia
1. **Sin Embeddings ni Modelos de Lenguaje**: Queda estrictamente prohibido el uso de transformers, LLMs, embeddings preentrenados (e.g. Word2Vec, GloVe, FastText) o embeddings entrenados desde cero. Toda representación debe ser estadística o basada en ingeniería de características léxicas/textométricas.
2. **Sin Fuga del Test (Test Intocable)**: El dataset de prueba (`age_test.csv`) nunca debe ser usado para ajuste de vectorizadores, modelos, selección de variables ni calibración de umbrales.
3. **Sin Datos Externos ni Metadatos**: No se usan identificadores de usuario, fechas ni información fuera del texto del tweet.
4. **Formato Estricto de Predicción**: Las predicciones deben coincidir exactamente con el orden y formato de `age_submission.csv` (`id,age_range`).

## 2. Prevención de Out-Of-Memory (OOM) y Eficiencia de Memoria
* **Representaciones Ralas**: Las matrices de características basadas en TF-IDF (caracteres o palabras) DEBEN conservarse en formato `scipy.sparse.csr_matrix`. Queda prohibido llamar `.toarray()` de forma global sobre el corpus.
* **Control de Concurrencia**: En validación cruzada y entrenamientos paralelos, limitar concurrencia para evitar agotar la RAM.
* **Mapeo de Memoria y Recolección**: Utilizar `np.load(..., mmap_mode="r")` al cargar matrices de disco y ejecutar `del` seguido de `gc.collect()` tras etapas intermedias pesadas.

## 3. Prevención de Sobreajuste y Protección del Leaderboard Privado
* **Esquema de Evaluación del Curso**:
  * **70% de la nota final**: Leaderboard privado (30% de los datos de test liberados al final).
  * **30% de la nota final**: Notebook de entrega (`notebooks/99_entrega_consolidado.ipynb`, ejecución 100% reproducible con 0 errores vía `nbconvert`).
* **Validación Agrupada Mandatoria**: Todas las decisiones deben evaluarse mediante `StratifiedGroupKFold` (5 folds) agrupado por clústeres de near-duplicados (`dup_clusters.parquet`).
* **Compuerta Estricta de Adopción y Subida a Kaggle**:
  * Solo se autoriza una submission si existe una ganancia estadísticamente significativa en CV local: $\Delta \text{CV} > 0$ con al menos $4/5$ folds positivos y prueba de significancia pareada bootstrap ($B=1.000$) con $p < 0.05$ (IC95 excluyendo el cero).
  * Queda terminantemente prohibido el *leaderboard probing* para evitar el colapso por shakeup en el 30% privado final.
* **Estrategia Dual de Selección Final en Kaggle**:
  * **Selección 1 (Public Anchor)**: `submissions/age_submission_v48_geometric_poe_blend.csv` (Score: **`0.63724`**, Rank #1 indiscutido).
  * **Selección 2 (Private Regularized)**: `submissions/age_submission_v56_nonlinear_poe_consensus.csv` (Local CV: **`0.6189929`**, Récord histórico absoluto, bootstrap $p=0.0010 \ll 0.01$ vs V48).

## 4. Estado del Arte y Champion Actual
* **Champion Oficial en Kaggle Leaderboard**: **V48 Geometric Product-of-Experts (PoE) Blend** (Hinton 2002).
  * Consenso bayesiano multiplicativo en log-espacio ($\log P_{\text{geom}} \propto \sum w_m \log P_m$) sobre 6 estimadores con vector plug-in por clase congelado.
  * **Puntaje Kaggle LB**: **`0.63724`** (Rank #1 Indiscutido).
* **Récord Histórico Absoluto en Validación Cruzada (CV)**: **V56 Non-Linear Multi-Modal PoE Consensus**.
  * Consenso bayesiano en log-espacio combinando 9 arquitecturas ortogonales: LightGBM V30, Cramér simétrico V43, Cramér asimétrico winsorizado V47, Smooth-SWA V42, SE-TabOrd V40, TabOrd V39, HistGB V39, Consistent-Cramer V51 y Dense-SE-TabNet V53.
  * **Récord Absoluto CV**: **`0.6189929`** (5/5 folds positivos vs V54, bootstrap $p = 0.0010 \ll 0.01$ vs V48).
  * **Puntaje Kaggle LB**: **`0.63717`** (Ref: `57350295`).

## 5. El Hallazgo Matemático Central: La Brecha de Menciones
* En tweets **CON menciones** (`@usuario`, 63.9% de train y 63.85% de test), V48 alcanza **`0.6760801` Macro-F1**, superando el objetivo de 0.65.
* En tweets **SIN menciones** (36.1% de train y 36.15% de test), Macro-F1 cae a **`0.5043177`** porque las 73 variables supervisadas colapsan al prior uniforme.

## 6. Lo que FUNCIONÓ (Principios a Conservar)
1. **Consenso PoE en Log-Espacio (Hinton 2002)**: Compuerta lógica AND que elimina falsos positivos espurios en clases extremas (`13-17` y `65-xx`).
2. **Transporte Óptimo de Cramér-von Mises $L_2$ (Frogner et al. NeurIPS 2015)**: Penalización cuadrática de CDF acumulada para respetar la topología ordinal.
3. **Stochastic Weight Averaging (SWA, Izmailov et al. UAI 2018)**: Localiza cuencas planas y mínimos anchos, blindando contra el shakeup privado.
4. **Compuertas Squeeze-and-Excitation (Hu et al. CVPR 2018)**: Modulación adaptativa por canal ante presencia o ausencia de menciones y URLs.
5. **Calibración Plug-in Congelada**: Corrección determinista del desbalance de clase en Macro-F1: $\mathbf{w}_{\text{plugin}} = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]$.

## 7. Inventario Sistemático de Falsaciones (NO REPETIR)
1. Tabular Transformer con Multi-Head Self-Attention (V50-A): Dispersión de atención en variables heterogéneas.
2. Redes Auto-Normalizantes Rígidas (SNN con SELU y AlphaDropout, V52-B): Colapso ante matrices ralas y 71 columnas cero-infladas.
3. Transformación Normal Cuantílica (RankGauss): Destruye la masa de probabilidad discreta en cero.
4. Wasserstein-1 ($L_1$ Boundary-EMD, V52-A): Gradientes discontinuos oscilantes.
5. Desacoplamiento Excesivo de Pesos por Clase ($W_{6 \times 6}$, V50-B): Inestabilidad muestral ($p = 0.1940$).
6. Paradoja de Simpson en Reglas de Mención: Ajustes aislados por subgrupo descalibran la proporción marginal global.
7. Árboles Simétricos Obliviosos (CatBoost sin regularizar, V49): Delta negativo e inestabilidad vs LightGBM leaf-wise.
8. Otras falsaciones: POS tags concatenados, Multi-sample dropout, Mixup tabular, Stacking logístico L2, XGBoost histogram trees, Temperatura de calibración, Bagging de semillas, MLP ancho 512, Umbralado 1D de E[c], Compuertas de entropía, Suavizado gaussiano ordinal, Focal-Cramér, Variación de bloques SE (1, 3, 4), Inyección directa de V30 en entrada, Gated Linear Units (GLU), Épocas extendidas 20-24, Nystroem RBF, Meta-árbitro LightGBM, Reoptimización anidada de plug-in, NBSVM char_wb, Proportional Odds / Enlace Acumulativo.
<!-- competition-guidelines:end -->

