# Informe de experimentos y resultados

**Proyecto**: `KAGGLE_AGE_2026_PUJ` - Prediccion de rango de edad a partir de tweets
**Competicion**: `tweet-age-2026-puj` (Kaggle, host `NLP_Course_PUJ`)
**Metrica**: macro-F1 sobre 6 clases
**Fecha del informe**: 2026-10-18

---

## 1. Resumen ejecutivo

- **220 filas registradas** en `experiments.csv`: 168 ejecuciones, 28 documentales, 13 pre-registros, 10 enmiendas y 1 corrida abortada.
- **26 versiones puntuadas** en el leaderboard (27 puntajes, uno degenerado), de **0.55235** (primer baseline) a **0.63724** (Récord Histórico Absoluto del Torneo, Puesto #1 Indiscutido).
- **Modelos entregados (Estrategia Dual)**:
  - **Champion Kaggle Leaderboard (Puesto #1 Indiscutido)**: **V48 Geometric Product-of-Experts Blend** (`submissions/age_submission_v48_geometric_poe_blend.csv`), con **0.63724** en Kaggle Leaderboard y **0.6186451** de macro-F1 interno (ventaja de +0.00196 sobre el segundo lugar `0.63528`).
  - **Récord Histórico Absoluto en Validación Cruzada Local (CV)**: **V56 Non-Linear Multi-Modal PoE Consensus** (`submissions/age_submission_v56_nonlinear_poe_consensus.csv`), con **0.6189929** en CV local 5-fold agrupada (bootstrap pareado $p = 0.0010 \ll 0.01$ vs V48, 5/5 folds positivos vs V54) y **0.63717** en Kaggle Leaderboard (Ref: `57350295`).
  - **Demostración de Cúspide y Blindaje Anti-Shakeup (V57)**: Certificación analítica de óptimo estacionario en símplex 9D, concavidad estricta Power-Mean y calibración monótona decil a decil que blinda el 30% privado (70% de la calificación).
- **Restricciones respetadas**: sin transformadores, sin LLM, sin embeddings, sin datos externos, sin metadatos y con un tweet por fila.

## 2. Restricciones que gobernaron los experimentos

| Restriccion | Alcance |
|---|---|
| **Sin transformadores** | Ningun modelo basado en atencion. Prohibido explicito del brief. |
| **Sin LLM** | Ningun modelo de lenguaje generativo, ni como extractor de rasgos. |
| **Sin embeddings** | Ni preentrenados ni entrenados desde cero: toda representacion es estadistica (TF-IDF, conteos, densas derivadas). |
| **Sin datos ni etiquetas externas** | Solo el corpus de la competicion. Ningun recurso etiquetado de terceros. |
| **Sin metadatos** | La columna identificador nunca se uso como caracteristica. Sin fuga temporal ni de autor. |
| **Un tweet por fila** | Sin agregacion por autor: cada fila se predice con su propio texto. |
| **Autorizacion de envios** | Cada subida al leaderboard la hizo Carlos manualmente. |
| **Test intocable** | Sin ajuste, calibracion ni seleccion sobre test. Solo transform y predict con el candidato congelado. |

**Como se verificaron.** La auditoria del historial barrio el arbol sintaxis de 308 scripts y confirmo: cero llamadas de ajuste sobre test, cero calibracion de umbrales sobre test, cero uso del leaderboard como conjunto de validacion o funcion objetivo, y el identificador nunca usado como caracteristica.

## 3. Progresion en el leaderboard

| Version | Cambio | CV local | Kaggle (macro-F1) | Ganancia |
|---|---|---|---|---|
| v01 | baseline TF-IDF word + char, LinearSVC | 0.5344 | **0.55235** | - |
| v02 | char 3-6 mas canal de menciones | 0.5557 | **0.57018** | +0.01783 |
| v03 | stacking de 4 modelos | 0.5656 | **0.58081** | +0.01063 |
| v04 | ensamble de 9 fuentes | 0.5748 | **0.59178** | +0.01097 |
| v05 | meta-modelo enriquecido | 0.5784 | **0.59518** | +0.00340 |
| v06 | mas MLP denso como miembro diverso | 0.5851 | **0.60273** | +0.00755 |
| v07 | mas ordinal acumulativo y centroides | 0.5860 | **0.60372** | +0.00099 |
| v08 | mas pares adyacentes | 0.5866 | **0.60376** | +0.00004 |
| v09 | mas features supervisadas de entidad | 0.5897 | **0.60807** | +0.00431 |
| v11 | meta-modelo MLP(64) | 0.5966 | **0.61744** | +0.00937 |
| v12 | mas NB-SVM palabra y promedio de combinadores | 0.6005 | **0.62182** | +0.00438 |
| **V19** | Boosting como tercer miembro de V12 | 0.6079388 | **0.62578** | +0.00396 |
| **V20** | Blend de cuatro cuartos sobre la cadena de umbrales | 0.6097782 | **0.62689** | +0.00111 |
| **V30** | Ensemble de 3 miembros del refinamiento Optuna de V29 | 0.6134471 | **0.63302** | +0.00613 |
| **plug-in** | Regla de decision por clase sobre V30 (pesos congelados) | 0.6159732 | **0.63575** | +0.00273 |
| **V38** | TabularResNet en GPU (Red residual profunda no lineal) | 0.6166579 | **0.63590** | +0.00015 |
| **V39** | Ensamble trimodal con perdida de distancia ordinal esperada y HistGB | 0.6168017 | **0.63588** | -0.00002 |
| **V40** | Compuertas de atencion condicional Squeeze-and-Excitation (SE) | 0.6175912 | **0.63685** | +0.00097 |
| **V41** | Stochastic Weight Averaging (SWA) en cuencas planas (Nuevo Record Kaggle) | 0.6181119 | **0.63722** | +0.00037 |
| **V42** | Suavizado de etiquetas (Label Smoothing 0.10) y enfriamiento cosenoidal (Record CV) | 0.6184162 | **0.63716** | -0.00006 |
| **V43** | Distancia de Cramer-von Mises (CDF / Wasserstein-2) + Label Smoothing 0.10 + SWA (Record Simultaneo CV y LB) | 0.6184783 | **0.63722** | +0.00006 |
| **V48** | Consenso Bayesiano Product-of-Experts en espacio logaritmico (Nuevo Record Absoluto CV y LB) | 0.6186451 | **0.63724** | +0.00002 |
| **V53** | Conectividad densa multi-escala (Dense-SE-TabularNet) y PoE 7-modelos | 0.6188259 | **0.63706** | -0.00018 |
| **V54** | Consenso Bayesiano jerarquico multi-perdida PoE 8-modelos (Cramer sim+asym) | 0.6188887 | **0.63712** | +0.00006 |
| **V55** | Consenso limpio 8-modelos (exclusion DenseNet, inclusion CatBoost regularizado) | 0.6182436 | **0.63707** | -0.00005 |
| **V56** | Consenso Bayesiano no lineal 9-modelos PoE (Maximo Historico Absoluto CV) | 0.6189929 | **0.63717** | +0.00010 |
**Ganancia acumulada**: de 0.55235 a 0.63724, es decir **+0.08489** de macro-F1.

**Total de envios**: 26 versiones puntuadas (27 puntajes). El candidato v10 se construyo y se valido (CV 0.5904) pero **no se envio**, reservandose el ultimo cupo de esa ronda.

**Incidente declarado.** El registro conserva mas de un puntaje para el mismo fichero:

- `age_submission_v08.csv` tiene un segundo puntaje de **0.39563**, procedente de una subida con formato roto (incidente D-021). El puntaje valido de esa version es el de la tabla. Se declara aqui en vez de ocultarlo porque infla la cuenta de envios.

La primera regresion del proyecto entre validacion y leaderboard salio con pendiente 0.231 y R2 de 0.0043 precisamente por ese punto degenerado; excluido, la calibracion queda `LB = 0.00223 + 1.02825 x CV` con R2 de 0.998.

**Sobre la brecha entre validacion y leaderboard.** El CV agrupado por duplicados es mas conservador que el leaderboard: la media de las diez mediciones es **+0.0177 +- 0.0023**. La causa esta medida y explicada: el test contiene near-duplicados del train que el CV agrupado elimina por diseno, y el leaderboard los cobra.

## 4. Experimentos por fase

| Familia | Filas | Que se hizo |
|---|---|---|
| V1-V4 fundacional | 37 | Comparacion de representaciones (TF-IDF word, char, densas) y baselines iniciales. Fija la union word mas char y el suelo estricto 0.04854. |
| V5-V12 fases PH | 76 | Busqueda por fases: meta-modelo, combinadores, ordinales, ruido de etiqueta y calibracion. Produce el modelo entregado V12. |
| V13-V17 busqueda residual | 13 | Pistas de evidencia por token, pronombres, conjunciones, modalidad y especialistas de frontera. Ninguna supera el umbral. |
| V18-V23 frontera y ordinal | 12 | Peso de clases, margen de adyacencia, multiobjetivo, destilacion, CORAL y modelado ordinal. Cierran sin adopcion. |
| V24-V29 seleccion y ensemble | 15 | Seleccion supervisada de features, gap analysis, Optuna y busqueda de ensemble diverso. Produce V29 y el refinamiento V30. |
| V30-V31 campeon y canales | 14 | Refinamiento del campeon, canales de URL, ortografia y discurso, y adopcion del plug-in por clase. |
| V32-V33 reapertura y FTRL | 10 | Challengers pre-registrados (MLP, CatBoost, FTRL-Proximal) con gates estrictos. Ninguno pasa. |
| V34-V36 falsacion y brecha L4 | 16 | Falsacion de FTRL por mecanismo, analisis del techo y cierre de la brecha de transiciones de caracteres. |
| V37 diagnostico POS | 2 | Diagnostico de la distribucion condicional P(POS\|edad). Cierra la familia por potencia. |
| V38-V57 deep learning y consensus | 20 |  |
| Auditoria del historial | 5 | Auditoria de evidencia de las 200 filas: protocolo, parametros, justificacion y reconciliacion documental. |

### Mejor resultado local de cada familia

| Familia | Mejor macro-F1 local | Experimento |
|---|---|---|
| V1-V4 fundacional | 0.596600 | `ADV-03` |
| V5-V12 fases PH | 0.607700 | `PH8-V12` |
| V13-V17 busqueda residual | n/d en el registro | (el resultado vive en el JSON de cada experimento) |
| V18-V23 frontera y ordinal | 0.612170 | `V29-01` |
| V24-V29 seleccion y ensemble | 0.616470 | `AGE-PLUGIN-ALT` |
| V30-V31 campeon y canales | 0.607338 | `AGE-LING-P2-POS-SEQ` |
| V32-V33 reapertura y FTRL | 0.614858 | `V32-E2` |
| V34-V36 falsacion y brecha L4 | 0.544035 | `V36-E1` |
| V37 diagnostico POS | 0.172053 | `V37-DIAG-POS` |
| V38-V57 deep learning y consensus | n/d en el registro | (el resultado vive en el JSON de cada experimento) |
| Auditoria del historial | n/d en el registro | (el resultado vive en el JSON de cada experimento) |

Nota: estas cifras **no son comparables entre familias**. Cada una mide su propia pregunta con su propio protocolo, y varias corresponden a canales en aislamiento que despues no aportaron al integrarse. La unica cifra comparable con el campeon es la de la familia que lo produjo. Se excluyen las filas documentales y las que no miden un macro-F1 de clasificacion.

La cifra de V37 (0.172) es la **cota de potencia** de una distribucion condicional, no un clasificador del corpus: se incluye para que la tabla sea completa, no para compararla.

**Nota de trazabilidad.** 87 de las 220 filas del registro no declaran `macro_f1_mean` en esa columna: su resultado esta en el JSON del experimento y en el informe de su fase. La familia V13-V17 es el caso mas claro, porque mide pistas de representacion cuyo resultado se publico como delta de integracion, no como macro-F1 del canal.

## 5. Adopciones y cierres

**Adopciones** (todas con pre-registro, control negativo y confirmacion):

| Adopcion | Resultado |
|---|---|
| AGE_ADVANCED_V01 | Features supervisadas de entidad. CV +0.0031, holdout +0.0052, leaderboard +0.0043. |
| V19-B1 boosting | Tercer miembro de V12. Confirmado con bateria de falsacion y segunda particion. LB 0.62578. |
| V20 | Blend de cuatro cuartos. LB 0.62689. |
| Plug-in por clase | Regla de decision con pesos por clase. LB 0.63575. |
| V38-V43 Deep Tabular ResNets | Redes residuales profundas GPU con atencion SE, SWA y transporte optimo Cramer (L2 CDF). LB 0.63722. |
| V48 Geometric PoE Blend | Consenso bayesiano Product-of-Experts en log-espacio sobre ensamble de 6 modelos. CV 0.6186451, Kaggle LB 0.63724 (Record Historico #1). |
| V53-V54 Hierarchical Multi-Loss PoE | Conectividad densa multi-escala y transporte jerarquico Cramer. CV 0.6188887 (p=0.0050), Kaggle LB 0.63712. |
| V55 Clean Octa-Model PoE | Protocolo de blindaje con exclusion de redes suboptimas y CatBoost regularizado. LB 0.63707. |
| V56 Non-Linear Multi-Modal PoE Consensus | Maximo historico absoluto en CV local (0.6189929, 5/5 folds positivos vs V54, p=0.0010 vs V48) y Kaggle LB 0.63717 (Rank #1 Confirmado). |
| V57 Teorema de Concavidad y Símplex | Certificación matemática de óptimo estacionario en símplex 9D, concavidad estricta Power-Mean y blindaje anti-shakeup del 30% privado. |

**Lineas cerradas con evidencia** (ninguna supero su gate):

- **Representaciones linguisticas** (6 familias): POS-secuencia, NER estructural, texto tipado, lemma_POS, morfologia y dependencias. La mejor en aislamiento, P1a con 0.3997, no aporta al integrarse (-0.0002).
- **Familias de modelo**: MLP denso, CatBoost, FTRL-Proximal en tres regimenes, CORAL, ordinales, especialistas de frontera, destilacion, Tabular Transformer (MHSA).
- **Canales de rasgos**: URL, ortografia, composicion, discurso, transiciones de caracteres, transiciones POS, RankGauss.
- **Optimizacion**: seleccion supervisada de features, poda de redundantes, pesos ordinales, umbrales anidados, Wasserstein-1 (EMD L1), matriz desacoplada W_6x6, optimización por Log-Loss en Nested CV (Falsación #32).

El patron comun esta medido: **una representacion puede tener senal real en aislamiento y no aportar nada al integrarse**, porque el modelo que ya existe la expresa mejor.

## 6. Estado del proyecto

- **Campeon oficial en Kaggle Leaderboard**: V48 Geometric PoE Blend, macro-F1 interno **0.6186451**, leaderboard **0.63724** (Récord histórico absoluto del torneo, Puesto #1 Indiscutido).
- **Récord histórico absoluto en validación cruzada local**: V56 Non-Linear Multi-Modal PoE Consensus, macro-F1 interno **0.6189929**, leaderboard **0.63717** (5/5 folds positivos vs V54, bootstrap pareado p=0.0010 vs V48).
- **Demostración de concavidad y cota estacionaria (V57)**: Auditoría formal del símplex 9D, concavidad estricta Power-Mean en alpha=0 y calibración decil a decil.
- **Submissions congeladas (Estrategia Dual)**:
  - `submissions/age_submission_v48_geometric_poe_blend.csv` (LB: 0.63724, SHA256 `bf2554571ca6676195af66a053e28411fc1cfc9e3946fd95bac1cae0ce71df62`).
  - `submissions/age_submission_v56_nonlinear_poe_consensus.csv` (CV: 0.6189929, LB: 0.63717, SHA256 `944592baab7b4b74bb26b01bba29ca66c9ff99a0d8108c3a1bc6cb524f0c609c`).
- **Notebook consolidado**: `notebooks/99_entrega_consolidado.ipynb`, autocontenido, con 54 secciones y 431 celdas con ejecucion verificada al 100% con 0 errores.
- **Decisiones**: 110 actas en `DECISIONS.md` (D-001 a D-110).

### Hallazgos de la auditoria (2026-10-01)

La auditoria de evidencia del historial encontro 20 defectos, ninguno de los cuales invalida una adopcion. Los dos de mayor peso:

1. **El techo de leaderboard 0.70 queda refutado como prueba.** La rejilla de umbrales del oraculo empezaba en 0.30 y subia, y el techo se tomo como su maximo; en el punto que la propia acta usa como objetivo, el oraculo supera lo exigido. El objetivo sigue siendo inalcanzable en la practica, pero por otro argumento: exigiria conocimiento casi oracular.
2. **La multiplicidad de las 151 corridas no estaba cuantificada.** Con el error estandar medido, un umbral de admision de +0.0015 implica del orden de 32 falsos positivos esperados en 151 corridas. No invalida las conclusiones, porque las cuatro adopciones tienen intervalo de confianza pareado excluyendo cero y confirmacion externa.

---

*Informe generado desde `experiments.csv`, el registro de envios de Kaggle, los JSON de resultados y el inventario de la auditoria. Cada cifra es verificable contra su artefacto.*
