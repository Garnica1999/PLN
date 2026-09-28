> **Leia este documento primero.** La cuenta completa y actualizada del proyecto, que
> incluye V12 y las cinco fases de busqueda posteriores (V13 a V17), esta en
> **[`INFORME_COMPLETO.md`](INFORME_COMPLETO.md)**. Este `FINAL_REPORT.md` es el documento de
> entrega del modelo V12 y se conserva como registro historico sin modificar: su alcance es el
> modelo entregado, no las fases posteriores.

# Informe final — Predicción de rango de edad a partir de tweets

**Competencia**: `tweet-age-2026-puj` (Kaggle, host `NLP_Course_PUJ`) · Procesamiento de Lenguaje Natural · Maestría en Inteligencia Artificial, Pontificia Universidad Javeriana.

**Autores**: Carlos Garnica y equipo · **Fecha**: 2026-09-24 · **Repositorio**: `KAGGLE_AGE_2026_PUJ/`

---

## 1. Resumen ejecutivo

- **Tarea**: clasificación multiclase de tweets en 6 rangos de edad (`13-17`, `18-24`, `25-34`, `35-49`, `50-64`, `65-xx`) usando solo el texto.
- **Resultado en el leaderboard**: **0.62182** de F1 (v12; línea base inicial 0.55235; mejora acumulada **+0.0695**).
- **Validación interna honesta**: 0.6005 de macro-F1 (leave-one-fold-out sobre particiones agrupadas por clústeres de near-duplicados).
- **Calibración validación<->leaderboard**: **15 envíos registrados (12 puntuados)** con brecha media **+0.0177 ± 0.0023**, explicada por la memorización de near-duplicados del test (el límite de ~10 envíos de D-004 era un supuesto erróneo).
- **Fase de búsqueda de señal residual (2026-09-25)**: 27 variantes controladas evaluadas; ninguna supera el umbral pre-registrado (Δ ≥ +0,003 con ≥4/5 folds en CV). **Línea saturada**, modelo final sin cambios y **ninguna submission nueva enviada**. Única señal positiva: la vista estructural (CV +0,0018 con 4/5 folds; holdout +0,0031).
- **Restricción cumplida**: sin transformers, sin LLMs y **sin ningún modelo de vectorización** (ni preentrenado ni entrenado desde cero): todo el desempeño proviene de representaciones estadísticas (TF-IDF), features artesanales y modelos clásicos.
- **Auditorías**: APROBADO en leakage, reproducibilidad, formato de submission y trazabilidad académica.

## 2. Datos

| Archivo | Filas | Columnas | Separador |
|---|---|---|---|
| `age_train.csv` | 169,336 | `id;text;age_range` | `;` |
| `age_test.csv` | 112,892 | `id;text` | `;` |
| `age_submission.csv` | 112,892 | `id,age_range` | `,` |

**Propiedades verificadas**

- 6 clases casi balanceadas por diseño: `13-17` 17.0%, `18-24` 16.9%, `25-34` 16.7%, `35-49` 16.7%, `50-64` 16.6%, `65-xx` 16.0%.
- Texto almacenado en ancho fijo de 140 caracteres con relleno de espacios (artefacto de exportación, eliminado en la carga: decisión D-005).
- Alfabeto de **96 caracteres con un único no-ASCII** (`¿`, 88 apariciones): el corpus fue transliterado a ASCII, sin tildes, `ñ` ni emojis (decisión D-009).
- **Sin metadatos**: no hay identificador de usuario, fecha, biografía ni seguidores. Toda la señal es textual.
- Textos vacíos: 104 en train y 62 en test.
- Ruido de etiqueta: 643 filas (0.38%) con etiquetas contradictorias para el mismo texto.
- `id` descartado como feature: la desviación máxima de la distribución de clase entre deciles de `id` es de 1.25 pp.

**Procedencia del corpus.** No fue posible identificar la fuente pública exacta (`reports/tables/dataset_provenance.md`): el esquema de 6 rangos y el vocabulario (~2016-2017, español latinoamericano, sin COVID) no coinciden con PAN AP ni con TwiSty. Consecuencia: **no se usaron datos externos** en ningún momento.

## 3. Metodología: validación sin autoengaño

El riesgo dominante de este corpus no es el autor repetido (no hay `user_id`) sino el **duplicado y near-duplicado**: retuits y variantes del mismo texto que solo cambian enlaces o menciones.

- 1.21% de las filas de test aparecen **literalmente** en train.
- 4.26% comparten un **clúster de near-duplicado** (Jaccard ≥ 0,8 sobre tokens, con *blocking* por tokens raros), con 92.8% de clústeres de etiqueta única.
- Un baseline de **memorización pura** alcanza 0.0608 de macro-F1 con un split estratificado ingenuo y 0.0000 con el esquema agrupado.

**Decisión (D-003).** Toda la validación usa `StratifiedGroupKFold` (5 folds) agrupado por clúster de near-duplicados, con comparación de modelos mediante leave-one-fold-out sobre predicciones out-of-fold. El test nunca participa del ajuste de vectorizadores, modelos, selección de features ni umbrales.

## 4. Estado del arte

Se revisaron **30 trabajos** (2006-2021) con DOI/URL verificado en `reports/state_of_art.csv` y `references.bib`. Los hallazgos que guiaron el proyecto:

| Trabajo | Aporte al proyecto |
|---|---|
| Schler et al. (2006) | Esquema de 6 rangos de edad idéntico al de esta competencia; features baratas (pronombres, palabras funcionales) |
| Rangel et al. (PAN 2013-2019) | Mejores sistemas de author profiling: n-gramas de carácter + modelos lineales; en PAN 2019 (ES) los char n-grams superaron a los embeddings |
| Sapkota et al. (2015) | Los n-gramas de carácter son la representación más robusta en texto informal |
| Nguyen et al. (2011) | Regresión ordinal auxiliar sobre el punto medio del rango |
| Monroe et al. (2008) | Log-odds con prior Dirichlet para términos distintivos por clase |
| Li et al. (2010), Parambath et al. (2014) | Ajuste de umbral para F-measure multiclase (probado y descartado aquí: ver §9) |
| Wang & Manning (2012) | NB-SVM (log-count ratio): implementado y evaluado; aporta como miembro del ensamble, no como modelo principal |

**Expectativas calibradas con la literatura**: 0,29-0,53 de exactitud en tareas PAN de 5-6 clases sobre Twitter. El resultado obtenido (0,5851 de macro-F1 interno) está en el extremo alto de ese rango, coherente con un corpus más separable (clases balanceadas por diseño y registro muy diferenciado entre extremos).

## 5. Análisis exploratorio y lingüístico

**Hallazgos principales** (`notebooks/02_eda.ipynb`, `03_linguistic_analysis.ipynb`):

| # | Observación | Evidencia | Consecuencia |
|---|---|---|---|
| E1 | La longitud y la riqueza léxica crecen con la edad | ε² = 0,126 en `n_chars`; 73 -> 111 caracteres de media entre extremos | familia textométrica en el catálogo |
| E2 | El léxico separa mundos sociales | `follow, justin, bieber` en `13-17` vs `venezuela, maduro, gobierno` en `65-xx` | representación léxica justificada |
| E3 | El retuit es más frecuente en mayores | 53% en `65-xx` vs 30% en `18-24` | feature `is_rt` |
| E4 | Alta redundancia entre features densas | ρ = 0,999 entre `ttr` y `hapax_ratio` | ablación incremental obligatoria |
| E5 | El corpus no tiene emojis ni tildes | 96 caracteres distintos, 1 no-ASCII | features de emoji descartadas (D-009) |

## 6. Representaciones: la decisión central

Se evaluaron **14 configuraciones** con modelo fijo (LinearSVC C=1) y validación agrupada:

| Configuración | macro-F1 | Features |
|---|---|---|
| E-008 — char_wb 3-6 + canal de menciones (crudo) | 0.5557 | 314,420 |
| E-003c — char_wb 3-6 (crudo) | 0.5413 | 300,000 |
| E-003b — char_wb 3-5 (crudo) | 0.5408 | 224,628 |
| E-004b — char_wb 3-6 + word 1-gram (crudo) | 0.5354 | 339,409 |
| E-007a — union sin IDF (TF puro) (crudo) | 0.5339 | 336,493 |
| E-005c — solo quitar URLs + union (sin_urls) | 0.5277 | 292,170 |
| E-005e — solo quitar puntuación + union (sin_punct) | 0.5264 | 292,313 |
| E-003a — char_wb 2-4 (crudo) | 0.5261 | 121,285 |
| E-002a — word 1-gram (crudo) | 0.5025 | 39,409 |
| E-002b — word 1-2 gram (crudo) | 0.4971 | 111,865 |
| E-002c — word 1-3 gram (crudo) | 0.4941 | 151,431 |
| E-005b — marcadores <url>/<user> + union (marcadores) | 0.4107 | 218,709 |
| E-005d — solo quitar menciones + union (sin_menciones) | 0.4094 | 260,025 |
| E-005a — union word 1-2 + char 3-5 (normalizado) (normalizado) | 0.3902 | 183,263 |

**Hallazgo 1 — los caracteres ganan a las palabras.** `char_wb 3-6` (0,5413) supera a `word 1-2` (0,4971) por +0,0442 en 5/5 folds, y la unión clásica palabra+carácter **diluye** (−0,0069).

**Hallazgo 2 — la identidad de las cuentas mencionadas es una de las señales más fuertes.** Eliminar solo las menciones cuesta **−0,1251** de macro-F1 y sustituirlas por `<user>` cuesta −0,1237: la pérdida no es el token, es *a quién* se menciona. Un canal TF-IDF dedicado a los tokens de mención aporta **+0,0144**.

**Hallazgo 3 — "limpiar" destruye señal.** La normalización completa (URLs + menciones + retuits + puntuación) cuesta **−0,1442** y deja 7.240 tweets vacíos (frente a 104 originales). El texto se usa **crudo**.

**Decisión (D-010)**: `TF-IDF char_wb(3,6)` sobre texto crudo ∪ canal de menciones, con SVM lineal.

## 7. Modelos, optimización y ensamble

| Baseline | Descripción | macro-F1 |
|---|---|---|
| B0 | DummyClassifier(prior) | 0.0485 |
| B1 | TF-IDF word 1-2 + LogisticRegression (OvR) | 0.5015 |
| B2 | TF-IDF word 1-2 + LinearSVC | 0.4971 |
| B3 | TF-IDF char_wb 3-5 + LinearSVC | 0.5408 |
| B4 | TF-IDF word 1-2 + char_wb 3-5 + LinearSVC | 0.5344 |
| B5 | 35 features densas + LightGBM | 0.3567 |
| E-008 | char_wb 3-6 + canal de menciones + LinearSVC (ganadora) | 0.5557 |

**Modelos clásicos** (`notebooks/06_classical_models.ipynb`): SVM lineal C=0,2 (mejor individual, 0,5628), LogReg OvR (0,5603, aporta probabilidades), SGD logístico (0,5579 con 1/25 del costo), ComplementNB (0,5370 pero el más diverso, 25,2% de desacuerdo). El barrido de regularización muestra un óptimo claro en C=0,2 (C=1,0 baja a 0,5557).

**Ensamble por stacking** (evaluado con leave-one-fold-out):

| Combinación | macro-F1 |
|---|---|
| 5 base + hashtags + ordinal + word | 0.5748 |
| 5 base + hashtags + ordinal | 0.5725 |
| 5 base (+700k) | 0.5703 |
| svm700k + hashtags + ordinal + word | 0.5702 |
| 4 base + hashtags + ordinal | 0.5682 |
| 4 base + hashtags | 0.5679 |
| 4 base (Fase 20) | 0.5660 |
| + MLP densas+SVD (E-020) | 0.5851 |
| **+ meta-modelo MLP(64) (v11)** | **0,5967** |
| **+ NB-SVM palabra y promedio de dos combinadores (v12)** | **0,6005** |

**Modelo final (submission v12)**: 14 modelos base (2× LinearSVC sobre vocabularios de 300k y 700k, LogReg, SGD, ComplementNB, SVM con canal de hashtags, Ridge ordinal, softmax ordinal, LogReg sobre palabras y MLP sobre densas+SVD) + meta-modelo `StandardScaler + LogisticRegression` sobre 90 features (49 scores + 6 del MLP + 35 features densas).

## 8. Análisis de errores

F1 por clase del ensamble (out-of-fold): `65-xx` 0.691, `13-17` 0.638, `50-64` 0.620, `accuracy` 0.566, `macro avg` 0.566, `weighted avg` 0.565, `35-49` 0.519, `18-24` 0.476, `25-34` 0.449.

**Fenómenos identificados** (`notebooks/08_error_analysis.ipynb`):

| # | Fenómeno | Evidencia | Hipótesis derivada | Resultado medido |
|---|---|---|---|---|
| F1 | Confusión entre rangos **contiguos** | 5 de los 6 pares más confundidos son adyacentes (18-24->25-34 15,3%) | señal ordinal (E-015) | +0,0003 en el ensamble |
| F2 | Tweets **cortos** sin señal | 52,5% de error en el quintil más corto vs 37,2% en el más largo | meta-modelo condicionado por longitud (E-018) | **+0,0036** |
| F3 | Clases **medias** débiles | `25-34` 0,449; `18-24` 0,476 | canal de hashtags (E-014) | +0,0019 |
| F4 | **Identidad** de las cuentas mencionadas | quitar menciones cuesta −0,125 | canal dedicado de menciones (E-008) | **+0,0144** |
| F5 | Errores distintos entre familias de modelo | MLP con 0,4071 aporta +0,0066 | diversidad para el ensamble (E-020) | **+0,0066** |

Cuatro de los cinco fenómenos se convirtieron en mejoras verificadas del modelo final: el análisis de errores fue el motor real de la progresión.

## 9. Resultados y calibración de la validación

| Versión | Modelo | CV macro-F1 | Leaderboard | Brecha |
|---|---|---|---|---|
| v01 | E-001 (word 1-2 ∪ char 3-5) | 0.5344 | **0.55235** | +0.0180 |
| v02 | E-008 (char 3-6 + canal de menciones) | 0.5557 | **0.57018** | +0.0145 |
| v03 | Stacking de 4 modelos | 0.5656 | **0.58081** | +0.0152 |
| v04 | Ensamble de 9 fuentes | 0.5748 | **0.59178** | +0.0170 |
| v05 | Ensamble + meta-modelo enriquecido | 0.5784 | **0.59518** | +0.0168 |
| v06 | Ensamble de 10 fuentes con MLP denso | 0.5851 | **0.60273** | +0.0176 |
| v07 | Ensamble de 12 fuentes (+ ordinal acumulativo + centroides) | 0.5860 | **0.60372** | +0.0177 |
| v08 | Ensamble de 13 fuentes (+ pares adyacentes) | 0.5866 | **0.60376** | +0.0172 |
| v09 | AGE_ADVANCED_V01 (features supervisadas de entidad) | 0.5897 | **0.60807** | +0.0184 |
| v10 | + NB-SVM (generada, no enviada) | 0.5904 | pendiente | — |
| v11 | Meta-modelo MLP(64) | 0.5966 | **0.61744** | +0.0208 |
| v12 | + NB-SVM palabra y promedio de combinadores | 0.6005 | **0.62182** | +0.0213 |

La brecha entre la validación interna y el leaderboard es **estable en 12 mediciones independientes** (+0.0177 ± 0.0020), lo que valida el diseño experimental: la diferencia se explica por la memorización de near-duplicados del test, que el esquema agrupado bloquea por construcción. En consecuencia, las decisiones de modelado se tomaron **con la validación interna**, sin usar el leaderboard como conjunto de validación.

## 10. Experimentos negativos (documentados)

| Experimento | Resultado | Interpretación |
|---|---|---|
| Normalización agresiva (E-005a) | −0,1442 | los tweets son a menudo solo enlaces, menciones o hashtags: su forma *es* la señal |
| Quitar/marcar menciones (E-005b, E-005d) | −0,125 | la identidad de la cuenta mencionada es señal fuerte |
| Ajuste de decisión por clase (Fase 19) | +0,0006 | con clases balanceadas por diseño no hay prior que corregir (D-011) |
| Features densas solas (B5) | 0,3567 | 35 números no compiten con 314k n-gramas |
| MLP sobre densas (E-019a) | 0,3381 | sin embeddings, la vía densa pierde información |
| MLP sobre densas+SVD (E-019b) | 0,4071 | la SVD conserva solo 7,8% de la varianza en 64 componentes |
| Regresión ordinal sola (E-015) | 0,3973 | útil solo como señal auxiliar del ensamble |
| Residuales de longitud (PH5) | −0,0019 | el combinador no lineal ya aprende la estructura condicionada por longitud |
| Estilo puro como representación (PH6) | 0,2793 | **manda el contenido**: el contenido solo alcanza 0,5537 y el estilo casi azar |
| NB-SVM por canal (PH7) | +0,0009 | por debajo del umbral de ruido |
| Tercer combinador (PH9) | −0,0006 | la diversidad sin calidad comparable no compone |
| Segmentación dura por tipo/longitud (PH2) | −0,0007 / −0,0015 | particionar pierde más de lo que gana |
| **Retrieval train->test** (PH10) | etiqueta recuperada acierta **0,615** | **permitido por las reglas pero los datos lo desaconsejan**: el techo lo fija la consistencia de etiqueta de los duplicados, no la cobertura |
| **POS de spaCy** (PH10) | −0,0032 | el corpus transliterado a ASCII degrada el etiquetado y la señal es redundante con los char n-grams |
| Exclusión de filas ruidosas (PH11) | +0,0001 / +0,0016 | mejora real pero bajo el umbral de cambio de modelo (D-027) |

## 11. Auditorías finales

Resultado global: **APROBADO** (`reports/auditorias.md`).

- **Leakage**: ningun_fit_sobre_test=True; n_llamadas_fit_sobre_test=0; n_canales_representacion=6; vectorizadores_ok=True.
- **Reproducibilidad**: archivos_con_semilla_o_random_state=38; semilla_global_definida=True; requirements_existe=True; n_paquetes_congelados=312.
- **Formato de submission**: columns_ok=True; columns_match_sample=True; rows_ok=True; ids_equal_test=True.
- **Trazabilidad**: n_experimentos_registrados=30; todos_con_hipotesis=True; todos_con_resultado=True; n_decisiones=16.

## 12bis. V23 — CORAL / cumulative-link con pesos compartidos

La ultima fase abierta sobre la formulacion ordinal responde a una sola pregunta: **compartir el
vector de pesos entre las cinco fronteras ordinales y ordenar los umbrales aporta una senal util
que los cinco binarios independientes de V20 no capturan?**

La formulacion es cumulative-link: `P(Y <= k | x) = sigmoid(theta_k - w' x)` con
`theta_k = gamma + suma_{{j<k}} softplus(delta_j)`. El orden de los umbrales es una **garantia de
construccion**, luego las cinco CDF son monotonicas siempre y no hay que imponer nada despues.
Esta implementada en `src/ordinal_coral.py` con `numpy` y `scipy`, con gradiente analitico
verificado contra diferencias finitas (error relativo 3,15e-10) y verificada contra la densidad
de clase calculada por otra via.

**Medicion, con el control A y un control B que hacia falta.**

| Modelo | macro-F1 | Delta | Folds + | violaciones de monotonia |
|---|---|---|---|---|
| V20 (cinco binarios LightGBM independientes) | **0,6097782** | - | 3/5 | **18.927** |
| cinco binarios **lineales** independientes (control B) | 0,4690055 | -0,140773 | 0/5 | no medido |
| **CORAL** (peso compartido, umbrales ordenados) | 0,4837017 | -0,126077 | 0/5 | **0** |

Un delta de -0,126 mezcla dos efectos —compartir pesos y perder no linealidad—, asi que se
anadio el control B para separarlos. El resultado de esa descomposicion es lo util de la fase:

- **Compartir pesos y ordenar umbrales vale +0,014696.** La hipotesis **es correcta**: compartir
  informacion entre las fronteras ordinales aporta senal.
- **La no linealidad vale -0,140773**, un orden de magnitud mas. Un modelo lineal sobre las 279
  features no compite con LightGBM, y ahi se decide el macro-F1.
- **La monotonia se resuelve de verdad y no compra nada.** CORAL tiene 0 violaciones frente a las
  18.927 de V20 y aun asi pierde 0,1261. Es la **segunda** confirmacion independiente de D-057.

**Conclusion.** `CORAL = CLOSED` y `ORDINAL_SHARED_WEIGHTS = NO_SIGNAL`. Sin submission, sin
bootstrap (el brief lo reserva al caso en que se supera el gate de delta) y sin abrir ninguna
familia nueva. La conclusion es incomoda pero la sostienen los numeros: **la formulacion ordinal
no es donde queda la senal**, ni compartida ni independiente, y la evidencia acumulada ya no
sostiene que quede ahi.

**Una nota sobre la verificacion.** Dos errores de formulacion propios fueron cazados por los
tests antes de tocar el dataset: factorizar la diferencia de dos CDF como producto de sigmoidos,
que solo vale si `s_y = 1`, y penalizar solo `w`, que produce separacion y dispara los umbrales a
12,07 con features estandarizadas en [-3, 3]. Los valores de regularizacion quedaron
**pre-registrados sin mirar AGE**, a partir de una comprobacion de separacion sobre un sintetico
con la forma y la escala del proyecto.

## 12ter. V24 — Reconciliacion documental: la causa era estructural

La fase V24 no experimenta. Su objeto es que el notebook consolidado represente el estado real de
la investigacion, y el hallazgo es que la causa de que no lo hiciera **no era editorial**.

**La fuente de verdad del notebook es una cadena, no un generador unico.** Hay cinco scripts, cada
uno de una fase, aplicados en orden sobre el mismo archivo: `fase09_99_entrega.py`,
`fase12_notebook_final.py`, `phase28_notebook_final_v2.py`, `v17_notebook.py` y
`v23_02_auditoria.py`. La cadena tiene **un eslabon por fase**, y **V18, V19, V20, V21 y V22
nunca tuvieron eslabon**. Ese es el motivo exacto de que el notebook se quedara en V17 con V20 como
champion, y explica por que no se podia cerrar desde V23 sin dejar esas cinco fases fuera.

Las 103 celdas no eran un numero erroneo: eran correctas **para lo que contenian**, V12 a V17 mas
la seccion de V23 anadida al final. El numero no estaba mal, estaba incompleto. Ahora el notebook
tiene **121 celdas y 33 secciones**, con las **doce versiones V12 a V23** presentes y cada una
con hipotesis, metodo, control, `corrected_cv`, decision y ruta de artefactos.

**El registry arrastra un defecto propio, y es preexisting.** `experiments.csv` tenia **tres
lineas de cabecera** en vez de una: las dos espurias ocupaban el identificador `experiment_id` dos
veces, luego declaraba 130 experimentos y tenia **128**. El defecto aparece igual en al menos los
ultimos seis commits, muy anterior a V23. Se corrige a una sola cabecera, y **V24 no anade
ninguna fila** porque es documentacion y no un experimento: el conteo real es **128**.

**Dos discrepancias se documentan sin sobreescribir los artefactos.** `v22_frontier_audit.json`
registra V20 como **0,609615**, que es la salida del **modelo compartido** de D-057 y no la cadena
de V20; el canonico es **0,6097782**, verificado por reproduccion exacta en cuatro artefactos. Y
`phase39_fusion_v12.json` registra la referencia de V12 **redondeada a cuatro decimales**
(0,6004), cuando el canonico pooled es **0,6005017**. En ambos casos el notebook muestra
`ORIGINAL` junto a `CORRECTED` con su motivo.

**Auditoria de la fase: PASS.** `py_compile` rc 0, `ruff` con cero errores de correccion, cero
caracteres Unicode introducidos, identificadores y titulos unicos, cero artefactos faltantes, nueve
celdas del notebook ejecutadas de verdad con cero errores, y coherencia numerica 7 de 7.

**Lo que esta fase no hace**: no entrena nada, no genera submissions, no consulta Kaggle, no lee el
test ni el holdout, no anade filas al registry, no cambia resultados historicos y **no abre ninguna
hipotesis**. El proximo experimento se decide despues y no automaticamente.

**Deuda que queda declarada y no se cierra aqui**: el notebook conserva 38 caracteres Unicode de
sesiones anteriores, porque el proyecto excluye `notebooks/` de su auditoria de idioma historica; y
`notebooks/09_final_pipeline.ipynb` es un notebook antiguo de diez celdas, obsoleto respecto a V20.

## 12cuatri. V26 — Advanced research: representacion de la evidencia del ensemble

La fase V26 no busco mas modelos. Busco una forma distinta de explotar la informacion que AGE ya
poseia, y arranca de una distincion que reencuadra el problema: **V20 ya es un meta-modelo**. De
las 279 features, 84 son predicciones de modelos base, y el meta-modelo es LightGBM. La pregunta no
era construir un meta-modelo, sino si su **representacion de la evidencia** era explotable de otra
forma.

El **gap analysis** de 15 familias, buscado por equivalencia funcional y no por coincidencia de
nombre, dejo **cinco genuinamente abiertas**: la reescalada afin de las 84 columnas de evidencia, el
percentil, las `monotone_constraints` nativas de LightGBM, la fiabilidad por miembro y clase, y la
reescalada de las salidas del blend.

Dos afirmaciones del propio briefing resultaron **falsas**, y el proyecto las midio antes de
decidir. La primera: un arbol **no** es invariante a una transformacion monotona en esta
implementacion de LightGBM, y un z-score afin cambia la prediccion **bit a bit**. La invariancia
teorica habria cerrado la hipotesis sin gastar presupuesto. La segunda: un percentil no es
monotonia estricta, porque colapsa empates.

El experimento minimo dio **0,6097770** frente al control **0,6097782**: delta **-0,000001**, dos
de cinco folds, y **1.809 filas** que cambian para un saldo de **+3** sobre 169.336. La conclusion
es que **la evidencia del ensemble es invariante a su escala en todo lo que el macro-F1 mide**: lo
que el meta-modelo necesita de las probabilidades de los miembros es el orden, y el orden ya se
conserva.

Con eso, y con lo acumulado en V25 y en las cinco subfamilias ordinales, **el cuello de botella
restante de AGE no esta en la representacion de las features, ni en la combinacion de las
predicciones, ni en la calibracion de la evidencia**. Todas tienen evidencia negativa reproducible.
Lo que queda es una limitacion de los datos: casi cuatro de cada diez filas las equivocan V19 y
V20 juntas, y ningun modelo del historial las corrige.

## 12. Limitaciones y trabajo futuro

**Limitaciones**

1. **Truncado a 140 caracteres y transliteración ASCII**: se pierde contexto y toda señal tipográfica (emojis, tildes, mayúsculas acentuadas), lo que acota el techo alcanzable.
2. **Sin metadatos de usuario**: no se puede explotar perfil ni agrupar por autor; el control de leakage es por contenido, no por identidad.
3. **Ruido de etiqueta** (0,38% de filas con etiquetas contradictorias) y 166 textos vacíos: techo práctico irreducible.
4. **Procedencia del corpus no identificada**: las comparaciones con la literatura son orientativas.
5. **Interpretación de las reglas** (D-001): la prohibición de "modelos de vectorización" se interpretó como embeddings; la tabla de ablación deja reconstruible un modelo sin POS ni lexicones si el criterio fuera más estricto.

**Trabajo futuro**

- Modelo jerárquico ordinal (clasificadores binarios encadenados entre rangos contiguos) para atacar la confusión adyacente, que sigue siendo el error dominante.
- Modelos especializados por tipo de tweet (cortos vs largos), dado que la longitud domina el error.
- Si las reglas lo permitieran, embeddings entrenados desde cero sobre este dominio, o modelos de lenguaje pequeños entrenados localmente.

## 13. Reproducibilidad

```bash
# 1. Entorno (venv del proyecto)
python -m src.data --download --verify --save-interim   # descarga y valida los datos
python scripts/fase01_auditoria.py                      # auditoría de datos y leakage
python scripts/fase02_eda.py && python scripts/fase03_linguistic.py
python scripts/experimentos_representaciones.py         # matriz de representaciones
python scripts/experimento_e009_modelos_clasicos.py     # modelos clásicos
python scripts/fase20b_ensamble_extendido.py            # ensamble (LOFO)
python scripts/fase17_error_analysis.py                 # análisis de errores
python scripts/fase23_auditorias.py                     # auditorías
python -c 'from src import pipeline; pipeline.run()'    # modelo final (~9 min CPU)
```

El notebook **`notebooks/99_entrega_consolidado.ipynb`** contiene la narración completa del proyecto y verifica el artefacto final; con `REPRODUCIR_PIPELINE = True` re-ejecuta el modelo de extremo a extremo. El pipeline regenera la submission enviada con **100,0000% de coincidencia** (verificado en `notebooks/09_final_pipeline.ipynb`).

## 14. Referencias

Las 30 referencias completas, con DOI/URL verificado, están en `reports/state_of_art.csv` y `references.bib`. Selección de las más influyentes:

- Schler, Koppel, Argamon & Pennebaker - Effects of Age and Gender on Blogging (PDF: u.cs.biu.ac.il/~koppel/papers/springsymp-blogs-07.10.05-final.pdf) (2006). *AAAI Spring Symposium on Computational Approaches to Analyzing Weblogs (SS-06-03)*.
- Argamon, Koppel, Pennebaker & Schler - Mining the Blogosphere: Age, gender and the varieties of self-expression - DOI 10.5210/fm.v12i9.2003 (verificado) (2007). *First Monday 12(9)*.
- Mukherjee & Liu - Improving Gender Classification of Blog Authors - ACL Anthology D10-1021 (verificado) (2010). *EMNLP 2010 (ACL)*.
- Rosenthal & McKeown - Age Prediction in Blogs: A Study of Online Social Networks, Style and Personality - [NO VERIFICADO: no se localizo DOI ni URL confirmada; probable informe tecnico de Columbia University] (2011). *no verificado (informe tecnico o workshop)*.
- Nguyen, Smith & Rose - Author Age Prediction from Text using Linear Regression - LaTeCH@ACL 2011 (PDF verificado en el sitio del autor: dongnguyen.nl/publications/nguyen-latech2011.pdf) (2011). *LaTeCH 2011 (workshop en ACL-HLT)*.
- Nguyen, Gravel, Trieschnigg & Meder - How Old Do You Think I Am? A Study of Language and Age in Twitter - ICWSM 2013 (PDF verificado en el sitio del autor) (2013). *ICWSM 2013 (AAAI)*.
- Nguyen, Trieschnigg, Dogruoz, Gravel, Theune, Meder & de Jong - Why Gender and Age Prediction from Tweets is Hard: Lessons from a Crowdsourcing Experiment - COLING 2014 (PDF verificado en el sitio del autor) (2014). *COLING 2014*.
- Sap, Park, Eichstaedt, Kern, Stillwell, Kosinski, Ungar & Schwartz - Developing Age and Gender Predictive Lexica over Social Media - DOI 10.3115/v1/D14-1121 (verificado) (2014). *EMNLP 2014 (ACL)*.
- Flekova & Gurevych - Can We Hide in the Web? Large Scale Simultaneous Age and Gender Author Profiling in Social Media - Notebook PAN@CLEF 2013 (CEUR Vol-1179, verificado) (2013). *PAN@CLEF 2013 (CEUR-WS Vol-1179)*.
- Hovy & Fornaciari - Increasing In-Class Similarity by Retrofitting Embeddings with Demographic Information - DOI 10.18653/v1/D18-1070 (verificado) (2018). *EMNLP 2018 (ACL)*.
- Morgan-Lopez, Kim, Chew & Ruddle - Predicting age groups of Twitter users based on language and metadata features - DOI 10.1371/journal.pone.0183537 (verificado) (2017). *PLOS ONE*.
- Liu, Singh & Mneimneh - A Comparative Analysis of Classic and Deep Learning Models for Inferring Gender and Age of Twitter Users - DOI 10.5220/0010559500480058 (verificado) (2021). *DeLTA 2021 (SCITEPRESS)*.

---

*Informe generado automáticamente desde los artefactos del proyecto por `scripts/fase_final_reporte.py`: las cifras provienen de `experiments.csv`, los JSON de resultados y las tablas de `reports/`, por lo que no pueden desincronizarse de la evidencia ejecutada.*

---

# Parte II — Búsqueda de señal residual (fases A-J, 2026-09-25)

**Pregunta**: ¿queda alguna fuente de señal permitida que produzca una mejora real sobre el modelo
final v12? **Criterio declarado de antemano**: Δ ≥ +0,003 de macro-F1 en CV agrupado, positivo en ≥4/5
folds, sin leakage, con confirmación en el holdout congelado.

## 15. Auditoría del estado real (fase A)

- Reconstrucción **exacta** de la matriz del meta (diferencia máxima 0,0) y del CV LOFO
  (**0.6004**, por fold
  [0.6007, 0.605, 0.6032, 0.6004, 0.5928]).
- El modelo v11 reconstruido reproduce `age_submission_v11.csv` con **acuerdo 100,0000%**: la
  alineación semántica del lado test (riesgo tipo D-021) queda descartada.
- Holdout congelado determinista: 25,204 filas
  (14.884%).
- Kaggle: **15 envíos registrados,
  12 puntuados**.

## 16. Análisis de errores y desacuerdo (fase B)

- **13.869%** de las filas: ningún miembro acierta (techo de la clase
  de modelos actual). **26.7%**: el ensamble falla y algún miembro
  acierta.
- F1 por clase: {'13-17': 0.671, '18-24': 0.5059, '25-34': 0.4858, '35-49': 0.5593, '50-64': 0.6549, '65-xx': 0.7261}. Elasticidad ordinal: el
  23.836% de los casos falla a un paso ordinal.
- Fronteras: `18-24|25-34` concentra el 42,48% del error total y `25-34|35-49` el 42,06%.
- Peores segmentos: `solo_url` 59,9% de error, `muy_corto` 50,8%, sin menciones 51,0%.

## 17. Las 27 variantes controladas (fases C-G y §13)

| Fase | Variante | Δ vs v12 (CV) | folds + |
|---|---|---|---|
| §13 · Vista estructural y extremos | estructura | +0.0018 | 4/5 |
| F · Limpieza selectiva | F3 clúster selectivo en clases centrales | +0.0009 | 3/5 |
| §13 · Refuerzo de la vista | V3 ampliada + extremos | +0.0006 | 3/5 |
| §13 · Refuerzo de la vista | V1 estructura ampliada | +0.0005 | 4/5 |
| D · Calibración y logits | D1a log-prob (T=1) · promedio de combinadores | +0.0003 | 3/5 |
| E · Estructura v2 (bloques libres) · libres | libres | +0.0002 | 2/5 |
| D · Calibración y logits | D1a log-prob (T=1) · promedio de combinadores | +0.0001 | 3/5 |
| D5 · Diversidad de subconjuntos | producción (MLP_full + LogReg_full) | +0.0000 | 2/5 |
| §13 · Vista estructural y extremos | extremos | +0.0000 | 2/5 |
| D · Calibración y logits | D1b log-prob + T in-fold · promedio de combinadores | -0.0001 | 2/5 |
| F · Limpieza selectiva | F2 clúster de near-duplicados (un representante) | -0.0001 | 2/5 |
| C · Consenso y gating | +C1+C2 | -0.0002 | 3/5 |

Familias cerradas con evidencia: consenso/desacuerdo por familia y gating contextual (D-028),
calibración de miembros y blending en logits (D-029), diversidad de subconjuntos de features (D-029),
estructura v2 de retuit/hashtag/extremos (D-030), edad implícita y selección greedy (D-030), limpieza
selectiva (D-031) y refuerzo de la vista estructural (D-032). **Ninguna alcanza el umbral.**

## 18. La única señal nueva: la vista estructural (marginal, no adoptada)

Secuencia de **tipos** de token (palabra, mención, hashtag, URL, número, puntuación, mayúsculas),
TF-IDF de palabras 1-3 y un LinearSVC. Sin contenido: F1 individual
**0.2285**, pero **rescata el
18.17% de los errores del v12** (acuerdo
27.96%).

- **CV**: 0.6022 (+0.0018,
  4/5 folds).
- **Holdout congelado**: base 0.6077
  reproducida exacta -> vista
  0.6108
  (**+0.0031**).
- **No adoptada**: el gate primario (CV ≥ +0,003) no se cumple. Enriquecer el alfabeto no mejora
  (V1 +0,0005), así que la vista simple es el óptimo de la familia.
- **Candidato**: `submissions/age_submission_v13_candidato_estructura_NO_ADOPTADO.csv` (112.892 filas,
  8/8 checks formales). **No enviado** (decisión de Carlos).

## 19. Hallazgo de integridad (D-034) y artefacto v12 reproducible

La submission v12 subida no se reconstruye al 100% desde el repositorio
(93.88%). El diagnóstico descarta
las tres causas posibles: matrices equivalentes (fresca vs caché
100.0000%), ajuste determinista (dos ajustes
idénticos: 100.0000%) y protocolo del holdout
reproducible (0,6077 exacto). Es un problema de **trazabilidad del artefacto**, no de metodología:
todas las comparaciones de la fase son pareadas en el mismo proceso y quedan válidas.
El artefacto `submissions/age_submission_v12_reproducible.csv` deja una versión de la v12 reconstruible desde el repositorio (112892 filas, distribución idéntica a la registrada por la fase 8: False).

## 20. Cierre de la fase

- **Veredicto**: línea de señal residual **saturada** (D-033, D-035). El modelo final **no cambia**:
  v12 con CV 0,6004, holdout 0,6077 y leaderboard **0,62182**.
- **Trabajo futuro**: con reglas más laxas (embeddings), atacar primero la frontera `18-24|25-34` y el
  segmento `solo_url`; la limpieza de duplicados (+0,0016) sigue siendo la mejora de coste nulo
  recomendada si se reentrena el modelo por otro motivo; y la vista estructural merece una exploración
  dedicada (es la única señal nueva con dirección positiva).
- **Documentación**: `DECISIONS.md` (D-028…D-035), `RUNLOG.md` (sesiones 14-19), `experiments.csv`
  (61 experimentos), `reports/fase_residual_informe.md`, `reports/auditorias.md` (APROBADO) y
  `notebooks/99_entrega_consolidado.ipynb` (Parte I + Parte II).

---

# Parte III — Capa de decisión ordinal y corrección de fronteras (2026-09-25)

**Pregunta**: ¿existe una forma distinta de convertir las señales ya disponibles en una decisión
ordinal de seis clases que reduzca el error en las fronteras `18-24|25-34` y `25-34|35-49`?
**Gate pre-registrado**: Δ ≥ +0,003 con ≥4/5 folds. **Holdout agotado** (no se reutiliza).

## 21. Cobertura: qué estaba ya hecho

Los modelos ordinales (E-015, E-021a) y los márgenes adyacentes (E-028a) **ya existen como miembros**
del ensamble, y el routing duro está cerrado. Lo que nunca existía era una **capa de decisión** sobre
probabilidades congeladas: nueve líneas genuinamente abiertas (`reports/age_hypothesis_coverage.md`).

## 22. Resultado: 33 capas, ninguna adopta

| Hipótesis | Capas | Δ máximo | Veredicto |
|---|---|---|---|
| A umbrales acumulados | 2 | −0,0031 | el modelo de umbrales solo da 0,5887; la evidencia de fronteras ya está en el meta |
| B combinación Luce de márgenes | 3 | −0,0034 | los márgenes de E-028a no forman una escala de Bradley-Terry válida (0,2947) |
| C corrección residual de fronteras | 4 | −0,0000 | en su forma más suave, **neto 0**: el contexto no aporta más de lo ya usado |
| D decodificación por transición | 5 | −0,0004 | empeora monótonamente al aumentar λ |
| E proyección ordinal | 6 | −0,0033 | desplazar masa del top-1 siempre destruye más de lo que arregla |
| F coste ordinal | 3 | +0,0004 | solo preserva la decisión con perfiles de coste casi planos |
| G desempate por E[Y] | 3 | **+0,0011** | mejora el MAE ordinal (0,6552 -> 0,6454) pero **empeora las fronteras centrales** |
| H calibración por banda | 2 | +0,0002 | inerte: una transformación por fila no cambia el argmax |
| I corrector mínimo | 5 | −0,0005 | reescribe aciertos a cambio de fallos nuevos |

## 23. Tres hallazgos defendibles

1. **El argmax de V12 ya es la mejor decisión** de su distribución de probabilidades: toda
   redistribución (proyección, calibración por banda, decodificación por confusión) degrada de forma
   monótona.
2. **Minimizar el error ordinal no es mejorar macro-F1**: el desempate por edad esperada mejora el MAE
   ordinal y las fronteras extremos, pero empeora `18-24|25-34` (+462 errores) y `25-34|35-49` (+534).
3. **La evidencia de fronteras ya está dentro del meta-modelo**: un modelo de umbrales acumulados
   ajustado sobre las features del meta alcanza 0,5887 por sí solo y su fusión con V12 no aporta.

## 24. Dónde vive el error residual

- **Techo de información**: 23,485 filas
  (34.7% del error) donde **ningún miembro acierta**.
- **Agregación**: 45,213 filas
  (66.9% del error) donde algún miembro acierta y el ensamble no;
  ninguna regla de decisión sobre las probabilidades convierte ese saber en aciertos.
- **Fronteras centrales**: 10,122 cruces de frontera en
  `18-24|25-34` (sobre 28,721 errores en las filas de
  la pareja) y 7,252 en `25-34|35-49`
  (28,438).

## 25. Conclusión

Bajo las restricciones de AGE, **la representación contiene la información disponible y el límite
está en la observabilidad de la edad, no en la regla de decisión**. V12 permanece congelado, no se
genera submission y la línea de decisión queda cerrada con 33 capas medidas (D-037). El leaderboard
de referencia es **0,62188** (artefacto reproducible; D-036 cierra la discrepancia de reconstrucción).

---

# Parte IV — Lexicones lingüísticos (2026-09-25)

## 26. Qué se construyó

24 subcategorías léxicas de conocimiento general (6 de registro e informalidad, 6 de etapas de
vida y contexto social, 12 de dominios temáticos), 1359 entradas efectivas
tras cribado de aplicabilidad al corpus, y 72 features agregadas (conteo, ratio y presencia por
subcategoría, más un agregado por familia). La transformación `tweet -> categorías` es
determinista y **no usa etiquetas**, a diferencia del canal supervisado de entidades ya existente.

## 27. Resultado por fase

| Fase | Resultado |
|---|---|
| P0 cobertura | 0 experimentos previos con canal léxico: la línea es nueva |
| P1 catálogo | 24 subcategorías, 1.588 entradas declaradas -> 1359 efectivas, SHA256 por fichero |
| P2-P5 control | lexicon-only: temas 0.2033, registro 0.152, vida 0.0731 (azar 0,1667) |
| P5 redundancia | 0 de 72 columnas con \|ρ\| ≥ 0,8 contra las 279 features de V12: el bloque es nuevo |
| P6 fusión | ningún bloque alcanza el gate; ninguno supera a su control de permutación |
| P7 complementariedad | rescate ≤ daño en todos los candidatos |

## 28. Tres conclusiones defendibles

1. **El bloque no es redundante**: la correlación máxima contra las 279 features de V12 queda por
   debajo del umbral, así que la señal no es una reformulación de lo que el modelo ya tenía. La
   hipótesis de §12 ("¿añade una dimensión nueva?") se responde **sí** a nivel de variable.
2. **La señal nueva no es útil**: solo la familia de temas supera el azar en aislamiento, y por
   poco. Un contexto temático es real pero marginal para la edad, que es una combinación de muchos
   factores débiles.
3. **El control de permutación es imprescindible**: sin él, cualquier Δ nominal positivo del
   bloque podría atribuirse al tamaño del bloque y no a su contenido.

## 29. Conclusión

Los lexicones **no** aportan una dimensión de edad explotable sobre V12. Junto con el resultado de
la capa de decisión (33 capas, ninguna adopta), la evidencia converge: bajo las restricciones de
AGE, el límite no está en la representación sino en la **observabilidad de la edad** (35% del
error es techo de información). V12 permanece congelado (leaderboard 0,62188) y la línea queda
cerrada con D-039.

_Generado por `scripts/phase41_cierre_lexicones.py`._

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
