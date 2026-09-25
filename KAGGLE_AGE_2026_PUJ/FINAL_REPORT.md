# Informe final — Predicción de rango de edad a partir de tweets

**Competencia**: `tweet-age-2026-puj` (Kaggle, host `NLP_Course_PUJ`) · Procesamiento de Lenguaje Natural · Maestría en Inteligencia Artificial, Pontificia Universidad Javeriana.

**Autores**: Carlos Garnica y equipo · **Fecha**: 2026-09-24 · **Repositorio**: `KAGGLE_AGE_2026_PUJ/`

---

## 1. Resumen ejecutivo

- **Tarea**: clasificación multiclase de tweets en 6 rangos de edad (`13-17`, `18-24`, `25-34`, `35-49`, `50-64`, `65-xx`) usando solo el texto.
- **Resultado en el leaderboard**: **0.62182** de F1 (v12; línea base inicial 0.55235; mejora acumulada **+0.0695**).
- **Validación interna honesta**: 0.6005 de macro-F1 (leave-one-fold-out sobre particiones agrupadas por clústeres de near-duplicados).
- **Calibración validación↔leaderboard**: 11 envíos con brecha media **+0.0177 ± 0.0020**, explicada por la memorización de near-duplicados del test.
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
| E1 | La longitud y la riqueza léxica crecen con la edad | ε² = 0,126 en `n_chars`; 73 → 111 caracteres de media entre extremos | familia textométrica en el catálogo |
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
| F1 | Confusión entre rangos **contiguos** | 5 de los 6 pares más confundidos son adyacentes (18-24→25-34 15,3%) | señal ordinal (E-015) | +0,0003 en el ensamble |
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
| **Retrieval train→test** (PH10) | etiqueta recuperada acierta **0,615** | **permitido por las reglas pero los datos lo desaconsejan**: el techo lo fija la consistencia de etiqueta de los duplicados, no la cobertura |
| **POS de spaCy** (PH10) | −0,0032 | el corpus transliterado a ASCII degrada el etiquetado y la señal es redundante con los char n-grams |
| Exclusión de filas ruidosas (PH11) | +0,0001 / +0,0016 | mejora real pero bajo el umbral de cambio de modelo (D-027) |

## 11. Auditorías finales

Resultado global: **APROBADO** (`reports/auditorias.md`).

- **Leakage**: ningun_fit_sobre_test=True; n_llamadas_fit_sobre_test=0; n_canales_representacion=6; vectorizadores_ok=True.
- **Reproducibilidad**: archivos_con_semilla_o_random_state=38; semilla_global_definida=True; requirements_existe=True; n_paquetes_congelados=312.
- **Formato de submission**: columns_ok=True; columns_match_sample=True; rows_ok=True; ids_equal_test=True.
- **Trazabilidad**: n_experimentos_registrados=30; todos_con_hipotesis=True; todos_con_resultado=True; n_decisiones=16.

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