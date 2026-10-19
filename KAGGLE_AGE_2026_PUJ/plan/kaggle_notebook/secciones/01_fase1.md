# FASE 1 — Comprensión del problema

## 1.1 La competición, la tarea y la métrica

La competencia plantea un problema de clasificación de texto supervisada: dado el texto de un tuit en
español, predecir a cuál de **seis rangos de edad ordenados** pertenece su autor (`13-17`, `18-24`, `25-34`,
`35-49`, `50-64`, `65-xx`). El corpus oficial tiene **169.336 tuits de entrenamiento** y **112.892 de
prueba**; el texto es corto (mediana de 93 caracteres) y está normalizado a ASCII (sin tildes ni eñes), lo
que elimina parte de la señal morfológica del español.

La métrica oficial del leaderboard es **F1 Score**, que en esta competencia coincide con la **macro-F1
sin ponderar sobre las seis clases**, la misma que el proyecto adoptó como métrica de trabajo: es la
lectura más exigente y estable para comparar modelos, porque no deja que las clases mayoritarias oculten
los errores en las minoritarias.
Todas las decisiones de adopción se tomaron sobre esta métrica en validación cruzada, y la calibración
posterior contra el leaderboard (sección 5.5) confirmó que es un estimador conservador y bien comportado
del puntaje real.

La competición puntúa sobre dos particiones del test: un **leaderboard público** (una fracción del test)
durante el torneo y un **leaderboard privado** (el resto) al cierre. El proyecto decidió desde el inicio
no usar el leaderboard como herramienta de búsqueda — ni para elegir modelos ni para calibrar umbrales —
porque eso sobreajusta la partición pública y castiga el resultado privado.

## 1.2 Restricciones inviolables

El brief del curso y las indicaciones del profesor fijaron restricciones que condicionaron cada decisión
técnica del proyecto:

| Restricción | Alcance en este proyecto |
|---|---|
| **Sin transformadores ni LLM** | Ningún modelo de atención (BERT, RoBERTa…) ni modelo generativo, tampoco como extractor de características. |
| **Sin embeddings** | Ni preentrenados (Word2Vec, GloVe, FastText) ni entrenados desde cero. Toda representación es estadística o de ingeniería léxica. |
| **Sin datos ni etiquetas externas** | Solo el corpus de la competición. Ningún recurso etiquetado de terceros. |
| **Sin metadatos** | El identificador nunca se usó como característica; no hay fuga temporal ni de autor. |
| **Test intocable** | El archivo de prueba jamás se usó para ajustar vectorizadores, modelos, selección de variables ni umbrales. |

Estas restricciones no son un detalle administrativo: explican por qué el proyecto invirtió tanto esfuerzo
en **representaciones clásicas** (TF-IDF (Salton y Buckley, «Term-Weighting Approaches in Automatic Text Retrieval», Information Processing & Management, 1988) de palabras y de caracteres (Sapkota, Bethard, Montes y Solorio, «Not All Character N-grams Are Created Equal: A Study in Authorship Attribution», NAACL-HLT 2015), rasgos estilométricos, canales
léxicos (Sap et al., «Developing Age and Gender Predictive Lexica over Social Media», EMNLP 2014)) y en **combinadores** de modelos lineales y de árboles, en lugar de recurrir al camino corto de
un modelo preentrenado (Liu, Singh y Mneimneh, «A Comparative Analysis of Classic and Deep Learning Models for Inferring Gender and Age of Twitter Users», DeLTA 2021).

## 1.3 Gobernanza experimental

El proyecto se condujo con disciplina de pre-registro: **cada experimento se definió y congeló antes de
ejecutarse** (pregunta, diseño, criterio de decisión y presupuesto), y solo después se corría. Un
experimento se adoptaba únicamente si superaba un **gate** estricto, que combinaba varias condiciones:

1. **Mejora en validación agrupada:** delta positivo sobre el campeón vigente, medido en la partición
   canónica de 5 folds agrupados por clústeres de near-duplicados (sección 2.4; Kaufman, Rosset, Perlich y Stitelman, «Leakage in Data Mining: Formulation, Detection, and Avoidance», ACM Transactions on Knowledge Discovery from Data, 2012).
2. **Consistencia entre folds:** la mejora debía aparecer en la mayoría de los folds, no en uno solo.
3. **Significancia estadística:** intervalo de confianza del 95 % por *bootstrap* (Efron, «Bootstrap Methods: Another Look at the Jackknife», The Annals of Statistics, 1979) pareado que excluyera
   el cero, y verificación en una **segunda partición** con semilla distinta.
4. **Control negativo:** cada línea de investigación incluía un gemelo de control (por ejemplo, el mismo
   modelo con las filas o las variables barajadas) para separar señal real de ruido de selección.

Cuando una línea no superaba el gate, se **cerraba con evidencia** y quedaba registrada — el proyecto
acumuló decenas de cierres documentados, que son parte central del valor del trabajo: los fracasos
medidos son lo que permitió no repetir caminos y concentrar el esfuerzo en lo que sí aportaba.

Cada adopción quedó registrada en un **acta de decisión** (con contexto, decisión, alternativas y
consecuencias) y cada corrida en un **registro experimental** con su configuración y su resultado. Este
notebook reconstruye esa cadena completa y verifica que los números reproducidos coinciden con los
documentados.
