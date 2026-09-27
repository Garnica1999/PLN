# PROJECT_STATE — KAGGLE_AGE_2026_PUJ

> Memoria persistente del proyecto. Leer SIEMPRE al inicio de sesión o tras compactación.
> Última actualización: 2026-09-26 (cierre de V13 ADVANCED RESEARCH).

## Entrega del trabajo (2026-09-25)
- **Commit y merge a `main`**: PR **#26** (`kaggle-age-2026-puj`) fusionado. Commit `3a518ac`,
  merge `27238e7`. 255 archivos versionados; datos, features y modelos quedan fuera de git por
  `.gitignore`.
- **Notebook consolidado regenerado** (`99_entrega_consolidado.ipynb`): narra las 12 fases con
  **aserciones de verificación** contra los artefactos, de modo que las cifras no pueden
  desincronizarse de lo ejecutado (0 errores al ejecutar).
- **Informe final** regenerado (252 líneas) y auditorías **APROBADAS** en las cuatro dimensiones.
- **INCIDENCIA DE SEGURIDAD**: el token de GitHub se compartió en el chat; debe **revocarse y
  regenerarse** en https://github.com/settings/tokens (no quedó escrito en ningún archivo del repo,
  verificado con `grep`).

## V14 (2026-09-26) CERRADA — relajar §3 no cambia el mecanismo (D-044)
- **D-042**: gate revisado. El efecto baja a **≥ +0,001** y los folds a **≥3/5**, pero el IC95 del
  bootstrap pareado **sigue teniendo que excluir el cero** y se añade la **segunda partición** (≥3/4
  folds de una partición de 4 agrupados, independiente). Con 169.336 filas un Δ de +0,001 da un IC95
  de ±0,0015, así que el listón de efecto no puede rebajarse sin el de significación.
- **D-043**: §3 relajado para permitir un recurso lingüístico externo: el analizador español
  `es_core_news_md` de spaCy (no transformer, no LLM, no embeddings de clasificación, sin etiquetas).
- **V14a**: 37 variables nuevas (morfología verbal y nominal, lemas, entidades, sintaxis). Control
  **LogReg 0,2409 / LinearSVC 0,2270**: el **mejor bloque aislado del proyecto**, por encima de los
  lexicones (0,2033), los pronombres (0,1918) y la modalidad (0,1270). Cribado: 4 de 37 redundantes, y
  son las de longitud y diversidad, como P0b predijo para la textometría.
- **V14b**: fusión con 33 columnas -> **0,5990, Δ −0,0015, 1/5 folds**. Falla 5 de los 6 criterios.
  Los tres controles permutados son **peores** que la candidata (señal real), y la **segunda
  partición da 0/4 folds con media −0,0031**. La frontera `18-24|25-34` empeora de 0,4963 a 0,4919.
- **Conclusión de alcance**: **la restricción de recursos no era el cuello de botella**. Las mejores
  variables disponibles degradan V12 de forma reproducible en dos particiones. El cuello es
  **arquitectónico**, y el gate revisado quedó validado por su primer uso: aceptó distinguir señal de
  ruido (el bloque supera a sus controles) y rechazó al candidato.
- **Decisión**: cerrar V14. No abrir la lista léxica externa de una sola función: el precedente dice
  que otro bloque del mismo tipo daría lo mismo.

## V13 ADVANCED RESEARCH (2026-09-26) CERRADA — 0 adopciones (D-041)
- **16 experimentos, 8 hipótesis pre-registradas, 0 candidatos adoptados.** V12 sigue congelado.
- **Seis pistas de variables cerradas**: P1 evidencia por token (control 0,3997, el doble del mejor
  bloque léxico; fusión −0,0002 con 3/5 folds y controles permutados indistinguibles), P17 pronombres
  (0,1918; rescate neto −64.075), P20 modalidad (0,1270, bajo el azar; 1,6 % de cobertura), P19 árboles
  de conjunciones (0,1539, bajo el azar; 0,64 conjunciones por documento), y las dos líneas anteriores
  (D-037 capa de decisión, D-039 lexicones).
- **P3a, el especialista binario de frontera, empeora**: 0,4963 -> 0,4944 y 0,4974 -> 0,4942; global
  0,6005 -> 0,5993. No es la línea de D-037: aquí se entrena un modelo nuevo con sus propios datos.
- **Tres mecanismos medidos que explican todos los fracasos**: (1) saturación del espacio de variables
  (P0b: VIF mediano 18,87, 170 variables por encima de 10, 87 de 278 direcciones para el 95 % de la
  varianza); (2) información nueva inaccesible al combinador (P16c: el R2 de la reconstrucción desde
  V12 es 0,522 pero su control de clasificación es 0,0485, y P16d muestra que el máximo de 90 puertas
  por confianza es 0,6005 con 0 filas desviadas); (3) el modelo está en el límite de su espacio (P3a).
- **P0b, auditoría de idioma y D-040**: la matriz completa de 279 variables tiene 15 duplicados
  exactos, 401 pares con |rho| >= 0,80 y 1 columna constante; `numpy 1.26.4` sobre `Python 3.14.7`
  calcula mal `X @ X.T` (D-040), sin impacto en ninguna métrica del proyecto, con defensa permanente
  en `src/verificacion_numerica.py`. Todo el proyecto está en español y sin alfabetos ajenos.
- **Infraestructura que queda**: criba de redundancia contra la matriz completa (`criba_redundancia`),
  autocomprobación numérica del entorno y protocolo de pre-chequeo de rescate, que evitó pagar unos
  70 minutos de CPU en dos pistas.
- **Entregables**: `reports/v13_final_research.md`, `reports/v13_reproducibility.md`,
  `reports/v13_hypothesis_registry.csv` (las 8 hipótesis falsadas), `reports/v13_master_results.csv`,
  `reports/v13_candidate_comparison.csv`. `experiments.csv` con 89 experimentos.
- **Lo único que queda abierto es una decisión de alcance**, no un experimento: relaxing §3 para
  permitir un recurso lingüístico externo, o cambiar la métrica y redefinir el gate §24.

## V15 · CORRELACION DE ENTRADA Y PODA (2026-09-26) CERRADA — la redundancia es funcional (D-045)
- **Petición de Carlos**: medir qué variables de entrada están correlacionadas entre sí, cuáles se
  asocian con la salida, y **eliminar las redundantes**.
- **V15a (estructura y asociación, in-fold sobre las 312 variables = 279 de V12 + 33 de V14)**:
  2 columnas constantes, **402 pares** con |rho| >= 0,80, **245** con |rho| >= 0,90, **15
  duplicados exactos**, **58 clusters que cubren 171 de 310 variables**. Asociación con la salida
  por eta cuadrado: mediana 0,0306, máximo 0,5599, y **112 variables con eta2 < 0,01 (110 de las 310 analizables; el recuento de 112 sale sobre las 312 columnas de la matriz, e incluye las 2 constantes, que son 0 por definicion)**. La
  estructura se calculó **dentro de cada fold** y el índice de variables conservadas se guarda por
  fold, porque elegir variables por su correlación con la etiqueta sobre todo el train sería una
  fuga de información.
- **La redundancia es perfectamente reproducible**: el **Jaccard de la estructura de pares entre
  folds es 1,0000** (mínimo 1,0000), así que podar no es una decisión dependiente de la partición.
- **V15b (poda evaluada con el gate de D-042)**: un representante por cluster, el de mayor eta2
  dentro del fold → **197 variables**. Macro-F1 **0,5965** frente a 0,6005 de V12: **delta -0,0040
  con 0 de 5 folds positivos**; segunda partición **0 de 4** con media -0,0065; IC95 pareado
  [-0,0054; -0,0027]. **NO ADOPTAR**.
- **Lectura**: las variables casi idénticas no sobran; funcionan como un conjunto de estimadores
  débiles parcialmente independientes cuyo promedio reduce la varianza. La colinealidad afecta a la
  **interpretación** de los coeficientes, no a la predicción, que es lo que P0b insinuaba y V15
  confirma con evaluación completa en vez de una inferencia.
- **Entregables**: `reports/v15_correlacion.md`, `reports/v15_poda.md`,
  `reports/v15_correlacion_entradas.csv`, `reports/v15_clusters.csv`,
  `reports/v15_asociacion_salida.csv`, `reports/v15_estabilidad.csv`,
  `experiments/v15a_correlacion.json`, `experiments/v15b_poda.json`, `features/v15b_oof.npy`,
  `features/v15_conservadas_fold{0..4}.npy`. `experiments.csv` con 93 experimentos.

## V16 · ARQUITECTURA DEL META-MODELO (2026-09-26) CERRADA — en las fronteras sí, en la global no (D-046)
- **Petición de Carlos**: añadir el paso de la arquitectura del meta-modelo, después del de las
  variables. Con todos los diagnósticos de V13, V14 y V15 apuntando al meta-modelo como cuello de
  botella, la palanca que quedaba era la **forma de combinar** las dos cabezas, que V12 hace con un
  promedio fijo 0,5 LogReg + 0,5 MLP, una decisión de diseño de la primera versión que **nunca se
  optimizó**.
- **Tres arquitecturas sobre las mismas 279 variables y el mismo protocolo, sin variables nuevas**:
  | Arquitectura | macro-F1 | Delta | Folds positivos |
  |---|---|---|---|
  | A0 promedio fijo (producción) | 0,6005 | - | - |
  | A1 peso aprendido por fold | 0,6002 | -0,0003 | 0/5 |
  | A2 combinador logístico (apilamiento) | **0,6011** | **+0,0006** | **3/5** |
- **A1 queda descartado**: los pesos que elige por fold son 0,1 · 0,5 · 0,5 · 0,4 · 0,5, así que el
  0,5 de producción se elige solo en 3 de 5 folds. El promedio fijo no era una decisión
  desafortunada, era la correcta.
- **A2 es el primer delta positivo del proyecto**, en la partición de 5 folds (3/5) y en la segunda
  partición (3/4), pero el **IC95 pareado [-0,0003; +0,0015] contiene el cero** y el efecto queda
  bajo +0,001. **NO ADOPTAR**.
- **El hallazgo que sí queda**: las dos fronteras centrales mejoran de forma notable,
  `18-24|25-34` de 0,4963 a **0,5065** y `25-34|35-49` de 0,4974 a **0,5131**, es decir **+1,0 y
  +1,6 puntos**, tres a cinco veces el efecto global. El problema de este conjunto de datos está en
  las fronteras de edad, y ahí el meta-modelo tiene algo que aportar; lo que no tiene es magnitud
  suficiente para que la media global la registre.
- **Lección operativa**: una mejora de +0,0157 en la frontera `25-34|35-49` es **invisible** en la
  macro-F1 global. Todo trabajo futuro dirigido a las fronteras tiene que medir sobre ellas desde el
  principio, no como métrica secundaria.
- **Entregables**: `reports/v16_arquitectura.md`, `reports/v16_arquitectura_comparativa.csv`,
  `experiments/v16_arquitectura.json`, `features/v16_oof_*.npy`, y
  `scripts/v15_v16_tablas.py` (repara la tabla de fronteras de V15b y V16, sin tocar cifras).
  `experiments.csv` con **94 experimentos**. Auditoría de idioma: 0 errores.

## V17 · ESPECIALISTAS DE FRONTERA Y DECISION LOCAL (2026-09-26) CERRADA — sin adopciones (D-047)
- **Pregunta**: ¿la senal que V16 encontro en las dos fronteras centrales puede convertirse en
  ganancia global con especialistas entrenados para esas fronteras?
- **Respuesta: no, y el motivo esta medido.** La senal existe y es grande; lo que no existe es una
  forma de identificarla.
- **Referencia reconciliada**: 0,6004 y 0,6005 son **dos estimadores** del mismo macro-F1 (pooled
  0,6005017473 frente a media de folds 0,6004103057; diferencia 9,14e-05, efecto Jensen).
  **Canonico 0,6005 pooled.** A2 de V16 se reproduce.
- **La complementariedad local es real y grande**: el especialista de pares **ya existia** (E-028a)
  y saca **0,6864 y 0,7218** en las filas de cada frontera frente a **0,4963 y 0,4974** de V12.
  Resuelve 14.356 y 15.571 filas, rompe 3.519 y 2.872. Techo con enrutamiento: **0,7481 y 0,7726**.
- **El mecanismo**: los errores de V12 en la frontera son **errores confiados** (separacion 0,809
  cuando el especialista acierta frente a 0,539 cuando acierta V12), y solo el 12,7 % y el 8,0 % de
  las victorias del especialista caen dentro de la region de frontera. Por eso toda compuerta de
  ambiguedad falla: no hay banda de confianza que discrimine.
- **Ninguna formulacion funciona**: compuerta +0,0002 · regla conjunta +0,0000 · **P9** arbitro
  aprendido (47 columnas existentes, protocolo anidado) **-0,0001 y 0/5 folds** · **P5**
  combinador local **-0,0002, 1/5 folds, y las fronteras empeoran**. El techo con arbitro perfecto
  seria **+0,0107**, que supera el gate de adopcion: el problema nunca fue el techo.
- **Leccion accionable**: los margenes de pares **ya son entradas de V12** y V12 no los explota,
  cuando por si solos valen 0,6864 y 0,7218 en las filas de su frontera. Eso no se arregla con otra
  compuerta: **exige cambiar la funcion objetivo**, que hoy es macro-F1 sobre seis clases y no sobre
  las fronteras.
- **Entregables**: `reports/v17_final_research.md`, `v17_reproducibility.md`, `v17_compliance.md`,
  `v17_candidate_comparison.csv`, `v17_master_results.csv`, `v17_hypothesis_registry.csv`,
  `v17_gap_audit.{md,csv}`, `v17_member_boundary_matrix.csv`, `v17_error_intersection.csv`,
  `v17_boundary_analysis.csv`, informes de cada fase y **10 figuras**.
  `experiments.csv` con **102 experimentos**. Auditoria de idioma: 0 errores.
- **Pendiente declarado**: los resultados no se han volcado al notebook consolidado. Es trabajo de
  redaccion, no de medicion, y queda declarado en `reports/v17_compliance.md` en vez de darse por
  hecho.

## V18 · FUNCION DE ENTRENAMIENTO (2026-09-26) CERRADA — la senal esta disponible y no se cobra (D-049)

- **La premisa de la fase quedo invertida, y fue el hallazgo mas importante de V18.** P0 concluyo que
  la ventaja del especialista de frontera venia de la **representacion**, apoyandose en el 0,4944 que
  V13 P3a publico. **Ese numero no era la exactitud del especialista**: `filas_actuables: 57.019` es el
  recuento sobre todo el dataset, y dentro de las 57.023 filas del par solo 38.424 eran actuables
  mientras 18.599 quedaban congeladas con la prediccion de V12. La mezcla daba exactamente 0,4944.
  Ejecutando el codigo y las variables exactos de P3a, el especialista sobre las **279 variables de
  V12** saca **0,7136** y **0,7493**, mas que el **0,6864** y **0,7218** del TF-IDF disperso, y sin
  fuga (las 72 columnas supervisadas usan el mismo `make_folds` agrupado). **D-048**: la senal de
  frontera estaba en el propio espacio de variables del modelo todo el tiempo, y el combinador de
  V12 extrae 0,4963 de 0,7136 disponibles.
- **Cuatro vias de funcion objetivo, cuatro mecanismos distintos, cuatro negativos**:

  | Via | Mecanismo | Resultado |
  |---|---|---|
  | P1 | `sample_weight` en las tres clases centrales | nivel moderado 0,6001 (-0,0004, 3 de 5 folds); nivel fuerte con frontera hasta 0,5786 (**+0,082**) y global **0,5841** (-0,0164); intercambio de suma cero exacto: `18-24` +0,0170, `13-17` -0,0166 |
  | P3 | margen de adyacencia en el objetivo | frontera 1 **+0,0392** y frontera 2 **-0,0552** en la cabeza lineal (+0,0209 y -0,0268 en el combinado): estan acopladas por la clase `25-34` |
  | P4 | termino auxiliar **aditivo, cero parametros** | la frontera se mueve **+0,0038** como maximo en la cabeza global y +0,0017 en el combinado: dos ordenes de magnitud menos que P1 con el mismo numero de parametros |
  | P5 | destilacion del teacher, **cross-fitted** | mejora sobre su propio control **+0,0008**, 0 de 5 folds, y la frontera recupera el **0,8 %** de lo disponible |

- **P5 en detalle**: teacher = el especialista sobre las 279 variables (el de la misma representacion
  que el student, que es la condicion para que pueda expresar lo que se le ensena); `teacher_OOF`
  obtenido en cinco inner folds **agrupados por duplicados**, con cada ejemplo predicho por un teacher
  que no lo uso para entrenarse; predicciones del teacher sobre `outer_validation` **solo para
  diagnostico**; cuadrica de seis combinaciones de lambda y temperatura **seleccionada dentro del
  train exterior**; etiqueta verdadera siempre como objetivo principal, sin pseudo-etiqueta dura.
  Los seis brazos destilados superan a su propio control, luego **la senal no es inerte**, pero no
  llega al gate exploratorio ni de lejos.
- **La explicacion comun de los cinco negativos** (P1, P3, P4, P5 y el caso de V17) es una sola y es
  medible: **la macro-F1 de seis clases es una media, y la frontera es una minororia de la masa**.
  Todo mecanismo que la mejore pagando fuera de ella pierde; todo mecanismo que la mejore sin pagar
  fuera no consigue moverla. No es un problema de informacion (D-048 lo descarto) ni de arquitectura
  (P4 lo descarto al no anadir parametros y tampoco rendir).
- **Veredicto por opcion**: A. cuello del objetivo, **parcial** (mueve la frontera pero no la cobra);
  B. cuello de informacion, **falso**; C. cuello de arquitectura, **no**; D. el conocimiento del
  especialista se destila, **marginal** (0,8 %); E. la senal sigue siendo local, **cierto**;
  F. no queda senal explotable, **cierto en la practica**.
- **Lo que este trabajo recomienda no es un experimento**: cambiar la funcion objetivo **a la que se
  optimise**, no la que se informa, o trenear y seleccionar **sobre las fronteras directamente**.
- **Consecuencia que queda a mano y tiene nombre propio**: **V12 comprime la salida de cada miembro a
  seis numeros, y el especialista binario recupera 0,7136 de esas mismas seis columnas por miembro**.
  Esa compresion es en si misma un objeto de estudio.
- **Entregables**: `reports/v18_final_research.md`, `reports/v18_compliance.md`,
  `reports/v18_reproducibility.md`, `reports/v18_master_results.csv` (21 filas),
  `reports/v18_hypothesis_registry.csv` (9 hipotesis), `reports/v18_candidate_comparison.csv` (7
  candidatos), `reports/v18_gap_audit.md`, `reports/v18_p0_reproduccion.md`,
  `reports/v18_p1_peso_clases.md`, `reports/v18_p3_margen.md`, `reports/v18_p4_multiobjetivo.md`,
  `reports/v18_p5_destilacion.md`, `reports/v18_distillation.csv`, 7 figuras.
- **8 de 14 experimentos previstos, 0 adopciones.** P2, P6, P7 y P9 no ejecutados: la condicion del
  brief para P5 era que P1-P4 demostraran senal del objetivo, y no se cumplio.
- **V12 intacto**: 0,6005017473 pooled, 0,62188 en el leaderboard, sin submission nueva.

## Fase actual
- **FASE DE LEXICONES (2026-09-25) CERRADA — línea cerrada (D-039).** 24 subcategorías léxicas
  (registro, etapas de vida, dominios temáticos), 1.369 entradas efectivas y 72 features
  agregadas, construidas como función determinista del texto (sin etiquetas). Control *lexicon-only*:
  temas **0,2033**, registro **0,1520**, vida **0,0731** frente a un suelo de azar de 0,1667.
  Redundancia contra las 279 features de V12: **0 de 72 columnas**. Fusión con V12: **ningún bloque
  alcanza el gate** (Δ ≥ +0,003 con ≥4/5 folds) ni supera a su control de permutación; la
  complementariedad es nula o negativa (rescate ≤ daño).
- **Modelo final: V12 congelado.** Leaderboard de referencia **0,62188**. **No se generó ni se
  envió ninguna submission nueva.**
- **D-038**: `transformers`, `sentence-transformers` y `torch` están instalados en el venv, pero su
  uso en el camino de modelado es 0 (único import: detección de GPU). Se documenta sin maquillar.
- **Conclusión**: bajo las restricciones de AGE, la línea de representación léxica está agotada.
  El criterio que queda abierto es la **frontera** (18-24|25-34 y 25-34|35-49), no más señal.
- **Entregables**: `reports/age_lexicon_hypothesis_coverage.md`, `reports/age_lexicon_catalog.csv`,
  `reports/age_lexicon_ablation.csv`, `reports/age_lexicon_redundancy.csv`,
  `reports/age_lexicon_results.csv`, `reports/age_lexicon_complementarity.md`,
  `reports/age_lexicon_error_analysis.md`, `reports/age_lexicon_audit.md`, `lexicons/**`.

## Estado según el formato del brief §83
- **CURRENT BEST CV**: **0,6005** (LOFO) — candidato v12 (promedio MLP+LogReg sobre 279 features).
  Con limpieza de textos duplicados en el entrenamiento sube a **0,6020** (+0,0016, por debajo del
  umbral de cambio de modelo; D-027)
- **CURRENT BEST HOLDOUT**: **0,6077** (LogReg 0,5983 · MLP 0,6034 en el mismo holdout)
- **CURRENT BEST KAGGLE**: **0,62182** (submission v12)
- **CURRENT MODEL**: ensamble de **14 fuentes + meta-modelo MLP(64)** sobre **273 features**
  (78 scores + 35 densas + 87 estructuradas + 73 supervisadas)
- **CURRENT FEATURES**: `features/age_structured_features.parquet` (87 locales) y
  `features/age_supervised_features.npy` (73, calculadas in-fold)
- **NEW SIGNAL DISCOVERED**: menciones supervisadas `P(clase|mención)` (+0,0016), cuenta fuente de
  RT (+0,0007), rareza/novedad léxica (+0,0003); hashtags neutros (−0,0001)
- **FAILED EXPERIMENTS**: especialistas por longitud (+0,0000), BM25 (−0,0002), kNN sobre SVD,
  meta LightGBM (−0,0002), jerarquía gruesa (+0,0003 redundante)
- **NEXT EXPERIMENT**: ninguno. La frontera 18-24|25-34 se investigó y se cerró como **techo de información** (D-019): tres arquitecturas de especialización fallaron. Líneas cerradas con evidencia: incertidumbre/interacciones (+0,0002), selección de features (0,0000), NB-SVM como modelo (0,5641) aunque sí aporta como miembro (+0,0007).
- **PRESUPUESTO DE ENVÍOS**: 11 envíos aceptados por la plataforma (el límite de ~10 del
  supuesto inicial no se aplicó). La competencia conserva la mejor puntuación, así que enviar no
  perjudica; sí perjudica enviar candidatos sin confirmación previa.
- **RULE RISKS**: ninguno pendiente. Las dos líneas bloqueadas por reglas se autorizaron y se
  evaluaron (D-026): retrieval train->test **permitido pero descartado por evidencia** (la etiqueta de
  los duplicados exactos solo coincide 61,5%, por debajo del modelo) y POS de spaCy **descartado**
  (−0,0032; el corpus ASCII degrada el etiquetado).

## Fases terminadas
| Fase | Notebook | Estado | Evidencia |
|---|---|---|---|
| 0 — Auditoría de datos y leakage | `01_data_audit.ipynb` | OK | `experiments/audit_01_resumen.json`, 4 figuras, `reports/data_dictionary.csv` |
| 1 — Estado del arte (2 corrientes) | — | OK | `reports/state_of_art.csv` (30 trabajos, 29 verificados), `references.bib`, `reports/tables/dataset_provenance.md` |
| 2 — EDA cuantitativo | `02_eda.ipynb` | OK | `experiments/eda_02_resumen.json`, 4 figuras, 5 tablas |
| 3 — Análisis lingüístico | `03_linguistic_analysis.ipynb` | OK | `reports/tables/top_terms_by_class.csv`, `reports/hipotesis_gate_a.md` |
| 4-11 — Representaciones y features | `04_feature_engineering.ipynb` | OK | matriz de 14 configuraciones, descomposición de normalización, `reports/feature_catalog.csv` |
| 12 — Baselines B0-B5 | `05_baselines.ipynb` | OK | `reports/tables/baselines_comparativa.csv` |
| 22 (parcial) — Submission v01 | `submissions/age_submission_v01.csv` | OK | LB = 0,55235 |
| 22 (parcial) — Submission v02 | `submissions/age_submission_v02.csv` | en curso pendiente subida | CV 0,5557 |

## Gate A — cerrado (2026-09-22)
1. **Tarea**: clasificación multiclase (6 clases balanceadas) de tweets en español ASCII, grano
   tweet, sin metadatos, texto truncado a 140 caracteres.
2. **Leakage cuantificado**: 4,26% de test con near-duplicado en train (92,8% de clústeres
   puros); el CV estratificado regalaría **0,061 de macro-F1** por memorización -> validación
   agrupada por clúster obligatoria (D-003).
3. **Señal confirmada**: textometría (ε² hasta 0,126 en `n_chars`), riqueza léxica, registro
   formal/informal, léxico temático (fandoms/MTV en 13-17 vs política/noticias en 65-xx),
   ortografía (char 4-grams: `foll`, `cion`, `ndro`).
4. **Corpus ASCII puro**: 96 caracteres distintos, sin emojis ni tildes -> features de emoji
   descartadas por varianza nula (D-009).
5. **Hipótesis**: 12 priorizadas en `reports/hipotesis_gate_a.md`, cada una con experimento.

## Expectativas calibradas con la literatura (Fase 1)
- PAN/Twitter 5-6 clases: accuracy 0,29 (PAN 2014) -> 0,53 (PAN 2016); humanos ≤ clasificador.
- **Rango realista de macro-F1: 0,30-0,45**. Un valor > 0,50 debe activar sospecha de leakage.
- En PAN 2019 (ES), char n-grams + Random Forest superaron a embeddings (0,8972 vs 0,8444).
- Procedencia del corpus: **irresoluble**; compatible con barrido propio del organizador
  (~2016-2017, español latinoamericano), re-indexado y equilibrado por clase.

## Modelo actual
- **Ensamble de 13 fuentes** (`src/pipeline.py`, submission v08): CV LOFO 0,5866 ± 0,0038;
  mejor envío v07 con **0,60372**. **Línea declarada saturada** (D-016 y su addendum).
- **E-001 (baseline de referencia)**: TF-IDF sublinear word(1-2) + char_wb(3-5) (300k features
  cada uno) + LinearSVC(C=1). Script: `scripts/experimento_e001_wordchar_linearsvc.py`.
- Verificación anti-leakage superada (`experiments/E-001_verificacion_leakage.json`):
  permutación 0,1672 (= azar), accuracy 0,5363 en el tramo limpio de similitud (<0,85).

## Features actuales
- 38 features densas en `features/dense_{train,test}.parquet` (textometría + estilo), de las
  cuales las de emoji son constantes (se excluirán).
- `features/dup_clusters.parquet` (clústeres de near-duplicados para la validación).
- Representación dispersa de E-001: 391.257 columnas (word 1-2 ∪ char_wb 3-5).

## Mejores resultados locales
| Exp | Representación + modelo | macro-F1 (CV agrupado) | Notas |
|---|---|---|---|
| **Fase 20** | **Stacking (SVM C=0.2 + LogReg + SGD + ComplementNB -> meta LogReg)** | **0,5656** | LOFO; gana al mejor individual 5/5 folds |
| E-013 | LinearSVC C=0.2 + char 3-6 + menciones | 0,5628 ± 0,0032 | mejor individual |
| E-012 | LinearSVC C=0.3 | 0,5624 ± 0,0036 | |
| E-009 | LogisticRegression (OvR, C=4) | 0,5603 ± 0,0033 | probabilidades para Fase 19 |
| E-011 | SGDClassifier (log_loss) | 0,5579 ± 0,0045 | |
| E-008 | LinearSVC C=1 + char 3-6 + menciones | 0,5557 ± 0,0052 | envío v02 |
| E-010 | ComplementNB (alpha=0.3) | 0,5370 ± 0,0046 | aporta diversidad al ensamble |
| B5 | 35 features densas + LightGBM | 0,3567 ± 0,0036 | |
| B0 | DummyClassifier(prior) | 0,0485 | suelo estadístico |

Referencias: azar = 0,1667 · memorización de duplicados = 0,061 · **leaderboard v02 = 0,57018**.

## Estado del leaderboard
| Versión | Modelo | CV macro-F1 | LB público |
|---|---|---|---|
| v01 | E-001 (word 1-2 ∪ char 3-5) | 0,5344 | 0,55235 |
| v02 | E-008 (char 3-6 + menciones) | 0,5557 | **0,57018** |
| v03 | Stacking de 4 modelos | 0,5656 | **0,58081** |
| v04 | Ensamble extendido de 9 fuentes | 0,5748 | 0,59178 |
| v05 | Ensamble 9 fuentes + meta enriquecido | 0,5784 | 0,59518 |
| v06 | Ensamble 10 fuentes (incluye MLP densas+SVD) | 0,5851 | **0,60273** |
| v07 | Ensamble 12 fuentes (+ ordinal acumulativo + centroides) | 0,5860 | **0,60372** |
| v08 | Ensamble 13 fuentes (+ pares adyacentes) | **0,5866** | **pendiente de subida manual** |

Brechas CV<->LB (10 mediciones): +0,0179 · +0,0145 · +0,0152 · +0,0170 · +0,0168 · +0,0176 ·
+0,0177 · +0,0184 · +0,0208 · **+0,0213** -> media **+0,0177 ± 0,0023**. Las dos últimas están por
encima del rango inicial: el combinador no lineal y su promedio aprovechan algo más en el test
(probablemente mejor explotación de los near-duplicados, que el CV agrupado bloquea). El desplazamiento es estable y explicado (near-duplicados del test),
por lo que el CV agrupado es un estimador insesgado del leaderboard.
Progreso: 0,55235 -> 0,57018 -> 0,58081 -> 0,59178 -> 0,59518 -> 0,60273 -> 0,60372 -> 0,60376 ->
0,60807 -> 0,61744 -> **0,62182** (mejora acumulada **+0,0695**).

## Análisis de errores (Fase 17, OOF del ensamble)
- F1 por clase: `13-17` 0,638 · `18-24` **0,476** · `25-34` **0,449** · `35-49` 0,519 ·
  `50-64` 0,620 · `65-xx` 0,691. **Las clases medias son el cuello de botella.**
- La confusión es casi toda **adyacente**: 18-24->25-34 15,3% · 35-49->25-34 14,4% ·
  35-49->50-64 14,2% · 25-34->35-49 12,5% · 65-xx->50-64 12,7%.
- **La longitud domina el error**: quintil más corto 52,5% de error vs más largo 37,2%.
- Retuits: 38,5% de error vs 46,2% en no retuits. Near-duplicados: 27,5% vs 44,2%.
- Figuras: `reports/figures/fase17_matriz_confusion.png`, `fase17_error_longitud.png`.

Barrido de C (LinearSVC): 0,1 -> 0,5595 · **0,2 -> 0,5628** · 0,3 -> 0,5624 · 0,5 -> 0,5606 ·
1,0 -> 0,5557. Fase 19 (ajuste de decisión por clase): ganancia LOFO de solo +0,0006 -> **no se
adopta** (D-011).

- Mejora v01->v02 en el LB: **+0,01783** (CV: +0,0213). Brecha CV<->LB: +0,0179 (v01) y +0,0145 (v02):
  estable en el rango que explica la memorización de near-duplicados del test.
- Progreso acumulado del leaderboard: 0,55235 -> **0,57018**.

## Pendientes (ninguno crítico)

## Problemas detectados
- Truncado a 140 chars y transliteración ASCII: limitan POS y lexicones con tildes.
- Ruido de etiqueta: 643 filas (0,38%) con etiquetas contradictorias -> techo práctico.
- 166 textos vacíos (104 train, 62 test): predicción por prior.
- Sin `user_id`: el control de leakage es por clústeres de near-duplicados, no por autor.

## Decisiones abiertas
- Métrica oficial macro vs weighted (se resuelve con el envío v01, D-002).
- Separabilidad de `50-64` vs `65-xx` (PAN 2015 las fusiona y gana precisión): decidir con la
  matriz de confusión, **sin colapsar etiquetas** en la submission.

## Próxima acción recomendada
- **Ninguna dentro del marco actual.** V12, V15, V16, V17 y V18 estan cerradas con medicion, y
  ninguna de las cinco ha dejado una pista que supere su propio gate. Reanudar solo tiene sentido
  **cambiando el marco**: optimizar o seleccionar **sobre las fronteras directamente**, en lugar de
  la macro-F1 de seis clases que las diluye. Esa es la accion que `reports/v18_final_research.md`
  recomienda, y es una decision de diseno, no un experimento mas.
- Antes de esa, un objeto de estudio con nombre propio que este trabajo deja medido y sin explotar:
  **la compresion a seis probabilidades por miembro**. El especialista binario saca 0,7136 de esas
  mismas seis columnas por miembro que el combinador de V12 promedia.

## Entregables finales (verificados 2026-09-24)
- Notebooks: `01_data_audit` · `02_eda` · `03_linguistic_analysis` · `04_feature_engineering` ·
  `05_baselines` · `06_classical_models` · `07_modelos_densos` · `08_error_analysis` ·
  `09_final_pipeline` · **`99_entrega_consolidado`** (artefacto calificado).
- Reportes: `FINAL_REPORT.md` (235 líneas) · `reports/auditorias.md` (APROBADO) ·
  `reports/state_of_art.csv` (30 trabajos) · `references.bib` · `reports/feature_catalog.csv` ·
  `reports/tables/dataset_provenance.md` · 15 tablas y 11 figuras.
- Memoria: `PROJECT_STATE.md` · `RUNLOG.md` · `DECISIONS.md` (D-001..D-015) · `experiments.csv`
  (30 experimentos) · `requirements.txt` (312 paquetes).
- Submissions: `age_submission_v01..v06.csv` + `age_submission_final.csv` (reproduce la v06 al
  100,0000%).
