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

## V21 · PRIMERA FRONTERA ORDINAL (2026-09-27) CERRADA — B1 no se puede mejorar (D-056)

- **`B1_SPECIALIST = NO_SIGNAL`.** La unica via que quedaba abierta en la familia ordinal esta cerrada,
  y con ella las **tres subfamilias**: reconstruccion (D-054), pesos (D-055) y **clasificadores de B1**.
- **Auditoria historica hecha buscando codigo**: ningun experimento habia sustituido especificamente B1,
  entrenado un modelo dedicado a `13-17` frente a `>=18`, ni usado una representacion especifica para B1.
  El antecedente cercano **E-028 no es equivalente**: entrena un `LinearSVC` sobre el **par adyacente**
  (dos clases, 57.539 filas), no sobre el umbral **anidado** (todas las filas, 17/83).
- **V20 como control exacto**: reconstruido desde sus artefactos, da **0,6097782 frente a 0,6097782**, con
  SHA256 `69b574f737c8cf43...`. El script se aborta si no cuadra.

  | Candidato | macro-F1 | Delta vs V20 | B1 balanceada | B1 AUC | Neto | Estado |
  |---|---|---|---|---|---|---|
  | **V20 (control)** | **0,6097782** | +0,0000 | **0,7632** | 0,9284 | 0 | CONTROL |
  | LinearSVC C 0,2 | 0,5992098 | **−0,0106** | 0,6782 | 0,9178 | −2.457 | `NO_SIGNAL` |
  | LinearSVC C 1,0 | 0,5992058 | **−0,0106** | 0,6784 | 0,9178 | −2.457 | `NO_SIGNAL` |
  | boosting especializado | 0,6097377 | **−0,0000** | 0,7606 | 0,9272 | −6 | `NO_SIGNAL` |

- **La razon es estructural, no estadistica**: el `LinearSVC` pierde **0,085** de exactitud balanceada y
  el boosting especializado queda **por debajo** del B1 de V20 (0,7606 contra 0,7632). **El B1 del
  LightGBM multiclase de V20 ya esta a su nivel**, porque las cinco CDF comparten las 279 features y el
  mismo modelo, y la primera sale de un ajuste multiclase que ya la resuelve.
- **Respuesta a la pregunta central de V21**: **no.** Mejorar especificamente `13-17` frente a `>=18`
  **no produce una mejora real en el objetivo final de seis clases**, medido con control exacto y dos
  mecanismos distintos.
- **Sin submission**: ningun candidato mejora el OOF de V20.
- **Entregables**: `scripts/v21_b1.py`, 1 informe, 3 CSV, 1 figura. `experiments.csv` pasa a **124**.

## V20 · PESOS ORDINALES (2026-09-27) CERRADA — mejorar la frontera debil destruye la global (D-055)

- **`ORDINAL_WEIGHTS` cerrada con numero.** Se probo **una sola configuracion** (`class_weight` balanced
  en el primer umbral, los otros cuatro intactos) porque P6 mostro que la frontera 1 pesa **siete veces
  mas** que la 4 y es la de menor exactitud balanceada.
- **La hipotesis era correcta y aun asi perdio.** La exactitud balanceada del primer umbral sube
  **0,7632 -> 0,8430 (+0,0797)**, pero el blend de V20 cae a **0,5932020 (-0,0166)**, con **0 de 5
  folds**, IC95 [-0,0178, -0,0153] y **neto -1.340**.
- **Es el tercer negativo limpio del mismo tipo, y el mas instructivo** porque aqui la hipotesis acerto:
  **la exactitud de una frontera no es el cuello de botella**. El blend ya extraia de ese umbral lo que
  necesitaba, y forzarlo a separar mejor el grupo pequeno **descalibra las cinco CDF a la vez**, porque
  las cinco comparten el mismo espacio de features y el mismo modelo.
- **Consecuencia para la direccion**: **no es reponderar fronteras**. La unica via abierta dentro de la
  familia ordinal es cambiar **los clasificadores**, no la reconstruccion (D-054) ni los pesos (aqui). El
  candidato natural es **mejorar el primer discriminador con un modelo distinto**, que es un cambio de
  mecanismo y no de hiperparametro.
- **No se abre ninguna otra familia** mientras la ordinal tenga un challenger confirmado por leaderboard.

## V26 · ADVANCED RESEARCH (2026-09-28) — H02 CLOSED — la evidencia es invariante a su escala (D-061)

- **Fase cerrada con una sola hipotesis ejecutada.** `V26_SELECTED_TRACK = H02`, y el resultado es
  **H02 CLOSED**: T1 da **0,6097770** frente al control **0,6097782** que reproduce exacto. Delta
  **-0,000001**, **2/5** folds, neto **+3** sobre 1.809 filas cambiadas (1,07%). Gate de V26
  (+0,003 y 4/5) no cumplido por las dos vias. **Sin submission.** **V20 sigue siendo el champion**
  con LB **0,62689** y hash sin cambio.
- **La distincion que reencuadra la fase: V20 YA es un meta-modelo.** De las 279 features, **84 son
  predicciones de modelos base** (78 de los 14 miembros de V12 y 6 de NB-SVM) y el meta-modelo es
  **LightGBM**. La pregunta de V26 no era construir un meta-modelo, sino si su **representacion
  de la evidencia** era explotable de otra forma.
- **Dos afirmaciones medidas en vez de suponer, y las dos contrarias a la intuicion**: un arbol
  **NO** es invariante a una transformacion monotona en esta LightGBM (el z-score cambia la
  prediccion **bit a bit**, diferencia maxima **0,219** sobre 135.468 filas), y un percentil no es
  monotonia estricta porque colapsa empates (**0,166**). La invariancia teorica habria cerrado H02
  sin gastar presupuesto, y era **falsa**.
- **Gap analysis**: 15 familias por equivalencia funcional, **5 genuinamente abiertas** (H02, H03,
  H11, H12, H14). H11 (`monotone_constraints` de LightGBM) tiene **cero usos** en el proyecto
  entero y es el unico mecanismo del punto 11 sin probar.
- **La lectura**: la transformacion **si cambia el modelo y no cambia la metrica**. La evidencia del
  ensemble es **invariante a su escala en todo lo que el macro-F1 mide**: lo que el meta-modelo
  necesita de las probabilidades de los miembros es el **orden**, y el orden ya se conserva.
- **El cuello de botella restante no esta en ninguna de las cuatro propiedades del punto 43**:
  representacion de features (V25, D-060), escala de la evidencia (V26-01), formulacion ordinal
  (cinco subfamilias, D-054 a D-058) y no linealidad (D-053, D-056). Todas con evidencia negativa
  reproducible. Lo que queda es una limitacion de los **datos**.
- **Abiertas y NO ejecutadas**: H03, H11, H12, H14. El punto 41 prohibe convertir una observacion en
  otro experimento sin autorizacion.
- **Entregables**: `scripts/v26_00_gap_analysis.py`, `scripts/v26_01_rank_representacion.py`,
  `scripts/v26_02_auditoria.py`, 7 informes en `reports/`, 5 figuras. `experiments.csv` pasa a **132**.

## V25 · SELECCION SUPERVISADA DE FEATURES (2026-09-27) CERRADA — elimina senal, no ruido (D-060)

- **`SUPERVISED_FEATURE_SELECTION = CLOSED`.** T1 (mutual information, top 50 por ciento,
  K = **139**) da **0,6091492** frente al control T0 **0,6097782** que reproduce exacto. Delta
  **-0,000629**, **1/5** folds positivos, neto **-101**, 3.545 filas cambian. Gate no cumplido.
  **V20 sigue siendo el champion** con LB **0,62689** intacto.
- **La procedencia de las 73 label-informed se verifico ANTES de entrenar**, por codigo, por
  comparacion array a array de los grupos de duplicado y por comparacion de los folds: se
  construyen con `y_train` del training fold y usan la **misma particion** que V20. **LEGAL.**
- **Hubo que estandarizar antes del mutual information**: las 73 supervisadas llegan a magnitud
  **59.439** frente a probabilidades en [0, 1], y el estimador por vecino mas cercano no es
  invariante a escala. Se estandariza dentro de cada training fold.
- **El hallazgo es el por bloque, no el delta**: sobreviven con frecuencia **1,000** las **78
  probabilidades del ensamble** y las **6 de NB-SVM**; **nunca** sobreviven **74 de las 87
  estructuradas** ni **30 de las 35 densas**. **El mutual information alto no es ruido: es la
  prediccion de los miembros que sostiene el modelo.** La seleccion **no elimina ruido, elimina
  senal**, y por eso no mejora.
- **La seleccion es estable**: 137 features en los cinco folds, 138 en ninguno. No es ruido
  descartado, es la mitad que sostiene el modelo.
- **La distincion que ya habia confundido dos veces queda registrada en el artefacto**: la cadena
  de cinco umbrales en solitario da **0,6046893** y el **blend** de cuatro cuartos da
  **0,6097782**. Cualquier comparacion contra V20 tiene que hacerse contra el blend.
- **Cuatro bugs propios cazados por `ruff` y no por `ast`**: un parametro mal escrito
  (`cumuladas` sin "a"), una variable de comprehension que no sobrevive al scope, un indice
  fuera de array y dos identificadores CJK. `ast.parse` no detecta ninguno.
- **Entregables**: `scripts/v25_00_inventario.py`, `scripts/v25_01_seleccion.py`,
  `scripts/v25_02_notebook.py`, `reports/v25_feature_inventory.csv`,
  `reports/v25_feature_selection.md`, `reports/v25_feature_selection.csv`,
  `reports/v25_feature_selection_frequency.csv`. `experiments.csv` pasa a **130**.

## V24 · RECONCILIACION DOCUMENTAL (2026-09-27) CERRADA — la causa era estructural (D-059)

- **Fase actual: V24, reconciliacion documental. Sin experimento, sin submission, sin hipotesis
  nueva.** **V20 sigue siendo el champion** (LB **0,62689**) y **V23 sigue cerrado**.
- **La causa del notebook incompleto era estructural, no editorial.** La fuente de verdad es una
  **cadena de cinco scripts** con **un eslabon por fase**, y **V18, V19, V20, V21 y V22 nunca
  tuvieron eslabon**. Por eso el notebook se quedaba en V17 con V20 como champion.
- **Notebook reconciliado**: de **103 a 121 celdas**, de 32 a **33 secciones**, con las **doce
  versiones V12 a V23 presentes**. Cada fase integrada con hipotesis, metodo, control,
  `corrected_cv`, decision y ruta de artefactos.
- **Renumeracion minima**: V24 pasa a la seccion **31** y V23 a la **32**, que es la cronologia
  correcta. Se renumera una sola seccion existente.
- **El registry tenia un defecto propio y preexisting**: **tres lineas de cabecera** en vez de una,
  con dos espurias ocupando `experiment_id` dos veces. Declaraba 130 experimentos y tenia **128**.
  Corregido a una cabecera; **V24 no anade filas** porque es documentacion.
- **Dos discrepancias documentadas sin tocar artefactos**: V20 registrado como 0,609615 (que es el
  modelo compartido de D-057, no la cadena de V20; canonico **0,6097782**) y V12 registrado como
  0,6004 (redondeado; canonico **0,6005017**).
- **Auditoria: PASS.** py_compile rc 0, ruff 0 errores de correccion, **0 Unicode introducido**,
  ids y titulos unicos, 0 artefactos faltantes, **9 celdas ejecutadas con 0 errores**, coherencia
  **7/7**.
- **Deuda declarada en el propio notebook, no escondida**: 38 caracteres Unicode de sesiones
  anteriores, y `notebooks/09_final_pipeline.ipynb` obsoleto respecto a V20.
- **Entregables**: `scripts/v24_notebook_reconciliacion.py`,
  `experiments/v24_reconciliacion.json`, `reports/v24_reconciliacion.md`,
  `reports/v24_tabla_maestra.csv`, `reports/figures/v24_evolucion_corrected_cv.png`.

## V23 · CORAL / CUMULATIVE-LINK (2026-09-27) CERRADA — compartir pesos funciona y no alcanza (D-058)

- **`CORAL = CLOSED` y `ORDINAL_SHARED_WEIGHTS = NO_SIGNAL`.** Delta **-0.126077**
  frente a V20, **0/5 folds** positivos, sin submission. **V20 sigue en 0,62689** como champion y
  no se ha tocado.
- **La hipotesis de la fase era correcta y no alcanza.** CORAL vale **+0.014696** frente
  al control B (cinco binarios lineales independientes). Compartir informacion entre las cinco
  fronteras **si** aporta. Pero la no linealidad de LightGBM vale **-0.140773**, un
  orden de magnitud mas, y ahi se decide el macro-F1.
- **La monotonia se resuelve de verdad y no compra nada.** CORAL: **0** violaciones frente a las
  **18.927** de V20. Aun asi pierde 0,1261. **Segunda confirmacion independiente de D-057**: la
  consistencia ordinal no era el cuello de botella, y resolverla gratis no compra nada.
- **Dos errores de formulacion que los tests cazaron**, ambos propios: (1) factorizar la
  diferencia de dos CDF como producto de sigmoidos, que **solo vale si `s_y = 1`**; el objetivo
  daba 678,80 contra 895,41 de la referencia; (2) penalizar solo `w` produce **separacion** y
  dispara los umbrales a 12,07 con features en [-3, 3].
- **Pre-registro sin mirar AGE**: TEST B demuestra que a la escala del proyecto no hay
  separacion, luego `lambda_w = 1,0` y `lambda_umbral = 0,0` quedan congelados antes de ver el
  resultado.
- **El control B hizo falta**: sin el, -0,126 mezclaba compartir pesos con perder no linealidad.
  Ademas recibe las features **crudas** en la primera version, no converge y devuelve la
  iteracion parcial: un control que no converge no es un control.
- **Auditoria de la fase**: PASS. Sintaxis, import, ejecucion, artefactos, serializacion, unicode
  (0 caracteres introducidos), notebook (16 celdas ejecutadas, 0 errores) y coherencia numerica
  (9/9). La validacion de ejecucion encontro **3 bugs reales** en las celdas del notebook.
- **Deuda documentada**: el notebook consolidado llega hasta la seccion 30 (V17) y **no contiene
  V18 a V22**, aunque el champion es V20. Se integra V23 como pide el brief y se deja constancia
  de lo demas.
- **Entregables**: `src/ordinal_coral.py`, `scripts/v23_00_tests_coral.py`,
  `scripts/v23_01_coral.py`, `scripts/v23_02_auditoria.py`, `reports/v23_coral_analysis.md`,
  `reports/v23_coral_results.csv`, `reports/v23_coral_audit.md`. `experiments.csv` pasa a **130**.

## V22 · FRONTIER AUDIT + COSTE ORDINAL (2026-09-27) — familia ordinal cerrada (D-057)

- **`ORDINAL_NESTED_THRESHOLDS = CLOSED`.** La familia ordinal queda cerrada en sus **cuatro**
  subfamilias: reconstruccion (D-054), pesos (D-055), clasificadores de B1 (D-056) y **ahora
  representacion compartida**. V20 sigue en LB **0,62689**.
- **Forense de V20 resuelto contra el codigo**: los labels son **binarios**, el target tiene **2**
  clases, HIERBA **declara** `multiclass` con `num_class 6` pero el booster tiene seis clases y las
  columnas 2 a 5 de `predict_proba` son **cero exacto**. Las seis columnas salen y la 0 es
  P(lado alto) y la 1 es P(lado bajo). Cada CDF se obtiene **directamente** del clasificador binario.
  No hay multiclase transformado: **CASO A**.
- **La especificacion declarada cuantificada**: da **0,6097782** y una binaria correctamente
  especificada da **0,6091232**. Correlacion entre las dos CDF de **0,998** en las cinco fronteras
  (mismo ranking), calibracion que difiere hasta **0,33**, y la declarada es **0,00066 mejor**.
  **No es un bug que haya que arreglar**: es una version que funciona mejor. Se documenta y se
  conserva.
- **PASO 5, el modelo compartido**: un **unico LightGBM multiclase** con las cinco CDF **por
  suma**. Al ser suma de probabilidades no negativas, **las cinco son monotonicas siempre**: se
  miden **0 violaciones** de 169.336 frente a las 18.927 de V20. Blend **0,6096150**, delta
  **-0,000163**, 3 de 5 folds, IC95 [-0,00078, +0,00045], neto **-13**, **4.451 cambios**.
- **PASO 3/4, los tres candidatos de B1**: los tres tienen B1 **por debajo** del de V20 (0,6782,
  0,6784 y 0,7606 contra 0,7632). **`NO_SIGNAL` en los tres** y el mecanismo queda cerrado por el
  criterio del brief.
- **PASO 14, el coste ordinal**: cambiar el argmax de V20 por el coste esperado minimo con
  `C[i][j] = |i-j|^alfa`. alfa 1 da **0,5949198** (-0,0149) y alfa 2 da **0,5392168** (-0,0706).
  **El coste ordinal destruye el macro-F1**: los errores del modelo no estan concentrados a grandes
  distancias, y forzarlos a evitarlas degrada la frontera y la global a la vez. Es la cuarta
  confirmacion de la leccion de P1.
- **Errores compartidos**: **64.825 filas (38,3 % del train) las equivocan V19 y V20 juntas**. V20
  rescata 1.474 de V19 e introduce 1.193: **neto +281**. La clase con mas errores compartidos es
  `25-34` con **14.913**.
- **Diversidad real**: el modelo historico mas complementario con V20 es **NB-SVM word**, que
  rescata **10.462** de sus 66.018 errores (**15,85 %**) — **pero ya es un miembro de V20**. Los
  modelos buenos son altamente redundantes: los tres primeros comparten mas del **82 %** de sus
  errores con V20.
- **La respuesta a la pregunta central**: **no se ha encontrado una fuente nueva con evidencia de
  existir.** La unica fuente grande medida (los errores compartidos) esta agotada; la que queda
  (representaciones a nivel de caracter) es **estructuralmente ilegal**. Compartir representacion
  **si** elimina la monotonía por construccion, pero **no mejora el macro-F1 final**: la consistencia
  ordinal **no era el cuello de botella**.
- **Entregables**: `scripts/v22_frontier_audit.py`, `scripts/v22_b1_specialist.py`,
  `scripts/v22_ordinal_cost.py`, `CHAMPION.md`, `reports/v22_frontier_audit.md`,
  `reports/v22_top3.md`, `reports/v22_frontier.csv`. `experiments.csv` pasa a **126**.

## V20 · CONFIRMED_CHALLENGER (2026-09-27) — 0,62689 (D-054)

- **V20 es el nuevo referente de leaderboard de AGE: LB 0,62689**, un delta de **+0,00111** sobre V19
  (0,62578). **V19 queda como baseline historico** y no se reemplaza ningun artefacto suyo.
- **La hipotesis ordinal queda validada externamente**: la cadena de cinco discriminadores no solo mejoro
  el OOF sino que **transfirio positivamente a Kaggle**. `ORDINAL_NESTED_THRESHOLDS` pasa de
  `WEAK_SIGNAL` a **`CONFIRMED_CHALLENGER`**, y la prioridad de investigacion queda **invertida**: ya no
  se buscan clases de funcion ni ensembles genericos, sino el maximo rendimiento de este mecanismo.
- **Auditoria de las cinco fronteras**: exactitud balanceada **0,7632 / 0,8264 / 0,8465 / 0,8373 /
  0,8112**, AUC **0,9284 / 0,9224 / 0,9274 / 0,9340 / 0,9456**, y **descalibracion entre -0,0003 y
  +0,0002**: **las cinco estan esencialmente perfectamente calibradas** y no hay margen en recalibrar.
- **Consistencia ordinal**: **11,18 %** de las filas violan la monotonia, con magnitud media **0,002457**
  y maxima **0,518606**. V20 lo resolvio con el **maximo acumulado**.
- **Impacto de cada frontera sobre el blend**: la 1 cuesta **-0,0129**, la 2 **-0,0072**, la 3 **-0,0040**,
  la 4 **-0,0018** y la 5 **-0,0024**. **La primera frontera es el cuello de botella en impacto** y es
  ademas la de menor exactitud balanceada: **la mas debil es la que mas importa**. Y los umbrales 4 y 5
  apenas importan.
- **`ORDINAL_RECONSTRUCTION = CLOSED`.** Las cuatro variantes, con el **control A reproduciendo V20 al
  100 %** (0,6097782 frente a 0,6097782):
  - **A control**: +0,0000, 0/5, neto 0.
  - **B proyeccion monotona por PAVA**: +0,0000, 3/5, neto **+5**. **Identica a A**, y era lo esperable
    con una magnitud de violacion de 0,002457.
  - **C conteo de umbrales**: **-0,0114**, 0/5, neto **-3.319**, con fronteras **0,545 y 0,5456**.
  - **D edad esperada**: **-0,0263**, 0/5, neto **-6.414**, con fronteras **0,541 y 0,5738**.
- **La respuesta a la pregunta de la fase**: **la mejora de V20 procede de usar cinco fronteras ordinales,
  y NO existe margen en la forma de convertir esas cinco probabilidades acumuladas en las seis clases
  finales.** Lo dicen tres senales independientes: descalibracion cero, violaciones de magnitud
  despreciable, y proyeccion monotona identica al control.
- **El hallazgo que vale mas que el resultado**: **C y D mejoran las fronteras de forma espectacular y
  destruyen el macro-F1**. Es la **leccion de P1 repetida dentro de la familia ordinal**, y la
  confirmacion mas limpia de que **la macro-F1 de seis clases es la metrica que manda y la frontera es una
  minoria de la masa**.
- **No se propone submission nueva todavia**: ninguna de las cuatro variantes mejora el OOF de V20, y el
  criterio exige mejora local antes de enviar.
- **Entregables**: `scripts/v20_p0_fronteras.py`, `scripts/v20_p1_reconstruccion.py`, 2 informes P0, 2
  informes P1, 4 CSV, 3 figuras. `experiments.csv` pasa a **122**.

## V20 · MECANISMOS NUEVOS (2026-09-27) — dos familias cerradas, una viva (D-053)

- **V19 sigue congelado en 0,62578.** No se toco su configuracion, sus features, su modelo persistido,
  su submission ni su hash.
- **Auditoria previa**: 117 experimentos cruzados contra 28 mecanismos. **Nunca** probados:
  **RandomForest/ExtraTrees, HistGradientBoosting, CatBoost** y la **transformacion de clase
  condicional** de las features numericas. Y el campo **`id` cerrado con medicion en un comando**: es el
  indice de fila, 1 a 169.336, con correlacion de rango **0,00123** con la clase y distribucion plana
  por quintiles. **No hay senal de tiempo que extraer.**
- **F1, clase condicional de las 122 features numericas: `NO_SIGNAL`.** Bloque de 976 columnas
  construido in-fold y comparado **con el mismo arbol y los mismos hiperparametros**: las 279 dan
  **0,6072312**, las 122 **crudas 0,3440624**, las 976 del bloque **0,3438942**. El transform no
  aporta (**-0,0002**). **Consecuencia que importa mas que el resultado**: acota por que funciona el
  bloque de 73, y es **no** que las transformaciones de clase condicional sean buenas en general, sino
  que las entidades y la rareza **son** discriminantes y las features de forma y posicion **no lo son**.
- **F2, clase de funcion alternativa: `NO_SIGNAL`, y el negativo mas informativo.** ExtraTrees da
  **0,5836218** sobre las **mismas 279**, **0,0236 por debajo** del LightGBM de V19, con 0 de 5 folds.
  **El efecto de V19 no es "cualquier combinador no lineal": es del BOOSTING**, que ajusta residuo a
  residuo, y eso explica por que el bloque de 73 se cobra.
- **F3, cinco umbrales anidados: `WEAK_SIGNAL`, el unico mecanismo vivo post-V19.** El especialista de
  D-048 saca 0,7136 siendo un binario de dos clases, luego con **el 34 % de las filas**; el umbral
  anidado hace **la misma decision con el 100 %**. No es un duplicado de E-028a (pares adyacentes, el
  comparador) ni de ORD-1 (CDF desde la salida de V12, sin clasificadores nuevos, dio 0,5887).
  - Exactitud balanceada de los cinco discriminadores: 0,7632 / 0,8264 / 0,8465 / 0,8373 / 0,8112, y
    **superan a V12 en la misma fila en tres de los cinco**.
  - **La cadena en solitario: 0,6046893, por encima del 0,6005017 de V12 y sin un solo modelo
    multiclase.**
  - Como cuarto miembro: **0,6097782, +0,0018, 5 de 5 folds, IC95 [+0,0012, +0,0025]**, neto **+294**,
    5.674 cambios.
- **Candidato preparado y NO ENVIADO**: `submissions/age_submission_v20_candidato_umbrales.csv`, **3.546
  de 112.892 filas (3,14 %)** frente al envio de V19, 8 de 8 checks, SHA256 `2c964af7...`. Mejora
  esperada **+0,0019** en LB, y el residuo de la calibracion sobre V19 fue de **-0,0016** con
  oscilacion de **0,0034**: la mejora esperada es **del mismo orden que la incertidumbre del aparato**.
- **La direccion con mas recorrido que queda abierta** es la de **descomposiciones de decision**, no la
  de clases de funcion ni la de transformaciones de features.
- **Entregables**: `scripts/v20_f1_f2.py`, `scripts/v20_f3_umbrales.py`,
  `scripts/v20_f3c_candidato.py`, 5 informes, 3 CSV, 2 figuras y 1 submission. El bloque de 1,3 GB queda
  **fuera de git** y se reconstruye con el script. `experiments.csv` pasa a **122**.

## V19.x · EXPLOTACION DEL CHAMPION (2026-09-27) CERRADA — el eje de parametros es plano (D-052)

- **`V19 PERMANECE CHAMPION` con LB 0,62578.** Sin submission nueva y **sin volver a V12**.
- **P0, configuracion congelada y verificada por el codigo**, no por el informe: LightGBM multiclass,
  300 arboles, `lr` 0,05, `num_leaves` 8, `max_depth` 3, `min_child_samples` 200, `colsample` 0,5,
  `subsample` 0,8, `reg_lambda` 1,0, semilla 42; combinacion a un tercio con LogReg y MLP(64),
  pre-registrada. Las 279 variables de V12, sin ninguna nueva y sin seleccion.
- **Atribucion**: las 279 dan 0,6072312, sin las 73 quedan en 0,586956 y las 73 solas dan 0,5597155.
  Anadir las 73 a las 206 aporta **+0,0203** y a la inversa **+0,0475**: es **interaccion y no suma**.
- **P1 corrige la lectura de D-051**: el bloque label-informed aporta el **9,1 % de la ganancia con el
  28,6 % de los cortes**, luego cada corte suyo vale **4,0 veces menos** que uno del bloque de 206. Se
  usan **271 de 279** columnas, el top 10 concentra el **39,3 %**, y **las 12 variables mas importantes
  son todas margenes de miembros del ensemble**, con `oof_E014_margenes_0` al 7,99 %. El bloque de 73 es
  un **socio de interaccion**, no una variable primaria.
- **P2**: hueco train a OOF de **+0,007781** (fold 0) y **+0,012609** (media de 5 folds); 1.800
  arboles, 2.400 hojas, 1.841.063 bytes. Modelo **persistido con SHA256** en
  `models/v19_champion_lightgbm.txt`, luego V19 es reproducible sin reajustar.
- **Consulta al especialista: cerrar**. Tres argumentos medidos: la **tasa base del subespacio de
  parametros es +0,0008** frente a un gate de +0,0030; la **resolucion del aparato es +0,0011**
  (0,6090431 en 4 folds contra 0,6079388 en 5, sin cambiar nada); y el **residuo de la calibracion sobre
  V19 es -0,0016**, con oscilacion de 0,0034 entre modelos vecinos, luego un +0,002 en CV puede
  aterrizar entre -0,000 y +0,004 en LB.
- **Puerta de coste cero superada**: la ocupacion del techo de cortes es del **97,5 %** (1519 de 1800
  arboles), luego la profundidad es el limite real y no las hojas. El eje de capacidad **no reabre**.
- **Las dos unicas configuraciones autorizadas, y ninguna adopta**:

  | Configuracion | CV | Delta contra V19 | Folds pos. | 4 folds | Neto de error | Submission |
  |---|---|---|---|---|---|---|
  | A, `max_depth` 4 con `num_leaves` 8 | 0,6079208 | **-0,0000** | 3/5 | -0,0000 | **+3** | **no** |
  | B, `min_child_samples` 800 | 0,6076698 | **-0,0003** | **0/5** | -0,0002 | **-50** | **no** |

- **0 de 2** cumplen el criterio de submission, que exige los cuatro puntos: delta CV >= +0,003, 5 de 5
  folds, IC95 que excluya el cero y confirmacion >= +0,002 en 4 folds.
- **`CHAMPION.md` creado**, con los tres scores separados: V12 official 0,62182, V12 reproducible
  0,62188, **V19 0,62578**. No volver a llamar 0,62188 "LB oficial".
- **Lo que reabre AGE**, y es solo tres cosas: fuente de informacion nueva, regla nueva del concurso o
  evidencia externa. La unica linea que el registro dejaba abierta, E-022 sobre las 279 de la era V12
  con regularizacion explicita, es exactamente lo que V19 ya cobro.
- **Entregables**: `CHAMPION.md`, `scripts/v19_p0_auditoria.py`, `scripts/v19_x1_explotacion.py`,
  `reports/v19_p0_auditoria.md`, `reports/v19_x1_explotacion.md`, `reports/v19_importancia.csv`,
  `reports/v19_complejidad.csv`, `reports/v19_x1_comparativa.csv`,
  `reports/v19_x1_complementariedad.csv`, 3 figuras. `experiments.csv` pasa a **117**.

## V19 · B1 (2026-09-27) — EL COMBINADOR DE V12 DESPERDICIABA UN BLOQUE DE 73 VARIABLES (D-051)

- **ADOPTADO como challenger. Submission construida y validada, NO ENVIADA.** V12 sigue siendo el
  champion hasta que el leaderboard diga otra cosa.
- **El hallazgo**: un boosted tree pequeno y regularizado sobre las **279 variables de V12** da
  **0,6072312** en OOF, y combinado a partes iguales con las dos cabezas de V12 da **0,6079388
  (delta +0,0074, 5 de 5 folds, IC95 [+0,0064, +0,0085])**, frente a los **0,6005017** de V12. El
  MLP(64) solo da 0,5989481 y la LogReg sola 0,5915679: **el combinador de V12 promedia algo peor que
  su mejor parte**.
- **El origen, acotado por bateria de falsacion**: el efecto **desaparece sin las 73 columnas
  supervisadas** (0,5870, por debajo de V12). No es que el arbol sea mejor: es que ese bloque,
  **agregados label-informed** (`mention_mean_c0` a `c5`, con valores de hasta 59.439) y no
  probabilidades, necesita combinacion no lineal, y el promedio lineal no puede darla.
- **Confirmado en particion independiente**: la de **4 folds**, con la combinacion ya fijada, da
  **0,6090431 (+0,0085)** con los cuatro folds sobre el gate y las fronteras en 0,5083 y 0,5018
  frente a las 0,4963 y 0,4974 de V12.
- **Sin fuga**: el bloque aislado da 0,5597 y su media de folds coincide con el pooled en **1,5e-06**.
  **Alineacion verificada por firma**: el indice 255 es el de mayor magnitud del bloque en train y en
  test.
- **Lo que cambia**: **11.392 filas (10,09 %)** de la submission, con **rescate 4.622, dano 3.311 y
  neto +1.311**. La primera frontera sube **+0,0120** y esta vez **sin intercambio de suma cero**, que
  era el fallo que hundia la global en P1.
- **Las otras tres familias se descartaron con numero antes de ejecutar**: retrieval (cobertura in-fold
  0,00 % y techo +0,0010), label cleaning (PH11 +0,0016 y PH18 +0,0009), cascada ordinal (ORD-1
  -0,0118 y ORD-B -0,3057).
- **Riesgo declarado y no medible**: el bloque supervisado tiene firma distinta entre train
  (cross-fitted) y test (ajustado sobre todo el train). El supervisor de V12 comparto ese sesgo, luego
  **solo el leaderboard puede resolverlo**. LB previsto **0,62734** frente a **0,62182**: prediccion,
  no medida.
- **D-050 queda superado**: su veredicto de cierre era correcto sobre las cuatro familias propuestas y
  falso como veredicto general. Y su accion recomendada ("optimizar sobre las fronteras") **no era la
  unica via**: aqui la frontera mejora como consecuencia de arreglar el combinador.
- **Entregables**: `submissions/age_submission_v19_challenger_boosting.csv`, 4 scripts `v19_b1_*`, 4
  JSON, 4 informes, 2 figuras. `experiments.csv` pasa a **115**.

## AGE CERRADO (2026-09-26), SUPERADO POR V19 (2026-09-27) — V12 permanece champion (D-050)

- **Decision**: `CLOSE AGE - V12 REMAINS CHAMPION`. **Sin experimento adicional y sin submission
  nueva.** V12 intacto en **0,6005017473** pooled, **0,62182** en el envio `age_submission_v12.csv` y
  **0,62188** en la replica reproducible.
- **La hipotesis de compresion de outputs muere por construccion, no por evidencia.** Los 62
  artefactos de train de `features/` estan consumidos o cerrados con numero, y **todos** los OOF de
  miembros estan ya comprimidos a seis columnas o menos, porque cada miembro es un modelo de seis
  clases y no puede emitir mas. La compresion **no la introduce V12**: se introdujo al construir cada
  miembro. La hipotesis no se puede formular.
- **D-049 estaba mal razonado, y ahora esta medido.** Escribio que la frontera es una *minoria de la
  masa*: `18-24` es el 16,93 % y `25-34` el 16,74 %, luego juntas son **33,67 %**, un tercio. Y la
  lever es grande: corregir 1.000 filas mal clasificadas de un par a su etiqueta verdadera vale
  **+0,0061** y **+0,0063** de macro-F1 global, medido por contrafactual sobre el OOF existente. El
  gate de +0,003 **se cruza con 478 correcciones** de las 28.721 y 28.438 que V12 comete en cada par.
  El cierre **no** puede sostenerse en la magnitud.
- **La razon correcta es la identificabilidad.** La senal existe (D-048: 0,7136 y 0,7493), la
  magnitud sobra, y la funcion objetivo no se puede re-pesar sin transferir (P1: intercambio de suma
  cero). Lo que falta es un **selector** que identifique cuales de esas 478 filas corregir, y cinco
  selectores de V17 mas la pagina de confianza de V16 (90 puertas, **cero filas desviadas**)
  demuestran que la senal de confianza no contiene esa informacion. Los errores de V12 en frontera son
  ademas **confiados**: margen 0,809 cuando el especialista acierta contra 0,539 cuando V12 acierta.
- **El gap CV -> LB no da puntos gratis.** `LB = +0,00223 + 1,02825 x CV`, **R2 = 0,9968**, con los
  cuatro envios cuyos dos extremos son verificables en disco (v01, v07, v08, v12). Un gate de +0,003
  en CV vale **+0,00308** en LB y llevaria a 0,62278. El gap es un offset aditivo estable, de
  +0,01716 a +0,02132. Los gates locales estan **calibrados**.
- **El veredicto del agente `kaggle specialist` es cerrar**, con las tres condiciones no negociables
  del brief en contra: la hipotesis no es competitiva, la informacion no es nueva y ya fue probada.
  Su sonda de prior-shift (+0,0010) **no dejo artefacto en disco**, luego no es verificable; aun
  tomandola por cierta, queda tres veces por debajo del gate.
- **AGE no se reabre** salvo que aparezca una fuente de informacion nueva, una regla nueva del
  concurso o evidencia externa. En particular y de forma permanente: no mas fases, ni busqueda de
  hiperparametros, ni variantes de perdidas, de features, de ensembles, de destilacion ni de
  especialistas.
- **Lo que el trabajo deja recomendado no es un experimento**: optimizar o seleccionar **sobre las
  fronteras directamente**, lo que exige cambiar la metrica efectiva del entrenamiento. Es un problema
  distinto del que se cerro, porque la lever es grande y lo que falta es un selector, no informacion.
- **Entregables**: `reports/age_decision_ejecutiva.md`, `reports/age_cierre_auditoria.md`,
  `reports/age_cierre_auditoria.csv`, `reports/age_lb_calibration.csv`,
  `figures/age_elasticidad_frontera.png`, `scripts/age_cierre_auditoria.py`,
  `scripts/age_decision_ejecutiva.py`. `experiments.csv` pasa a **111**.

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
- **La familia ordinal esta cerrada en sus cuatro subfamilias** (D-054 reconstruccion, D-055 pesos, D-056 clasificadores de B1, D-057 representacion compartida) (D-054 reconstruccion, D-055 pesos, D-056
  clasificadores de B1). No queda ningun eje dentro de ella, y **V20 sigue en 0,62689** como champion.
- **Lo que queda abierto ya no es un eje de la familia ordinal**: la unica pregunta que sobrevive es de
  **analisis**, no de modelo: **por que el primer umbral tiene la menor metrica y el mayor impacto en el
  blend**. No se responde con otro experimento de B1, porque los dos mecanismos posibles ya fallaron.
- **Antes de abrir otra familia** convendria resolver esa pregunta, porque indicaria si la senal de la
  primera frontera esta ya agotada o si hay una lectura distinta por hacer.
- **No enviar nada sin mejora local previa.**: las unicas configuraciones de bajo riesgo autorizadas no llegaban ni al gate
  de adopcion, y el eje de parametros resulto plano. El detalle esta en `CHAMPION.md` y en
  `reports/v19_x1_explotacion.md`.
- Lo que reabre AGE son solo tres cosas, ninguna de ellas un experimento: **fuente de informacion
  nueva**, **regla nueva del concurso** o **evidencia externa** que cambie el espacio de busqueda.
- Históricamente, y solo como registro de lo que se descartó: V12, V15, V16, V17 y V18 estan cerradas con medicion, y
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

## V31+ ADVANCED RESEARCH (2026-09-30) CERRADA - la senal linguistica es real y V30 ya la tenia

- **Baseline corregido**: el operativo de V30 es **0,6134471** (pesos medios, lo que se envio y lo
  que reproduce), no los 0,6130051 publicados, que venian de pesos por fold nunca persistidos.
  Error de transferencia contra el LB real 0,63302: **+0,00001**. Ver D-066.
- **Track A**: prevalencia de URL **monotona en la edad** (33,4 % -> 50,0 %), pero **99 % `t.co`**,
  **0 status URLs**, sin semantica de dominio que extraer. Enmascarar la URL **cuesta** 0,006969.
  El **retrieval de contenido queda QUARANTINED**: las reglas de la compilacion no estan
  disponibles y no se hizo ninguna llamada de red.
- **Banco de representaciones** (modelo, folds, semilla y vectorizador fijos): el canal
  `ortografia+composicion+discurso` da **+0,006820 con 5/5 folds**. Cerradas negativas: posicion
  (-0,002303) y forma de palabra (-0,002984).
- **Integracion**: como 4o miembro **+0,0000642** sobre el control reoptimizado; como 48 features
  **+0,0002948** sobre el control barajado. Ninguna via supera el gate de +0,0015.
- **Track J**: el **90 % de las confusiones de V30 son entre bandas adyacentes**. 9.780 errores de
  alta confianza (15,0 % de los errores), pero el problema es de **resolucion de escala**.
- **Resultado**: **0 finalistas, 0 submissions**. V30 sigue siendo el mejor resultado validado.
  Ver D-067.
- **Auditorias**: compliance 0 hallazgos sobre 9 scripts con 6 comprobaciones positivas;
  notebook 242 celdas con la seccion V31+ en **10/10 verde**; idioma 586 ficheros **0 errores**.

---

## G14 - AUDITORIA DE INFORMACION DE ENTRADA (2026-10-06) - NOT_ACTIONABLE

- **Pregunta**: existe algun campo de entrada legal, presente en prediccion y que ningun modelo
  haya consumido?
- **Esquema real**: el ZIP contiene exactamente `age_train.csv` (`id`, `text`, `age_range`),
  `age_test.csv` (`id`, `text`) y `age_submission.csv`. Sin campos ocultos.
- **El `id` no es un snowflake**: 1..169.336 en train y 1..112.892 en test, **6 digitos** en el
  maximo, monotono y sin huecos. Y **el 100 % de los ids del test reaparece en train** para filas
  distintas: es un indice de fila por split, no un identificador de contenido. Correlacion con la
  clase: **-0,00084**.
- **Matriz campo-por-modelo**: `text` USED/DERIVED 9 de 9; `id` NOT_USED 9 de 9; `age_range`
  PROHIBITED 9 de 9.
- **Traza de `text`**: toda dimension no capturada por el TF-IDF (case, puntuacion, elongaciones,
  posicion) fue cubierta por los diez canales de V31-03 y cerrada por V31-06. Caso C, no canal
  nuevo.
- **Decision**: **NOT_ACTIONABLE**. No se rescata el `id` con modulo, hash, posicion ni patrones de
  digitos. Ver D-068.
- **Estado de la investigacion**: seis lineas cerradas y la **entrada enumerada por completo**. El
  techo esta medido: 27.007 de los 65.403 errores de V30 no los corrige ninguna de las 26 fuentes,
  y no hay campo sin explotar que pudiera hacerlo.

---

## CIERRE DE LA INVESTIGACION DE AGE (2026-10-07) - `AGE_RESEARCH_CLOSED`

- **Baseline final**: V30 = **0.6134471** corrected_cv, **0.6143347** en la particion alterna,
  **0.63302** en el leaderboard. Hash del submission `f543a715...` sin cambio.
- **Input gap**: cerrado por G14 (`NOT_ACTIONABLE`).
- **Mechanism gap**: cerrado por esta auditoria. **39 mecanismos auditados**, 23 usados, 9
  cerrados, 1 invalidado, 1 no accionable y 5 sin usar que se cierran con evidencia.
- **Particion de errores**: A = **27,430** (41,9 %) sin ninguna fuente que los corrija; B =
  **37,973** (58,1 %) con al menos una. De B, **8,358** son rescates unicos y
  **30,038** compartidos: **la informacion esta repartida**, no encerrada en una fuente.
- **Rescates exclusivos**: **8,029** entre **23 fuentes**, concentracion 20,5 %, sin
  estructura comun en longitud, confianza ni URL. Es ruido de modelo.
- **Cinco vias de integracion cerradas con control**: feature, miembro de mezcla, residuo global,
  residuo local y residuo con pendiente. En las cinco el control gana o empata.
- **Conclusion**: no quedan mecanismos de aprendizaje metodologicamente defendibles y no recorridos
  dentro de las restricciones del proyecto. Ver D-069.
- **Correcciones declaradas**: A = 27.430, no 27.007; rescates exclusivos = 8.029 con esta
  definicion, no 122 con la del inventario previo.

---

## REVISION BIBLIOGRAFICA (2026-10-08) - `NO_NEW_METHOD`

- **Baseline**: V30 = **0.6134471** corrected_cv, leaderboard **0.63302**, sin cambios.
- **Pregunta**: existe un mecanismo publicado, compatible con el reglamento, genuinamente distinto
  de las 39 familias auditadas y con evidencia de mejora en macro-F1 sobre texto?
- **Hallazgo central**: **macro-F1 es una metrica nominal y la literatura ordinal optimiza metricas
  distancia-conscientes**. Vargas et al. (IEEE TKDE 2026) y Kasai et al. (arXiv 2405.11775)
  confirman el trade-off de forma explicita: la mejora ordinal se paga con metricas nominales.
- **Huecos de ejecucion reales, declarados**: AGE nunca corrio targets suaves ni label smoothing
  (0 scripts), ni EMD (0), ni proyeccion de topicos (0). Solo hay 2 perdidas custom en todo el
  proyecto: margen (V18-P3) y destilacion (V18-P5).
- **Cinco candidatos evaluados, ninguno justifica reabrir**: C1 (perdidas distancia-conscientes),
  C2 (ruido ordinal), C3 (unimodal), C4 (estilo/contenido), C5 (geometria). C1 es el unico con
  hueco real; se registra y **no se ejecuta**.
- **Duplicacion algebraica**: EMD2 es la familia acumulativa CORAL con perdida L2.
- **Registros**: 5 candidatos, 21 duplicaciones, 25 metodos rechazados con motivo.
- **Conclusion**: no existe razon cientifica para reabrir. Ver D-070.

---

# CIERRE FORMAL DE AGE (2026-10-09) - `AGE_RESEARCH_CLOSED`

> Estado congelado definitivo. Esta seccion es el acta de cierre: si algo de lo anterior
> contradice lo de aqui, manda esto.

## Los tres numeros

```text
V30 corrected_cv      = 0.6134471
V30 particion alterna = 0.6143347
V30 leaderboard       = 0.63302
```

No son intercambiables. 0.6134471 es la unica configuracion reproducible desde los artefactos **y**
la que se envio. 0.6143347 es una **reejecucion con reentrenamiento** sobre los mismos grupos con
otra semilla: su diferencia con la primera es reentrenamiento en el 99 %, medido. 0.63302 es
evidencia de transferencia y nunca se uso para seleccionar nada.

## Estado de cada linea

| Linea | Veredicto |
|---|---|
| G14 auditoria de informacion de entrada | **NOT_ACTIONABLE** |
| P33 residuo global | **CLOSED** |
| P34 correccion local por pares | **INVALIDATED** |
| G12 pendiente sobre el margen | **CLOSED** |
| Revision bibliografica 2026 | **NO_NEW_METHOD** |
| C1 perdidas distancia-conscientes | **LATENT / NO EJECUTADA** |

## La conclusion, con sus tres negaciones

> **No se afirma que V30 sea optimo.**
> **No se afirma que AGE sea imposible de mejorar.**
> Se afirma que, dentro de las restricciones vigentes, los inputs legalmente disponibles y los
> mecanismos auditados, **no queda actualmente una hipotesis nueva con justificacion suficiente
> para reabrir la investigacion**.

## La cobertura que sostiene la afirmacion

```text
39 mecanismos auditados
23 USED          9 CLOSED          1 INVALIDATED
 1 NOT_ACTIONABLE 5 NOT USED       0 GENUINELY_NEW con justificacion suficiente
```

Los cinco `NOT USED` se cierran con medicion: focal loss, hard example mining y label smoothing
apuntan a la poblacion A (27.430 filas sin ninguna de 26 fuentes que las corrija) y a la frontera,
donde P34 midio que el ruido aleatorio gana. `ranking objective` no mueve macro-F1. `bootstrap de
grupos` es varianza con otra unidad, y el bagging ya se cerro en -0,000312.

## C1, latente y con su control

**No se ejecuta.** Su expectativa honesta es no cruzar el gate, y la evidencia publicada dice que el
trade-off ordinal/nominal va en contra de macro-F1 (Vargas et al., IEEE TKDE 2026; Kasai et al.,
texto). Queda con su control disenado: tres brazos (CE one-hot, label smoothing, kernel con
etiquetas permutadas) y gate doble (delta >= +0,0015 con 4/5 folds **y** delta > 8,9e-04).

## Las equivalencias que impiden reabrir por error

1. `EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2` -> es la familia acumulativa CORAL, cerrada.
2. El minimizador esperado de **Ord-NLL** es un target suave -> colapsa a C1.
3. Los **targets suaves** son un **superconjunto** de label smoothing -> cerrado por argumento,
   y ahora por evidencia.

## Correcciones de auditoria conservadas

| Se reporto | Correcto | Causa |
|---|---|---|
| A = 27.007 | **27.430** | enmascarado en los miembros de V30 |
| 122 rescates exclusivos | **8.029** | definiciones distintas, ninguna erronea |
| OOF de V31-03 como probabilidades | son **`decision_function`** | dividir por la suma invierte el orden: 0,5048 -> 0,0445 |

## Como se reabriria

Con **evidencia nueva** que cambie el diagnostico, un **cambio de restriccion** (los metadatos o los
modelos preentrenados son donde la literatura situa la mayor parte de la ganancia), o una
**hipotesis cientificamente distinta** que no este entre las 39 auditadas. Cambiar el `beta` de una
familia cerrada no cumple ninguna de las tres.

---

## PLUG-IN POR CLASE PARA MACRO-F1 (2026-10-11) - `PROMISING_CONFIRMED`

- **Que es**: cambiar la **regla de decision** de `argmax` a `argmax_c(w_c * p_c)` con
  `w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]`. **No** cambia el modelo, la
  representacion ni los artefactos de V30.
- **Por que**: las clases de train son perfectamente balanceadas (0,160-0,170) y la prediccion de
  V30 no (0,146-0,187). `argmax` maximiza accuracy, no macro-F1.

| Particion | V30 | plug-in | delta | folds+ | IC95 | alta conf. |
|---|---|---|---|---|---|---|
| canonica | 0.6134471 | 0.6154643 | **+0,0020172** | 5/5 | [+0,00101, +0,00295] | **0** |
| alternativa | 0,6144120 | 0.61647 | **+0,0020580** | 5/5 | [+0,00112, +0,00306] | **0** |

- **Diferencia entre particiones: 0,0000408.** Y el `nested` (pesos re-derivados en la particion
  alternativa) da +0,0022884: el mecanismo aparece solo.
- **Prediccion falsable confirmada dos veces**: cero cambios en filas de alta confianza. Un
  multiplicador no puede alterar el orden con margen grande. Es un cambio de regla, no de
  calibracion.
- **Matiz declarado**: el control `permutado` cambio de signo entre particiones; parte del efecto es
  generica (diversificar el reparto) y la asignacion correcta anade ~+0,0006. El permutado no es un
  control negativo perfecto.
- **Fuga descartada**: `congelado - nested = -0,0002304` en la particion nueva; los congelados no se
  inflan.
- **Clasificacion**: NO mejora de representacion, NO mejora del modelo, **SI mejora de la regla de
  decision**. Sube macro-F1 bajando accuracy (0,61464 -> 0,61254).
- **Estado**: no `FINALIST` (delta bajo +0,003) y no `ADOPTED` (la particion alternativa es
  reejecucion, no independiente). **Submission: ninguna. Leaderboard: no consultado.**
- **Correccion al registro**: `V20-F3-UMBRALES` era una **arquitectura ordinal** (cadena de
  discriminadores binarios anidados), no una regla de decision. Ver D-072.

---

## SUBMISSION EXPERIMENTAL PREPARADA (2026-10-12)

```text
V30          = FROZEN
plug-in      = PROMISING_CONFIRMED
submission   = PREPARED
```

- **Fichero**: `submissions/age_submission_plugin_candidate.csv`
- **SHA256**: `29cbe6cca4312a65f5176bad25c7d4ae4624d348f3e55586dfa36767a3abd6c0`
- **112.892 filas** | columnas `id`, `age_range` | **10/10 checks** | auditoria **APROBADA**
- **Regla**: `class = argmax_c(w_c * p_c)` con `w = [1.00, 1.17, 1.27, 1.16, 1.00, 0.81]` sobre las
  probabilidades de V30 **sin transformar**.

### Los tres numeros, porque confundirlos fue un error

| | Valor | Delta | Que es |
|---|---|---|---|
| (A) nested | 0,6154643 | **+0,0020172** | **la estimacion honesta del mecanismo** |
| (B) congelados global | 0,6159732 | +0,0025261 | lo que ejecuta la submission |
| (C) redondeados | 0,6159908 | +0,0025437 | variante de la anterior |

(B) > (A) es la fuga declarada en el pre-registro: los pesos vieron todas las filas al derivarse.

### En test

**7.408 filas cambian (6,56 %)** respecto de `argmax` puro, con confianza media **0,3361** frente a
**0,6247** del conjunto, y **cero cambios con confianza >= 0,60**. La prediccion falsable se cumple
tambien en test.

### Estado

**No subida a Kaggle** y sin score por anticipado: no se proyecta el leaderboard. Si se obtiene, se
clasificara como `POSITIVE_TRANSFER` / `NEUTRAL_TRANSFER` / `NEGATIVE_TRANSFER` contra 0,63302, sin
reentrenar ni generar una segunda sonda.

---

# ESTADO FINAL DE AGE

```text
V30     = HISTORICAL_FROZEN_BASELINE   LB = 0.63302
PLUG-IN = ADOPTED                      LB = 0.63575
plug-in delta LB = +0.00273

research_status   = CLOSED
candidate_status  = ADOPTED
external_transfer = POSITIVE_TRANSFER
```

| | CV canonico | CV alternativo | leaderboard |
|---|---|---|---|
| V30 | 0.6134471 | 0.6143347 | 0.63302 |
| **plug-in** | **0.6154643** | **0.6164700** | **0.63575** |
| delta | **+0.0020172** | **+0.0020580** | **+0.00273** |

**Regla**: `argmax_c(w_c * p_c)` con `w = [1.00, 1.17, 1.27, 1.16, 1.00, 0.81]`.
**No cambia** el modelo, la representacion, los hiperparametros ni la semilla.

**Fichero**: `submissions/age_submission_plugin_candidate.csv`, SHA256
`29cbe6cca4312a65f5176bad25c7d4ae4624d348f3e55586dfa36767a3abd6c0`. No sobrescrito.

**Conclusion**: la mejora proviene de la **regla de decision**, no de una representacion ni de un
modelo nuevo. Afirmacion acotada: para este problema y esta configuracion de probabilidades. **No**
se afirma que V30 fuera incorrecto ni que `argmax` no sea optimo en general.

**Investigacion cerrada.** Sin grid nueva, sin pesos nuevos, sin submission nueva, sin
tuning con el leaderboard. P33/P34/G12/G14 no se reabren salvo evidencia externa o cambio de reglas.

---

# FASE ABIERTA - LINGUISTIC STRUCTURED ENRICHMENT (2026-09-29)

Pregunta: existe informacion linguistica estructural (secuencias tipadas) no explotada de forma
equivalente? El candidato adoptado (V30 + plug-in, LB 0,63575) permanece como FINAL_REFERENCE.

| Paso | Estado | Resultado |
|---|---|---|
| P0 gap analysis por operacion | **COMPLETADO** | Secuencias tipadas L1-L7 = GENUINELY_NEW; formas agregadas y regex cerradas con medicion |
| P1 auditoria de legalidad | **COMPLETADO** | Anotador es_core_news_md = AMBIGUOUS (conflicto interno + vectores estaticos internos) |
| P2 POS estructurado | **GATEADO** | No se ejecuta hasta resolver la clasificacion (opciones A/B/C) |

Reglas especificas capturadas: `references/age_rules_capture_2026-09-29.md`.
Artefactos: `reports/linguistic_enrichment_gap_analysis.{md,csv}`,
`reports/age_linguistic_legality_audit.md`, `reports/linguistic_annotation_registry.csv`.
Sin test, sin holdout, sin submissions.

---

## Resolucion de legalidad (D-076, 2026-09-29)

El anotador queda **AUTORIZADO condicionalmente** (etiquetas si, vectores no) con las seis
condiciones de D-076, y se anade `es_core_news_lg` como modelo de comparacion. P2 (POS
estructurado) queda habilitado con pre-registro previo. El candidato adoptado (V30 + plug-in,
LB 0,63575) permanece como FINAL_REFERENCE.

---

## P2 ejecutado: canal POS-secuencia NEGATIVO (2026-09-29)

- Canal `pos_seq` (md y lg): senal aislada real (**0.2751 / 0.2725** frente al suelo 0.1667),
  pero la integracion neutral pre-registrada (alfa 0.5, pesos del plug-in congelados) da delta
  **-0.0086357 / -0.0084168** con **0/5 folds** y con el control barajado rindiendo **mejor**
  (-0.0039528 / -0.0041620). Veredicto del arbol pre-registrado: **NEGATIVO**.
- Respuesta de la fase para POS-secuencia: **C** - existe senal nueva pero la integracion no
  mejora; el modo de fallo es dilucion, no ausencia de informacion.
- Decision: **POS-secuencia se cierra documentado**; el candidato adoptado (V30 + plug-in,
  LB 0.63575) permanece intacto. Sin test, sin holdout, sin submission.

---

## P6 ejecutado: NER estructural (2026-09-29)

- Resultado: NINGUN BRAZO PASA: NER estructural se cierra documentado; el campeon intacto.
- N1/N1b/N2 x md/lg con control barajado por el pipeline completo; baseline 0.6159732.
- Sin test, sin holdout, sin leaderboard, sin submission.

---

## P7 ejecutado: texto tipado (2026-09-30)

- Resultado: CLOSED. T solo (md) 0.4941735; delta real/barajado por anotador en el informe.
- Baseline 0.6159732; sin test, sin holdout, sin submission; campeon intacto.

---

## Ciclo final ejecutado: P3/P4/P5 (2026-09-30)

- Resultado: ningun brazo pasa; serie completa de seis representaciones medida.
- Baseline 0.6159732; sin test, sin holdout, sin submission; campeon intacto.

---

## V32-E1 ejecutado: perdida surrogate de macro-F1 (2026-09-30)

- Reapertura V32 (candidatos: perdida surrogate > CatBoost > FTRL). P1 ejecutado con
  pre-registro congelado; revisiones ml-architect / kaggle-specialist / dl-specialist; gate
  ml-reviewer + ml-qa (GO); ES estricto (decision de Carlos) en GPU (D-007).
- Resultado: **NO_ATRIBUIBLE**. A1 (soft-macro-F1) delta -0.0034165 (0/5 folds, IC95
  [-0.0044246, -0.0023612]); A2 (CE gemelo) -0.0009284 (1/5); A3 (barajado) -0.0009011 (0/5).
  Sonda de determinismo 0.0; sin leakage; campeon intacto.
- Decision: P1 cerrada sin segunda particion (D-081). Sin test, sin holdout, sin submission.

---

## V32-E2 ejecutado: CatBoost queda CLOSED (2026-09-30)

- Ciclo 2 de la reapertura (P2 CatBoost MultiClass; gemelo LightGBM con el ancla afinada
  V29/V30 como control decisional; control negativo barajado). Pre-registro con enmienda
  pre-ejecucion (TotalF1 -> TotalF1:average=Macro). Revisiones y QA completos; ejecucion CPU
  en 1149 s; determinismo exacto; sin truncamiento.
- Resultado: **CLOSED**. A1 delta -0.0011258 (1/5 folds); A2 -0.0012871; A3 -0.0012138;
  control A1 vs A2 no favorable. Sin segunda particion (D-082).
- Lista priorizada de la reapertura agotada; P3 (FTRL) solo con autorizacion. Sin test, sin
  holdout, sin submission; campeon intacto.
