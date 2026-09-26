# Plan vivo — V13 ADVANCED RESEARCH (KAGGLE_AGE_2026_PUJ)

> **Se actualiza al cerrar cada fase.** Creado: 2026-09-25 · Rama: `kaggle-age-2026-puj-v13`
> (base `kaggle-age-2026-puj-lexicons`, commit `c8a2c30`)

## Referencia congelada

| Artefacto | Valor | Estado |
|---|---|---|
| V12 corrected CV | **0,6004** ± 0,0042 (per fold 0,6007 / 0,6050 / 0,6032 / 0,6004 / 0,5928) | intocable |
| Holdout histórico | 0,6077 | **agotado**, solo referencia documental |
| `submissions/age_submission_v12_reproducible.csv` | **LB 0,62188** | **no tocar** |
| Errores OOF | 67.610 | 34,7% techo de información · 66,9% agregación |
| Fronteras centrales | `18-24\|25-34` 10.122 cruces · `25-34\|35-49` 7.252 cruces | foco de la fase |

## Pregunta científica (§40)

¿V12 falla por **A** (no posee información), **B** (la tiene y no la usa), **C** (solo existe en
una familia de modelos), **D** (la capa de decisión no puede explotarla) o **E** (hace falta un
especialista explícito)?

## Reglas fijadas (pre-registradas)

| # | Regla |
|---|---|
| R-1 | **Vocabulario cerrado válido SI y SOLO SI lo construimos nosotros desde cero**: palabras funcionales, árboles SFL, modalidad, subjetividad, especificidad, patrones gramaticales. Cada lista versionada con procedencia, SHA256, nº de entradas, sin datos externos, sin etiquetas de edad, sin embeddings |
| R-2 | **P13 se implementa con diagnóstico explícito** de underfitting, overfitting y leakage (§4 del plan) |
| R-3 | Sin push, sin PR, sin merge, sin modificar remotes, sin tocar `age_submission_v12*.csv` |
| R-4 | Test = transform/predict/submission **nada más**. Holdout intacto. Sin pseudo-labels, sin embeddings, sin vectorizador aprendido nuevo, sin corpus/labels/lexicones externos |
| R-5 | Toda estadística supervisada `P(clase\|token)` se calcula **dentro del train fold**, nunca antes de crear los folds |
| R-6 | Gate: Δ ≥ +0,003 **y** ≥4/5 folds **y** IC95 bootstrap que excluya 0 **y** no explicable por una sola fold **y** segunda partición para los 2–3 finalistas |
| R-7 | Doble métrica siempre: macro-F1 global **y** accuracy/F1 de `18-24<->25-34` y `25-34<->35-49` |
| R-8 | Control de permutación cuando la hipótesis dependa de asociación con la etiqueta |

## Predicción falsable pre-registrada (del artículo de referencia)

> Moreno-Sandoval et al. (2021) conclude que sus features lingüísticas **ayudan a los modelos
> clásicos y perjudican a los profundos**. V12 es un promedio LogReg+MLP, así que el bloque nuevo
> **debe subir la rama LogReg y diluirse en la MLP**. Se medirá por cabeza (misma comprobación DD-8
> de la fase de lexicones). Si no se cumple, se documenta como evidencia contra la literatura.

## Estado de fases

| Fase | Contenido | Entregable | Estado |
|---|---|---|---|
| **P0** | Gap audit: inventario, cerradas, nuevas, nested, reutilizables, potencial, coste, leakage, orden | `reports/v13_gap_audit.md` + `.csv` | **cerrada** (commit 3c473e9) |
| **P0a** | Auditoría de idioma: todo el proyecto en español y sin alfabetos ajenos | `reports/v13_auditoria_idioma.md` | **cerrada** (0 errores, 0 avisos) |
| **P0b** | Prueba de colinealidad de las 279 variables + criba reutilizable | `reports/v13_colinealidad.md` | **cerrada** (matriz completa, sin reentrenar) |
| **P15** | Diagnóstico: matriz de error por miembro (**antes** de P3/P4) | `reports/v13_member_error_matrix.csv` | **cerrada** |
| **P1** | Evidencia de edad por token, in-fold | idem | **cerrada** (0,3997 aislado; fusión -0,0002, 3/5 folds) |
| **P17** | Pronombres de persona (V14–V18 del artículo) | idem | **cerrada** (0,1918; pre-chequeo -64.075, sin fusión) |
| **P19** | Árboles SFL (sin POS) | idem | **cerrada** (árbol 0,1539, bajo el azar) |
| **P20** | Incertidumbre y modalidad | idem | **cerrada** (0,1270, bajo el azar; sin fusión) |
| **P3** | Especialistas binarios de frontera | idem | **cerrada** (el especialista empeora: -0,0017 y -0,0029) |
| **P24** | Forma de la distribución de evidencia | idem | - |
| **P4** | Boundary gating | idem | - |
| **P5** | Entrenamiento pairwise | idem | - |
| **P18** | Palabras funcionales | idem | - |
| **P21** | Subjetividad + especificidad | idem | - |
| **P2** | Evidencia supervisada de hashtag/URL | idem | - |
| **P6** | CentROIDES de clase en el espacio TF-IDF | idem | - |
| **P22** | Queísmo / dequeísmo | idem | - |
| **P26** | PCA in-fold del bloque nuevo | idem | - |
| **P8/P9** | Topología de menciones / URL / hashtag | idem | - |
| **P10** | Estilometría residual (≤30, solo ausentes) | idem | - |
| **P11** | Vista dual (menciones normalizadas) | idem | - |
| **P12** | Micro-lexicón derivado del target | idem | - |
| **P23** | Legibilidad en español | idem | - |
| **P16** | Tipología de error en fronteras | idem | - |
| **P25** | Significancia por atributo (diagnóstico) | idem | - |
| **P13** | Hard-boundary training (**condicional**, con diagnóstico R-2) | idem | - |
| **P14** | Combinación (**solo** con dos positivos independientes) | idem | - |
| **V15a** | Correlacion entrada-entrada y asociacion con la salida, **in-fold** | `reports/v15_correlacion.md` | **cerrada** (58 clusters, Jaccard de pares 1,0000) |
| **V15b** | Poda de variables redundantes (1 representante por cluster) | `reports/v15_poda.md` | **cerrada, NO ADOPTAR** (delta -0,0040, 0/5 y 0/4) |
| **V16** | Arquitectura del meta-modelo (peso aprendido y combinador logistico) | `reports/v16_arquitectura.md` | **en curso** |
| **Cierre** | Informes §31 + memoria | `reports/v13_final_research.md` | - |

**Presupuesto**: ≤24 experimentos principales + 8 reservas.

## Orden de ejecución (inmutable salvo evidencia)

1. **P0** gap audit — sin entrenar nada
2. **P15** diagnóstico del error por miembro (decide si P3/P4 se dirigen a una familia concreta)
3. **P1 · P17 · P19 · P20** — las cuatro familias de información nueva
4. **P3** especialistas de frontera (prioridad máxima del error)
5. **P24 · P4 · P5** — forma de la evidencia, gating, pairwise
6. **P18 · P21** — vocabulario cerrado adicional
7. **P2 · P6 · P22 · P26**
8. **P8/P9 · P10 · P11 · P12 · P23**
9. **P13** condicional · **P14** solo con dos positivos independientes

## Hallazgos de P0a, P0b y P15 (verificados)

**P0a · idioma.** 251 ficheros auditados (`.py`, `.md`, `.txt`, `.csv`): 6 ideogramas CJK y 260
emoji/flechas, todos residuos de la fase de lexicones. Corregidos; las tres matrices de la familia
se regeneraron **idénticas bit a bit**, luego la limpieza es cosmética (D-040 documenta además el
fallo numérico encontrado por el camino).

**P0b · fallo numérico del entorno (D-040).** La correlación vectorizada daba `+0,0001` donde
`scipy` daba `+0,9985`. Causa: `X @ X.T` (auto-transpuesta) devuelve valores erróneos en este
entorno (`numpy 1.26.4` sobre `Python 3.14`), con 1 hilo y con 32, en cualquier layout y tamaño.
`np.corrcoef`/`np.cov` con `rowvar=True` caen en el mismo fallo. **Ninguna métrica del proyecto
está afectada** (los 10 sitios que usan producto matricial o `corrcoef` usan rutas sanas, y el
`MLPClassifier` de producción da predicciones idénticas con la ruta parcheada). Se añade
`src/verificacion_numerica.py`: `autotest_numerico()` detiene el experimento si una ruta en la que
el proyecto se apoya falla, y `grama_exacta()` da la vía fiable (coincide con `spearmanr` con
1,2e-16). **La primera ejecución de P0b dio "0 pares correlacionados" y era un artefacto: se
descartó.**

**P15 · quién acierta cuando V12 falla.** Diagnóstico sobre OOF, sin entrenar:
- V12 acierta 60,07 %; 67.610 errores; **45.213 (66,87 %) los rescata al menos un miembro** y
  22.397 (33,13 %) son techo de información para todo el ensemble.
- **El rescate está repartido, no concentrado**: hacen falta **13 de 14 miembros** para cubrir el
  66,9 % del error. Los tres mejores (`E019b` denso, `E015_softmax` y `E028a` ordinal) cubren solo
  el 13,86 % del error de V12.
- **La confianza de ningún miembro identifica las filas que va a rescatar**: la separación de
  margen (confianza al rescatar menos confianza al fallar) es **negativa en los 13 miembros**
  (de -0,010 a -0,348). Por tanto una puerta §37 por confianza sería ruido, y el techo de
  0,6821 que se obtiene eligiendo el miembro más confiado es **un oráculo no realizable**.
- `svm_lineal` es la familia más redundante (2,00 % exclusiva: sus tres miembros aciertan casi lo
  mismo); `ordinal` aporta 22,53 % exclusiva, `prototipos` 19,87 % y `denso` 19,27 %.
- En las fronteras, `E015_softmax` resuelve el 37,99 % de los cruces `18-24|25-34` y el 36,29 % de
  `25-34|35-49`, pero daña 15.232 y 10.137 filas del par respectivamente: de ahí la necesidad
  obligatoria de la forma `gate(especialista, V12)` y de no aplicarlo globalmente.

**P0b · colinealidad de la matriz completa (sin reentrenar).** La referencia de V12 se lee de
`phase39_fusion_v12.json` y una comprobacion independiente reprodujo sus cinco folds uno a uno
(0,6007/0,6050/0,6032/0,6004/0,5928), asi que la matriz medida es exactamente la de produccion:
- 401 pares con |rho| >= 0,80; 229 redundantes; **15 duplicados exactos** (`punct_ratio` =
  `char_symbol_ratio`, `wlen_median` = `wlen_p50`, `compression_ratio` = -`compression_gain`, 9
  pares de supervisadas); 54 clusters.
- **1 columna constante**: `supervisada_49` nunca varia.
- Rango efectivo estricto 263 de 278: las 15 direcciones que faltan son exactamente los 15
  duplicados, no hay mas degeneracion estructural. Pero **87 de 278 direcciones bastan para el
  95 % de la varianza**.
- **170 variables con VIF >= 10** y VIF mediano 18,87: la interpretacion de coeficientes es
  inservible, la prediccion no se resiente (0,6004 con 15 pares identicos).
- Reparto: `scores de miembros` VIF mediano 528,7 (catorce modelos coinciden en que clase dan como
  mas probable: correlacion intra-clase mediana 0,7014, hasta 0,9987), `densas` 42,8,
  `estructuradas` 12,6, `supervisadas` 3,1, `nb_svm` 0,77.
- **Consecuencia para el orden de fases**: las familias de textometria estan saturadas; las de
  evidencia supervisada in-fold y NB-SVM son las unicas con margen. Esto refuerza poner P1
  (evidencia `P(class|token)` in-fold) por delante de mas variables deterministas.
- La criba quedo **validada de forma cruzada**: aplicada a las 72 variables lexicas de la fase
  cerrada reproduce el resultado ya calculado por otra ruta de codigo (3 redundantes en ambos
  casos, diferencia maxima 0,0005), lo que confirma D-040 por segunda via.

**P17 · pronombres de persona (cerrada).** Bloque de 26 columnas con vocabulario cerrado y
propio (R-1), solo con formas no ambiguas: `el` y `los` se declaran aparte por ser artículo o
pronombre, y `el` es la palabra más frecuente del corpus. Cribado **0 de 26 redundantes**
(máximo 0,4757): ocupa direcciones libres. Pero el control da 0,1918 (suelo 0,1667) y el
pre-chequeo de rescate da **neto -64.075** (rescate 12.043, daño 76.118), negativo también en las
dos fronteras. El bloque no es solo débil: es mucho peor que V12. La pista se cierra sin fusionar.

**P20 · incertidumbre y modalidad (cerrada).** 23 columnas, vocabulario cerrado y propio, cribado
0 de 23 redundantes (máximo 0,4067). El control da **0,1270, por debajo del suelo de azar**, y la
cobertura lo explica: solo el 1,6 % de los documentos tiene algún marcador de duda y el 0,4 % alguno
de certeza. No es que la variable no discrimine, es que **el corpus no contiene la variable**: son
tweets de 140 caracteres sobre noticias y televisión, no comentarios de fans como el artículo de
referencia. Pre-chequeo: neto -70.023. Pista cerrada sin fusionar.

**P1 · evidencia por token (cerrada tras la fusión).** El control dio 0,3997, el doble que el
mejor bloque léxico cerrado, y P1c mostró que el 57 % de esa mejora es evidencia léxica y no
reconocimiento de cuentas. Pero la fusión con V12 da **-0,0002 con 3/5 folds**, y el primer control
de permutado da -0,0005: la mejora del bloque aislado se pierde al combinarla con V12. Los miembros
de V12 son clasificadores de TF-IDF, es decir ya calculan una versión de `P(clase|token)`.

**P1b · fusion (NO ADOPTAR).** Delta **-0,0002** con 3/5 folds. Los tres controles permutados
(0,6000, 0,6003, 0,6000) son **indistinguibles del bloque real** (0,6003): la contribucion es ruido.
Ninguna de las dos cabezas mejora (LogReg 0,5915 -> 0,5909; MLP 0,5978 -> 0,5973). Falla 5 de los 6
criterios del gate. El unico sitio constructivo son las dos fronteras centrales (neto +294 y +248),
que es donde el plan queria, pero muy por debajo del gate.

**P16 · por que falla toda fusion (diagnostico, tres versiones).** P16 AttemptPlain salio
inconcluyente por un fallo de diseno mio (ecuaciones normales sin tipificar columnas: la gramatica de
V12 tiene una direccion que acapara la traza, 8,99e12 de 8,99e12, y el estimador se dispara). P16c lo
corrige parametrizando el ajuste por **grados de libertad efectivos** con columnas tipificadas:
- El **R2** (accesibilidad de varianza) es alto: mediana 0,522 en el bloque de tokens, maximo 0,959,
  con 21 de 36 columnas por encima de 0,5. La varianza del bloque es en buena parte una funcion de
  pocas direcciones de V12.
- El **control** (informacion de clase) es 0,0485 en los tres bloques, por debajo del azar.

Los dos indicadores responden a preguntas distintas y divergen: **la redundancia lineal no explica el
fracaso**. El bloque tiene R2 de 0,96 en su mejor columna y aun asi una reconstruccion con 5 grados de
libertad clasifica por debajo del azar, porque la varianza que V12 reproduce no es la que discrimina.

**P16b · donde es util.** El bloque sube de 0,4028 a 0,5194 de exactitud segun su actividad (tokens
conocidos), y el techo con puerta perfecta llega a 0,677 global y 0,7221 en las filas con cinco o mas
tokens. Pero el **neto de rescate es negativo en todas las franjas** (-33.515 a -1.202).

**P16d · hay puerta realizable? No.** Regla sin oraculo: usar el bloque cuando su confianza supera
`t_b` y la de V12 no supera `t_v`. La confianza del bloque **si** separa sus aciertos (AUC 0,7173), pero
el maximo de las 90 combinaciones da **0,6005 con 0 filas desviadas**: la mejor puerta es no usar el
bloque. Ambas mitades de folds eligen eso de forma independiente. El techo de +0,077 es inalcanzable
porque las filas que el bloque acierta son en su mayoria filas que V12 ya acierta.

**P3a · especialistas de frontera (cerrada).** Modelo **nuevo** por par, entrenado con las filas
del par del train exterior y con las mismas 279 variables, aplicado en la forma
`gate(especialista, V12)`: solo redecide entre las dos clases cuando V12 ya está en el par, así que por
construcción no puede tocar las demás filas. No es la línea que D-037 cerró (allí, reglas a
posteriori; aquí, un modelo con sus propios datos de entrenamiento).
- `18-24|25-34`: exactitud en el par **0,4963 -> 0,4944** (LogReg) y 0,4946 (MLP). Neto -109 y -96.
- `25-34|35-49`: **0,4974 -> 0,4942** y 0,4945. Neto -180 y -163.
- Global: 0,6005 -> 0,5993 con 6.944 filas modificadas.

Un clasificador binario dedicado al par es **peor** que la predicción de seis clases de V12 restringida
a ese par. No queda estructura específica de frontera que explotar en ese espacio de variables.

**Consecuencia operativa para P3.** Un especialista de frontera **no puede apoyarse en los
miembros actuales**: hay poco rescate exclusivo, no hay señal de confianza que active la puerta y
aplicar cualquier miembro globalmente tiene neto muy negativo. P3 debe construirse sobre variables
nuevas (P1, P17, P19, P20), no sobre una recombinación de las existentes.

## Hallazgos de P0 (verificados, no opinados)
- **72 experimentos** inventariados por bloque y etapa.
- **18/18 mapeos verificados** contra los archivos reales de `features/`: el script aborta si un
  nombre de columna declarado no existe.
- De las **18 variables** del artículo de referencia: **14 ya existen** en V12 (2 inertes por el
  corpus sin emojis) y **5 AUSENTES**: las cinco de **pronombres de persona** (V14–V18). El
  artículo declara la 3ª persona singular como su variable más predictiva.
- De las **7 familias de estilo** del curso: 3 abiertas (funcionales, SFL, y las 5 de complejidad
  medidas), 4 cerradas (POS, morfología, palabras inestables, idiosincrasias).
- De las **10 dimensiones** de la taxonomía NLP: 3 abiertas (incertidumbre, subjetividad,
  especificidad), 5 cerradas o prohibidas, 2 ya existentes.
- **7 pistas exigen nested validation** (P1, P2, P6, P12, P13, P24, P26) porque calculan
  magnitudes dependientes de la etiqueta.
- Potencial: solo hay dos palancas con masa propia — **información que V12 no tiene** (P1, P17,
  P19, P20, P21) u **objetivo que V12 no optimiza** (P3, P5). Recolocar probabilidades ya está
  demostrado como ruido (33 capas + 3 familias de lexicon, todas con rescate ≈ daño).

## Matriz de cobertura de las técnicas del PDF (cerradas sin ejecutar)

| Técnica del PDF | Decisión | Motivo |
|---|---|---|
| % de palabras sin tilde | **INERTE** | corpus ASCII puro sin tildes (D-009): la feature sería constante |
| Errores ortográficos (hayga, lledo) | **PROHIBIDO** | requiere diccionario externo |
| Concordancia / gerundio / "donde" | **CERRADO** | requiere POS o parser (D-026) |
| Neologismos y verbos inventados | **CERRADO** | requiere diccionario; y reabre la línea de lexicones (D-039) |
| N-gramas de POS | **CERRADO** | D-026, −0,0032 |
| Morfología (prefijos/sufijos) | **CERRADO** | redundante: `char_wb(3-5)` lo cubre |
| Palabras inestables (traducción ida-vuelta) | **PROHIBIDO** | requiere modelos de traducción externos |
| Idiosincrasias (slang colombiano) | **CERRADO** | D-039 |
| Sentiment / swear words / netspeak | **CERRADO** | reabriría la línea de lexicones |
| Voz pasiva | **CERRADO** | requiere parser |
| Preprocesamiento con stopwords / stemming | **CERRADO** | implicaría un vectorizador nuevo (§4) |
| Ponderación booleano / frecuencia / tf-idf | **CERRADO** | ya en V12 |
| Feature selection | **CERRADO** | ya evaluado (0,0000) |
| "Característica mágica" (largo, TTR, hapax) | **CERRADO** | el propio curso los declara obsoletos; ya existen |
| Medidas de complejidad y diversidad | **CERRADO** | ya existen y son superiores |
| Breakeven point | **ADOPTADO como métrica de reporte** de fronteras | coste cero, sin riesgo |
| V1–V13 léxico/simbólico/participación | **CERRADO** | ya existen en V12 |
| **V14–V18 pronombres de persona** | **P17 ABIERTA** | no existen; el artículo los declara predictivos |
| **Palabras funcionales** | **P18 ABIERTA** | solo 2 agregados actuales |
| **Árboles SFL** | **P19 ABIERTA** | no existen y no requieren POS |
| **Uncertainty / Subjectivity / Specificity** | **P20 · P21 ABIERTAS** | no existen |
| **Queísmo / dequeísmo** | **P22 ABIERTA** | bigramas deterministas |
| **Readability en español** | **P23 ABIERTA** | índice clásico, heurística de sílabas |
| Tendencia central y dispersión (media, mediana, moda, skew, std, curtosis) | **P24 ABIERTA** aplicadas a la distribución de evidencia por token | en V12 solo sobre longitudes |
| PCA / MCA sobre el bloque nuevo | **P26 ABIERTA** | in-fold |
| Modelo de significancia por atributo | **P25 ABIERTA** | diagnóstico de interpretabilidad |

## Fases V15 y V16 (peticion de Carlos, 2026-09-26)

V13 y V14 cerraron con la misma conclusion: el cuello de botella **no son los datos**. Carlos pidio
entonces dos cosas que cierran el circulo por los dos extremos.

### V15 · Correlacion de entrada y poda (cerrada)

**Lo que se midio.** Sobre la matriz completa de **312 variables** (279 de V12 y 33 linguisticas de
V14), y **dentro de cada fold**: la correlacion de Spearman entrada-entrada, agrupada en clusters por
union-find con umbral |rho| >= 0,90; y la asociacion de cada variable con la salida mediante eta
cuadrado, que es la medida adecuada cuando la salida es categorica. Elegir variables por su
correlacion con la etiqueta sobre todo el train seria una fuga, asi que **toda la estructura y toda
la asociacion se calculan in-fold** y el indice de variables conservadas se guarda por fold.

**Lo que salio.**

| Medida | Valor |
|---|---|
| Columnas constantes | 2 (`v12_supervisada_49`, `v14_pasiva_ratio`) |
| Pares con \|rho\| >= 0,80 | 402 |
| Pares con \|rho\| >= 0,90 | 245 |
| Duplicados exactos | 15 |
| Clusters | 58, que cubren 171 de 310 variables |
| Variables conservadas tras podar | 197 de 312 |
| eta2 mediana / maximo | 0,0306 / 0,5599 |
| Variables con eta2 < 0,01 | 112 |
| **Jaccard de la estructura de pares entre folds** | **1,0000** (minimo 1,0000) |

El Jaccard de pares es 1,0000: la estructura de redundancia es **perfectamente reproducible** al
cambiar de fold, asi que podar no es una decision dependiente de la particion. (El Jaccard sobre
los **miembros** de cada cluster da 0,0157, pero es un criterio tan estricto que no informa:
exige que el cluster tenga exactamente los mismos miembros, y un solo par que cambie reordena el
cluster entero por la transitividad de union-find.)

**La poda, evaluada con el gate de D-042.** Un representante por cluster, el de mayor eta2 dentro del
fold. Resultado: macro-F1 **0,5965** frente a 0,6005 de V12, **delta -0,0040, 0 de 5 folds
positivos**; segunda particion **0 de 4** con media -0,0065; IC95 pareado [-0,0054; -0,0027].
**NO ADOPTAR** (ver D-045).

**Lo que el resultado afirma.** La redundancia existe y se mide con precision, pero **es funcional,
no cosmetica**. Las variables casi identicas forman un conjunto de estimadores debiles
parcialmente independientes cuyo promedio reduce la varianza: quitar 115 variables, entre ellas 15
duplicados exactos, cuesta entre 0,4 y 0,65 puntos de macro-F1 de forma reproducible. La
colinealidad afecta a la interpretacion de los coeficientes, no a la prediccion, y esa idea ya
habia aparecido en P0b; V15 la confirma con una evaluacion completa en vez de una inferencia.

### V16 · Arquitectura del meta-modelo (en curso)

Si el cuello de botella es arquitectónico, la palanca que queda es la **forma de combinar las dos
cabezas**. V12 usa un promedio fijo 0,5 LogReg + 0,5 MLP, una decision de diseno de la primera
version del proyecto que **nunca se optimizo**. V16 mide tres arquitecturas sobre las mismas 279
variables y el mismo protocolo:

- **A0**: promedio fijo 0,5 + 0,5 (produccion; su OOF ya estaba calculado, no se reentrena).
- **A1, peso aprendido**: sigue siendo un promedio, pero el peso se elige por fold sobre una
  particion **interior** del train exterior (20 %, agrupada por duplicados), de modo que la
  eleccion no ve la validacion. En la prueba de humo el peso optimo fue 0,4, no 0,5.
- **A2, combinador logistico**: las doce probabilidades de las dos cabezas alimentan una LogReg,
  es decir apilamiento en dos niveles. Es la que puede dar peso distinto a cada clase y a cada
  cabeza, y la que de verdad cambia la arquitectura y no solo un parametro.

Ninguna introduce variables nuevas. La arquitectura que gane en la particion de 5 folds se
confirma en la **segunda particion de 4 folds**, que es control de seleccion y no una medida mas.
El test no se toca.

## Referencias

- **Moreno-Sandoval, L.G., Pomares-Quimbaya, A. y Alvarado-Valencia, J.A. (2021).** *Celebrity
  profiling through linguistic analysis of digital social networks.* Computational Social Networks,
  8:16. DOI 10.1186/s40649-021-00097-w. — 18 variables (V1–V18) en 5 grupos; corpus PAN@CLEF2019;
  F1 0,65 fama / 0,88 género / 0,37 año nac. / 0,57 ocupación; **+0,14 accuracy media** con modelos
  clásicos; **empeora en modelos profundos**.
- **Material del curso** `Clasificacion Texto.pdf` (78 págs.): atribución de autoría y author
  profiling, 7 familias de características de estilo, árboles SFL, y las variables V1–V18 del
  artículo anterior.
- **Zhou, L. y Zafarani, R. (2020).** *A Survey of Fake News: Fundamental Theories, Detection
  Methods, and Opportunities.* ACM Computing Surveys. DOI 10.1145/3395046 — taxonomía de 10
  dimensiones de features textuales.

---
_Este archivo es el plan vivo de la fase. Se actualiza al cerrar cada pista._
