# V13 ADVANCED RESEARCH — Informe final

**Proyecto:** KAGGLE_AGE_2026_PUJ · Clasificación de edad a partir de texto en español
**Rama:** `kaggle-age-2026-puj-v13` (commits locales, sin publicar)
**Fechas:** 25 y 26 de septiembre de 2026
**Resultado de cabecera:** **ningún candidato adoptado. V12 permanece congelado en macro-F1 0,6004
(CV) y 0,62188 (leaderboard).**

---

## 1. Resumen ejecutivo

V13 se abrió para buscar señal supervisada dentro del propio train que las dos fases anteriores no
habían explotado, sobre todo en las fronteras centrales `18-24|25-34` y `25-34|35-49`, sin tocar V12
y bajo un conjunto estricto de restricciones (sin embeddings, sin modelos externos, sin corpus ni
lexicones externos, sin pseudo-etiquetas, sin nuevos vectorizadores aprendidos).

Se ejecutaron **16 experimentos** en seis pistas de variables, dos diagnósticos de estructura de
decisión y cuatro diagnósticos de causa, con un total de **8 hipótesis pre-registradas** que este
informe somete a falsación. **Ninguna se confirmó en su forma accionable.** El resultado no es que
las hipótesis fueran falsas sin más: es que se identificaron tres mecanismos distintos por los que una
mejora aislada se evapora al combinarla con V12, y esos mecanismos **explain también** por qué las dos
fases anteriores ya se habían cerrado.

El hallazgo más útil de la fase no es un modelo: es la medición de que la matriz de 279 variables de
V12 tiene un VIF mediano de 18,87 con 170 variables por encima de 10, y que 87 de sus 278 direcciones
explican el 95 % de su varianza. Sobre ese bloque, añadir variables nuevas no es una apuesta neutral:
es meter señal en un espacio ya saturado.

## 2. Por qué existía V13

El diagnóstico previo era desfavorable y conviene dejarlo escrito antes de los resultados, porque
condiciona cómo se leen:

- El 33,13 % de los errores de V12 (22.397 filas de 67.610) **no los resuelve ningún miembro** del
  ensemble: es techo de información para todo el conjunto.
- El error se concentra en las fronteras centrales: la exactitud de V12 dentro del par
  `18-24|25-34` es 0,4963 y dentro de `25-34|35-49` es 0,4974, frente a 0,6004 global.
- Las dos líneas anteriores (capa de decisión ordinal y lexicones temáticos) se habían cerrado tras
  33 capas de decisión y 72 variables léxicas respectivamente, sin alcanzar el gate.

V13 se a probar si el hueco estaba en **más señal** (supervisada, lingüística) o en **otra
forma de decidir** (frontera, puerta). La respuesta medida es que no está en ninguno de los dos, y el
informe explica por qué con evidencia en lugar de con conjeturas.

## 3. Las pistas y lo que dio cada una

| Pista | Qué se construyó | Control aislado | Resultado sobre V12 | Veredicto |
|---|---|---|---|---|
| **P1** evidencia por token | `P(clase\|token)` in-fold, 36 columnas, ajuste cruzado anidado | **0,3997** (suelo 0,1667) | Δ **−0,0002**, 3/5 folds | **NO ADOPTAR** |
| **P17** pronombres de persona | 26 columnas, vocabulario cerrado propio | 0,1918 | pre-chequeo **neto −64.075** | cerrada sin fusionar |
| **P20** incertidumbre y modalidad | 23 columnas, vocabulario cerrado propio | **0,1270** (bajo el azar) | — | cerrada |
| **P19** árboles de conjunciones | 36 columnas, 9 funciones, árboles de profundidad 4/6/8 | **0,1539** (bajo el azar) | — | cerrada |
| **P3** especialistas de frontera | modelos binarios nuevos sobre las 279 variables | — | **0,6005 -> 0,5993** | cerrada |
| **Lexicones** (fase anterior) | 72 variables léxicas | 0,2033 (mejor bloque) | −0,0001 a −0,0018 | cerrada en D-039 |
| **Capa de decisión** (fase anterior) | 33 reglas sobre probabilidades | — | +0,0011 con 4/5 folds, neto de rescate **−117** | cerrada en D-037 |

Los cuatro diagnósticos que explican estos resultados:

| Diagnóstico | Pregunta | Respuesta medida |
|---|---|---|
| **P15** | ¿Qué miembro acierta cuando V12 falla? | El 66,87 % del error es rescatable, pero hacen falta **13 de 14** miembros y **ninguno** tiene señal de confianza (separación de margen negativa en los 13) |
| **P16c** | ¿V12 ya contenía esa información? | La **varianza** sí (R² mediano 0,522, máximo 0,959), la **información de clase** no (control de la reconstrucción 0,0485) |
| **P16d** | ¿Había una puerta realizable? | Techo con puerta perfecta 0,677, pero el máximo de 90 combinaciones de umbrales es **0,6005 con 0 filas desviadas**: la mejor puerta es no usar el bloque |
| **P3a** | ¿Sirve un especialista entrenado para la frontera? | **No**: 0,4963 -> 0,4944 y 0,4974 -> 0,4942, es decir peor que V12 restringido al par |

## 4. Los tres mecanismos que explican el fracaso

**Mecanismo 1 — saturación del espacio de variables.** P0b medió que las familias de textometría de
V12 están saturadas (VIF mediano 12,6 en las estructuradas y 42,8 en las densas) y que las únicas con
margen son la evidencia supervisada (3,1) y NB-SVM (0,77). Un bloque nuevo que no añada una
dirección nueva no puede aportar. La criba de redundancia, obligatoria a partir de P0b, detectó que
tres de los cuatro bloques nuevos eranoxigenadamente no redundantes (0 de 36, 0 de 26, 0 de 23
variables con |rho| ≥ 0,80) — es decir, **no se trata de redundancia**.

**Mecanismo 2 — la información nueva no está donde el combinador puede usarla.** El resultado más
delicado de la fase es la divergencia entre dos medidas de P16c: la regresión lineal desde V12
explica bien la **varianza** de las columnas del bloque de P1 (R² mediano 0,522) pero una
reconstrucción con solo 5 grados de libertad clasifica a 0,0485, por debajo del azar. La varianza que
V12 reproduce no es la que discrimina clases. Y P16d cierra la puerta a la salida alternativa: la
confianza del bloque **sí** separa sus aciertos de sus fallos (AUC 0,7173), pero las filas que el
bloque acierta son en su mayoría filas que V12 ya acertaba, así que desviar el bloque no mejora nada.

**Mecanismo 3 — el modelo ya está en el límite de su espacio.** P3a es el más contundente: un
clasificador binario **entrenado específicamente** para cada frontera, con las mismas 279 variables y
sus propios datos de entrenamiento, es **peor** que la predicción de seis clases de V12 restringida a
ese par. No es la línea que D-037 cerró (allí, reglas a posteriori; aquí, un modelo nuevo). Es la
evidencia de que dentro de este espacio de variables no queda estructura de frontera que explotar.

## 5. La predicción pre-registrada y su veredicto

El plan pre-registró, a partir del artículo de referencia, que *un bloque lingüístico debe subir la
rama LogReg y diluirse en la MLP*. El reporte por cabeza (DD-8) no pudo confirmar esa predicción,
pero **tampoco se cumplió su alternativa**: ninguna de las dos cabezas mejora con el bloque de P1
(LogReg 0,5915 -> 0,5909; MLP 0,5978 -> 0,5973). No había nada que disolver. Se registra tal cual,
porque una predicción pre-registrada que no se confirma es información, no un inconveniente.

## 6. Hallazgos de calidad metodológica

Dos hallazgos ajenos a la hipótesis scientistà principal(resultaron igual de útiles que los anteriores):

**D-040 — el entorno calcula mal un producto matricial.** `numpy 1.26.4` sobre `Python 3.14.7`
devuelve valores erróneos en el patrón `X @ X.T`: la diagonal de una matriz de Gram 2000×5 sale
0,7392 en lugar de 1.999,0, y ocurre con 1 hilo y con 32, en layout C y F, y con datos enteros
pequeños. `np.corrcoef` y `np.cov` con `rowvar=True` caen en el mismo fallo. **Ninguna métrica del
proyecto está afectada**: los diez sitios del repositorio que usan producto matricial usan rutas
sanas, y el `MLPClassifier` de producción da predicciones idénticas con la ruta parcheada. Se
documenta, se añade `src/verificacion_numerica.py` con una autocomprobación que detiene cualquier
experimento si una ruta en la que el proyecto se apoya falla, y se valida la criba por segunda vía
contra un cálculo anterior hecho con otro código (diferencia máxima 0,0005).

**Auditoría de idioma.** Todo el proyecto (301 ficheros) está en español y sin caracteres de otros
alfabetos: se detectaron y corrigieron 6 ideogramas CJK y 260 emoji y flechas, y la comprobación
automática se ejecuta antes de cada commit. La limpieza se verificó como cosmética regenerando las
tres matrices de la fase de lexicones **idénticas bit a bit**.

## 7. Lo que queda sin probar, y por qué no se abrió

| Pista | Razón para no abrirla |
|---|---|
| P2 (dominios de URL), P6 (centroides), P18, P21, P24 | Son **más variables del mismo tipo** que P1, P17, P20 y P19. Los cuatro bloques ya medidos dieron señal aislada y nada sobre V12, y P16c ha mostrado que añadir variables a esta matriz no es una apuesta neutral. El §28 pide no ejecutar por cantidad. |
| P4 (boundary gating), P5 (entrenamiento pairwise) | Son **estructura de decisión**, y P3a ya ha medido que un modelo nuevo y dedicado a la frontera empeora. El techo de puerta de P16d muestra que la complementariedad existe pero no hay gatillo. |
| Legibilidad (P23), morfología, risa | Cerradas antes de empezar: la legibilidad no está medida, la morfología es redundante con `char_wb(3-5)`, y los marcadores de risa reabrirían la línea de lexicones que D-039 cerró. |

## 8. Limitaciones y amenazas a la validez

1. **Restricción fuerte de la muestra.** Los tres bloques que más se parecen al español se construyen con vocabulario
   cerrado propio; con las restricciones de §3 no se pueden usar lexicones externos, y un estudio con
   recursos léxicos reales (por ejemplo, un diccionario de variantes del español y redes sociales) podría
   encontrar señal que estas listas no alcanzan. Es la limitación más grande del trabajo.
2. **Un solo corpus.** Todo se mide en 140 caracteres de tweets sobre noticias y televisión en
   español. Las conclusiones sobre ausencia de señal (P20, P19) son conclusiones **sobre este
   corpus**, no sobre la función gramatical en general.
3. **El umbral de adopción es exigente a propósito.** Δ ≥ +0,003 con 4/5 folds positivos e IC95 que
   excluya el cero descarta mejoras de +0,001, que en una competencia podrían ser soumise a Cuentas.
   Con la LB en 0,62188, +0,001 sigue siendo relevante; V13 no lo ha buscado porque el gate del plan lo
   prohíbe deliberadamente.
4. **Dos experimentos=P16 (P16, P16c)-great** corrections.Tres correcciones metodológicas suyas
4. **Tres correcciones metodológicas** (término independiente, tipificación de columnas y
   diagnóstico fuera utilizable. Se documentan las tres, porque un resultado que necesitó tres correcciones para no ser falso es
   un resultado cuyo valor hay que tomar con cautela.

## 9. Conclusión y recomendación

V13 cierra con **cero adopciones** y con algo más útil que cero adopciones: una explicación medida de
por qué V12 no se puede mejorar por ninguno de los dos caminos que quedaban abiertos, más
infraestructura reutilizable que se queda detrás (criba de redundancia contra la matriz completa,
autocomprobación numérica del entorno, y el protocolo de pre-chequeo de rescate que evitó pagar unos
70 minutos de CPU en dos pistas).

La recomendación es **no abrir P19 ni las pistas de variables restantes** (ya está hecho: P19 se
ejecutó y cerró) y tratar V12 como el punto de parada del proyecto, salvo que se acepte un cambio de alcance de una
   de estas dos formas:

1. **Relajar §3** para permitir un recurso lingüístico externo o un corpus de referencia. Eso cambiaría
   la naturaleza del proyecto y hay que decidirlo explícitamente, no por defecto.
2. **Cambiar la métrica de evaluación** a una tolerancia a ganancias pequeñas (por ejemplo, exactitud en
   lugar de macro-F1), lo que exigiría redefinir el gate §24 y reabrir formalmente la línea.

Ninguna de las dos es una decisión que deba tomarse dentro de un experimento: es una decisión de
alcance, y corresponde a quien dirige el proyecto.

---

## Anexo A — Reproducibilidad

Todos los experimentos son scripts completos y ejecutables de forma independiente, cada uno con su
resultado en `experiments/*.json` y su informe en `reports/*.md`. El orden de ejecución es el del
plan (`plan/V13_ADVANCED_RESEARCH_PLAN.md`).

| Paso | Script | Entregable |
|---|---|---|
| 1 | `scripts/v13_p0a_auditoria_idioma.py` | `reports/v13_auditoria_idioma.md` |
| 2 | `scripts/v13_p0b_colinealidad.py` | `reports/v13_colinealidad.md` |
| 3 | `scripts/v13_p15_matriz_errores_miembros.py` | `reports/v13_p15_member_error_matrix.md` |
| 4 | `scripts/v13_p1a_evidencia_tokens.py` | `reports/v13_p1_evidencia_tokens.md` |
| 5 | `scripts/v13_p1c_identidades.py` | `reports/v13_p1_identidades.md` |
| 6 | `scripts/v13_p1b_fusion_tokens.py` | `reports/v13_p1_fusion.md` |
| 7 | `scripts/v13_p17a_pronombres.py` | `reports/v17_pronombres.md` |
| 8 | `scripts/v13_p17b_pronombres.py` | `reports/v17_pronombres_fusion.md` |
| 9 | `scripts/v13_p20_modalidad.py` | `reports/v20_modalidad.md` |
| 10 | `scripts/v13_p3a_especialistas.py` | `reports/v3_especialistas.md` |
| 11 | `scripts/v13_p19_conjunciones.py` | `reports/v19_conjunciones.md` |
| 12 | `scripts/v13_p16c_reconstruccion.py` | `reports/v16c_reconstruccion.md` |
| 13 | `scripts/v13_p16b_activacion.py` | `reports/v16b_activacion.md` |
| 14 | `scripts/v13_p16d_puerta.py` | `reports/v16d_puerta.md` |
| 15 | `scripts/v13_cierre.py` | `reports/v13_master_results.csv` y las dos tablas agregadas |

Los pasos 12 a 15 son diagnósticos y no dependen del orden entre sí. Los pasos 4 a 6 sí: P1b consume
las matrices por fold que produce P1a, y P16c y P16d consumen las de P1a.

**Restricciones de recursos.** Cada evaluación del combinador de producción (LogReg + MLP sobre 279
variables, 5 folds) tarda entre 2 y 3 minutos por fold. Los experimentos de fusión con control de
permutado (P1b) son los caros: cuatro evaluaciones completas, unos 90 minutos. Los diagnósticos
(P0b, P15, P16b, P16c, P16d) tardan menos de 5 minutos porque no entrenan nada.

## Anexo B — Cumplimiento de las restricciones

| Restricción | Cumplimiento |
|---|---|
| Sin embeddings, transformers, LLM o modelos externos | OK El único import de `torch` es `src/gpu.py` para detectar la GPU (D-038) |
| Sin corpus, datasets ni labels externos | OK Todo se deriva del train |
| Sin lexicones externos | OK Los tres vocabularios cerrados (P17, P20, P19) están escritos en los scripts (R-1) |
| Sin pseudo-etiquetas ni labels de test | OK Ninguna; el test solo se transforma para generar submission, y en V13 no se generó ninguna |
| Sin nuevos vectorizadores aprendidos | OK Se reutiliza `src.lexicons.tokeniza`, el tokenizador del vectorizador de V12 |
| Fold train/val/test irreproducible desde fuera | OK Un único `cv.make_folds(..., scheme="group")` y las features supervisadas con ajuste cruzado dentro de cada fold |
| Gate de adopción completo (§24) | OK Aplicado en P1b: 6 criterios, incluidos 3 controles de permutado y bootstrap pareado |
| Doble métrica (§36) | OK Global y las dos fronteras en cada experimento que evalúa un modelo |
| Reporte por cabeza (DD-8) | OK LogReg y MLP por separado en P1b |
| Control de permutación (DD-6) | OK 3 semillas en P1b |
| Auditoría de idioma y de numeración | OK 0 errores en 301 ficheros; `autotest_numerico()` en la primera línea de cada script |
| Sin push remoto | OK Todo son commits locales; el push está prohibido por indicación expresa |