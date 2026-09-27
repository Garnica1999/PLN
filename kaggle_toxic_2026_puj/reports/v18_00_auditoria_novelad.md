# V18-00: auditoria de novedad documental, correccion selectiva (P0)

Alcance: lectura de codigo, informes y registro. No se entrena, no se generan artefactos,
no se consulta el leaderboard ni se envia nada. Ninguna cifra de este informe es nueva.

Pregunta: existe ya una regla que use informacion auxiliar para decidir **cuando** aceptar
el cambio propuesto por un challenger, en vez de modificar el modelo global?

## 1. Correccion de un nombre erroneo en el encargo

El encargo cita `PLN-v11/kaggle_toxic_2026_puj/experiments/v11_p7_corrector_margenes.py`.
Ese fichero **no existe** en ninguna worktree (`find` sobre `/home/carlos/Git` no devuelve
ninguna coincidencia). Los dos ficheros reales y distintos son:

- `experiments/v8_08_corrector_margenes.py`, creado en v8, presente desde v8 hasta v18.
- `experiments/v11_07_correccion_prior.py`, creado en v11, correccion de odds por shift.

El nombre `v11_p7` procede de la **submission** que produjo el primero,
`submissions/toxic_submission_v11_p7_corrector_margenes.csv`, no del script.

## 2. Los dos indicios, perseguidos

### 2.1 El corrector sobre margenes: V8-08, ejecutado y cerrado

Informe `reports/v8_corrector_margenes.md`, con `.json` y `.csv`gemelos, generado por
`experiments/v8_08_corrector_margenes.py` el 2026-09-26T03:49:43, duracion 110,6 s.
**Se ejecuto**, con las submissions escritas y verificadas (integridad 7 de 7 correcta).

Mecanismo: un segundo modelo Ridge de cuatro variables de entrada por etiqueta, alimentado
por el margen con signo de V03 y, en una variante, por el desacuerdo entre cinco modelos de
pliegue con particiones distintas. El alfa se elige con sub-pliegues internos dentro de cada
pliegue. Los umbrales de V03 quedan congelados.

| rama | metrica | delta | IC95 | pliegues | celdas | acierto | veredicto |
|---|---|---|---|---|---|---|---|
| corrector_margen | 0.775148 | -0.002063 | [-0.01089, +0.00705] | 0/5 | 252 | 0.3611 vs minimo 0.5381 | no pasa |
| corrector_margen_y_desacuerdo | 0.775148 | -0.002063 | [-0.01089, +0.00705] | 0/5 | 252 | 0.3611 vs 0.5381 | no pasa |
| solo_desacuerdo | 0.417940 | -0.359271 | [-0.38528, -0.33291] | 0/5 | 2185 | 0.1648 | no pasa |
| promedio_de_repeticiones | 0.782277 | +0.005066 | [-0.00371, +0.01414] | 4/5 | 182 | 0.5495 vs 0.5527 | no pasa |

Veredicto registrado en D-064, Decision 3: "P7 confirma el patron por cuarta vez con el
corrector... La geometria de los margenes no lleva informacion que la decision de V03 no haya
usado". El filtro de detectabilidad de V8-02 exige acierto 0.5166; el observado es 0.3611.

**LB: no enviado.** En `resultados/v16_00_prevalencia_y_curva_lb.csv` la fila
`v11_p7_corrector_margenes` tiene la columna `lb` **vacia**, con 192 celdas distintas de V03
y un `lb_ajustado` de 0.7858 que es una estimacion interna, no un score de Kaggle. Por D-063
el envio era manual y recayo en la otra rama del mismo script, el promedio de repeticiones,
que si se envio como `v12_p7_promedio_repeticiones` con **LB 0.77122**, es decir -0.01184
respecto a V03.

**Lo importante para V18**: el corrector de V8-08 es un modelo que **sustituye** la
prediccion de forma incondicional, no una regla que decide celda a celda si el cambio del
challenger se acepta. No hay mascara, no hay interruptor, no hay aceptacion condicional.

### 2.2 El eje "error correction" de v12: ejecutado, no solo catalogado

`reports/v12_gap_audit.md` y `.csv`, generados por `experiments/v12_00_auditoria_brecha.py`
(2026-09-26T22:27:50). La fila del eje **error correction** es la familia
"Corrector sobre margenes y desacuerdo entre pliegues", con `experiment_ids` = **V8-08**,
`best_result` = "192 de 3462 celdas cambian, en 161 filas", estado **PARTIALLY_TESTED**.

La distincion pedida es exacta aqui y hay que sostenerla: **no es un hueco catalogado**.
V12 lo cataloga como parcialmente probada porque el mecanismo base si se midio y fallo
(-0.002063, tasa 0.3611, 0 de 5 pliegues), y unicamente deja abierta una forma distinta:
"interacciones **multiplicativas** entre margenes, que el corrector lineal no puede
representar". Es decir, lo que V12 registro como pendiente es una extension del corrector,
no una regla de aceptacion. Y no existe script ni informe posterior a v12 que implemente
esas interacciones: `v13` a `v17` no vuelven a abrir el eje, y `v13_gap_audit.md` cierra la
pista de forma explicita, con la entrada "Corrector o combinacion de margenes en inferencia |
V8-08 | SI para la parte de P3 que combina scores en inferencia".

## 3. El resultado de v15, localizado con precision y correctamente clasificado

Ficheros exactos:

- Script: `experiments/v15_01_gap_y_ortogonalidad.py` (P1 matriz de gap, P3 ortogonalidad,
  P4 gateado por P3), worktree `PLN-v15`.
- Informe de la medicion: `reports/v15_01_gap_y_ortogonalidad.md`, con
  `reports/v15_01_gap_y_ortogonalidad.json` en `PLN-v15`.
- Informe de cierre: `reports/v15_02_informe_cierre.md`, generado por
  `experiments/v15_02_informe_cierre.py`, secciones D senal residual y E resultado de V15.
- Inventario previo: `experiments/v15_00_inventario_senal.py` y
  `reports/v15_00_inventario_senal.md`.
- Entrada del registro: fila `V15-01` de `experiments.csv`, decision "NO ADOPTADA. 0 candidatas".

Cifras confirmadas, las tres que cita el encargo:

| rasgo | AUC en estratos de V03 | p empirico | macro-F1 | delta | celdas | acierto entre las modificadas |
|---|---|---|---|---|---|---|
| primer_token_es_mencion | 0.5547 | 0.0000 | 0.774344 | -0.002868 | 50 | 0.0000 |
| posicion_primera_mencion | 0.5491 | 0.0000 | 0.775860 | -0.001352 | 14 | 0.0000 |
| mencion_en_primeros_dos | 0.5547 | 0.0000 | 0.774174 | -0.003037 | 57 | 0.0000 |

Ademas una cuarta, `proporcion_antes_de_la_mencion` con AUC en estratos 0.4453, clasificada
**YA VISIBLE EN V03** y por tanto descartada en P3, sin llegar a P4.

**La distincion que pedia el encargo, resuelta**: el trabajo de v15 fue **anadir features a
V03**, no una regla que decida cuando aceptar un cambio de otro modelo. La evidencia es
literal en el codigo y en el informe. En `experiments/v15_01_gap_y_ortogonalidad.py` la
funcion se llama `_adicion_a_v03`, y la nota del JSON dice "La columna se estandariza con la
media y el desvio del pliegue de entrenamiento, y se anade al final de la matriz. Todo lo
demas queda congelado." El informe lo describe igual: "Se mantienen congelados la
representacion, el modelo, la constante y los umbrales: lo unico que cambia es que la matriz
lleva una columna mas, con el mismo escalar que el resto."

Es decir: **V03 mas una feature**, no **V03 mas un challenger mas un gate**. El cambio se
aplica globalmente a la matriz, no condicionalmente por fila. Lo que v15 mide es si la senal
residual **existe** (AUC estratificado por el margen de V03 con permutacion como null), y
despues si **anadida como columna** mejora. El acierto 0.0000 se produce porque la feature
empuja en el sentido contrario en el punto de corte del umbral congelado, no porque una
regla de aceptacion rechazara las celdas: no habia ninguna regla que las rechazara.

## 4. Tabla de mecanismos candidatos a correccion selectiva

Estado: **ejecutado** = corrio y dio veredicto; **catalogado y no ejecutado** = existe la
entrada de inventario pero no hay script ni informe que lo implemente; **descartado** =
ejecutado con veredicto negativo registrado.

| mecanismo | fase | script | estado | veredicto | LB |
|---|---|---|---|---|---|
| Corrector lineal sobre margenes de V03, 4 variables | v8 (V8-08) | `experiments/v8_08_corrector_margenes.py` | ejecutado y descartado | -0.002063, tasa 0.3611, 0/5 pliegues. D-064 Decision 3 | no enviado, celda `lb` vacia |
| Corrector de margenes mas desacuerdo entre cinco pliegues | v8 (V8-08) | mismo script | ejecutado y descartado | identico al anterior, el desacuerdo no aporta | no enviado |
| Solo desacuerdo entre pliegues como senal | v8 (V8-08) | mismo script | ejecutado y descartado | -0.359271, 2185 celdas | no enviado |
| Eje "error correction" del inventario v12 | v12 (V12-00) | `experiments/v12_00_auditoria_brecha.py` | ejecutado como auditoria; la familia que cataloga ya estaba ejecutada | PARTIALLY_TESTED, cerrada por V8-08; solo queda abierta la interaccion multiplicativa entre margenes | n/a |
| Interacciones multiplicativas entre margenes | v12 | ninguno | **catalogado y no ejecutado** | sin script ni informe en v13 a v17 | n/a |
| Correccion de odds por shift de prevalencia | v11 (V11-07) | `experiments/v11_07_correccion_prior.py` | ejecutado y descartado | VIA CERRADA, macro medio -0.000293, 0/3 intervalos excluyen el cero, gate A y B no superados | n/a, no genera submission |
| Anadir rasgos de posicion a la matriz de V03 | v15 (V15-01) | `experiments/v15_01_gap_y_ortogonalidad.py` | ejecutado y descartado | -0.002868, -0.001352, -0.003037, acierto 0.0000 en las tres | no genera submission |
| Correccion de HS con propagation por near-duplicados | v9 (V9-03) | `experiments/v9_03_near_dup_test.py` | ejecutado | 8 pares nivel A, 4 tuits de test, **0 celdas que cambian**; sonda escrita | no enviado, celda `lb` vacia |
| Post-proceso jerarquico con HS predicho como llave | v6, remedido en v12 (V12-02) | `experiments/02_baseline_tfidf_lr.py`, remedido en `experiments/v12_02_perdida_y_estructura.py` | ejecutado y descartado | -0.004187; la llave falla en 143 de 1483 filas | n/a |
| Mascara con HS verdadero (oraculo, no adoptable) | v12 (V12-02) | mismo script | ejecutado como diagnostico | +0.046271, techo mas alto medido; no adoptable por usar la verdad | n/a |
| Umbral local por rebanada con shrinkage | v2 (E2-16) | registrado en `experiments.csv` | ejecutado y descartado | 0.7750, CERRADA sin mejora | n/a |
| Especialistas por rebanada | v2 (E2-20) | registrado en `experiments.csv` | ejecutado y descartado | 0.7749, los especialistas rinden -0.045 dentro de su rebanada | n/a |
| Factorizacion condicional suave entre etiquetas | v6 (V6-B) | `experiments/v6_02_label_factorization.py` | ejecutado y descartado | B1 -0.0107, B2 -0.0008, B3 -0.0115, 1 rescatada frente a 35 rotas | n/a |
| Seleccion de rasgos por grupo con criterio de informacion | v8 (V8-12) | `experiments/v8_12_seleccion_por_grupo.py` | ejecutado y descartado | grupo_25 +0.005324 pero solo 2/5 repeticiones, tasa 0.5057 | submission escrita, no recomendada |
| Meta-modelo sobre scores OOF, 44 a 65 columnas | v2 (E2-02, E2-18) | `experiments/17_stacking_v2.py` | ejecutado y descartado por transferencia | 0.7853 y 0.7874 local, adoptados en su gate, **LB 0.78163 frente a 0.78306** | si enviado, 0.78163 |
| Correccion de la estimacion de HS en cascada sobre TR y AG | v13 | `experiments/v13_00_auditoria_hs.py` | ejecutado | la correccion deja TR y AG con delta **exactamente 0.000000** | n/a |

Sobre la fila E2-02 y E2-18: es el precedente mas duro contra V18, porque es el unico caso
del proyecto en que un modelo auxiliar **supera** a V03 en local (+0.0102) y **pierde** en el
externo (-0.00143). La interseccion de errores entre las bases se midio en 12 filas.

## 5. Terminos buscados que NO devolvieron nada

Busqueda sobre `reports/`, `experiments/`, `experiments.csv`, `DECISIONS.md`, `CHAMPION.md`,
`PROJECT_STATE.md`, `RUNLOG.md` en las worktrees `PLN-v7`, `-v8`, `-v11`, `-v12`, `-v13`,
`-v14`, `-v15`, `-v16`, `-v17`, `-v18`. Recuento de ficheros que coinciden:

| termino | ficheros |
|---|---|
| correccion selectiva | 0 |
| selective correction | 0 |
| capa de correccion | 0 |
| cascada | 0 |
| cascade | 0 |
| abstain | 0 |
| abstencion | 0 |
| desistir | 0 |
| conmutador | 0 |
| interruptor | 0 |
| switch | 0 |
| if-then | 0 |
| criterio de aceptacion | 0 |
| regla de aceptacion | 0 |
| acepta el cambio | 0 |
| aceptar el cambio | 0 |
| solo cuando | 0 |
| unicamente cuando | 0 |
| subconjunto de filas | 0 |
| en las filas donde | 0 |
| condicionada por | 0 |
| gate por feature | 0 |
| test de aceptacion | 0 |
| cambia solo | 0 |
| cambia unicamente | 0 |
| unicamente en | 0 |
| solo en las filas | 0 |
| reemplaza la prediccion | 0 |
| anula la prediccion | 0 |
| aplique el cambio | 0 |

Terminos que **si** devolvieron algo, y que son los que sostienen la ausencia anterior:
`corrector` (V8-08 y sus referencias), `correccion` (V11-07, V13, catalogs), `error
correction` (solo el eje del inventario v12), `routing` (V6-B, post-proceso jerarquico),
`post-proceso` y `postproceso` (42 y 16 ficheros, siempre el post-proceso de HS como llave),
`mascara` (77 ficheros, siempre mascara de regimen o de oraculo), `llave` (28 ficheros, la
llave de HS), `propagacion` y `propagar` (V9-03, near-duplicados), `near-duplicado` (68
ficheros, grupos de CV y V9-03), `aceptacion` (3 ficheros, `embedding_00_scaffold_v5.py`,
`corpus_manifest_v5.json` y `embedding_compliance.md`, los tres sobre cumplimiento de
licencia de corpus, sin relacion con aceptar una prediccion).

Esta distincion importa y hay que sostenerla: la ausencia de "correccion selectiva" como
frase no es la prueba, porque el proyecto no nombra sus mecanismos asi. La prueba es que
**todos** los mecanismos que si existen y que se aproximan a la idea fueron ejecutados, y
que ninguno tiene la forma de una regla de aceptacion condicional por fila. Los dieciocho
mecanismos de la seccion 4 son el universo, y todos son modelos, reglas de umbral, mascaras
de regimen o reglas logicas de etiqueta, ninguno es un gate de aceptacion de un challenger.

## 6. Veredicto

**La formulacion de V18 es genuinamente nueva en este proyecto, y es importante distinguirla
de V8-08, que es lo unico que se le parece en el nombre y no en el mecanismo: V8-08
ejecuto un corrector que **sustituye** la prediccion de forma incondicional, y v15-01
ejecuto **anadir features a V03**, no una regla de aceptacion; ninguno de los dos decide
celda a celda si acepta el cambio propuesto por un challenger, y las dieciseis familias
restantes del registro tampoco.** El precedente que mas pesa en contra no es de este eje
sino de otro: E2-02 y E2-18, donde un modelo auxiliar que ganaba +0.0102 en local perdio
0.00143 en el externo, y donde la interseccion de errores entre bases se midio en 12 filas
de 1483.

Dos advertencias para la fase, dictated por la evidencia y no por cautela generica:

1. El techo de este proyecto es de **+0.000148** como mejor ganancia anidada contra un gate
   de +0.002, con una banda de ruido entre particiones equivalentes de **0.017184** y un
   desacuerdo medio entre cinco modelos de **0.92**. Si la regla de V18 mueve menos que eso,
   no se distinguira del ruido, y el registro lo dice sin ambiguedad.
2. La unica senal residual demostrable del proyecto, la posicion de la primera mencion, ya
   dio acierto **0.0000** en las celdas modificadas. Que una senal ordenar bien los ejemplos
   y aun asi empujar en el sentido contrario en el punto de corte es un precedente directo
   contra laypothesis de que "AUC alto dentro de los estratos" implique "cambio aceptable".
   Si V18 usa esa senal como regla de aceptacion, tiene que medir el acierto entre las
   celdas donde la regla **acepta** y no solo entre las que el challenger **cambia**.

## 7. Donde queda el hueco, con la distincion pedida

- **Ejecutado y descartado**: corrector sobre margenes (V8-08), correccion de prior (V11-07),
  features de posicion (V15-01), routing duro y mascara de HS (V12-02), facturacion
  condicional (V6-B), rebanadas con umbral o modelo propio (E2-16, E2-20), meta-modelo con
  rechazo por transferencia (E2-02, E2-18), seleccion por grupo (V8-12).
- **Catalogado y no ejecutado**: la interaccion multiplicativa entre margenes, entrada de
  `v12_gap_audit.csv` sin script ni informe posterior. Esto es lo unico que el registro deja
  abierto dentro del eje error correction, y es una extension del corrector, no una regla de
  aceptacion.
- **Genuinamente sin probar, y es el que V18 propone**: una regla que use informacion
  auxiliar para decidir **cuando aceptar** el cambio de un challenger, celda a celda. Ningun
  termino la nombra y ningun mecanismo registrado la implementa.

Fecha: 2026-09-26. Auditoria documental, sin entrenamiento y sin consulta al leaderboard.
