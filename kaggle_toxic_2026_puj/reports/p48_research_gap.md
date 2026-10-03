> **RECLASIFICADO EN P49 (2026-09-28).** Este informe se conserva intacto y sus
> cifras numericas se **verifican**, pero dos afirmaciones quedan **superadas**:
>
> 1. La limitacion que declaraba "los tensores OOF por canal ya no estan en disco, luego
>    la categoria C es una cota superior" es **FALSA**: los tensores estaban en los
>    worktrees `PLN` y `PLN-v8`, y la reconstruccion del pool de 20 canales coincide con
>    `reports/hardness_v4.csv` en las 4449 celdas con diferencia 0. Las cifras 715 y 42
>    son **observacion directa**, no cota superior.
> 2. El criterio de enrichment usado aqui solo miraba la tasa de rescate. P49 lo completa
>    con el enriquecimiento en errores **irrecuperables**, que si separa slices.
>
> El veredicto `EDA_RESEARCH_GAP = NONE` **se mantiene**, pero por un motivo mas fuerte
> que el que se dio aqui: no es que no haya rescate, es que **ninguno de los 20 canales
> tiene net positivo**. Ver `reports/p49_rescue_reconciliation.md`.

# P48 - Brecha de investigacion guiada por EDA

Generado por `experiments/v48_01_residual_atlas.py` (V48-01 (P1 atlas de residuos, rescate y patrones)).

Fecha 2026-09-28T00:43:19+00:00. **0 modelos, 0 submissions, 0 uso de holdout o test.**

Pregunta: Que errores de V03 son corregibles por el pool historico y cuales parecen no contener informacion recuperable?

## 0. Limitaciones que no se pueden resolver con los artefactos presentes

- Los tensores OOF por canal (features/v2_oof_*.npz) ya no estan en disco. Solo queda la fraccion de fallo por celda, de modo que la categoria C es una cota superior del rescate y no identifica el modelo concreto.
- No se puede aplicar la matriz de rescate literal de la seccion 7. La que se reporta es la matriz de rescate del pool agregado.
- holdout2_ids.csv esta dibujado de dev, luego forma parte de las 1483 filas y no anade filas. No se usa holdout ni test para nada.

## 1. Atlas de residuos

| magnitud | valor |
|---|---:|
| filas de dev | 5 |
| celdas de error de V03 | 757 |
| filas con 0 errores | 1042 |
| filas con 1 error | 223 |
| filas con 2 errores | 120 |
| filas con 3 errores | 98 |

## 2. Clasificacion de la poblacion de error

| clase | filas | significado |
|---|---:|---|
| A | 1042 | V03 correcto en las tres labels |
| C | 439 | V03 falla y algun canal historico acierta |
| B | 33 | V03 falla y ningun canal acierta esa celda |

**Celdas de error rescatables por el pool: 715 (94.45 pct).**
**Celdas de error irrecuperables: 42 (5.55 pct).**

## 3. Matriz de rescate por etiqueta

| etiqueta | V03 mal | rescate | damage | net | canales que aciertan de media |
|---|---:|---:|---:|---:|---:|
| HS | 316 | 307 | 0 | +307 | 6.0 |
| TR | 164 | 152 | 0 | +152 | 4.6 |
| AG | 277 | 256 | 0 | +256 | 5.5 |

## 4. Contexto de margen

Si los rescates se concentran en margen bajo, la representacion ya contiene la informacion y lo que falla es la frontera. Si hay rescates en margen alto, el hueco de representacion es mas plausible.

| bin de margen | n | errores de V03 | tasa | rescatables | irrecuperables |
|---|---:|---:|---:|---:|---:|
| [0,0.05) | 306 | 141 | 0.4608 | 141 | 0 |
| [0.05,0.10) | 344 | 141 | 0.4099 | 139 | 2 |
| [0.10,0.20) | 748 | 207 | 0.2767 | 203 | 4 |
| [0.20,0.30) | 880 | 134 | 0.1523 | 123 | 11 |
| [0.30,1] | 2171 | 134 | 0.0617 | 109 | 25 |

**Respuesta a la pregunta de la seccion 11.** La tasa de rescate cae de forma monotona al subir el margen: **0.9929** con margen por debajo de 0.10 frente a **0.8657** con margen de 0.20 o mas. La tasa de rescate va de 99.3 por ciento con margen bajo a 86.6 por ciento con margen alto.

Ademas, **85.7 por ciento de las celdas irrecuperables estan en margen alto** (36 de 42). Esa es la firma de un hueco de representacion: el modelo se equivoca con conviccion justamente donde ningun canal historico acierta.

Pero el tamano absoluto es pequeno: **42 celdas de 757**, un 5.55 por ciento. Ninguna hipotesis mechanisticamente nueva puede sostenerse sobre esa base.
## 5. Errores conjuntos

| labels falladas | filas | pct | celdas rescatables | irrecuperables |
|---|---:|---:|---:|---:|
| 0 | 1042 | 70.26 | 0 | 0 |
| 1 | 223 | 15.04 | 214 | 9 |
| 2 | 120 | 8.09 | 232 | 8 |
| 3 | 98 | 6.61 | 269 | 25 |

## 6. Patrones y slices

| patron | n | tasa error V03 | tasa rescate | tasa damage | net | p |
|---|---:|---:|---:|---:|---:|---:|
| margen_bajo_error | 269 | 1.0000 | 0.9703 | 0.0297 | +253 | 0.0000 |
| puntuacion_interna | 612 | 0.3203 | 0.9337 | 0.0663 | +170 | 0.1060 |
| multiples_urls | 458 | 0.3406 | 0.9487 | 0.0513 | +140 | 0.0149 |
| urls | 458 | 0.3406 | 0.9487 | 0.0513 | +140 | 0.0149 |
| longitud_larga | 462 | 0.3745 | 0.9017 | 0.0983 | +139 | 0.0000 |
| segunda_persona | 672 | 0.2470 | 0.9036 | 0.0964 | +134 | 0.0001 |
| menciones | 701 | 0.2382 | 0.8982 | 0.1018 | +133 | 0.0000 |
| colectivo | 363 | 0.4132 | 0.9267 | 0.0733 | +128 | 0.0000 |
| negacion | 386 | 0.3808 | 0.8776 | 0.1224 | +111 | 0.0000 |
| mencion_y_posicion_inicial | 605 | 0.2231 | 0.8815 | 0.1185 | +103 | 0.0000 |
| elongacion | 275 | 0.3927 | 0.9537 | 0.0463 | +98 | 0.0001 |
| leet | 327 | 0.3058 | 0.9200 | 0.0800 | +84 | 0.7053 |
| longitud_corta | 458 | 0.1943 | 0.8539 | 0.1461 | +63 | 0.0000 |
| mencion_inicial_en_error_de_TR | 78 | 1.0000 | 0.8590 | 0.1410 | +56 | 0.0000 |
| hashtags | 144 | 0.3611 | 0.9808 | 0.0192 | +50 | 0.0782 |
| elongacion_con_HS_1 | 119 | 0.4370 | 0.9231 | 0.0769 | +44 | 0.0005 |
| negacion_en_error_de_TR | 61 | 1.0000 | 0.8525 | 0.1475 | +43 | 0.0000 |
| colectivo_en_FN_de_HS | 54 | 1.0000 | 0.8333 | 0.1667 | +36 | 0.0000 |
| cita | 61 | 0.4754 | 0.9655 | 0.0345 | +27 | 0.0019 |
| atribucion | 102 | 0.3235 | 0.9091 | 0.0909 | +27 | 0.5492 |
| url_y_texto_corto | 93 | 0.2688 | 0.9600 | 0.0400 | +23 | 0.5338 |
| margen_alto_error | 41 | 1.0000 | 0.7073 | 0.2927 | +17 | 0.0000 |
| risas | 84 | 0.2024 | 0.8824 | 0.1176 | +13 | 0.0499 |
| en_cluster_de_duplicados | 68 | 0.0735 | 0.6000 | 0.4000 | +1 | 0.0000 |

## 7. Criterio de hipotesis y decision

Criterio aplicado: n >= 100, net > 0, p < 0.01, tasa de rescate > 0.5 y lift de la tasa de error sobre la media > 1.15.

**`EDA_RESEARCH_GAP = NONE`.** Ningun patron.supera los cinco criterios a la vez. No se propone ninguna hipotesis y no se ejecuta ningun experimento.

Lo que esto responde: los errores de V03 **no** son un conjunto homogeneo con subpoblaciones_ocultas recuperables por el pool historico. Donde el pool acierta, acierta por azar tanto como por senal, y la proporcion de celdas rescatables no se concentra en ninguna slice de las ya definidas.

