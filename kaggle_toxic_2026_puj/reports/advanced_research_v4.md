# Advanced Research v4 - Informe de cierre

Fecha: 2026-09-25. Rama: `proyecto-kaggle-toxic-2026-v4`. Spec:
`docs/advanced_research_v4_spec.md` (108 secciones). Decisiones: D-025 a D-029.

**Veredicto**: fase cerrada con la **condicion de terminacion B** de la spec
(§102): las hipotesis nuevas razonables se investigaron y se cerraron con
evidencia, **sin una mejora transferible**. V03 (LB 0.78306) se mantiene como
modelo y submission finales.

## 1. Protocolo y resultado de cabecera

| Elemento | Valor |
|---|---|
| Referencia (V03) | `corrected_cv` = 0.777211, LB 0.78306, congelada y verificada byte a byte en P0 |
| Holdout | 248 tweets; 4/5 evaluaciones usadas al abrir la fase; **5/5** al cerrarla (solo para el finalista) |
| Sondas de LB | 12 registradas; **0 nuevas** en v4 |
| Candidatos evaluados | 4 (A, B, C, D) con evidencia multiple |
| Candidatos que superan los 3 gates de D-016 | 1 (C) |
| Resultado de ese candidato en el holdout | **0.8163 vs 0.8285 de V03 (delta -0.0122): RECHAZADO** |

## 2. Resumen por track

| Track | Lineas | Resultado | Evidencia |
|---|---|---|---|
| A - Integridad de datos y ruido de etiqueta | V4-A01, A02, A03 | **cerradas**: la contradiccion textual es escasa (2 clusters, 2.8 % de los ejemplos) y **no** explica el error (el modelo acierta los ambiguos) | `reports/label_consistency_v4.csv`, `reports/hardness_vs_noise_v4.md` |
| B - Dureza de ejemplos y reweighting | V4-B01..B05 | **cerradas**: el hard example mining empeora (-0.003 a -0.011); solo mejora **bajar** el peso de los ejemplos que mas violan el umbral | `reports/reweighting_v4.csv` |
| C - Weighting de la representacion | V4-C01..C06 | **cerrada**: log-odds, chi2, MI, BM25 y binario **degradan**; la causa no es el desajuste con L2 (C creciente empeora mas) | `reports/supervised_weighting_v4.csv` |
| D - Representaciones sparse nuevas | V4-D01..D04 | **cerrada**: afijos, interacciones y ventanas como tokens no aportan | `reports/structural_sparse_v4.csv` |
| E - Modelo conjunto de etiquetas | V4-E01..E03 | **cerrada**: Label Powerset (6 y 4 clases) no supera a tres binarios independientes (mejor 0.7713) | `reports/joint_label_v4.csv` |
| F - Robustez y shift | V4-F01..F04 | **cerrada**: sin shift significativo (2/10 variables), la vecindad no explica el error, la propagacion por near-duplicado tiene techo nulo | `reports/distribution_shift_v4.csv` |
| G - Modelos estadisticos alternativos | V4-G01..G03 | **mixta**: la verosimilitud class-conditional aporta +0.0041 en fusion; el SVC RBF sobre SVD 300 alcanza 0.7904 con IC frágil | `reports/class_conditional_lm_v4.csv` |
| H - Fusion y calibracion del meta | V4-H01, H02 | **cerrada**: el shift de calibracion **existe pero es pequeño** (KS 0.062) y **no causa dano**; la hipotesis abierta de v2 queda refutada | `reports/meta_shift_v4.csv` |

## 3. Los cinco hallazgos que la fase aporta

### 3.1 El ruido de etiqueta es escaso y no es la fuente del error

Con cuatro criterios de agrupamiento hay 15 clusters de texto repetido (84
tweets, 4.85 %) y solo **2 contradictorios**, concentrados en un unico cluster de
48 tweets con el mismo patron de insulto (38 anotados `000` frente a 10 con
`110`/`111`). El error OOF de V03 es **menor** dentro de los clusters
contradictorios (HS 0.046 vs 0.218). Conclusion: la ambiguedad de anotacion
existe pero el modelo la resuelve bien, y las tres estrategias de la spec §8
(downweight +0.0005, remove -0.0004) no mueven la metrica.

### 3.2 La dureza no se debe medir con la perdida BCE

Con `class_weight='balanced'` y umbrales por debajo de 0.5, la dificultad por
etiqueta ordena bien el error (`p_true` vs error: r = -0.795/-0.810/-0.801) pero
**agregar las tres etiquetas rompe el orden**: la BCE media correlaciona -0.136
con el numero de errores, mientras que la distancia a la frontera correlaciona
+0.345 y la violacion del umbral +0.811. La razon es de compensacion: la mediana
de la BCE baja de 1.34 (0 errores) a 0.58 (3 errores). La spec §11 proponia
ponderar por perdida OOF; en este corpus esa instruccion es tecnicamente
defectuosa y se sustituyo por la medida de frontera.

### 3.3 El reweighting funciona con el signo contrario al esperado

Subir el peso a los ejemplos duros empeora de forma consistente (`medium_up`
-0.0101 con IC negativo). Bajar el peso de los ejemplos que **mas violan** el
umbral mejora: +0.0036 en LR y +0.0086 en SVC sobre su propio control. La lectura
es que en este corpus los ejemplos que el modelo falla son frontera irreductible
con la representacion disponible, y excluirlos del entrenamiento permite un
limite de decision mas limpio.

### 3.4 Reponderar las features empeora, y no por la penalizacion

Log-odds por etiqueta 0.7571, chi2 0.6622, MI 0.7378 y BM25 0.7748-0.7755,
todos por debajo de la referencia 0.7772, con una degradacion monotone en la
agresividad del esquema. La hipotesis de que el problema era el desajuste con la
penalizacion L2 se refuto: con C = 2, 10, 50 y 200 el resultado cae de 0.7571 a
0.7435. El peso externo compite con lo que el clasificador ya aprende.

### 3.5 El shift de calibracion del meta no explica la no transferencia

La hipotesis abierta de `advanced_research_v2.md` §4.3 se midio por primera vez
dentro del desarrollo: el desplazamiento entre el regimen de entrenamiento del
meta (bases al 75 %) y el de inferencia (bases al 100 %) es pequeno (KS medio
0.062) y **no causa dano**: el meta con bases completas (0.7828) supera al meta
entrenado y evaluado en el mismo regimen (0.7791). La alineacion por cuantiles
recupera +0.00005 y Platt empeora -0.017. La no transferencia del meta v2 hay que
explicarla por otra via: meta y V03 son indistinguibles en el test.

## 4. Los cuatro candidatos y su destino

| Candidato | corrected_cv | delta | pliegues | IC95 pareado | correccion neta | destino |
|---|---|---|---|---|---|---|
| A `cost_sensitive_downweight` | 0.7854 | +0.0082 | 5/5 | [-0.0002, +0.0117] | +10 | IC toca 0: no pasa |
| B `meta36` | 0.7828 | +0.0056 | 4/5 | [+0.0031, +0.0166] | +22 | nivel menor que C: no finalista |
| **C `meta36_downweight`** | **0.7882** | **+0.0110** | 4/5 | **[+0.0054, +0.0200]** | **+34** | unico que pasa los 3 gates: **evaluado en holdout** |
| D `svd300_rbf` | 0.7904 | +0.0132 | 5/5 | [-0.0011, +0.0148] | +30 | IC toca 0: no pasa |

**Holdout del finalista C**: 0.8163 frente a 0.8285 de V03 en las mismas 248
tweets, delta **-0.0122**. El holdout estaba congelado desde la fase v2 y no
participo en ninguna decision de v4, por lo que es la evidencia externa mas
fiable disponible. El candidato se descarta; su submission (v12) se conserva
como artefacto reproducible, marcada `DESCARTADO` en el manifest.

## 5. Lo que la fase deja como evidencia general

1. **El corrected_cv no es un selector fiable en este problema.** Es el cuarto
   caso consecutivo de mejora local que no transfiere (Optuna 0.8039 -> LB
   0.77602; meta v2 0.7874 -> LB 0.78163; candidatos v09-v11; y ahora C). Con un
   error estandar del test de ~0.013 y mejoras seleccionadas de +0.005 a +0.011,
   la diferencia entre "mejor modelo local" y "mejor modelo en el test" es del
   orden del ruido.
2. **El rendimiento depende de la memorizacion en un subconjunto pequeno**: V03
   alcanza 0.9769 en los 68 ejemplos dentro de clusters de near-duplicado y
   0.7568 en los 1.415 textos unicos. La metrica agregada esta inflada por ese
   4.6 %.
3. **Lo que si aporta estructura nueva** (verosimilitud class-conditional,
   reponderacion por violacion, no linealidad sobre compresion) mejora entre
   +0.004 y +0.013 en desarrollo, siempre por debajo del margen de error del
   test; ninguno de esos mecanismos cambia la conclusion de que el techo actual
   es de datos y de representacion, no de modelo.

## 6. Cumplimiento de la spec y artefactos

- Matriz de cumplimiento por seccion: `reports/spec_compliance_v4.md`.
- Auditoria de reglas y reproducibilidad: `reports/rule_audit_v4.md`,
  `reports/reproducibility_v4.md`, `reports/auditoria_v4.json`.
- Cobertura de senales: `reports/signal_coverage_v4.csv` (actualizada).
- Gap analysis: `reports/v4_gap_analysis.md`; backlog: `reports/research_backlog_v4.md`.
- Revision bibliografica 2022-2026: `reports/estado_arte_v4.md` (38 papers
  verificados) y `.csv`.
- Scripts: `experiments/26..38` (13 scripts nuevos, ninguno modifica los previos).
- Registro: `experiments.csv` pasa de 128 a 143 filas; `DECISIONS.md` D-025..D-029.
