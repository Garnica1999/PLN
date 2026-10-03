# v6: analisis de brechas (P0)

Generado por `experiments/v6_00_gap_analysis.py` (V6-00 (P0, ANALYSIS)).
No entrena nada. Se ejecuta antes de gastar una sola unidad de la fase v6.

Cada afirmacion de la tabla se apoya en una **verificacion programatica** sobre el
codigo real del proyecto, registrada en `reports/v6_verificaciones_gap_v6.json`. Si
alguien implementa hoy un `char_weight`, la tabla pasara a marcar el track I como
duplicado sin que nadie tenga que acordarse.

## 1. Inventario de lo ya experimented

El registro tiene **174 filas**, de las cuales **6 estan
desalineadas** (seccion 2). El numero real de experimentos es por tanto
**168**: la especificacion de v6 estimaba 170 y la
diferencia son los experimentos de v5 y v5.1 mas los de esta fase. El proyecto
acumula **43 decisiones**. Por prefijo de identificador:

| prefijo | experimentos |
|---|---|
| `E2-*` | 58 |
| `EBD-*` | 20 |
| `V4-*` | 14 |
| `PRE-*` | 10 |
| `FIN-*` | 10 |
| `MOD-*` | 9 |
| `REP-*` | 6 |
| `E-*` | 5 |
| `ENS-*` | 4 |
| `HIB-*` | 4 |
| `DL-*` | 4 |
| `V2-*` | 4 |
| `AUD-*` | 3 |
| `E00-*` | 1 |
| `B1-*` | 1 |
| `TH-*` | 1 |
| `S02-*` | 1 |
| `S03-*` | 1 |
| `S04-*` | 1 |
| `B5-*` | 1 |
| `OPT-*` | 1 |
| `S05-*` | 1 |
| `R1-*` | 1 |
| `R2-*` | 1 |
| `S06-*` | 1 |
| `S07-*` | 1 |
| `PA-*` | 1 |
| `HOLD-*` | 1 |
| `S08-*` | 1 |
| `P8-*` | 1 |

## 2. Integridad del registro (hallazgo de esta auditoria)

**6 de las 174 filas de `experiments.csv` tienen las
columnas desplazadas**: el campo `experiment_id` contiene el nombre de la competicion
en lugar de un identificador, y el resto de campos ocupa una posicion de mas.

| linea | experiment_id | competition |
|---|---|---|
| 147 | `KAGGLE_TOXIC_2026_PUJ` | `2026-09-25` |
| 148 | `KAGGLE_TOXIC_2026_PUJ` | `2026-09-25` |
| 149 | `KAGGLE_TOXIC_2026_PUJ` | `2026-09-25` |
| 150 | `KAGGLE_TOXIC_2026_PUJ` | `2026-09-25` |
| 151 | `KAGGLE_TOXIC_2026_PUJ` | `2026-09-25` |
| 152 | `KAGGLE_TOXIC_2026_PUJ` | `2026-09-25` |

Las filas **no se tocan**: son evidencia historica y el principio del proyecto es no
reescribir resultados. Lo que se hace es (a) contarlas aparte para no inflar el
inventario y (b) verificar el formato en cada escritura a partir de ahora, que usa
`csv.DictWriter` con los nombres de campo y por tanto no puede desalinearse. El
inventario cuenta solo filas bien formadas.

## 3. Familias completamente cerradas

Agrupadas por mecanismo, no por identificador:

| familia |experimentos | veredicto con evidencia |
|---|---|---|
| Metrica y preprocesamiento de superficie | E00, PRE-00..09 | cerrada: 10 transformaciones deterministas, mejor 0.7623 frente a 0.7587 |
| Representacion TF-IDF y clasificador lineal | REP-01..06, MOD-01..04 | cerrada: la union word+char es V03 |
| Umbrales por etiqueta | TH-01, S02, S03 | cerrada: adoptados y congelados en V03 |
| Ensembles heterogeneos | ENS-01..04, S04 | cerrada: mejor anidado 0.7949 que no transfiere (LB 0.77510) |
| Features densas (afijos, registro, objetivo) | HIB-H11/H12/H14, B5-DENSE | cerrada: todas descartadas |
| Optimizacion de hiperparametros | OPT-01, S05 | cerrada: OOF 0.8039 que cae a LB 0.77602 |
| Control de capacidad | R1-C1, R2-MINDF5, S06, S07 | cerrada: el gap train-OOF no predice el LB |
| Deep learning con embeddings aleatorios | DL-01, DL-02, DL-03 | cerrada: CharCNN 0.7587 y BiLSTM por debajo de B1 0.7927 |
| Cadenas, meta y dependencias | E2-01, E2-18, S08 | cerrada: el meta no transfiere (LB 0.78163) |
| Canales sparse sintacticos y semanticos | E2-03..E2-10 | cerrada: todos entre -0.011 y -0.25 |
| NB-SVM, calibracion y target encoding | E2-13, E2-14, E2-15 | cerrada: entre -0.016 y -0.039 |
| Especialistas y espacios por etiqueta | E2-16, E2-20, E2-21 | cerrada: sin mejora |
| v4: ruido, dureza y ponderacion | V4-A01..A02, B01..B05, H01 | cerrada: el ruido es escaso; solo el downweight de frontera dio senal pequena |
| v4: sparse estructural | V4-D01..D04 | **cerrada**: interacciones como tokens 0.7426, ventanas 0.7734 |
| v4: modelo conjunto de etiquetas | V4-E01..E03 | **cerrada**: 12 variantes de label powerset, mejor 0.7713 |
| v4: vecindad y shift | V4-F01..F04 | **cerrada**: la afinidad no ordena el error |
| v5 y v5.1: representacion distribuida | E-CORP-01..09, EBD-00..27 | cerrada: 14 configuraciones, 0 adoptables |

## 4. La tabla de brecha

| familia | ya probada | realmente cerrada | prioridad |
|---|---|---|---|
| A - Ponderacion estructural | no | no | alta |
| A2 - Grupo + frontera | parcial | no | media |
| B - Factorizacion condicional de etiquetas | parcial | no | alta |
| C1 - Skip-grams | no | no | media |
| C2 - Canales posicionales | si | si | nula: no ejecutar |
| C3 - Tokens de interaccion | si | si | nula: no ejecutar |
| C4 - Forma x token | si | si | nula: no ejecutar |
| D - Secuencial sobre FastText | arquitectura si, input no | no | media-baja |
| E - kNN local sparse | parcial | no | baja: solo si el analisis de interseccion de errores muestra informacion recuperable |
| F - Ruido por par instancia-etiqueta | no | no | media |
| G - Condicionales en la capa final | casi | no | muy baja: no ejecutar |
| H - Bagging por grupos | no | no | alta |
| I - Ponderacion word/char | no | no | muy alta: la mas barata del proyecto |
| J - Aumento de datos | no | no | ultima |

El detalle de cada fila, con la hipotesis, los experimentos relacionados, el resultado
historico y la formulacion nueva, esta en `reports/v6_hypothesis_gap_analysis.csv`.

## 5. Verificaciones programaticas

Cada busqueda se hace sobre el codigo y los artefactos reales del proyecto:

| verificacion | resultado | ficheros |
|---|---|---|
| `I_word_char_weight` | **ausente** | - |
| `A_group_weight` | **ausente** | - |
| `H_group_bagging` | **ausente** | - |
| `C1_skipgram` | **ausente** | - |
| `C3_interacciones` | **presente** | structural_sparse_v4.csv (1), spec_compliance_v4.md (1) |
| `C2_ventanas` | **presente** | structural_sparse_v4.csv (1), spec_compliance_v4.md (1) |
| `C4_shape` | **presente** | 20_emocion_nbsvm_ortografia_v2.py (16) |
| `D_textcnn` | **presente** | 11_dl_desde_cero.py (1) |
| `B_factorizacion` | **ausente** | - |
| `E_knn_predictor` | **ausente** | - |
| `E_knn_medida` | **presente** | 28_hardness_v4.py (21), 27_label_consistency_v4.py (3), 35_discovery_v4.py (3) |
| `F_pair_noise` | **ausente** | - |
| `G_label_conditional` | **ausente** | - |
| `J_augmentation` | **ausente** | - |
| `J_normalizacion` | **presente** | 03_variantes_representaciones.py (24), 24_informe_final_candidatos.py (11), variantes_representaciones.csv (10) |

**Una asimetria que hay que entender**: la verificacion busca codigo, no el registro
historico. Que `G_label_conditional` salga ausente no significa que los espacios por
etiqueta sean nuevos: significa que en los scripts de v2 se llamaban de otra manera
(`E2-21`). Por eso la tabla cita **siempre** los identificadores de experimento, que
son la fuente autoritativa, y la verificacion programatica sirve para atrapar lo que
se implemento sin registrar. Cuando las dos fuentes discrepan, manda el registro.

Las lecturas que importan:

- `I_word_char_weight` **ausente**: nunca se pondero de forma explicita la escala
  entre los dos bloques de V03. Es el track mas barato del proyecto.
- `C3_interacciones`, `C2_ventanas` y `C4_shape` **presentes**: esos tres sub-tracks
  de C ya se ejecutaron en v4 y en v2. Ejecutarlos seria repetir un resultado
  negativo conocido.
- `J_augmentation` **ausente** pero `J_normalizacion` **presente**: lo que se probo
  diez veces fue normalizacion (quitar ruido), no aumento (crear instancias). Son
  cosas distintas, aunque el precedente de superficie es adverso.
- `E_knn_medida` **presente** y `E_knn_predictor` **ausente**: la afinidad de
  vecindad ya se midio y resulto nula para ordenar el error; usarla como predictor es
  genuinamente nuevo, pero parte de una afinidad que ya se vio fallar.

## 6. Duplicaciones explicitas de la especificacion de v6

La especificacion propone cuatro lineas que el proyecto ya cerro con evidencia
medida, porque fue redactada sin conocer el registro:

| track propuesto | ya existe en | por que no se ejecuta |
|---|---|---|
| C2 - Canales posicionales | V4-D (ventanas como tokens) | ventanas_hstack 0.7734 frente a V03 0.7772 (delta -0.0038); ventanas_standalone 0.5370 |
| C3 - Tokens de interaccion | V4-D (interacciones como tokens), E2-05 (canal simbolico), E2-06 (targeting) | interacciones_hstack 0.7426 (delta -0.0346); canal simbolico -0.1696; targeting -0.0416 |
| C4 - Forma x token | E2-10 (ORT-A, ORT-ABCD, ORT-AE) | -0.0286, -0.0238 y -0.0276 frente a V03 |

## 7. Hipotesis genuinamente nuevas

| familia | formulacion | prioridad |
|---|---|---|
| A - Ponderacion estructural | w = 1/sqrt(group_size) y w = 1/group_size, con los grupos calculados solo sobre el train del pliegue | alta |
| A2 - Grupo + frontera | Producto de los dos pesos en pocas configuraciones, sin rejilla amplia | media |
| B - Factorizacion condicional de etiquetas | p(TR|X)=p(HS|X)p(TR|HS=1,X) y p(AG|X)=p(HS|X)p(AG|HS=1,X)+(1-p(HS|X))p(AG|HS=0,X) | alta |
| C1 - Skip-grams | Bigramas con hueco acotado (max_gap=1, y 2 solo si aporta) | media |
| D - Secuencial sobre FastText | TextCNN 1D sobre los FastText ya entrenados, congelado y ajustado, sin encoder externo | media-baja |
| E - kNN local sparse | kNN como predictor y como fuente de un score para la capa final, no como medida de ambiguedad | baja: solo si el analisis de interseccion de errores muestra informacion recuperable |
| F - Ruido por par instancia-etiqueta | Score de sospecha por par con probabilidad, margen, desacuerdo entre modelos, consistencia estructural y densidad local; primero detectar, luego ponderar | media |
| G - Condicionales en la capa final | marginal: una sola interaccion extra, no un espacio nuevo | muy baja: no ejecutar |
| H - Bagging por grupos | Cinco modelos V03-equivalentes sobre muestras balanceadas por grupo, promediar probabilidades | alta |
| I - Ponderacion word/char | char_weight en {0.5, 1.0, 1.5, 2.0} con word_weight fijo en 1 | muy alta: la mas barata del proyecto |
| J - Aumento de datos | Crear instancias nuevas en train, nunca en validacion ni test, con transformaciones de elongation, espaciado y puntuacion | ultima |

## 8. Artefactos y scripts reutilizables

| recurso | para que sirve en v6 |
|---|---|
| `src/splits.py` | grupos fusionados y pliegues corregidos; es el unico split permitido |
| `src/evaluation.py` | metrica primaria y bootstrap pareado, ya auditados |
| `experiments/embedding_18_complementariedad_v5_1.py` | patron de salvage, damage y bootstrap de interaccion |
| `features/oof_v03_final.npz` | probabilidades, margenes y predicciones de V03 para las features condicionales |
| 19 modelos `.bin` de FastText | Track D, sin reentrenar |
| `reports/error_intersection_v2.csv` y `all_three_errors.csv` | marco del analisis de interseccion de errores |
| `reports/error_taxonomy_v4.md` | taxonomia de errores y subconjuntos robustos |
| `experiments/28_hardness_v4.py` | calculo de vecindad y medidas de dureza, reutilizable |

## 9. Presupuesto

- experimentation principal: **24**
- reserva: **6**
- total: **30**, y no es obligatorio consumirlo

Contabilidad separada: un experimento no es una unidad de entrenamiento ni una
evaluacion downstream, y los graficos, tablas, auditorias y reproducciones no cuentan.
Las familias que se cierren no consumen nada mas.

## 10. Criterio de adopcion (puerta de dos niveles)

La especificacion proponia `delta >= +0.003` y `4/5 folds positivos **o**
complementariedad robusta`. Ese gate es mas laxo que el criterio vigente desde
D-036, y D-029 ya registro que un candidato que paso los gates fue rechazado por el
holdout. Por decision del usuario el gate queda en dos niveles:

- **Gate 1**: `delta >= +0.003` **y** `IC95 que excluye el 0` **y** `>= 4/5 pliegues positivos`.
- **Gate 2**: segunda particion agrupada con otra semilla. Solo se aplica a los finalistas, no para elegir entre
  decenas de candidatos.
- **Adopcion**: exige superar Gate 1 **y** Gate 2.

El holdout no se usa: esta agotado (5/5) y el Gate 2 se resuelve con otra particion
agrupada.

## 11. Orden de ejecucion

Se sigue el orden de la especificacion (**B, A, C, D**), con el diagnostico previo que
justifica la eleccion de lineas:

1. **Diagnostico** (P0-bis): interseccion de errores y estructura de etiquetas. Barato
   y decide si D y E merecen gastarse.
2. **B**: factorizacion condicional, precedida del diagnostico de P(TR|HS) y P(AG|HS).
3. **A**: ponderacion por tamano de grupo.
4. **C**: solo **C1** (skip-grams). C2, C3 y C4 quedan cerradas con evidencia previa.
5. **D**: TextCNN sobre FastText, y D3 solo si D1 muestra senal.

Las lineas E, F, G, H, I y J esperan al resultado del diagnostico y de B y A, y se
priorizan segun lo que la evidencia mostre, no segun el orden del documento.

