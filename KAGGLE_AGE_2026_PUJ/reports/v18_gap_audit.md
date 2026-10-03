# V18 - P0 -- Auditoria de brechas

- **Rama**: `kaggle-age-2026-puj-v18` - **Fecha**: 2026-09-26 - **Fase**: P0, sin entrenar nada
- **Referencia canonica**: V12 pooled macro-F1 **0,6005017473** (`reports/v17_p0_reproduccion.md`)
- **Nota de formato**: la etiqueta de frontera lleva un guion vertical que Markdown parte como
  separador de columna. En las tablas de este informe se escribe `18-24\|25-34`; en el CSV se escribe
  `18-24__25-34` porque ahi el guion vertical delimita campos.
- **Regla de evidencia**: cada afirmacion sobre un resultado previo cita su artefacto. Lo que no se
  pudo comprobar aparece como `NO VERIFICADO`.

---

## 1. La respuesta a la pregunta central

### 1.1 Veredicto

**Respuesta (a): la ventaja del especialista es de la REPRESENTACION, no del objetivo.**

Formulada con el matiz que la evidencia obliga: la comparacion **si** es valida en cuanto a filas,
folds, entrenamiento y regla de decision, y en esa comparacion el unico factor que cambia de forma
sustantiva es la representacion (texto crudo disperso frente a las 279 variables agregadas de V12).
Queda **un confound residual, nombrado y no descartado**: familia de modelo y regularizacion cambian
a la vez. La seccion 1.6 dice como falsarlo con una celda de P0' de cinco minutos.

Lo que la evidencia **no** sostiene es la formulacion fuerte de la intencion ("cambiar el objetivo
sobre la representacion de V12 no tiene por donde tomar la senal") en sentido causal cerrado: lo que
queda demostrado es que **ninguna funcion de las 279 columnas separa la frontera mejor que V12**, y
eso se ha medido por tres vias independientes (especialista, capa de decision, arbitro), no por una.

### 1.2 Los dos especialistas, en protocolo comparable

| Elemento | `scripts/v13_p3a_especialistas.py` | `scripts/experimento_e028_ordinal_fino.py` | Coincide? |
|---|---|---|---|
| Particion | `cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")` (linea 100) | identica llamada (linea 76) | **si** |
| `y` y `groups` | `train[TARGET].astype(str)` y `dup_clusters.parquet["cluster"][:len(y)]` | identicos | **si** |
| `SEED` / repeticiones | 42 / 1 -> una sola repeticion, `random_state=42` | 42 / 1 | **si** |
| Filas de entrenamiento | `entrenamiento[en_par[entrenamiento]]` (linea 124) | `mask = (y_train==lower) \| (y_train==upper)` (linea 96) | **si** |
| `entrenamiento` | union de los `val_idx` de los otros cuatro folds (lineas 122-123) | `fold["train_idx"]` (linea 83) | **si** (`src/cv.py:364-374` devuelve indices complementarios; el mismo truncamiento se usa en `scripts/phase2_ponderacion.py:97`) |
| Filas evaluadas | `filas_par` = **57.023** y **56.577** (filas cuya etiqueta real cae en el par) | las mismas, por construccion (`experiments/v17_final_research.md` seccion 2.2) | **si** |
| Regla logito -> etiqueta | `modelo.predict()` = argmax de la funcion de decision binaria = umbral 0 | `np.where(margin > 0, upper, lower)` (linea 104) | **si** |
| Representacion | `build_features()` + `oof_ADV07_nbsvm_word.npy` = **279** columnas densas agregadas | `TfidfVectorizer(char_wb, 3-6, min_df=5, max_features=300.000, sublinear_tf)` + menciones `(1,1)`, `hstack` disperso | **NO** |
| Modelo | `StandardScaler + LogisticRegression(C=1.0, max_iter=3000)` y `MLPClassifier((64,), alpha=1e-3, batch_size=512, early_rate=1e-3, max_iter=40, early_stopping=True)` | `LinearSVC(C=0.2, max_iter=3000)` | **NO** |
| Vectorizador ajustado | no aplica (no ve texto) | **dentro del fold**, sobre `texts.iloc[train_idx]` (lineas 84-89) | correcto |

### 1.3 Los numeros, con su origen

| Magnitud | Valor | Fuente |
|---|---|---|
| E-028a, `18-24\|25-34`, por fold | 0,6877 - 0,6855 - 0,6856 - 0,6875 - 0,6856 -> media **0,68638** | `experiments/e028_ordinal_fino.json`, `E-028a.per_fold[].accuracy_por_frontera` |
| E-028a, `25-34\|35-49`, por fold | 0,7224 - 0,7284 - 0,7227 - 0,7243 - 0,7112 -> media **0,72180** | mismo |
| E-028a, media de las 5 fronteras | 0,7427950100503219 | mismo, `accuracy_fronteras_mean` |
| V12 en las mismas filas | **0,4963** y **0,4974** | `experiments/v3_p3a_especialistas.json`, `accuracy_v12_par`; identico en `reports/v17_p1_fronteras.md` |
| P3a especialista sobre 279, `18-24\|25-34` | **0,4944** (LogReg), 0,4946 (MLP) | `experiments/v3_p3a_especialistas.json` |
| P3a especialista sobre 279, `25-34\|35-49` | **0,4942** (LogReg), 0,4945 (MLP) | mismo |
| P3a global con la puerta | 0,5993 (delta **-0,0012**), 6.944 filas modificadas | mismo |
| ADV-07(a), especialista LogReg sobre las features del meta con compuerta top-2 | **-0,0036** | `DECISIONS.md` D-019 |
| V17: rescate / rotura / techo | 14.356 / 3.519 / 0,7481 y 15.571 / 2.872 / 0,7726 | `reports/v17_final_research.md` seccion 2.2 |

Las dos series de V12 (0,4963 y 0,4974) coinciden **cifra a cifra** entre P3a y V17, y las dos series
del especialista (0,6864 y 0,7218) coinciden **cifra a cifra** entre E-028a y V17. Los dos
experimentos miran las mismas 57.023 y 56.577 filas.

### 1.4 La compuerta NO explica el 0,4944

Este era el principal sospechoso de invalidez, y queda descartado con los numeros del propio JSON:

- **`18-24\|25-34`**: `filas_par` 57.023, `filas_actuables` **57.019**. Solo **4 filas** quedan
  congeladas en la prediccion de V12. La compuerta de P3a es un no-op al 99,993 %. El 0,4944 es
  enteramente del especialista.
- **`25-34\|35-49`**: `filas_par` 56.577, `filas_actuables` **51.160**, luego 5.417 filas congeladas
  (9,57 %). Aqui si hay un techo artificial, pero **no es el culpable**: sobre el mismo denominador
  (`accuracy_*_actuables`) el especialista saca **0,5465** (LogReg) y **0,5468** (MLP) frente al
  **0,55** de V12. Sigue perdiendo, por 0,0035.

Ademas, para que la compuerta explicase el 0,4944 haria falta que el especialista estuviera cerca de 1,0 en
las filas actuables, y el propio techo con la puerta (0,7481 en V17, con el especialista disperso)
demuestra que ~0,25 de las filas de frontera no las decide nadie.

### 1.5 Que confirma la mecanica, y por que no es un problema de objetivo

Las 279 variables de V12 son, segun `scripts/phase2_ponderacion.py:49-55` y `:61-81`:
**78 scores de miembro** (13 miembros x 6 columnas; el decimocuarto, `oof_E028a_pares.npy`, aporta 5) +
35 densas de textometria + 87 estructuradas + 73 supervisadas + 6 NB-SVM. Confirmado: 78+35+87+73+6
= 279, y `reports/v17_gap_audit.md` seccion 6 lo desglosa igual.

Es decir: **la salida de cada miembro entra comprimida a 6 numeros**, y el margen del propio
especialista de pares entra como **5 escalares**. Ese es el contenido operativo de la compresion que
la intencion senala. Y de esos 5 escalares, mas las otras 274 columnas, se ha intentado extraer la
decision de frontera por tres vias, todas medidas y todas negativas:

| Via | Experimento | Resultado |
|---|---|---|
| Specialist que redecide dentro del par | ADV-07(a) y V13 P3a | **-0,0036** y **-0,0019 / -0,0032** en las dos fronteras |
| Capa de decision a posteriori (33 variantes) | D-037 (PH31, PH32, PH33) | mejor +0,0011 con 4/5 folds y **neto -117**; ninguna alcanza el gate |
| Arbitro entrenado sobre 47 columnas derivadas | V17 P9 | **-0,0001**, 0/5 folds, 3 rescates contra 22 danos |

La asimetria que D-047 declaro abierta ("el margen del par ya es entrada de V12 y V12 no lo explota")
tiene ahora una explicacion medida: **esa entrada existe, y desde ella no se puede superar a V12 en su
propia tarea**. El margen es un escalar que no distingue "seguro y correcto" de "seguro y equivocado",
y V17 demostro que en la frontera V12 y el especialista estan ambos seguros y en desacuerdo
(separacion 0,809 frente a 0,539).

### 1.6 El confound residual, y como falsarlo

Representacion, familia de modelo y regularizacion cambiaron a la vez:

- E-028a: `LinearSVC(C=0.2)`, perdida hinge, 300.000 features dispersos normalizados L2 por fila.
- P3a: `StandardScaler + LogisticRegression(C=1.0)` sobre 279 columnas estandarizadas, y
  `MLPClassifier((64,), alpha=1e-3)`.

**No existe en el proyecto ninguna corrida que separe los dos factores**: nunca se ha entrenado un
`LinearSVC` sobre las 279 variables para la tarea de par, ni un `LogisticRegression` o un MLP sobre el
TF-IDF disperso para la misma tarea y los mismos folds. Por eso el veredicto es (a) **con confound
residual**, no (a) limpio.

**Falsacion propuesta, para P0' y no para V18**: una celda de cuatro ajustes, mismo protocolo, misma
tarea de par, mismos folds (los de `make_folds` de siempre):

| Celda | Representacion | Modelo | Si sale | Lectura |
|---|---|---|---|---|
| A | 279 densas | `LinearSVC(C=0.2)` | ~0,49 | (a) confirmado: el modelo no importa |
| B | 279 densas | `LinearSVC(C=0.2)` | ~0,68 | el modelo si importa; (a) revisado |
| C | TF-IDF disperso | `LogisticRegression(C=1.0, max_iter=3000)` | ~0,68 | (a) confirmado |
| D | TF-IDF disperso | `LogisticRegression(C=1.0, max_iter=3000)` | ~0,55 | (b) parcial: la ventaja es de modelo |

Coste estimado: 4 ajustes x 5 folds sobre ~45.000 filas de un par, mas el ajuste del vectorizador por
fold. E-028a entero tardo **169,5 s** (`experiments/e028_ordinal_fino.json`, campo `segundos`); P3a
entero tardo **154,6 s**. La celda completa cabe en menos de 10 minutos. Sin esto, la conclusion de
este informe queda con un "con confound declarado" en vez de un hecho.

### 1.7 Consecuencia directa para las ocho pistas

Si (a) se sostiene, la consecuencia no es "P1-P3 son inutiles" sino algo mas preciso:

- **P1, P2 y P3, tal como estan formuladas** (cambiar el objetivo de un modelo cuya representacion son
  las 279 variables) actuan sobre un espacio donde se ha medido tres veces que no hay informacion de
  frontera recuperable. Su mecanismo sigue siendo nuevo, pero la senal que lo Alimenta no esta ahi.
- **P4** es la unica pista cuya representacion el brief autoriza y en la que hay evidencia de que la
  informacion existe: el texto crudo disperso, con el que un especialista lineal alcanza 0,6864 y
  0,7218 en las mismas filas donde V12 saca 0,4963 y 0,4974.
- **P5, P6 y P7** dependen de la representacion del **student**, y el brief no la fija. Si el student
  vive en las 279 variables, la destilacion no puede transportar un margen lexico por la misma razon
  que P3a no pudo mejorar a V12. Esto es una pregunta abierta para Carlos (seccion 8).
- **P9** ya esta medido y publicado por V17; lo que queda es reejecutarlo por candidato.

---

## 2. Alcance y metodo

### 2.1 Leido completo

`PROJECT_STATE.md` - `DECISIONS.md` (D-020 a D-047) - `RUNLOG.md` (ultimas 110 lineas) -
`INFORME_COMPLETO.md` (no entero: secciones de V13, V16, V17 y tabla maestra) -
`plan/00_intent_v18.md` - `experiments.csv` (con `csv.DictReader`, seleccionando 24 filas) -
`reports/v17_final_research.md` - `reports/v17_gap_audit.md` (secciones 6, 7 y 8) -
`reports/v13_final_research.md` (via `PROJECT_STATE.md` y `experiments.csv`) -
`reports/v16_arquitectura.md` (via D-046) - `scripts/v13_p3a_especialistas.py` (256 lineas) -
`scripts/experimento_e028_ordinal_fino.py` (162 lineas) - `scripts/phase2_ponderacion.py` (229 lineas) -
`scripts/v17_p1_fronteras.py` (grep dirigido) - `scripts/experimento_e009_modelos_clasicos.py`
(lineas 45-115) - `scripts/advanced_02_features.py` (lineas 190-257) -
`scripts/advanced_07_frontera.py` (lineas 1-60) - `scripts/phase31_decision_ordinal.py` (grep dirigido) -
`src/cv.py` (332-375) - `src/config.py` - `src/decision_layer.py` (grep dirigido).

Inspeccion de artefactos, en modo lectura: formas y `dtype` de los 60+ `.npy` de `features/`
(`mmap_mode='r'`), distribucion de clases de `age_train.csv`, y contenido integro de
`experiments/v3_p3a_especialistas.json` y `experiments/e028_ordinal_fino.json`.

### 2.2 Inspeccionado con listados dirigidos

Todo `scripts/*.py` y `src/*.py` para: `sample_weight`, `class_weight`, `def .*loss`, `focal`,
`cost_matrix`, `head`, `auxiliary|aux_|multitask`, `destil|teacher|student|pseudo.?label`,
`LinearSVC`, `ordinal`, `CHAR_CONFIG =`, `age_supervised_features`, `dense_train.parquet`,
`make_folds`, `train_idx|val_idx`.

### 2.3 Lo que NO se pudo verificar

- **`INFORME_COMPLETO.md` no se leyo entero** (526 lineas). Se leyeron las secciones de V13, V16, V17
  y la tabla maestra mediante `PROJECT_STATE.md`, `DECISIONS.md` y `experiments.csv`. Cualquier cifra
  de las fases Tempranas (E-001 a E-019) que solo viva en ese informe queda `NO VERIFICADO`.
- **`notebooks/99_entrega_consolidado.ipynb` no se abrio.** Es una redaccion de artefactos ya
  ejecutados, pero no se ha descartado que contenga codigo con `class_weight` o `sample_weight` que no
  exista en `scripts/`. Afirmacion affected: "no hay ponderacion de clases en todo el proyecto" es
  `NO VERIFICADO` para el notebook.
- **No se ejecuto ningun script de experimento ni se entreno nada.** En consecuencia no hay ninguna
  cifra nueva en este informe: todas las que aparecen salen de artefactos.
- **No se comprobaron las 42 familias del informe `reports/v17_gap_audit.md` una por una.** Se leyeron
  sus secciones 6, 7 y 8 y se reutilizo su seccion de trampas de leakage, que se verfico contra el
  codigo citado. Las familias de las fases tempranas (E-001 a E-020) se tomaron de `experiments.csv`
  y `DECISIONS.md`, no de los informes individuales de cada experimento.
- **La clase de las fronteras de `25-34|35-49` como `filas_par` 56.577 no esta contrastada con el
  informe de V17 linea a linea**; se infiere de `experiments/v3_p3a_especialistas.json` y del recuento
  de V17, que coinciden.
- **Coste real de la celda de falsacion de la seccion 1.6**: estimado a partir de los 154,6 s y
  169,5 s de los dos experimentos originales. No medido.

---

## 3. Inventario por familia

28 familias. `estado` usa las cuatro etiquetas del encargo.

| # | Familia | Hipotesis | Experimento previo | Mecanismo | Resultado | Estado |
|---|---|---|---|---|---|---|
| F01 | Variables de evidencia por token al meta | El texto aporta evidencia por token que el meta no captura | V13 P1a, P1b | representacion | control 0,3997 (el doble del mejor bloque); fusion -0,0002, 3/5, controles permutados indistinguibles | CLOSED |
| F02 | Bloque linguistico externo (spaCy) | Un analizador morfologico y lexico externo anade senal | V14a, V14b (D-043, D-044) | representacion | control 0,2409 LogReg / 0,2270 LinearSVC, el mejor bloque aislado; fusion -0,0015, 1/5, segunda particion 0/4 con -0,0031 | CLOSED |
| F03 | Lexicons de edad | Listas de terminos por rango discriminan | PH36-PH40, D-039 | representacion | 0,2033 de control; la fusion no aporta sobre V12 | CLOSED |
| F04 | Pronombres, modalidad, conjunciones | Marcas de edad en el genero verbal y la modalidad | V13 P17, P20, P19 | representacion | 0,1918, 0,1270, 0,1539, los tres bajo el azar; cobertura 1,6 % y 0,64 por documento | CLOSED |
| F05 | Poda de variables redundantes | Quitar colinealidad libera al meta-modelo | V15b (D-045) | representacion | 0,5965 frente a 0,6005; delta -0,004, 0/5; segunda particion 0/4; IC95 [-0,0054; -0,0027] | CLOSED |
| F06 | Sincronizacion de variables (rho) | Elegir variables por correlacion con la etiqueta fuga informacion | V15a | representacion (metodo) | 112 variables casi informativas, 197 conservadas; el campo `variables_tras_poda` del JSON vale 81 y esta mal calculado (`RUNLOG.md`) | CLOSED |
| F07 | Colinealidad interna del meta (VIF) | El cuello de botella es el espacio de variables | V13 P0b | diagnostico | VIF mediano 18,87; 170 variables por encima de 10; 87 de 278 direcciones para el 95 % de la varianza | CLOSED |
| F08 | Suficiencia de las variables de entrada | Lo que sobra a V12 no se puede reclasificar | V13 P16b, P16c, P16d | arquitectura | R2 de reconstruccion 0,522 con control de clasificacion 0,0485; el maximo de 90 puertas por confianza es 0,6005 con **0 filas desviadas** | CLOSED |
| F09 | Forma de combinar las dos cabezas | El promedio fijo 0,5/0,5 no es la forma correcta | V16 A1, A2 (D-046) | arquitectura | A1 peso aprendido 0,6002 (-0,0003, 0/5); A2 combinador logistico 0,6011 (+0,0006, 3/5), fronteras 0,5065 y 0,5131, IC95 contiene el cero | CLOSED sin adoptar |
| F10 | Combinador de miembros (segmentado, suave) | El meta pondera mal porque usa pesos globales | PH2-L6 (D-020, D-022, D-025) | arquitectura | la ponderacion suave funciona; el promedio de combinadores es el optimo; la segmentacion dura falla | CLOSED |
| F11 | Calibracion, blending en logits, subconjuntos | Diversidad de calibracion entre miembros | D-029, PH15, PH15B | arquitectura | nulos | CLOSED |
| F12 | Limpieza de filas ruidosas | Quitar filas ruidosas del entrenamiento ayuda | D-027, D-031, PH11, PH18 | datos | mejora real pero acotada, sin adopcion | CLOSED |
| F13 | Retrieval y POS | Anadir senal de recuperacion o de parte de la oracion | D-026 | representacion | negativo | CLOSED |
| F14 | Multivista (contenido / estructura / extremos) | Ver el texto con mas vistas separa mas | PH21, PH22, D-032 | representacion | la vista estructural es la unica senal nueva, marginal, no adoptada | CLOSED |
| F15 | Senal ordinal auxiliar y seleccion greedy (Caruana) | La frontera central es ordinal y Caruana supera al meta | PH16-ORD-GREEDY (D-030) | seleccion | G1 0,6001 (-0,0003, 1/5); G2 0,5770 (-0,0234) | CLOSED |
| F16 | Modelos ordinales como miembros (K-1 umbrales) | Descomponer en "supera el umbral k?" captura la adyacencia | E-015, E-021a | arquitectura | E-015 0,3973 como modelo suelto; E-021a entra como miembro y no aporta; **entrenados con BCE binaria plana, sin coste** | CLOSED como miembro |
| F17 | Capa de decision ordinal a posteriori | Reglas sobre P(V12) congelada reducen la inconsistencia ordinal | PH31 (33 capas A-I, D-037) | decision | mejor G (tau 0,05) 0,6015 (+0,0011, 4/5) con **neto -117**; A umbrales 0,5887; ninguna alcanza el gate | CLOSED |
| F18 | Correccion de probabilidades, transicion, calibracion por banda | Los sesgos de frontera son sistematicos y corregibles | PH32, PH33 (D-037) | decision | mejor H 5 bandas 0,6006 (+0,0002, 3/5); D empeora monotonamente; I1 0,5999; B (Luce) 0,2947 | CLOSED |
| F19 | **Especialista de frontera sobre las 279 variables** | Un binario entrenado en las filas del par, con las mismas variables, supera a V12 dentro del par | **ADV-07(a)** y **V13 P3a** | arquitectura + objetivo restringido | -0,0036 (ADV-07) y 0,4963->0,4944 / 0,4974->0,4942 (P3a); global 0,6005->0,5993; neto -259 | **CLOSED, dos veces** |
| F20 | **Especialista de frontera sobre el TF-IDF disperso** | Un SVM entrenado solo con las dos clases concentra capacidad en la ambiguedad real | **E-028a** (`experimento_e028_ordinal_fino.py`) | representacion + objetivo restringido | **0,6864** y **0,7218** en las filas de cada frontera, sobre las mismas que V12 falla al 0,4963 y 0,4974; rescuing 14.356/15.571, techo 0,7481/0,7726 | **EXISTENTE, capacidad alta, no enrutable** |
| F21 | Enrutamiento del especialista disperso | Si se puede identificar cuando escuchar al especialista | ADV-07(b), V17 P1b, P1c, P9, P5 | decision | compuerta por ambiguedad +0,0002; regla conjunta +0,0000; arbitro de 47 columnas -0,0001 (0/5); combinador local -0,0002 y fronteras peores | CLOSED, cuatro formulaciones |
| F22 | Perfil de confianza de V12 | Alguna senal de confianza identifica las filas que el especialista arregla | V17 P8 | decision (diagnostico) | AUC: acuerdo entre miembros 0,7070/0,7300; margen del especialista 0,7294/0,7539; separacion de V12 0,5780/0,5268; entropia 0,4323/0,4230 | CLOSED como contribution |
| F23 | Errores de V12 en la frontera | Los errores son de baja confianza y, por tanto, alcanzables | V17 P1 + diagnostico de P9 | diagnostico | son **errores confiados**: separacion 0,809 donde el especialista acierta frente a 0,539 donde acierta V12; solo 12,7 % y 8,0 % de las victorias caen en la region de frontera | CLOSED como pista, ya publicado |
| F24 | Bloque supervisado in-fold (73 variables) | Variables que dependen de la etiqueta, calculadas dentro del fold | `advanced_02_features` | representacion | ya incorporadas a las 279; construidas **in-fold**, sin fuga | CLOSED (incorporado) |
| F25 | Embeddings y transformers | Un encoder preentrenado separaria los rangos medios | D-038 | representacion | el entorno los tiene instalados; el proyecto no los usa | **PROHIBIDO** por el brief |
| F26 | **Funcion de objetivo / perdida** | Cambiar la perdida en vez del decision layer mueve la frontera | **ninguno** | objetivo | `NO VERIFICADO` porque no existe: `grep` de `sample_weight`, `class_weight`, `def .*loss`, `focal` y `cost_matrix` sobre `scripts/*.py` y `src/*.py` devuelve **cero** coincidencias | **GENUINELY_NEW** |
| F27 | **Multi-objetivo con heads auxiliares** | Entrenar en las fronteras ademas de en las seis clases | **ninguno** | objetivo + arquitectura | `NO VERIFICADO` porque no existe: cero coincidencias de `head`, `auxiliary`, `aux_`, `multitask` en sentido de aprendizaje Multiple (los unicos `head(` son de pandas) | **GENUINELY_NEW** |
| F28 | **Destilacion / teacher-student / pseudo-etiquetas** | Un teacher binario puede transferir su margen a un student de seis clases | **ninguno** | objetivo | `NO VERIFICADO` porque no existe: cero coincidencias de `destil`, `teacher`, `student`, `pseudo.?label` en codigo (los dos unicos hits, en `phase36` y `phase37`, **declaran** `genera_pseudo_labels: False`) | **GENUINELY_NEW** |

**Recuento de estados**: 22 `CLOSED`, 0 `PARTIALLY_TESTED`, 1 `PROHIBIDO` (F25, equivalente a
`CLOSED` por restriccion), 3 `GENUINELY_NEW`, y 2 con estado propio (`EXISTENTE` F20,
`CLOSED sin adoptar` F09). F19 y F21 son `CLOSED` y son las que sostienen de la pregunta central.

### 3.1 El hallazgo que reencuadra la fase

Las tres familias `GENUINELY_NEW` (F26, F27, F28) son exactamente las tres cosas que el brief de V18
pide y que el proyecto **nunca ha hecho**: ninguna perdida modificada, ningun head, ninguna
destilacion. Y las dos familias que ya existen con capacidad alta (F19 sobre 279, cerrada dos veces;
F20 sobre TF-IDF, abierta pero no enrutable) se apoyan en la **representacion**. La fase, leida asi,
no es "probar mas cosas sobre el objetivo": es la primera vez que el objetivo se toca, y la unica
representacion con la que el objetivo ha demostrado algo es el texto crudo disperso.

---

## 4. Las ocho pistas de V18, una por una

### P1 -- Peso de clases en frontera (tres niveles, con control V12-equivalente)

- **Estado: `PARTIALLY_TESTED`**
- **Experimento previo mas cercano**: D-037, familias de capas **A** (umbrales acumulados, 0,5887),
  **C** (correccion residual de fronteras, 4 variantes, la mas suave con **neto exactamente 0**) y
  **H** (calibracion por banda de confianza, 2 variantes, mejor 0,6006 con +0,0002 y 3/5).
- **Por que no es `GENUINELY_NEW` en mecanismo puro**: no existe `class_weight` ni `sample_weight` en
  `scripts/` ni `src/`, luego el ajuste en entrenamiento es literal y mecanicamente nuevo. Pero la
  **pregunta** que responde ("subrayar las clases de la frontera mueve la macro-F1?") tiene su
  equivalente de decision medido 33 veces, y la unica palanca que los pesos de clase tendrian es
  **desplazar la frontera de decision**: exactamente lo que hicieron esas capas.
- **Hecho verificado que pesa mucho en contra**: **las clases estan balanceadas**.
  `13-17` 17,04 %, `18-24` 16,93 %, `25-34` 16,74 %, `35-49` 16,67 %, `50-64` 16,57 %, `65-xx` 16,04 %
  (razon 1,062). No hay desbalance de clases que corregir; el desbalance es de **confusion entre
  pares adyacentes**, no de prevalencia. Un peso de clase solo puede mover el prior y desplazar la
  frontera, y el desplazamiento a posteriori esta cerrado.
- **Coste**: un experimento. **Pronostico**: negativo, con prior alta.

### P2 -- Coste ordinal sensible en el entrenamiento, `coste(i,j) propto |i-j|`

- **Estado: `PARTIALLY_TESTED`**
- **Experimento previo mas cercano**: D-037 **familia F**, "coste ordinal (3 perfiles: plano,
 discounted y con castigo extra a distancia >= 3)", implementada como
  `coste[i,j] = 0 si d==0, 1 si d==1, p2 si d==2, p3 si d>=3` y `riesgo = oof[filas] @ coste`
  (`scripts/phase31_decision_ordinal.py:165-178`). Resultado registrado: **"F solo positiva con coste
  casi plano"**. Ademas **G** (desempate por edad esperada) y **B** (Luce sobre margenes adyacentes,
  0,2947).
- **Por que no es `GENUINELY_NEW`**: existe el analogo exacto de la matriz de coste, con tres perfiles
  barridos, sobre las probabilidades de V12. Y existe la conclusion mecanica de por que falla, escrita
  en D-037: *"Minimizar el error ordinal no es mejorar macro-F1"*: el desempate por E[Y] mejora el MAE
  ordinal (0,6552 -> 0,6454) y **empeora las dos fronteras centrales** (+462 y +534 errores).
- **Por que aun asi vale la pena una celda**: la version de decision opera sobre un vector de 6
  probabilidades que, en la frontera, esta **confiado y equivocado** (separacion 0,809). No puede
  expresar la frontera porque la informacion no esta en ese vector. La version de entrenamiento opera
  sobre algo distinto. La distincion es real y es la que el brief manda hacer.
- **Pronostico**: es la pista con el prior negativo **mas fuerte** de las ocho, porque un coste
  `|i-j|` es mas agresivo que el perfil "plano" que fue el unico positivo. Si se ejecuta, el perfil
  de coste que hay que explorar primero es el **plano** (0/1), no `|i-j|`.

### P3 -- Margen de adyacencia en entrenamiento (primero solo en las dos centrales)

- **Estado: `GENUINELY_NEW`**
- **Experimento previo mas cercano**: **E-015** y **E-021a** (descomposicion de umbrales K-1). Se
 : son la misma intuicion ordinal, pero como **arquitectura**, y entrenados con
  `LogisticRegression().fit(x_train, (y_index >= threshold).astype(int))` y **BCE binaria plana**,
  sin ningun termino de adyacencia ni matriz de coste. PH16-ORD-GREEDY (D-030) es el mas cercano en
 utanismo ("frontera central es ordinal") y esta cerrado: 0,6001 (-0,0003, 1/5) y 0,5770.
- **Por que no es un sinonimo de E-028a**: E-028a es un **modelo binario independiente por par** sobre
  su propia representacion. Un termino de margen de adyacencia es una **restriccion sobre un modelo
  de seis clases compartido**, en la que las cinco fronteras compiten por la misma representacion.
  No hay nada en el proyecto que imponga esa restriccion.
- **Nota a favor del mecanismo**: E-028a demuestra que la estructura de pares adyacentes **si** lleva
  senal, y muy grande, cuando hay texto crudo detras. Es el unico mecanismo de las ocho pistas con
  evidencia positiva directa de su principio.

### P4 -- Entrenamiento multi-objetivo: global + heads auxiliares de frontera

- **Estado: `GENUINELY_NEW`**
- **Experimento previo mas cercano**: **ninguno**. No hay `head`, ni `auxiliary`, ni `multitask` en
  sentido de aprendizaje multiple en todo el proyecto. Lo mas proximo es la descomposicion de
  E-021a, que es multi-salida pero **no multi-objetivo**: K-1 clasificadores independientes que
  comparten el resultado por construccion, no una perdida conjunta con un termino global y otro de
  frontera.
- **Representacion**: ninguna se ha usado con heads. El brief autoriza el TF-IDF
  ("las heads deben utilizar la representacion TF-IDF permitida"), que es la unica con evidencia
  (0,6864 / 0,7218). Y segun la seccion 1, es la unica coherente con el veredicto (a).
- **Por que el brief la deja abierta**: es la unica pista cuyo mecanismo no tiene antecedente
  negativo y cuya representacion esta medida. Ademas separa dos cosas que el brief quiere separadas
  (arquitectura de objetivo): P4.1 con heads sin perdida de frontera y P4.2 con ella.

### P5 -- Destilacion de conocimiento de los especialistas, con cross-fitting

- **Estado: `GENUINELY_NEW`**
- **Experimento previo mas cercano**: **ninguno**. Verificado: `destil`, `teacher`, `student` y
  `pseudo.?label` no aparecen en `scripts/*.py` ni `src/*.py` mas que en `phase36_auditoria_lexicones.py`
  y `phase37_catalogo_lexicones.py`, y **ambos declaran lo contrario**:
  `phase37_catalogo_lexicones.py:119` fija `"genera_pseudo_labels": False` y
  `phase36_auditoria_lexicones.py:153` dice que "no se usan para pseudo-labels".
- **No es `INVALIDLY_TESTED`**: la pista no se ha probado de forma invalida; sencillamente no se ha
  probado. No hay precedente que veta el mecanismo, solo la via de pseudo-etiquetas, que el brief
  prohibe y que aqui no existe.
- **Condicion que el brief no fija y que hay que decidir**: la representacion del **student**. Si el
  student vive en las 279 variables, no puede transportar un margen lexico, por el mismo motivo
  medido en F19. Si el student tambien es un modelo sobre texto, P5 es coherente con el veredicto (a).
  Es la pregunta abierta numero 1 de la seccion 8.

### P6 -- Destilacion de miembros seleccionados

- **Estado: `GENUINELY_NEW`**
- **Experimento previo mas cercano**: ninguno para la destilacion; para la **seleccion de miembros**,
  PH16-ORD-GREEDY (D-030, G2 Caruana 0,5770, -0,0234) y D-029 (subconjuntos y diversidad).
- **Por que esta doblemente bloqueada**: (a) el mecanismo de destilacion es nuevo, y (b) la
  seleccion de miembros que la pista necesita ya esta cerrada y **fuertemente** (G2 es el peor
  resultado de combinacion del proyecto). El brief ademas la condiciona a que P5 funcione.
- **Orden**: no abrir antes que P5.

### P7 -- Destilacion de residuo

- **Estado: `GENUINELY_NEW`**
- **Experimento previo mas cercano**: ninguno para destilacion. Para el diagnostico de residuo,
  V13 **P16c** (R2 de reconstruccion desde V12 0,522 con control de clasificacion 0,0485) y la
  consecuencia 2 de D-037 ("**65 % de agregacion**, 26,70 % de filas rescatables").
- **Por que no es un sinonimo**: P16c midio si las variables sobrantes se pueden reclasificar; no
  midio si se puede **ensenar** un residuo a un modelo por perdida. Ese es un mecanismo nuevo.
- **Advertencia**: hereda la misma restriccion de representacion que P5. Y con el 65 % de agregacion
  ya medido, la version con mas probabilidad depayer es la que ensena **el error de V12 en la
  frontera**, no un residuo generico.

### P9 -- Analisis de error y destino de las victorias del especialista

- **Estado: `PARTIALLY_TESTED`** (de hecho, **hecho y publicado**)
- **Experimento previo mas cercano**: es el mismo. V17 P1 y el diagnostio previo a P9 ya lo midieron
  todo: rescate 14.356 / 15.571, rotura 3.519 / 2.872, techo 0,7481 / 0,7726; en la region de frontera
  hay 11.130 filas, el especialista **gana 1.830 y pierde 2.485**, y escucharle ahi vale **-0,0080**;
  solo el 12,7 % y el 8,0 % de las victorias caen dentro de la region; separacion 0,809 frente a 0,539.
- **Por que `PARTIALLY_TESTED` y no `CLOSED`**: en el brief P9 no es una pista con hipotesis propia,
  es una **obligacion por candidato** ("obligatorio para cada candidato"). El analisis ya existe; lo
  que queda es reejecutarlo con el candidato delante, que es una carga de protocolo, no una
  hipotesis.
- **Advertencia de coste**: V17 ya gasto 8 de 16 experimentos previstos. Si P9 se reejecuta por
  candidato sobre las 6 variantes de P4 y las 7 de P5, es la partida mas cara de la fase en tiempo de
  CPU. Recomendacion: reejecutar P9 **solo** para los candidatos que superen el gate exploratorio.

### 4.1 Recuento de las ocho pistas

| Estado | Pistas |
|---|---|
| `CLOSED` | ninguna |
| `PARTIALLY_TESTED` | P1, P2, P9 |
| `INVALIDLY_TESTED` | ninguna |
| `GENUINELY_NEW` | P3, P4, P5, P6, P7 |

Ninguna queda `CLOSED` de entrada, y eso es un cambio real respecto de V17, donde 3 de 10 estaban
cerradas: **la dimension que V18 abre (la funcion de objetivo) no tiene ningun antecedente en el
proyecto**. Lo que si esta cerrado es la **representacion** sobre la que P1-P3 serian aplicadas.

---

## 5. Artefactos reutilizables

Rutas comprobadas con existencia y forma en el arbol actual. "Creacion verificada" dice si la
generacion esta documentada y comprobada.

| Artefacto | Contenido | Forma | Creacion verificada |
|---|---|---|---|
| `features/oof_ensemble_v12.npy` | probabilidades OOF de V12 | 169.336 x 6 | si: fase `phase8_holdout_v12`; reconstruida con diferencia maxima 0,0 y acuerdo 100,0000 % (`RUNLOG.md:440-441`) |
| `features/oof_E028a_pares.npy` | **los 5 margenes del especialista que da 0,6864 / 0,7218** | 169.336 x 5 | si: `scripts/experimento_e028_ordinal_fino.py:147`, con `LinearSVC(C=0.2)` |
| `features/oof_E028b_grupos.npy` | margenes del clasificador grueso de 3 grupos (0,7128) | 169.336 x 3 | si: mismo script, linea 148 |
| `features/oof_argmax_v12.npy` | argmax de V12, para emparejamiento fila a fila | 169.336 | si: fase avanzada |
| `features/meta_features_test_v12_fresca.npy` | **las 279 variables de V12 en TEST** | 112.892 x 279 | si: `advanced_06_submission_v10` y la reconstruccion de D-034 |
| `features/meta_features_train_v11.npy` | version de 273 columnas en TRAIN | 169.336 x 273 | si: `phase3_holdout_vista` |
| `features/v16_oof_combinador_logistico.npy` | OOF de A2, la pieza que gana en las fronteras (0,5065 / 0,5131) | 169.336 x 6 | si: `scripts/v16_arquitectura.py:188` |
| `features/v16_oof_peso_aprendido.npy` | OOF de A1 | 169.336 x 6 | si: mismo script |
| `features/v17_p9_oof.npy`, `features/v17_p5_oof.npy` | OOF de los dos candidatos fallidos de V17 | 169.336 x 6 | si: `scripts/v17_p9_arbitro.py`, `scripts/v17_p5_combinador_local.py` |
| `features/age_supervised_features.npy` | 73 variables supervisadas, **calculadas in-fold** | 169.336 x 73 | si: `scripts/advanced_02_features.py:207-232`. Patron de referencia para cualquier variable que dependa de la etiqueta |
| `features/age_structured_features.parquet` | 87 variables estructuradas | - | si: fase avanzada |
| `features/dense_train.parquet` | 35 variables densas de textometria | - | si: `src/textometry.py` |
| `features/dup_clusters.parquet` | clusters de near-duplicados (282.228 filas, train primero) | - | si: `src/cv.build_dup_clusters` |
| `scripts/phase2_ponderacion.build_features()` | **las 279 variables de V12** | - | si: `scripts/phase2_ponderacion.py:61-81`, `MEMBER_FILES` en 49-55, `DENSE_FEATURES` en 56-57. **No sustiturible por una lectura de `.npy`**: la matriz de TRAIN de 279 columnas no esta materializada en disco |
| `scripts/phase2_ponderacion.EncodedMLP` | la cabeza MLP de V12 (64, alpha 1e-3, batch 512, early stopping) | - | si: `scripts/phase2_ponderacion.py:106-125`. **No sustituir por `MLPClassifier` pelado** |
| `scripts/v13_p1b_fusion_tokens.PARES` | diccionario de frontera canonico del proyecto | - | si: linea 76. **La copia de `v13_p3a_especialistas.py:70` es identica y no se debe re-declarar en V18** |
| `src/cv.make_folds(...)` | unico split permitido (D-003) | - | si: `src/cv.py:332-375` |
| `src/verificacion_numerica.py` | `autotest_numerico()` y `grama_exacta()` | - | si: D-040. **Obligatorio** en la primera linea de todo script de V18 |
| `scripts/v13_p1b_fusion_tokens.metricas_frontera` / `bootstrap_diferencia` | doble metrica de frontera e IC95 pareado con 4.000 remuest multinomiales | - | si: lineas 145-167 y 170-205 |

### 5.1 La funcion que construye el TF-IDF disperso de E-028a

**Respuesta corta: no existe como funcion reutilizable en E-028a.** El script
`scripts/experimento_e028_ordinal_fino.py` la construye **en linea dentro de `main()`**, en las lineas
84-89, y las constantes viven en las lineas 47-50:

```python
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
# dentro de main(), por fold:
char = TfidfVectorizer(**CHAR_CONFIG)
mentions = TfidfVectorizer(**MENTION_CONFIG)
x_train = hstack([char.fit_transform(texts.iloc[train_idx]),
                  mentions.fit_transform(tokens_of(texts.iloc[train_idx], r"@\w+"))]).tocsr()
x_valid = hstack([char.transform(texts.iloc[valid_idx]),
                  mentions.transform(tokens_of(texts.iloc[valid_idx], r"@\w+"))]).tocsr()
```

**Si se puede reconstruir de forma identica sin fuga: si, y hay una funcion ya escrita que lo hace.**

`scripts/experimento_e009_modelos_clasicos.py:90-99` define

```python
def build_representation(train_texts: pd.Series, valid_texts: pd.Series):
    char = TfidfVectorizer(**CHAR_CONFIG)
    mentions = TfidfVectorizer(**MENTION_CONFIG)
    x_train = hstack([char.fit_transform(train_texts),
                      mentions.fit_transform(mention_tokens(train_texts))]).tocsr()
    x_valid = hstack([char.transform(valid_texts),
                      mentions.transform(mention_tokens(valid_texts))]).tocsr()
    return x_train, x_valid
```

con `CHAR_CONFIG` y `MENTION_CONFIG` **identicos byte a byte** a los de E-028a
(`scripts/experimento_e009_modelos_clasicos.py:48-51` frente a
`scripts/experimento_e028_ordinal_fino.py:47-50`), y `mention_tokens` identico a `tokens_of`
(ambos `preprocessing.collapse_spaces(series.str.findall(r"@\w+")...)` en minusculas). La firma es la
que V18 necesita: **recibe los dos subconjuntos de texto por separado y ajusta el vectorizador solo
con el primero**.

Propiedades que hacen que la reconstruccion sea identica y sin fuga:

- `TfidfVectorizer` no tiene aleatoriedad: dados los mismos `train_texts`, el vocabulario, el IDF y la
  matriz son deterministas. Repetir el ajuste sobre el mismo `train_idx` reproduce las mismas columnas.
- El vocabulario **cambia por fold**, y eso es lo correcto: el `min_df=5` y el `max_features=300.000`
  se calculan sobre el train del fold, nunca sobre todo el train.
- El modelo se entrena sobre `x_train[mask]` con `mask` de las dos clases del par
  (`experimento_e028_ordinal_fino.py:96-99`), o sea `outer_train INTERSECT par`.
- Requisito de V18: **el vectorizador debe ajustarse dentro de cada fold exterior**, y si P4 o P5
  necesitan un split interior, tambien dentro de el. Un ajuste fuera del fold seria una fuga
  silenciosa que no aparece en ninguna metrica agregada.

---

## 6. Riesgos y trampas de leakage concretos, en el terreno

1. **Ajustar el TF-IDF fuera del fold es la fuga que P4 y P5 van a cometer.** El vectorizador **ve el
   texto**, cosa que ninguna de las 279 variables hace. Si se ajusta una vez sobre todo el train y se
   reutiliza dentro de cada fold, el IDF y el `min_df` han visto la validacion exterior: la mejora
   seravisible y falsa, y ninguna metrica agregada la delatara. El patron correcto esta en
   `experimento_e028_ordinal_fino.py:84-89` y en `experimento_e009_modelos_clasicos.py:90-99`.
2. **Destilacion sin cross-fitting es fuga por construccion.** Si el teacher que produce la senal
   auxiliar se entreno con la fila que el student va a ver, la senal contiene la etiqueta. El brief ya
   lo exige ("cross-fitting obligatorio"), y el precedente correcto del proyecto es
   `scripts/advanced_02_features.py:207-232`, donde las variables supervisadas se construyen
   **in-fold**: se ajustan con `train_texts`/`y_train` del fold y se escriben solo en `valid_idx`.
3. **La matriz de 279 columnas de TRAIN no existe en disco.** `meta_features_train_v11.npy` tiene 273
   columnas y el unico artefacto de 279 es el de **test** (`meta_features_test_v12_fresca.npy`,
   112.892 x 279). Cualquier V18 que necesite la de train debe llamar a
   `scripts/phase2_ponderacion.build_features()`, que reconstruye las 279 leyendo 14 `.npy`, un
   `.parquet` de 87 y otro de 35. Es de las primeras cosas que cronometrar en P0'.
4. **`oof_ensemble_v12.npy` es la unica referencia de V12 que se puede usar.** Reentrenar V12 para
   obtener "su propia" referencia mete ruido de reentrenamiento en cada comparacion y rompe el
   emparejamiento fila a fila del bootstrap pareado
   (`scripts/v13_p1b_fusion_tokens.py:10-14`, `reports/v17_gap_audit.md` seccion 7 punto 9).
5. **`metricas_frontera` mide accuracy dentro del par, no macro-F1.** En
   `experiments/v16_arquitectura.json`, `fronteras_referencia["18-24|25-34"]` tiene `accuracy` 0,4963
   y `macro_f1` 0,5923. Los "+0,0102 y +0,0157" de D-046 son diferencias de **accuracy**. Mezclar las
   dos metricas daria una lectura falsamente grande del efecto, y V18 va a producir las dos.
6. **Las etiquetas de frontera estan en tres sitios.** `PARES` en
   `scripts/v13_p1b_fusion_tokens.py:76` (canonico, re-exportado por `scripts/v16_arquitectura.py:58-59`),
   una copia identica en `scripts/v13_p3a_especialistas.py:70`, y
   `PARES_ADYACENTES = "oof_E028a_pares"` como constante de string en `src/decision_layer.py:33`. V18
   debe **importar** el canonico, no re-declararlo, o las cifras de frontera dejan de ser comparables
   con las de V16 y V17 y el cruce del gate de D-042 queda invalido.
7. **Los controles permutados de la era V13 estan calibrados en la escala pooled** (0,6000 a 0,6003,
   `DECISIONS.md:915-916`). Un control calculado con la media por fold no es comparable, y D-042 exige
   que el IC95 del bootstrap pareado **excluya el cero** en la escala pooled.
8. **`A @ B.T` esta prohibido** (D-040): en este entorno (`numpy 1.26.4` sobre `Python 3.14.7`) el
   producto matricial de dos vistas de la misma matriz **falla en silencio y devuelve ceros**, es
   decir "no hay colinealidad", la conclusion exactamente opuesta a la real. La correlacion se calcula
   con `grama_exacta` o `einsum`, nunca con producto matricial.
9. **La particion interior debe seguir siendo agrupada por duplicados.** El precedente es
   `scripts/v16_arquitectura.py:82-90` (`partido_agrupado`), que reparte **grupos**, no filas. Con un
   `train_test_split` ingenuo, cualquier combinador, arbitro o teacher ve casi-duplicados de sus
   propias filas de entrenamiento y el IC95 del bootstrap deja de ser crecible. Es exactamente el
   riesgo que hace P5 y P7 mas fragiles de lo que parecen.
10. **`cruces_resueltos` vale 0 por construccion** en `metricas_frontera` para candidatos de frontera,
    porque la condicion exige que la prediccion sea la del par y la de la clase verdadera, y el par
    contiene precisamente las dos clases que se excluyen. No usar ese campo como medida de nada.
11. **El techo con puerta perfecta no es alcanzable.** El +0,0107 de V17 (y el 0,6770 de P16b y P15)
    requieren saber que filas falla V12. Usarlos como cota de un candidato real es un error de lectura
    que ya se cometio una vez (`reports/v16b_activacion.md:51` lo advierte).
12. **La matriz de 279 variables tiene 15 duplicados exactos, 401 pares con |rho| >= 0,80 y 1 columna
    constante** (`DECISIONS.md` D-041, P0b). Cualquier seleccion de variables o reshape dentro de V18
    tiene que pasar por `scripts/v15a_correlacion.matriz_completa`, que lleva la asercion
    `len(nombres) == 279`.
13. **`experiments/v15a_correlacion.json` tiene un campo erroneo publicado:** `variables_tras_poda`
    vale 81 y deberia ser **197** (la formula `310 - 171 - 58` resta los clusters una vez de mas).
    El 197 es el que se uso, y se puede comprobar en `features/v15_conservadas_fold{0..4}.npy`, que
    tienen 197 columnas. No citar el 81 (`RUNLOG.md`, seccion "Correccion de un campo de V15").
14. **`reports/tables/e028_combinaciones.csv` no existe** pese a estar citado
    (`reports/v17_gap_audit.md` seccion 7 punto 11). Si V18 lo necesita, hay que regenerarlo o leer
    `experiments/e028_ordinal_fino.json`.

---

## 7. Conclusion

### 7.1 Pistas que quedan abiertas, en orden de prioridad que la auditoria recomienda

Este orden **no** es el del brief. El brief pone P1, P2, P3 primero porque son baratas; la auditoria
los pone despues porque la pregunta central dice que su representacion no tiene la senal. El brief
autoriza el orden y la puerta ("P2 solo si P1 muestra senal", "P3 solo si P1-P2 muestran senal"), asi
que la secuencia real que propongo es:

| Orden | Pista | Justificacion | Riesgo |
|---|---|---|---|
| **0** (antes de P1) | **Celda de falsacion de la seccion 1.6** | Convierte el confound residual en un hecho. Cuatro ajustes, menos de 10 min. Sin ella, toda la fase se apoya en un veredicto con asterisco | coste nulo, y decide el resto |
| **1** | **P4** (heads de frontera sobre TF-IDF, con y sin perdida de frontera) | Unica pista sin antecedente negativo, con representacion medida y autorizada por el brief | P4.1 (sin perdida de frontera) tiene riesgo de ser casi identico a P4.2; designing las dos bien es el trabajo |
| **2** | **P5** (destilacion con cross-fitting), **solo si el student es sobre texto** | Mecanismo sin ningun antecedente en el proyecto, y la unica forma de que la senal de frontera entre por la puerta sin identificar filas | El brief no fija la representacion del student. **Pregunta 1 abierta** |
| **3** | **P3** (margen de adyacencia en entrenamiento) | Unico mecanismo con evidencia positiva directa de su principio (E-028a) | actuando sobre las 279 columnas tiene el problema de la seccion 1; actuando sobre TF-IDF es lo que tiene evidencia |
| **4** | **P1** (peso de clases) | Mecanicamente nuevo, pero las clases estan **balanceadas** (16,04 % a 17,04 %) y su unica palanca es desplazar la frontera, que D-037 cerro 33 veces | Pronostico negativo con prior alta. Un experimento |
| **5** | **P2** (coste ordinal `\|i-j\|`) | El analogo a posteriori esta medido con tres perfiles y solo el casi plano fue positivo; D-037 escribio que minimizar el error ordinal **no** es mejorar macro-F1 | El prior negativo mas fuerte de las ocho. Si se ejecuta, empezar por el perfil plano |
| **6** | **P7** (destilacion de residuo) | Mecanismo nuevo, pero hereda la restriccion de representacion de P5 y el 65 % de agregacion ya medido | Brief: solo si P5 funciona |
| **7** | **P6** (destilacion de miembros seleccionados) | Doblemente bloqueada: la destilacion es nueva y la seleccion de miembros esta cerrada y muy fuerte (G2 0,5770) | Brief: solo si P5 funciona. No abrir antes |
| transversal | **P9** | Ya hecho y publicado; reejecutar **solo** para candidatos que superen el gate exploratorio, por coste de CPU | Es la partida mas cara si se corre por todos los candidatos |

### 7.2 Pistas que quedan cerradas de entrada

- **Ninguna de las ocho queda `CLOSED` de entrada.** Es un cambio real respecto de V17 (3 de 10
  cerradas): la dimension que V18 abre no tiene antecedente.
- **Lo que si queda cerrado, y hay que decirlo explicitamente para no volver a pagarlo**: cualquier
  intento de extraer decision de frontera de las 279 variables de V12, por la via que sea. Tres
  precedentes medidos y negativos: especialista (ADV-07a, V13 P3a), capa de decision (D-037, 33
  variantes) y arbitro de 47 columnas (V17 P9). Si P1, P2 o P3 se formulan sobre esas columnas, lo
  que se esta midiendo es por tercera vez lo mismo.
- **Y queda cerrado por restriccion**: embeddings, transformers, vectorizadores aprendidos nuevos,
  corpus externo, etiquetas externas, pseudo-etiquetas, y todo lo que abra D-037 (lexicons, capas de
  decision, gating, pairwise generals, poda, meta-combinador generico, decodificacion de edad
  esperada, post-procesamiento ordinal), salvo que aparezca un defecto metodologico concreto y
  nombrado. La seccion 1 **nombra** uno: el campo `variables_tras_poda` de V15a, que esta mal
  calculado y no es defecto del pipeline.

### 7.3 Las tres conclusiones que el resto de la fase debe tener presentes

1. **El objetivo nunca se ha tocado.** `sample_weight`, `class_weight`, `focal`, `cost_matrix` y
   cualquier definicion de perdida propia: **cero coincidencias** en `scripts/` y `src/`. Todo lo que
   se ha llamado "cambio de objetivo" hasta ahora (D-030, D-037, V16) es **seleccion, combinacion o
   capa de decision**. V18 es la primera vez que la dimension esta realmente abierta, y por eso
   ninguna de las ocho pistas queda cerrada de entrada.
2. **La asimetria de D-047 ya tiene explicacion.** "El margen del par ya es entrada de V12 y V12 no
   lo explota" era el punto abierto que V17 dejo sin cerrar. La auditoria lo cierra en el lado
   corto: **esa entrada existe y desde ella no se puede superar a V12 en su propia tarea**, medido
   tres veces. La lectura simetrica es que la frontera vive en el texto crudo, no en las 279 columnas.
3. **La comparacion central aguanta, pero no del todo, y el hueco es barato de cerrar.** Fold, filas,
   entrenamiento y regla de decision coinciden; la compuerta de P3a es un no-op al 99,993 % en
   `18-24\|25-34` y sigue perdiendo en el denominador emparejado de `25-34\|35-49`. Lo unico que
   queda confuso es familia de modelo contra representacion, y se resuelve con cuatro ajustes que
   caben en la sesion de P0'.

### 7.4 Preguntas abiertas para Carlos

1. **Sobre que representacion vive el student de P5, P6 y P7?** Si es el modelo de 279 variables,
   la destilacion no puede transportar un margen lexico y las tres pistas caen por la misma razon que
   P1-P3. Si es tambien un modelo sobre texto, son coherentes con el veredicto (a) y con el texto del
   brief ("las heads deben utilizar la representacion TF-IDF permitida"). **El brief no lo fija y es
   la decision que mas condiciona el resto.**
2. **Se acepta la celda de falsacion de la seccion 1.6 como trabajo de P0', aunque el brief no la
   pida?** Cuesta menos de 10 minutos y es lo unico que convierte (a) de "con confound declarado" a
   "medido". Sin ella, P1-P3 se justifican por el veredicto y no por una medicion.
3. **Se reordena el orden del brief?** La auditoria propone P4 antes que P1-P3. El brief ya pone
   puertas ("P2 solo si P1 muestra senal"), asi que reordenar es legitimo, pero es una decision de
   Carlos, no de la auditoria.
4. **Se acepta el riesgo de budget de P9?** V17 gasto 8 de 16 experimentos. Reejecutar el analisis
   de error por cada uno de los candidatos de P4 y P5 es la partida mas cara de la fase.

> **Correccion posterior (D-048, 2026-09-26).** El numero de especialista que se cita aqui (0,4944 y 0,4942) **no es la exactitud del especialista en la frontera**: es una mezcla con la prediccion de V12 en las filas que la compuerta dejaba congeladas. Midiendo al especialista sin compuerta, con el codigo y las variables exactos de P3a, sale **0,7136 y 0,7493**. La conclusion de este informe sobre el especialista binario **queda corregida**: las 279 variables de V12 si contienen la senal de frontera, y mas que la representacion dispersa. El resultado de sistema con compuerta (delta negativo) se mantiene: el problema era la compuerta, no la informacion. Ver `reports/v18_p0_reproduccion.md` y D-048.
