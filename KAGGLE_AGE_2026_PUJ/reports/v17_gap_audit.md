# V17 - P0 - Auditoria de brechas

**Entrega previa a cualquier entrenamiento.** No se ha entrenado ningun modelo en esta fase: solo se
leen artefactos ya calculados, se inspecciona codigo y se reconcilia la referencia. Ninguna
submission se genera ni se toca nada de V12, V15 o V16.

**Regla de falsacion aplicada**: cada afirmacion sobre un resultado previo cita su artefacto
(`experiments/*.json`) o su informe (`reports/*.md`) y, cuando se puede, la linea del script que lo
produce. Lo que no se ha podido verificar aparece literalmente como `NO VERIFICADO`, sin sustituirlo
por una suposicion.

**Notacion de fronteras**: en las tablas de este informe la frontera `18-24|25-34` se escribe con el
guion vertical, igual que en el resto del proyecto. En `reports/v17_gap_audit.csv` el guion vertical
se sustituye por `__` porque delimita columnas (p. ej. `18-24__25-34`); la equivalencia se declara en
la primera fila de datos de `reports/v17_gap_audit.csv` (familia `nota de notacion`).
El CSV tiene 9 columnas, esa fila de notacion y **42 filas de familia**, una por cada fila
numerada de la tabla de la seccion 3.

---

## 1. Alcance y metodo de la auditoria

### 1.1 Que se leyo completo

| Fichero | Lineas | Para que |
|---|---|---|
| `plan/00_intent.md` | 116 | transcripcion operativa del brief de V17; define P0..P10, puertas y prohibiciones |
| `PROJECT_STATE.md` | 291 | estado por fases y cifras de cabecera |
| `DECISIONS.md` | 1090 | D-001..D-046; se leyeron enteras las D-019, D-020, D-025, D-026, D-027, D-028, D-029, D-030, D-031, D-032, D-033, D-034, D-035, D-036, D-037, D-038, D-039, D-040, D-041, D-042, D-043, D-044, D-045, D-046 |
| `RUNLOG.md` | 721 | sesiones 15-18 (fase residual), entradas 96-107 y 442-548; y las lineas de V13/V15/V16 |
| `FINAL_REPORT.md` | 448 | partes I a IV; en especial 15-20 y 21-25 |
| `experiments.csv` | 95 (94 experimentos) | indice; se leyo por `grep` dirigido y por las 6 filas de V15/V16 |
| `reports/v13_gap_audit.md` y `.csv` | 204 | precedente de formato de P0 y lista de pistas de V13 |
| `reports/v13_final_research.md`, `v13_p15_member_error_matrix.md`, `v16b_activacion.md`, `v16c_reconstruccion.md`, `v16d_puerta.md`, `v3_especialistas.md` | - | diagnosticos de V13 |
| `reports/v14_linguistico.md`, `v14_fusion_linguistico.md`, `v15_correlacion.md`, `v15_poda.md`, `v16_arquitectura.md` | - | informes de V14, V15 y V16 |
| `src/config.py`, `src/cv.py`, `src/data.py`, `src/verificacion_numerica.py`, `src/decision_layer.py` (contrato) | - | contratos de codigo reutilizable |
| `scripts/phase2_ponderacion.py` (`build_features`, `EncodedMLP`, `MEMBER_FILES`, `DENSE_FEATURES`, `evaluate_global`), `scripts/phase21_vista_estructural.py` (`evaluar`), `scripts/v13_p1b_fusion_tokens.py` (`PARES`, `logreg`, `mlp`, `metricas_globales`, `metricas_frontera`, `bootstrap_diferencia`, `complementariedad`), `scripts/v15a_correlacion.py` (`matriz_completa`), `scripts/v15b_poda.py`, `scripts/v16_arquitectura.py`, `scripts/v13_p3a_especialistas.py`, `scripts/advanced_07_frontera.py`, `scripts/phase32_correccion_probabilistica.py`, `scripts/phase5_residuales.py`, `scripts/experimento_e028_ordinal_fino.py` | - | representacion, metricas y protocolos |

### 1.2 Que se inspiro con listados dirigidos

`src/` (20 modulos), `scripts/` (~120 scripts), `notebooks/` (10 cuadernos), `reports/` (100 ficheros) y
`experiments/` (137 JSON). No se releyeron uno a uno: se inventariaron con `glob` y se localizaron por
nombre con `grep` sobre `experiments.csv` y `DECISIONS.md`.

### 1.3 Que NO se pudo verificar

1. **El brief de V17 completo (53 secciones) no esta en el repositorio.** La unica fuente es la
   transcripcion de `plan/00_intent.md`. Las pistas P6 a P10 aparecen alli **solo por su nombre** en
   la fila "P6 a P10" de la tabla de la seccion 4: `Residual`, `coeficientes especificos de
   frontera`, `perfil de confianza`, `experto selectivo`, `decodificacion local`. La auditoria de esas
   cinco pistas se ha hecho contra esos nombres de una linea, no contra su definicion completa. Si el
   brief define P6..P10 con un alcance distinto, las clasificaciones de la seccion 4 hay que revisarlas.
2. **La parte (b) de D-019 no es reproducible en el arbol actual.** `DECISIONS.md:386-387` documenta
   un "enrutamiento por el margen del par-SVM con umbrales tau en {0; 0,5; 1,0; 1,5}" con los
   resultados `tau=0 -> -0,0095`, `tau=0,5 -> -0,0000 (41 corregidas / 43 introducidas)`,
   `tau=1,0 -> 3.571 filas activadas con 0 correcciones y 0 danos` y `tau=1,5 -> 301 activaciones`.
   Ni `scripts/advanced_07_frontera.py` contiene codigo de enrutamiento (su `main` solo tiene el
   bloque "especialista con compuerta", lineas 153-199) ni `experiments/advanced_07_frontera.json`
   registra tales resultados (sus claves son `diagnostico`, `especialista`, `segundos`). El resumen
   `0,0000` si aparece en `reports/advanced_gap_analysis_v2.md:57` y en
   `reports/age_hypothesis_coverage.md:8`. **Los cuatro valores por tau se marcan `NO VERIFICADO`.**
3. **`reports/tables/e028_combinaciones.csv` no existe**, pese a que se cita en
   `reports/advanced_research_report.md:71` como fuente del 0,6856. El 0,6856 si es verificable, pero
   en `experiments/e028_ordinal_fino.json` (vease 3.1).
4. **No se ha reproducido A2 de V16.** Es la tarea de P0' y exige entrenamiento; queda fuera de P0.
5. **No se ha ejecutado nada.** Las dos unicas operaciones con el interprete fueron lecturas de
   artefactos ya existentes y el calculo de metricas inertes sobre ellos, sin ajustar ningun modelo
   (seccion 2.4). No se genero prediccion, no se toco el test y no se abrio ninguna submission.

---

## 2. Reconciliacion de la referencia: 0,6004 frente a 0,6005

### 2.1 Veredicto

**No son la misma magnitud redondeada.** Son **dos estimadores distintos de la misma magnitud**, y la
diferencia entre ellos (9,14e-5) es mayor que la unidad de redondeo a 4 decimales (5e-5), de modo que
no pueden ser el mismo numero escrito con redondeos distintos. Ademas los dos valores caen en lados
opuestos del umbral de redondeo 0,60045, asi que un mismo numero redondeado a 4 decimales jamas
daria los dos.

| Magnitud | Valor |
|---|---|
| (a) **Valor canonico para V17** | **0,6005** (macro-F1 *pooled*, la escala de `metricas_globales`) |
| (b) **Valor reproducido** | 0,6004 en la escala de media-por-fold, y 0,6005 en la pooled. **Los dos reproducen exactamente**; no hay ninguna reproduccion fallida |
| (c) **Motivo exacto** | estimador distinto: macro-F1 de la matriz de confusion *pooled* de las 169.336 filas frente a la media aritmetica de los cinco macro-F1 por fold |

### 2.2 Los dos estimadores, con su codigo

**Estimador POOLED = 0,6005.** `scripts/v13_p1b_fusion_tokens.py:132-142`, funcion
`metricas_globales`: hace `f1_score` una sola vez sobre las 169.336 filas OOF, es decir sobre la
matriz de confusion **acumulada de los cinco folds**.

**Estimador MEDIA-POR-FOLD = 0,6004.** Dos implementaciones, mismo resultado:
- `scripts/phase2_ponderacion.py:103`, `evaluate_global`: `return float(np.mean(scores))` sobre la
  lista de cinco macro-F1 por fold.
- `scripts/phase21_vista_estructural.py:142`: `"macro_f1": round(float(np.mean(puntajes)), 4)`.

La diferencia es **matematica, no de redondeo**: el macro-F1 es una funcion no lineal (razon de
conteos) de la matriz de confusion, asi que la media de los macro-F1 por fold no es el macro-F1 de la
matriz pooled. Es un efecto Jensen. Los folds tienen tamano casi igual (33.868, 33.868, 33.867, 33.866,
33.867), asi que la disparidad no viene del desbalance de folds sino de la no linealidad.

### 2.3 Verificacion inercial sobre el artefacto de produccion

Calculado leyendo `features/oof_ensemble_v12.npy` (169.336 x 6) y los folds de
`src.cv.make_folds(..., groups=..., n_splits=5, n_repeats=1, scheme="group")`. Sin entrenar nada.

| Magnitud | Valor a 10 decimales | Redondeo a 4 |
|---|---|---|
| macro-F1 pooled | 0,6005017473 | **0,6005** |
| media de los 5 macro-F1 por fold | 0,6004103057 | **0,6004** |
| diferencia | 0,0000914416 | -- |

Los cinco macro-F1 por fold a 8 decimales son `0,60065636`, `0,60502888`, `0,60320352`, `0,60039292`,
`0,59276984`, que redondeados dan exactamente el vector `0,6007 / 0,6050 / 0,6032 / 0,6004 / 0,5928`
que aparece en `experiments/v15b_poda.json`, `experiments/v16_arquitectura.json`,
`experiments/phase21_vista_estructural.json` y `plan/V13_ADVANCED_RESEARCH_PLAN.md:10`. Es decir:
**el vector por fold es el mismo en las dos tradiciones**, y solo cambia como se agrega.

### 2.4 Quien usa cada estimador (mapa de ficheros)

| Estimador | Quien lo usa | Valores de referencia registrados |
|---|---|---|
| **Pooled (0,6005)** | `scripts/phase9_combinadores.py:127`, `phase7_nbsvm_canal_meta.py:105`, `phase8_holdout_v12`, y desde V13 todo lo que importa `metricas_globales` | `experiments/phase8_holdout_v12.json` (`cv_promedio_combinadores` 0,6005)  -  `phase9_combinadores.json`  -  `v13_p1b_fusion_tokens.json` (`referencia.macro_f1` 0,6005)  -  `v3_p3a_especialistas.json` (`referencia_macro_f1` 0,6005)  -  `v14b_fusion_linguistico.json` 0,6005  -  `v15b_poda.json` 0,6005  -  `v16_arquitectura.json` 0,6005  -  `v16b_p16b_activacion.json` 0,6005  -  `v16d_puerta_real.json` 0,6005  -  `v17_p17b_pronombres.json` 0,6005 |
| **Media-por-fold (0,6004)** | `scripts/phase2_ponderacion.py:103`, `phase21_vista_estructural.py:142` y los scripts que importan su evaluador | `experiments/phase11_ruido_etiqueta.json`  -  `phase12_auditoria_estado.json:617,623`  -  `phase14_consenso_familias.json`  -  `phase15_calibracion_scores.json`  -  `phase15b_diversidad_subsets.json`  -  `phase16_ordinal_greedy.json`  -  `phase17_estructura_v2_libres.json`  -  `phase18_limpieza_selectiva.json`  -  `phase20_resumen.json`  -  `phase21_vista_estructural.json`  -  `phase22_estructura_ampliada.json`  -  `phase31_decision_ordinal.json`  -  `phase39_fusion_v12.json`  -  `v13_p0b_colinealidad.json` |

Documentos que imprimen 0,6004: `FINAL_REPORT.md:168,340`  -  `RUNLOG.md:442,471,488,501,548`  - 
`DECISIONS.md:548,575,598,619,647,666,693,854,895,994`  -  `reports/age_residual_audit.md` (via D-040)  - 
`reports/age_lexicon_audit.md:17`  -  `reports/fase_residual_informe.md:121`  - 
`reports/v13_entorno_numerico.md:67`  -  `reports/v13_final_research.md:6,38`  - 
`reports/v13_p15_member_error_matrix.md:8`  -  `reports/v13_reproducibility.md:47`  - 
`reports/v16b_activacion.md:51`  -  `plan/V13_ADVANCED_RESEARCH_PLAN.md:10,128,137`.

Documentos que imprimen 0,6005: `PROJECT_STATE.md:47,52,79,101,143`  -  `RUNLOG.md:364,388,452,687`  - 
`DECISIONS.md:458,506,595,891,1028,1058,1070`  -  `FINAL_REPORT.md:133`  - 
`reports/v13_auditoria_idioma.md:15`  -  `reports/v13_final_research.md:54,64`  - 
`reports/v16b_activacion.md`  -  `reports/v16d_puerta.md`  -  `plan/V13_ADVANCED_RESEARCH_PLAN.md:193,204,297`.

**Ninguno de los dos grupos esta equivocado**: cada documento es coherente con su propia epoca y con su
propio estimador. El conflicto es de nomenclatura, no de medicion.

### 2.5 Otros valores del proyecto que pueden confundirse con la referencia

| Valor | Que es realmente | Fuente |
|---|---|---|
| **0,6007** | **exactitud (accuracy) de V12, no macro-F1** | `experiments/v13_p1b_fusion_tokens.json` (`referencia.accuracy`)  -  `reports/v13_p15_member_error_matrix.md:8`  -  `reports/v16b_activacion.md` |
| 0,6015 | mejor capa de decision (D-037 G, tau 0,05), +0,0011 con 4/5 folds y **neto de rescate -117**; no adoptada | `DECISIONS.md:774-777`  -  `reports/age_boundary_models.csv` |
| 0,6020 | V12 con limpieza de duplicados (D-027), +0,0016; no adoptada | `PROJECT_STATE.md:144-145` |
| 0,6077 | holdout congelado, no CV | `experiments/phase8_holdout_v12.json`  -  `RUNLOG.md:442` |
| 0,62182 | leaderboard de `age_submission_v12.csv` (el enviado) | `DECISIONS.md:694`  -  `experiments/phase12_auditoria_estado.json:435` |
| 0,62188 | leaderboard de `age_submission_v12_reproducible.csv`, 93,88 % de acuerdo con el enviado (D-036) | `DECISIONS.md:751-759`  -  `experiments/phase34_resumen.json:46-47` |

Los dos valores de leaderboard **tienen una explicacion documentada y verificada** (dos artefactos
distintos con 93,88 % de coincidencia, `experiments/phase27_reproduccion_v12.json`) y **no son un
problema abierto**. No deben mezclarse con la discrepancia de CV.

### 2.6 Que valor se usa como referencia en los gates de V17

**0,6005, la escala pooled de `metricas_globales`.** Cuatro razones, todas verificables:

1. **Emparejamiento dentro de V16.** `scripts/v16_arquitectura.py:124` calcula la referencia con
   `metricas_globales` y las candidatas A1 (0,6002) y A2 (0,6011) con la misma funcion, en el mismo
   bucle. El delta de +0,0006 de A2 es `0,6011 - 0,6005`. Un candidato de V17 medido en la escala
   pooled restado a 0,6004_MEMORYwouldregoleganaria **+0,0001 regalado**.
2. **Coherencia con el criterio de significacion.** `bootstrap_diferencia`
   (`scripts/v13_p1b_fusion_tokens.py:170-205`) recalcula el macro-F1 de la confusion **pooled**
   remuestreada (`macro_f1_de_confusion`, lineas 208-223). El IC95 del criterio 3 del gate D-042 vive,
   por tanto, en la escala pooled. Los criterios 1 y 3 del gate deben hablar de la misma magnitud.
3. **La banda de ruido esta medida en pooled.** `DECISIONS.md:915-916` calibra el gate de D-042 con
   los controles permutados de P1b, que se mueven entre **0,6000 y 0,6003** en la escala pooled. El
   umbral de efecto de +0,001 se calibro contra ese ruido, no contra el otro estimador.
4. **Toda la linea moderna usa pooled.** V13 P1b, P3a, V14b, V15b y V16 miden su referencia en 0,6005.
   Cambiar de escala en V17 romperia la comparabilidad con las 5 fases anteriores.

**Regla operativa para V17**: todo candidato se mide con `metricas_globales` de
`scripts/v13_p1b_fusion_tokens.py` (pooled) y con `metricas_frontera` para las dos fronteras. Si en
algun informe de V17 aparece 0,6004, es porque se ha usado la media por fold, y entonces **el delta
publicado no es comparable con el de V16**. Cuando se quiera citar el historico, hay que escribir
"0,6004 (media por fold)" o "0,6005 (pooled)" de forma explicita.

### 2.7 Lo que esta reconciliacion NO arregla

`PROJECT_STATE.md:143` y `FINAL_REPORT.md:133,168` mezclan los dos valores en el mismo documento
(`FINAL_REPORT.md:133` dice 0,6005 y `:168` dice 0,6005 con LB 0,62182 mientras `:340` dice 0,6004 con
LB 0,62182). **No se corrige ninguno de esos ficheros en P0**: la regla de la fase es no tocar V12,
V15, V16 ni los informes ya publicados. La correccion es documental y corresponde a la actualizacion de
memoria del cierre de V17.

---

## 3. Inventario por familia

Estado: `CERRADA` = mecanismo valido y resultado estable negativo. `PARCIAL` = probada en una forma y
no en otra. `ABIERTA` = no medida. `VIGENTE` = forma parte del modelo de produccion.

| # | Familia | Hipotesis | Experimento previo | Mecanismo | Resultado | Estado |
|---|---|---|---|---|---|---|
| 1 | Representacion dispersa char+word | char n-gram + canal de menciones separa mejor que las alternativas | E-001 a E-013, D-010 | TF-IDF char_wb + LinearSVC/LogReg | 0,5628 mejor individual (E-013) | VIGENTE |
| 2 | Ensamble de 13-14 miembros | la diversidad de sesgosbate al individual | D-012 a D-016, D-025 | segundo nivel sobre scores de miembros | 0,5866; cerrado por D-016 y addendum | VIGENTE |
| 3 | Meta-modelo MLP(64) + LogReg promedio 0,5 | el promedio de dos cabeas de calidad comparable es el optimo | D-020, D-022, D-025, `phase9_combinadores` | promedio fijo de probabilidades | 0,6005; +Ridge 0,5999, +SGD 0,5676 | VIGENTE |
| 4 | **Arquitectura del meta-modelo** | la forma de combinar las dos cabeas estaba sin optimizar | **V16 A0/A1/A2** | A1 peso por fold; A2 apilamiento logistico de 12 probabilidades | A1 0,6002 (0/5); **A2 0,6011 (+0,0006, 3/5)**, fronteras 0,5065 y 0,5131, IC95 [-0,0003;+0,0015] contiene el cero | **PARCIAL**: D-046 deja sin medir la perdida de la combinacion y las arquitecturas de mas de dos niveles |
| 5 | Segmentacion dura vs ponderacion suave | condicionar el combinador al tipo y la longitud del tweet | D-020, `phase2_ponderacion` | segmentacion por tipo/longitud | segmentado peor; la ponderacion suave funciona | CERRADA |
| 6 | Multivista (estructura, extremos) | una vista sin contenido aporta al ensemble | D-032, `phase21`, `phase22` | vista de secuencia de tipos de token | estructura **+0,0018 (4/5)**, holdout +0,0031; extremos +0,0000 | CERRADA (marginal, no adoptada) |
| 7 | Variable supervisada in-fold `P(clase\|entidad)` | la supervision dentro del train sobre menciones, hashtags y RT aporta | ADV-02, D-017 | cuentas por clase calculadas in-fold | +0,0016 (menciones), +0,0007 (RT) | VIGENTE (ya en V12) |
| 8 | Evidencia por token `P(clase\|token)` | 36 columnas de probabilidad de clase por token | V13 P1, P1a, P1b | bloque in-fold con ajuste cruzado anidado | control 0,3997; fusion **-0,0002**, 3/5, controles permutados 0,6000/0,6003/0,6000 | CERRADA |
| 9 | Pronombres de persona | 26 columnas de pronombres | V13 P17 | recuento closed-vocab | control 0,1918; pre-chequeo **neto -64.075** | CERRADA |
| 10 | Modalidad e incertidumbre | 23 columnas de marcadores de duda | V13 P20 | recuento closed-vocab | control 0,1270, bajo el azar; cobertura 1,6 % | CERRADA |
| 11 | Arboles de conjunciones (SFL) | 9 funciones, 36 columnas | V13 P19 | arboles de profundidad 4/6/8 | control 0,1539, bajo el azar; 0,64 conjunciones por doc | CERRADA |
| 12 | Lexicones tematicos | 24 subcategorias, 1.369 entradas, 72 columnas | D-039, `phase39` | recuento closed-vocab | mejor bloque 0,2033; fusion -0,0001 a -0,0018; rescate <= dano | CERRADA |
| 13 | Variables linguisticas spaCy | 37 columnas de morfologia, lemas, entidades y sintaxis | V14a, V14b, D-043, D-044 | `es_core_news_md` determinista | control 0,2409 (el mejor bloque aislado del proyecto); fusion **-0,0015**, 1/5, segunda particion 0/4 | CERRADA |
| 14 | Colinealidad de la matriz de entrada | la redundancia lineal explicaria el fracaso de lasvariables nuevas | V13 P0b, `reports/v13_colinealidad.md` | matriz completa con `grama_exacta` | VIF mediano 18,87; 170 variables >10; 87 de 278 direcciones para el 95 % de la varianza | DIAGNOSTICO (no es una via de mejora) |
| 15 | **Poda de variables redundantes** | las variables casi identicas sobran | **V15a, V15b, D-045** | un representante por cluster de abs(rho) >= 0,90, elegido por eta2 in-fold | 312 -> 197 variables; **0,5965, delta -0,0040, 0/5 folds**, segunda particion 0/4, IC95 [-0,0054;-0,0027] | **CERRADA: NO ADOPTAR** |
| 16 | **Capa de decision ordinal (33 capas A-I)** | recolocar las probabilidades de V12 mejora macro-F1 | **D-037, `phase31`, `phase32`, `phase33`** | 33 capas: umbrales, correccion residual, transicion, proyeccion ordinal, coste ordinal, edad esperada, calibracion por banda, corrector minimo | ninguna llega al gate; mejor G (tau 0,05) 0,6015 con **neto -117**; C4 con **neto exactamente 0** | **CERRADA** |
| 17 | **Luce / Bradley-Terry sobre margenes adyacentes** | los 5 margenes de E-028a forman una escala probabilistica coherente | **D-037 linea B** (`age_boundary_models.csv` filas B1, B2) | combinacion probabilistica Luce de los margenes existentes | B1 solo **0,2947** (escala invalida); B2 + V12 0,5971 y 0,5867 | **CERRADA** |
| 18 | **SVM por par adyacente (E-028a)** | un LinearSVC por par,**entrenado en el par**, sobre la representacion dispersa | **E-028a, `experiments/e028_ordinal_fino.json`** | 5 LinearSVC C=0,2 sobre TF-IDF char_wb(3,6) + menciones, OOF | exactitud en el par **0,7178 / 0,6864 / 0,7218 / 0,7735 / 0,8145**; aporte al ensemble **+0,0006** | **VIGENTE como miembro**; su uso como especialista con compuerta **no esta probado** (vease 3.1) |
| 19 | Jerarquia gruesa -> fina (E-028b) | agrupar en 3 grupos y luego refinar | E-028b | clasificador grueso de 3 grupos | exactitud 0,7128; aporte +0,0003, peor que solo pares | CERRADA |
| 20 | **Especialistas binarios de frontera sobre las 279 variables** | un modelo nuevo entrenado en las filas del par mejora la decision dentro del par | **V13 P3a, D-041, `reports/v3_especialistas.md`** | LogReg y MLP(64) sobre las 279 variables, entrenados en `outer_train  INTERSECT  par`, aplicados en `outer_val  INTERSECT  par`, forma `gate(especialista, V12)` | `18-24\|25-34` 0,4963 -> **0,4944** (LogReg) / 0,4946 (MLP); `25-34\|35-49` 0,4974 -> **0,4942** / 0,4945; global **0,6005 -> 0,5993** con 6.944 filas; puerta: mejora >= 0,010 **no**, rescate neto >= 1000 **no** (-259), ambas fronteras **no** | **CERRADA** |
| 21 | Especialista con compuerta top-2 (ADV-07) | especialista + compuerta sobre el top-2 del ensemble | ADV-07 parte B, D-019 | LogReg sobre las features del meta, compuerta = el top-2 cae en el par; 3 fronteras | **-0,0036** (2.587 corregidos, 3.217 introducidos) sobre el ensemble v09 de 0,5906 | CERRADA |
| 22 | Enrutamiento por margen del par-SVM | enrutar por el margen del SVM del par con umbral tau | D-019 (b) | compuerta por margen con tau en {0; 0,5; 1,0; 1,5} | resumen 0,0000 en `advanced_gap_analysis_v2.md:57` y `age_hypothesis_coverage.md:8`; **los cuatro valores por tau son `NO VERIFICADO`** (ver 1.3.2) | CERRADA en el resumen, **details no verificables** |
| 23 | Especialistas por longitud | un modelo por tramo de longitud | E-023, E-024 | modelos por quintil/banda de longitud | **+0,0000** | CERRADA |
| 24 | **Gate de ambiguedad sobre un bloque nuevo** | una rejilla de umbrales de confianza puede activar un bloque solo donde es util | **V13 P16b, P16d, `reports/v16d_puerta.md`** | 90 combinaciones de umbrales (confianza del bloque, confianza de V12), eleccion por mitades de folds | techo con puerta perfecta 0,6770; AUC de la confianza del bloque **0,7173**; maximo de la rejilla **0,6005 con 0 filas desviadas** | **CERRADA para el bloque P1**; la maquinaria de umbrales si esta medida |
| 25 | Reconstruccion de un bloque desde V12 | V12 ya contiene la informacion del bloque? | V13 P16, P16c | R2 por grados de libertad efectivos | P1 tokens: R2 mediano 0,527, maximo 0,968, pero control de la reconstruccion **0,0485** (suelo 0,1667) | DIAGNOSTICO |
| 26 | Matriz de error por miembro | que miembro sabe, y donde? | `phase13`, **V13 P15**, `reports/v13_p15_member_error_matrix.md` | sobre las probabilidades OOF de los 14 miembros | 66,87 % del error de V12 tiene algun miembro correcto; separacion de margen **negativa en los 13**; por frontera el mejor es `oof_E015_softmax` con 37,99 % de los 10.122 cruces de `18-24\|25-34` | DIAGNOSTICO (parcialmente reutilizable por P1.1) |
| 27 | Calibracion de miembros y blending en logits | la heterogeneidad de escalas de los margenes pierde informacion | D-029, `phase15` | temperatura por miembro in-fold y promedio en logits | mejor +0,0003 (3/5) -> nulo; T in-fold -0,0001 | CERRADA |
| 28 | Limpieza de filas ruidosas | conservar un representante por duplicado | D-027, D-031, `phase18` | limpieza global y selectiva | +0,0016 (global) y +0,0009 (selectiva, 3/5); ninguna adopta | CERRADA sin adopcion |
| 29 | Seleccion greedy de miembros (Caruana) | la seleccion greedy optimiza la media simple | D-030 G2, `phase16` | media greedy con reemplazo, 10 pasos, elegido en train | **0,5770 (-0,0234)**: el combinador supera a la seleccion | CERRADA |
| 30 | Edad implicita ordinal | reexpresar los scores ordinales como edad esperada | D-030 G1, `phase16` | 15 features de edad implicita in-fold | 0,6001 (-0,0003, 1/5) | CERRADA |
| 31 | Umbrales acumulados ajustados sobre las features del meta | un modelo de umbrales por si solo y su fusion | D-037 linea A | masa de clase por diferencias de supervivencia | 0,5887 solo; fusion 0,5974 (-0,0031) | CERRADA |
| 32 | Correccion residual dentro del par adyacente | ajustar el logit del par top-1/top-2 con senales de contexto | **D-037 linea C**, `scripts/phase32_correccion_probabilistica.py` | `logit_final = logit_base + gamma * Ridge(8 contexto)`, Ridge ajustado en el residuo con filas del fold | C1 -0,0016  -  C2 -0,0005  -  C3 -0,0006  -  **C4 neto exactamente 0** con 1.397 filas modificadas | **CERRADA** (es el antecedente mas proximo a P5 y a P7) |
| 33 | Decodificacion por matriz de transicion | redistribuir masa entre clases adyacentes con una matriz | D-037 linea D | matriz de transicion con shrinkage | 5 variantes, de -0,0004 a -0,0073 | CERRADA |
| 34 | Proyeccion ordinal y coste ordinal | forzar la decision a ser monotona en la edad | D-037 lineas E y F | nudge al vecino adyacente; coste esperado por distancia | E: -0,0033 a -0,0306  -  F: la mejor **+0,0004 (4/5)**, no llega al gate | CERRADA |
| 35 | Confianza como senal de disparo | la confianza de V12 o de un miembro identifica a quien va a fallar | `advanced_04_incertidumbre` (51 features, **+0,0002**), **V13 P15** (separacion negativa en 13 miembros), **V13 P16d** (AUC 0,7173), D-037 linea H (2 y 5 bandas: 0,6003 / 0,6006) | margen, entropia, acuerdo, Platt por banda | ninguna version aporta | **CERRADA** |
| 36 | Residualizacion de features por longitud | el residuo de cada feature tras E[feature \| longitud] es senal ortogonal | `phase5_residuales`, `experiments/phase5_residuales.json` | 40 features residualizadas con 30 bins de cuantiles in-fold | **-0,0019**, 2/5 folds | CERRADA |
| 37 | Retrieval train->test y POS spaCy | dos vias prohibidas que se autorizaron | D-026 | recuperacion por duplicado; 26 recuentos POS | coincidencia de etiqueta del duplicado **61,5 %**; POS **-0,0032** | CERRADA |
| 38 | NB-SVM como modelo y como miembro | NB-SVM aporta al meta | ADV-05, ADV-07 | NB-SVM sobre palabra | como modelo **-0,0017**; como miembro **+0,0007** | CERRADA como modelo, VIGENTE como miembro |
| 39 | **Especialista sparse con compuerta sobre V12** | el SVM por par (E-028a) es mucho mejor que V12 dentro del par; si se usa con compuerta deberia ganar | **NINGUNO** | compuerta `gate(E-028a, V12)` por ambiguedad de V12 | sin medir. La pieza previa (familia 18) mide 0,6864 y 0,7218 en el par frente a 0,4963 y 0,4974 de V12 | **ABIERTA** (vease 5.b) |
| 40 | **Combinador local por frontera** | un meta-combinador entrenado solo con las filas del par, y activo solo en el par | **NINGUNO** | apilamiento por frontera sobre las 12 probabilidades de las dos cabeas | sin medir. El antecedente mas proximo es V16 A2, que es **global**, y D-037 C, que es una correccion residual sobre 8 variables de contexto | **ABIERTA** (vease 5.c) |
| 41 | Gate de ambiguedad de V12 con A2 como contenido | la ambiguedad de V12 separa las filas donde A2 gana | **NINGUNO** | hard gate y luego soft gate sobre el margen de V12 | sin medir. Ningun gate anterior tuvo un componente **mejor que V12 dentro del par** | **ABIERTA** (vease 5.a) |
| 42 | Reconciliacion de la referencia CV | 0,6004 y 0,6005 son el mismo numero redondeado o dos estimadores distintos | `phase2_ponderacion.evaluate_global:103`, `phase21_vista_estructural:142`, `v13_p1b_fusion_tokens.metricas_globales:132-142` | media aritmetica de los 5 macro-F1 por fold frente a macro-F1 de la confusion pooled de las 169.336 filas | **resuelto en esta auditoria**: pooled 0,6005017473, media por fold 0,6004103057, diferencia 9,14e-5. Canonico para V17: **0,6005 (pooled)** | **CERRADA** (vease seccion 2) |

### 3.1 El hallazgo que reencuadra P2, P4 y P5

Dos mediciones ya existentes y verificadas, que nunca se cruzaron:

| Frontera | especialista sparse OOF (E-028a) | V12 en las mismas filas | V13 P3a sobre las 279 variables |
|---|---|---|---|
| `18-24\|25-34` | **0,6864** | 0,4963 | 0,4944 |
| `25-34\|35-49` | **0,7218** | 0,4974 | 0,4942 |
| `13-17\|18-24` | 0,7178 | 0,5895 | no medido |
| `35-49\|50-64` | 0,7735 | 0,6124 | no medido |
| `50-64\|65-xx` | 0,8145 | 0,7184 | no medido |

Fuente del especialista sparse: `experiments/e028_ordinal_fino.json` (0,6856 y 0,7224 como medias por
fold) recalculado en modo inerte sobre `features/oof_E028a_pares.npy` (0,6864 y 0,7218 pooled; las dos
agregaciones coinciden en este caso). Fuente de V12: `metricas_frontera` sobre
`features/oof_ensemble_v12.npy`. Filas del par: 57.023 y 56.577, identicas a
`experiments/v16_arquitectura.json` (`fronteras_referencia.filas`).

**Las tres consecuencias, y son el corazon de V17**:

1. **Un clasificador por par, entrenado solo con el par y evaluado fuera de muestra, supera a V12
   dentro del par en +0,1900 y +0,2245.** La informacion existe y es medible.
2. **Los 5 margenes de E-028a son ya columnas de entrada de V12** (`MEMBER_FILES` en
   `scripts/phase2_ponderacion.py:49-55`; `src/decision_layer.py:33` los declara como sustrato de la
   capa de decision). Es decir, **V12 recibe esa informacion y no la aprovecha dentro del par**: su
   argmax de 6 clases es incompatible con las decisiones binarias que sus propias entradas lo admiten.
   Esto separa **(D) el problema es solo de decision** de **(A) V12 no tiene informacion**: la tiene.
3. **V13 P3a fallo porque uso la representacion equivocada, no porque los especialistas no sirvan.**
   P3a entreno LogReg y MLP sobre las 279 variables, que son una destilacion del ensemble y ya estan
   saturadas (VIF mediano 18,87). Un especialista sobre una matriz que resume al ensemble no puede
   superar al ensemble. La representacion dispersa si separa el par.

Por eso la formulacion literal de P2 en `plan/00_intent.md` linea 41 ("P2 con LogReg/LinearSVC sobre
pares de clases") es **un sinonimo de P3a** si se ejecuta sobre las 279 variables, y es **genuinamente
nueva** si se ejecuta sobre la representacion dispersa. La distincion no es academica: es la diferencia
entre repetir un resultado medido y medir algo que nadie ha medido.

---

## 4. Las diez pistas de V17, una por una

| Pista | Nombre (seccion 4 de `plan/00_intent.md`) | Clasificacion | Experimento previo mas proximo | Por que es o no es sinonimo |
|---|---|---|---|---|
| **P0** | Auditoria de brechas | **genuinamente nueva** (es esta fase) | `reports/v13_gap_audit.md` (V13 P0) | Mismo formato, corpus distinto: aquel inventario 33 pistas de variables; este inventaria 41 familias ya ejecutadas y clasifica 10 pistas de decision. Sin experimento asociado. |
| **P0'** | Reconciliar 0,6004/0,6005 y reproducir A2 | **parcialmente probada** | `experiments/phase27_reproduccion_v12.json` (reproduccion del artifacto de submission, 93,88 %) | La reconciliacion documental **esta hecha** en la seccion 2 de este informe. La **reproduccion de A2 sigue pendiente** y exige entrenamiento, luego es trabajo de P0' y no de P0. |
| **P1  -  P1.1** | Matriz de error por miembro en las fronteras y complementariedad local | **parcialmente probada** | **V13 P15** (`reports/v13_p15_member_error_matrix.md` seccion 5) y `complementariedad` de `scripts/v13_p1b_fusion_tokens.py:226-239` | P15 ya dio la tabla miembro x frontera sobre los 10.122 y 7.252 cruces, con el mejor miembro y su exactitud en el par. Y la complementariedad local **entre V12 y A2 ya esta medida**: `experiments/v16_arquitectura.json` da `18-24\|25-34` rescate 1.525 / dano 945 / **neto +580** y `25-34\|35-49` 1.516 / 627 / **neto +889**, frente a un neto global de **-295**. Lo que falta es la matriz por miembro **de los candidatos nuevos**, no la de V12. |
| **P2  -  P2.1  -  P2.2  -  P2.3** | Especialistas reales `18-24 vs 25-34` y `25-34 vs 35-49`, protocolo anidado | **parcialmente probada** (dividida en dos) | **V13 P3a** (`scripts/v13_p3a_especialistas.py`, D-041) y **ADV-07 parte B** (D-019) | Ver 5.b. La forma de 279 variables esta **CERRADA** con el mismo protocolo anidado y con dos modelos. La forma sparse esta **ABIERTA** y tiene evidencia a su favor. |
| **P3  -  P3.1-P3.3** | Gating de ambiguedad: hard gate, luego soft gate | **genuinamente nueva** (el mecanismo), **parcialmente probada** (la rejilla de umbrales) | **V13 P16d** (90 combinaciones de umbrales, maximo 0,6005 con 0 filas desviadas) y **ADV-07 (b)** (compuerta por margen) | La rejilla de umbrales comotal **ya se midio**, pero con un contenido que era peor que V12 en todas partes. Lo nuevo es que **A2 es mejor que V12 dentro del par** (+0,0102 y +0,0157), de modo que un gate tiene por primera vez algo que ganar. Ver 5.a. |
| **P4  -  P4.1-P4.2** | Entrenamiento pairwise real (adyacente primero) | **parcialmente probada** | **E-028a** (SVM por par, ya miembro, VIGENTE) y **D-037 linea B** (Luce/Bradley-Terry sobre esos margenes) | Ver 5.d. Lo que ya existe es el **clasificador** por par y su uso como miembro. Lo que no existe es un **objetivo** pairwise (rank o pairwise loss) ni un especialista por par **usado como decision**. Y lo que **esta cerrado** es la combinacion Luce/Bradley-Terry de esos margenes, que es una capa de decision, no un entrenamiento. |
| **P5  -  P5.1-P5.2** | Meta-combinador local por frontera con inner OOF | **genuinamente nueva** | **V16 A2** (combinador **global** de dos niveles) y **D-037 linea C** (correccion residual Ridge de 8 variables de contexto dentro del par) | Ver 5.c. A2 es un unico combinador para las 6 clases; P5 seria **un combinador por par**, entrenado con las filas del par y activo solo en el par. D-037 C es lo mas proximo en localizacion, pero sus variables son 8 senales de contexto y su objetivo es un residuo de logit, no un apilamiento de las dos cabeas. |
| **P6** | Residual | **cerrada** (en las dos lecturas) | **D-037 linea C** (correccion residual de la decision: neto exactamente 0) y **`phase5_residuales`** (residualizacion de features por longitud: -0,0019) | "Residual" admite dos lecturas y **las dos estan medidas y son negativas**: residuar la decision dentro del par (D-037 C) y residuar las features respecto a la longitud (`phase5`). Se cierra de entrada salvo que el brief precise un alcance distinto, que no consta en el repositorio (ver 1.3.1). |
| **P7** | Coeficientes especificos de frontera | **parcialmente probada** | **D-037 lineas A y C** | A1 (0,5887) ajusta un modelo de umbrales sobre las features del meta, es decir coeficientes por umbral; C1-C4 ajustan un Ridge **por par** (el Ridge se entrena en `seleccion_entreno`, filas cuyo par coincide con el del top-1/top-2). Es exactamente "coeficientes especificos de frontera" y da -0,0016 a 0,0000. Lo no probado es ajustarlos **sobre las probabilidades de las dos cabeas** en vez de sobre 8 variables de contexto. |
| **P8** | Perfil de confianza | **cerrada** | **`advanced_04_incertidumbre`** (+0,0002 con 51 features), **V13 P15** (separacion de margen negativa en los 13 miembros), **V13 P16d** (AUC 0,7173), **D-037 linea H** (2 y 5 bandas: 0,6003 y 0,6006) | Cuatro mediciones independientes, todas negativas o nulas, y una de ellas (P15) demuestra que el signo de la separacion es **negativo** en los 13 miembros: la confianza del miembro va en el sentido contrario al de su utilidad. Se cierra de entrada. |
| **P9** | Experto selectivo | **parcialmente probada** | **V13 P3a** (seleccionar un especialista por par), **D-030 G2** (seleccion greedy, -0,0234), **ADV-05** (seleccion de features, 0,0000), **V16 A1** (peso aprendido por fold, 0/5) | Elegir un modelo distinto por fila exige una senal de seleccion, y **la unica senal disponible es la confianza, que P8 acaba de cerrar**. P3a es ademas la forma "un experto por par" y fallo. Queda como mucho la variante "un experto por par activado por la ambiguedad de V12" (P3), que es P3 y no P9. |
| **P10** | Decodificacion local | **cerrada** | **D-037 lineas D, E y F** | Decodificacion por matriz de transicion (5 variantes, -0,0004 a -0,0073), proyeccion ordinal (6 variantes, -0,0033 a -0,0306) y coste ordinal (3 variantes, la mejor +0,0004 con 4/5, sin gate). Y la version **local** de la decodificacion es la correccion residual de la linea C, tambien cerrada. Se cierra de entrada. |

**Recuento de las diez pistas**: P0 es esta misma fase, sin experimento asociado, y no se clasifica.
De las nueve restantes: **cerradas 3** (P6, P8, P10), **parcialmente probadas 5** (P1, P2, P4, P7, P9)
y **genuinamente nuevas 2** (P3, P5). P0' se anade aparte: su mitad documental esta hecha en la
seccion 2 y su mitad de entrenamiento sigue pendiente, asi que esta **a medias**.

**Aviso sobre la tabla**: la clasificacion de P3 y de P5 es "genuinamente nueva" en cuanto al
**mecanismo**, pero ambas arrastran un precedente parcial que hay que respetar: P3 tiene precedents la
rejilla de 90 umbrales de P16d, y P5 tiene como antecedente la correccion residual de D-037 linea C.
Lo que es genuinamente nuevo en ambas es **el contenido del mecanismo** (A2 como contenido del gate, y
un combinador por par en vez de uno global), no el tipo de pieza. En el recuento de la seccion 8 se
tratan como dos de las dos pistas abiertas.

---

## 5. Cruce critico: por que cada pista nueva NO es una reformulacion de algo ya cerrado

### 5.a P3, gating de ambiguedad

**Lo cerrado**. `scripts/v13_p16d_...` (fichero `experiments/v16d_puerta_real.json`,
`reports/v16d_puerta.md`) midio una rejilla de **90 combinaciones de umbrales** sobre la confianza del
bloque P1 y la confianza de V12, con eleccion por mitades de folds. Resultado: techo con puerta
perfecta 0,6770, AUC de la confianza del bloque 0,7173, y **maximo de la rejilla 0,6005 con 0 filas
desviadas**, es decir la mejor puerta es no usar el bloque. ADV-07 (b) midio una compuerta por margen.

**Por que P3 no es sinonimo**.

1. **El contenido del gate es distinto y es la diferencia decisive.** En P16d el componente que se
   activaba era el bloque de evidencia por token, con exactitud 0,4028 frente al 0,6007 de V12: en
   **todo** el train era peor que V12, de modo que cualquier puerta solo podia perder. En P3 el
   componente que se activa es **A2**, que en las dos fronteras centrales es **mejor** que V12
   (0,5065 frente a 0,4963 y 0,5131 frente a 0,4974). Es la primera vez que un componente del gate es
   mejor que V12 en la region donde actua.
2. **La senal de activacion es distinta.** P16d y ADV-07 usaban la confianza del **otro** componente
   (el bloque, el margen del par-SVM). P3 usa la **ambiguedad de V12** (que su top-2 cae en el par y cual
   es su margen), que es la unica senal disponible en inferencia y que P15 no midio como disparador de
   un componente mejor.
3. **La firma empirica es distinta.** V16 `complementariedad` da **neto +580** en `18-24|25-34` y
   **neto +889** en `25-34|35-49`, con un neto global de **-295**: el componente gana filas exactamente
   donde se le quiere activar y pierde filas fuera. P16d no tenia nada que aktivarse; aqui si, y la
   asimetria esta medida.

**Lo que hay que aceptar como riesgo**: P8 esta cerrada, y la ambiguedad de V12 es un tipo de
confianza. Si el hard gate de P3.1 falla, P3.2 y P3.3 tienen una base probabilistica debil y lo honesto es cerrar
P3, no buscar otro gatillo.

### 5.b P2, especialistas de frontera (prestacion especial)

**Que se midio exactamente en V13 P3a** (`scripts/v13_p3a_especialistas.py`, informe
`reports/v3_especialistas.md`):

| Elemento | Valor en V13 P3a |
|---|---|
| **Representacion** | las **279 variables de V12**, sin variables nuevas: `build_features()` mas `oof_ADV07_nbsvm_word.npy` (linea 102-103). Ninguna representacion dispersa |
| **Modelos** | `especialista_logreg` (StandardScaler + LogReg C=1,0, lbfgs, linea 78-79) y `especialista_mlp` (StandardScaler + MLPClassifier(64), alpha 1e-3, batch 512, early stopping, max_iter 40, linea 85-88) |
| **Datos de entrenamiento** | `filas_entrenamiento = entrenamiento[en_par[entrenamiento]]` (linea 124): exactamente `outer_train  INTERSECT  {las dos clases del par}` |
| **Datos de aplicacion** | `objetivo = validacion[actuable[validacion]]` con `actuable = V12 predice una de las dos` (lineas 117 y 128): exactamente `outer_val  INTERSECT  par  INTERSECT  {V12 en el par}` |
| **Clases** | las dos fronteras centrales: `18-24\|25-34` y `25-34\|35-49` (`PARES`, linea 70) |
| **Forma de la puerta** | `gate(especialista, V12)` de la seccion 37, sin umbral de ambiguedad: actua sobre todas las filas actuables |
| **Resultado** | 0,4963 -> 0,4944 (LogReg) y 0,4946 (MLP); 0,4974 -> 0,4942 y 0,4945. Global 0,6005 -> 0,5993 con 6.944 filas modificadas. Red de -109, -96, -180, -163. Puerta: **0 de 3 criterios cumplidos** |
| **Decision** | cerrar la pista sin pagar la etapa 2 (D-041) |

**Que es distinto en V17**:

1. **La representacion.** Es la diferencia de fondo y no una sutileza. Sobre las 279 variables el
   especialista obtiene 0,4944, es decir **reproduce al ensemble conuna eficiencia del 99,6 %**: es una
   destilacion del ensemble y un modelo nuevo sobre ella no puede ganarle. Sobre la representacion
   dispersa el mismo tipo de clasificador (E-028a, LinearSVC C=0,2 sobre char_wb(3,6) + menciones)
   alcanza **0,6864** y **0,7218** en las mismas filas. Hay **+0,19 y +0,22** sin medir en la puerta.
2. **La linea temporal.** P3a se ejecuto el 2026-09-26, **antes** de V16. En ese momento no existia
   evidencia de que el combinador tuviera algo que aportar en las fronteras. V16 la dio. La hipotesis
   de P2 en V17 no es "un especialista mejora el par" (falso, medido), sino "un especialista sobre la
   representacion correcta, combinado con la senal de A2, mejora el par mas que A2 sola" (nunca medido).
3. **El tercer modelo.** P3a uso LogReg y MLP sobre la matriz densa. `plan/00_intent.md` linea 41 nombra
   **LinearSVC**, que es exactamente el estimador de E-028a, sobre una matriz donde si funciona. Anadir
   LinearSVC no es un sintono: es el estimador que ya demostro 0,6864.
4. **P2.3, la tercera frontera.** P3a y ADV-07 midieron `18-24|25-34` y `25-34|35-49`; ADV-07 midio
   tambien `35-49|50-64` pero con el especialista denso. Elgap de la tabla de la seccion 3.1 para
   `35-49|50-64` es **+0,1610** y para `50-64|65-xx` es **+0,0961** con el especialista sparse: nunca
   evaluado con compuerta sobre V12.

**Conclusion de P2**: **parcialmente probada, no cerrada**. La sub-pista "especialista sobre las 279
variables" se cierra de entrada por D-041. La sub-pista "especialista sparse con compuerta sobre V12"
es genuinamente nueva y es la de mayor valor esperado de toda la fase. **Ejecutar P2 como esta escrito
en el brief, es decir LogReg/LinearSVC sin especificar la representacion, repetiria P3a con un
estimador mas.**

### 5.c P5, meta-combinador local por frontera (prestacion especial)

**Lo cerrado**: el combinador de V16. `scripts/v16_arquitectura.py:160-163` hace `hstack` de las **12**
probabilidades (6 de la LogReg + 6 de la MLP), entrena **una** `LogisticRegression(C=1.0)` sobre ellas
y aplica `modelo.predict_proba` a las 6 clases (linea 102-104). El combinador es **unico y global**:
las mismas twelve columnas entran para las seis decisiones, y el mismo modelo decide `13-17` y decide
`18-24 vs 25-34`. El entranamiento ocurre en una particion interior agrupada del 20 % del train exterior
(`partido_agrupado`, lineas 82-90) y el modelo se aplica al train exterior completo (lineas 165-169).
Su resultado es 0,6011 con 3/5 folds, IC95 que contiene el cero, y fronteras +0,0102 y +0,0157.

**Por que P5 no es sinonimo**:

1. **La unidad de entrenamiento cambia.** P5 entrena un combinador **por par**, con las filas cuyo
   valor verdadero cae en el par, y lo activa **solo** en las filas cuyo top-2 de V12 cae en el par.
   V16 entrena un combinador con todas las filas y lo activa en todas. En un problema donde el 84,5 %
   del error se concentra en dos fronteras (`DECISIONS.md:698`), un modelo global spends su capacidad
   de discriminacion en las cuatro clases faciles, que es exactamente lo que dice el propio informe de
   V16: "la media ponderada por clases la dominan las clases FACILES, y ahi el combinador no tiene nada
   que ganar" (`reports/v16_arquitectura.md:74-75`).
2. **El conjunto de columnas es distinto.** El combinador local puede incluir, ademas de las 12
   probabilidades, la **decision binaria del especialista sparse** (el margen de E-028a para ese par) y
   la **confianza de V12**. Nada de eso es una reformulacion de A2: es informacion que A2 no recibe.
3. **El perfil de riesgo es el opuesto al de A2.** A2 tiene **neto global -295** con netos de frontera
   +580 y +889: gana donde se le quiere limitar y pierde fuera. Un combinador local elimina por
   construccion la parte que pierde. Esa asimetria es la que hace que la misma pieza que fallo como
   combinador global pueda funcionar como combinador local. **No es la misma hipotesis.**

**Antecedente mas proximo y por que no es sinonimo**: D-037 linea C (`scripts/phase32_correccion_probabilistica.py`,
`construir_ord4`, lineas 62-89). Es local, es por par y es una correccion residual. Difiere en tres
puntos verificables: (a) sus variables son **8 senales de contexto** (`contexto_extra`, lineas 48-53:
`n_chars`, `n_mention`, `n_hashtag`, `n_url`, `is_rt`, `es_grupo_solo_url`, `es_grupo_muy_corto`,
`es_grupo_sin_menciones`), no las probabilidades de las dos cabeas; (b) su objetivo es el **residuo**
`y - sigmoid(logit_base)` con un Ridge, no un apilamiento de dos niveles; (c) su mejor variante C4 llega
a **neto exactamente 0** con 1.397 filas, es decir no muevo nada.

### 5.d P4, entrenamiento pairwise

**Lo cerrado**:

- **D-037 linea B, Luce / Bradley-Terry.** `reports/age_boundary_models.csv` filas B1 y B2. B1 (Luce
  solo sobre los margenes de E-028a) da **0,2947**, una escala que no es valida; B2 (Luce + V12, peso
  0,25 y 0,5) da 0,5971 y 0,5867. Es una **capa de decision** sobre margenes ya calculados, y su
  veredicto en D-037 es explicito: "los margenes adyacentes de E-028a no forman una escala de
  Bradley-Terry valida".
- **E-028a, el SVM por par.** Esto **si** es entrenamiento pairwise real: cinco LinearSVC entrenados
  solo con las filas de su par, sobre la representacion dispersa, con salida OOF. Y funciona:
  0,6864 y 0,7218 dentro de los pares centrales. Es miembro de V12 y aporta +0,0006.
- **E-028b, la jerarquia gruesa -> fina**: 0,7128 de exactitud, +0,0003, cerrada.
- **G7 desempate por edad esperada** (D-037 linea G): mejora el MAE ordinal de 0,6552 a 0,6454 y
  empeora las dos fronteras centrales en +462 y +534 errores.

**Por que P4 no es sinonimo de nada de eso**:

1. **La combinacion Luce/Bradley-Terry es post-proceso; el entrenamiento pairwise es un objetivo.**
   Es la distincion que la propia auditoria de V13 ya establecio en su fila P5
   (`reports/v13_gap_audit.md:126`): "Luce, Bradley-Terry y proyeccion ordinal fueron capas de decision
   sobre probabilidades". D-037 cerro la capa; **el objetivo nunca se entreno**.
2. **E-028a ya aporta la mitad de lo que P4 pide, y su informacion esta desaprovechada.** Es el
   argumento fuerte: si la pregunta de P4 es "hay informacion de estructura ordinal en el par?", la
   respuesta ya esta medida y es **si**, en un margen enorme (+0,19 y +0,22 dentro del par). Lo que
   P4 tiene que aportar no es la existencia de la informacion, sino un **uso** de ella. Y aqui el
  Storage es el problema: V12 ya recibe los 5 margenes y sigue fallando el par.
3. **Lo que si queda genuinamente abierto en P4**, y es poco: (a) un **objetivo** rank o pairwise
   sobre la representacion dispersa en vez de un SVM binario por par; (b) un SVM por par
   **reentrenado dentro del fold exterior y usado como decision con compuerta**, que es la
   interseccion de P2 y P4. Con la evidencia de 3.1, **(b) tiene mas valor esperado que (a)**, y de
   hecho es la misma pieza. **Aviso de economy**: P4 y la sub-pista sparse de P2 son probablemente el
   **mismo experimento**; si se ejecuta P2 sparse, P4.1 no debe repetirse.

### 5.e P7, coeficientes especificos de frontera

D-037 lineas A y C ya ajustan coeficientes por frontera (el Ridge de `construir_ord4` se entrena en
`seleccion_entreno`, que son las filas cuyo par coincide; A1 ajusta umbrales sobre las features del
meta). Resultado: de -0,0016 a 0,0000, con C4 en **neto exactamente 0**. Lo no probado es ajustar esos
coeficientes **sobre las 12 probabilidades de las dos cabeas** en lugar de sobre 8 variables de
contexto. Eso es genuinamente nuevo, pero es una variante de P5, no una pista independiente: si P5 se
ejecuta con un Ridge en vez de con una LogReg, P7 queda cubierta por el mismo script.

---

## 6. Artefactos reutilizables

Todas las rutas verificadas con existencia y tamano en `features/`. La columna "creacion verificada"
dice si la generacion del artefacto esta documentada y comprobada en el arbol actual.

| Artefacto | Contenido | Tamano | Creacion verificada |
|---|---|---|---|
| `features/oof_ensemble_v12.npy` | probabilidades OOF de V12, 169.336 x 6 | 3,9 MB | si: `phase8_holdout_v12` (D-034) y `phase12_auditoria_estado` la reconstruye con diferencia maxima 0,0 y acuerdo 100,0000 % con la submission v11 (`RUNLOG.md:440-441`) |
| `features/oof_E028a_pares.npy` | 5 margenes de los SVM por par, 169.336 x 5 | 3,3 MB | si: `scripts/experimento_e028_ordinal_fino.py` con `LinearSVC(C=0.2)` sobre `char_wb(3,6)` + menciones; exactitud en el par 0,7178/0,6864/0,7218/0,7735/0,8145 |
| `features/oof_ADV07_nbsvm_word.npy` | scores NB-SVM de palabra | 3,9 MB | si: `advanced_07` (ADV-07) |
| `features/oof_argmax_v12.npy` y `oof_argmax_v12__*.npy` | argmax de V12 y de los bloques de la fase de lexicones | 168 KB c/u | si: fase de lexicones |
| `features/v16_oof_combinador_logistico.npy` | OOF de A2, la pieza que gana en las fronteras | 7,8 MB | si: `scripts/v16_arquitectura.py:188` |
| `features/v16_oof_peso_aprendido.npy` | OOF de A1 | 7,8 MB | si: mismo script |
| `features/v15b_oof.npy` | OOF de la poda (descartada) | 7,8 MB | si: `scripts/v15b_poda.py` |
| `features/v14b_oof.npy`, `features/v14_linguistico.npy`, `features/v14_linguistico_cache.parquet`, `features/v14_linguistico_nombres.json` | bloque linguistico de 37 columnas (33 tras el cribado) | - | si: `scripts/v14a_linguistico.py`, SHA256 de `meta.json` registrado en `reports/v14_linguistico.md:9` |
| `features/age_structured_features.parquet` | 87 features estructuradas | 19 MB | si: fase avanzada |
| `features/dense_train.parquet` | 35 features densas de textometria | 4,5 MB | si: `src/textometry.py` |
| `features/age_supervised_features.npy` | 73 features supervisadas, calculadas in-fold | 95 MB | si: fase avanzada |
| `features/length_residuals.npy` | 40 features residualizadas por longitud | 62 MB | si: `scripts/phase5_residuales.py` |
| `features/dup_clusters.parquet` | clusteres de near-duplicados (282.228 filas, train primero) | 24 MB | si: `src/cv.build_dup_clusters`, metodo documentado en el docstring de `src/cv.py` |
| `features/v13_p1_tokens_fold{0..4}.npy` y `v13_p1_tokens_nombres.json` | bloque de evidencia por token, por fold | 24 MB c/u | si: `scripts/v13_p1a_evidencia_tokens.py` |
| `src/verificacion_numerica.py` | `autotest_numerico()` y `grama_exacta()` | - | si: D-040. **Obligatorio** en la primera linea de todo script de V17 (`plan/00_intent.md:112-114`) |
| `src/cv.make_folds(y, groups, n_splits, n_repeats, scheme="group")` | unico split permitido (D-003) | - | si: `src/cv.py:332-375` |
| `src/config` (`CLASSES`, `TARGET`, `SEED`=42, `N_SPLITS`, `N_REPEATS`, `TEXT_COL`, `ID_COL`, rutas) | contrato central | - | si: `src/config.py:57-72` |
| `src/data.load_train()` | carga de `data/raw/age_train.csv` (`;`, 169.336 filas) | - | si: `src/data.py` |
| `scripts/phase2_ponderacion.build_features()` | las 279 variables de V12: 78 scores de miembro + 35 densas + 87 estructuradas + 73 supervisadas + 6 NB-SVM | - | si: `scripts/phase2_ponderacion.py:61-81`, con `MEMBER_FILES` en 49-55 y `DENSE_FEATURES` en 56-57 |
| `scripts/phase2_ponderacion.EncodedMLP` | la cabeza MLP de V12 (64, alpha 1e-3, batch 512, early stopping) | - | si: `scripts/phase2_ponderacion.py:106-125`. **No sustituir por `MLPClassifier` pelado**: invalidaria la comparacion |
| `scripts/v13_p1b_fusion_tokens.logreg()` y `.mlp()` | las dos cabezas de V12 con sus parametros exactos | - | si: `scripts/v13_p1b_fusion_tokens.py:79-93` |
| `scripts/v13_p1b_fusion_tokens.PARES` | `{(18-24, 25-34): "18-24\|25-34", (25-34, 35-49): "25-34\|35-49"}` | - | si: linea 76. **Diccionario de frontera canonico del proyecto** |
| `scripts/v13_p1b_fusion_tokens.metricas_globales` / `metricas_frontera` / `bootstrap_diferencia` / `complementariedad` | metricas pooled, doble metrica de frontera, IC95 pareado con 4.000 remuest multinomiales, rescate-dano-neto | - | si: lineas 132-142, 145-167, 170-205, 226-239 |
| `scripts/v15a_correlacion.matriz_completa` | 279 + 33 = 312 columnas con sus nombres, y asercion `len(nombres) == 279` | - | si: `scripts/v15a_correlacion.py:87-100` |

---

## 7. Riesgos y trampas de leakage detectados en el terreno

1. **Las etiquetas de frontera estan definidas en un diccionario, no en una constante de clase.**
   `PARES` esta en `scripts/v13_p1b_fusion_tokens.py:76` y **re-exportado** por
   `scripts/v16_arquitectura.py:58-59`. `scripts/v13_p3a_especialistas.py:70` define **su propia copia
   identica**, y `src/decision_layer.py:33` usa una constante de string distinta
   (`PARES_ADYACENTES = "oof_E028a_pares"`). Cualquier especialista, gate o combinador local de V17
   **debe usar las mismas etiquetas de `PARES`**; si se redefinen, las cifras de frontera de V17 no
   seran comparables con las de V16 y el cruce de criterio de la puerta de D-042 quedara invalido.
   Para P2.3 hay que **anadir** el tercer par al diccionario de forma explicita y declarada, no
   derivarlo de otro sitio.
2. **`metricas_frontera` mide `accuracy` dentro del par, no `macro_f1`.** En
   `experiments/v16_arquitectura.json`, `fronteras_referencia["18-24|25-34"]` tiene `accuracy` 0,4963 y
   `macro_f1` 0,5923. Los_boundary de +0,0102 y +0,0157 de D-046 son diferencias de **accuracy**. mixing
   las dos metricas de frontera daria una lectura falsamente grande del efecto.
3. **`cruces_resueltos` vale 0 por construccion** en `metricas_frontera` para estos candidatos, porque
   la condicion exige que la prediccion sea la del par Y la de la clase verdadera, y el par contiene
   precisamente las dos clases que se excluyen. No usar ese campo como medida de nada.
4. **Los controles permutados de la era V13 estan calibrados en la escala pooled** (0,6000 a 0,6003,
   `DECISIONS.md:915-916`). Un control de permutado calculado con la media por fold no es comparable.
5. **La specialist sparse de P2 tiene un riesgo de leakage propio que P3a no tenia**: el
   `TfidfVectorizer` debe ajustarse **dentro de cada fold exterior** sobre el train del fold, y el
   especialista debe entrenarse en `outer_train  INTERSECT  par`. `scripts/experimento_e028_ordinal_fino.py:84-89`
   lo hace correctamente (ajusta `char` y `mentions` con `texts.iloc[train_idx]`), y ese archivo es el
   patron a copiar. A diferencia de P3a, aqui el vectorizador **ve el texto**, asi que un ajuste
   fuera del fold seria una fuga silenciosa que no apareceria en ninguna metrica agregada.
6. **La particion interior de V16 es agrupada por duplicados y debe seguir siendolo.**
   `partido_agrupado` (`scripts/v16_arquitectura.py:82-90`) reparte **grupos**, no filas, entre
   interior_train e interior_valid. Si P5 se implementa con un `train_test_split` ingenuo, el
   combinador local vera casi-duplicados de sus propias filas de entrenamiento y el IC95 del bootstrap
   dejara de ser creible.
7. **El `SEED + 7` de la segunda particion de V16** (`scripts/v16_arquitectura.py:197`) esta fijado y
   no debe cambiarse sin registrarlo: es lo que hace la segunda particion comparable con la de V15b
   (0,6013 / 0,5998 / 0,6039 / 0,6048).
8. **`A @ B.T` esta prohibido** (D-040): la correlacion se calcula con `grama_exacta` o `einsum`, nunca
   con producto matricial cuando los dos operandos son vistas de la misma matriz. En este entorno
   `numpy 1.26.4` sobre `Python 3.14.7` falla en silencio y devolveria ceros, es decir "no hay
   colinealidad", la conclusion opuesta. `plan/00_intent.md:114` lo reitera.
9. **La frontier set de V12 que se usa como referencia debe ser `features/oof_ensemble_v12.npy`, no un
   OOF reentrenado**, si se quiere el emparejamiento fila a fila que hace comparables los deltas
   (argumento de `scripts/v13_p1b_fusion_tokens.py:10-14`). Reentrenar V12 para obtener "su propia"
   referencia introduce ruido de reentrenamiento en cada comparacion.
10. **`ADV-05` y `SRC-ORACLE`**: el techo con **puerta perfecta** de 0,6770 que P16b y P15 calculan
    requiere saber que filas falla V12. Ninguna regla realizable puede acercarse a ese numero; usarlo
    como cota de un candidato real es un error de lectura que ya seSERVER committed una vez
    (`reports/v16b_activacion.md:51` lo advierte explicitamente).
11. **`reports/tables/e028_combinaciones.csv` no existe** pese a estar citado. Si V17 necesita la tabla
    de combinaciones de pares, hay que regenerarla o leer `experiments/e028_ordinal_fino.json`.
12. **No existe un artefacto con la parte (b) de D-019.** Si P3.1 o P2SPARSE necesitan el antecedente
    del enrutamiento por margen, hay que generarlo desde cero o accepting que el dato es
    `NO VERIFICADO`.

---

## 8. Conclusion

### 8.1 Pistas que quedan genuinamente abiertas, en orden

El orden no es el de la tabla de `plan/00_intent.md` sino el que sale de la evidencia, y hay que decirlo
porque es una discrepancia con el brief: **P1.1 y P2.3 no tienen por que ser las tres primeras**.

| Orden | Pista | Por que aqui | Depende de |
|---|---|---|---|
| **1** | **P0'** (reproducir A2) | es la puerta del brief y es barata: sin A2 reproducible, todo lo demas no tiene referencia | - |
| **2** | **P1.1** (miembro y complementariedad local) | diagnostico de 20-40 min que **decide** si P2 y P3 se cierran sin entrenar, exactamente como hizo P15 antes de P3a. Ademas es el unico entregable que mide la firma empirica que V17 necesita (neto de frontera frente a neto global) | P0' |
| **3** | **P2SPARSE** (especialista sparse con compuerta, las dos fronteras centrales) | mayor valor esperado de la fase: **+0,19 y +0,22 medidos** en el par y jamas usados con compuerta. Modelos: LinearSVC (el de E-028a) y LogReg **sobre la representacion dispersa**, no sobre las 279 | P0' y P1.1 |
| **4** | **P3.1** (hard gate de ambiguedad con A2 como contenido) | primera vez que el contenido del gate es mejor que V12 en la region. Es el unico uso que la asimetria de `complementariedad` (neto global -295, netos de frontera +580 y +889) hace plausible | P0' |
| **5** | **P5** (combinador local por frontera con inner OOF agrupado) | culmination: si P2SPARSE y P3.1 dan senal, el combinador local es la pieza que la convierte en decision. Ver 5.c | P2SPARSE y P3.1 |
| **6** | **P2.3** (tercera frontera `35-49|50-64`) | solo si el enfoque funciona; el gap sparse es +0,1610 | P2SPARSE |
| **7** | **P7** | probablemente cubierta por P5 si el combinador local es un Ridge | P5 |
| **8** | **P4.1** | probablemente **el mismo experimento** que P2SPARSE; no debe repetirse | P2SPARSE |

### 8.2 Pistas que quedan cerradas de entrada

| Pista | Cierre | Evidencia |
|---|---|---|
| **P6** Residual | cerrada en las dos lecturas | D-037 linea C: neto exactamente 0 con 1.397 filas  -  `experiments/phase5_residuales.json`: -0,0019 con 2/5 folds |
| **P8** Perfil de confianza | cerrada | `advanced_04_incertidumbre` +0,0002 con 51 features  -  P15: separacion **negativa** en los 13 miembros  -  P16d: AUC 0,7173  -  D-037 linea H: 0,6003 y 0,6006 |
| **P10** Decodificacion local | cerrada | D-037 lineas D (5 variantes, -0,0004 a -0,0073), E (6 variantes, -0,0033 a -0,0306) y F (3 variantes, mejor +0,0004 sin gate); la version local es la linea C |
| **P2 sobre las 279 variables** | cerrada | V13 P3a: 0,4963 -> 0,4944 y 0,4974 -> 0,4942, con el **mismo protocolo anidado**; ADV-07 parte B: -0,0036 |
| **P4 en su forma de capa de decision** | cerrada | D-037 linea B: B1 0,2947 (escala invalida), B2 0,5971 y 0,5867 |
| **P9** | cerrada en la practica | depende de P8 (senal de seleccion) y de P2 (el experto por par); ambas cerradas |

### 8.3 Las tres conclusiones que el resto de la fase debe tener presentes

1. **La referencia de V17 es 0,6005 (pooled), no 0,6004.** No es un redondeo distinto: es otro
   estimador, con una diferencia medida de 9,14e-5, y la escala pooled es la que usan A2, el bootstrap
   del IC95 y toda la linea V13-V16. Un delta publicado contra 0,6004 lleva **+0,0001 regalado**.
2. **El especialista de frontera no fallo: fallo la representacion.** Un clasificador por par sobre la
   representacion dispersa saca 0,6864 y 0,7218 dentro de los pares centrales, frente a los 0,4963 y
   0,4974 de V12 y a los 0,4944 y 0,4942 del especialista de V13 P3a sobre las 279 variables. Y los 5
   margenes de eseclasificador **ya son entradas de V12**, que sigue fallando el par. Eso convierte la
   pregunta de V17 de "hay mas informacion?" en "por que la decision de 6 clases no coherente con las
   decisiones binarias que ya recibe?".
3. **Ejecutar P2 tal como esta escrito en el brief seria repetir P3a.** `plan/00_intent.md` linea 41
   nombra "LogReg/LinearSVC sobre pares de clases" sin fijar la representacion. Sobre las 279 variables
   eso es P3a, con un estimador mas, y su resultado esta medido: peor que V12. La version
   genuinamente nueva exige la representacion dispersa, el protocolo anidado de `plan/00_intent.md` linea
   78-79 y, si se quiere capturar el +0,19, el `gate` de la forma `gate(especialista, V12)`.

---

_Generado en P0 de V17. Sin entrenamiento, sin predicciones, sin submissions. Unica operacion con el
interprete: lectura de artefactos existentes y calculo de metricas inertes sobre ellos (seccion 2.3 y
3.1). No se ha modificado ningun script ejecutado, ni V12, ni V15, ni V16, ni las submissions._

> Nota sobre las tablas: el guion vertical de las etiquetas de frontera se escapa
> como `\|` en las filas de tabla para que markdown no lo interprete como separador de columna
> (mismo criterio que el que aplico `scripts/v15_v16_tablas.py` en V16). Fuera de las tablas se
> escribe sin escapar. No se modifico ninguna cifra.
