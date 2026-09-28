# P49 - Reconciliacion forense del rescate OOF

Generado por `experiments/v49_00_rescate_forense.py` (V49-00 (P0 reconciliacion forense del rescate OOF)).

Fecha 2026-09-28T00:53:18+00:00. **0 modelos, 0 submissions, 0 consultas a Kaggle.**

Pregunta: Que significa 715, que significa 147, y la cifra grande es senal o un artefacto de azar?

## 1. Los tensores SI eran recuperables

- ubicaciones: /home/carlos/Git/PLN/kaggle_toxic_2026_puj/features
- canales reconstruidos: **20**, con la misma regla de exclusion de `experiments/28_hardness_v4.py` (se descartan `TXT+*` y `CH-TGT+DIST`)
- validacion contra `reports/hardness_v4.csv`: max |diferencia| = **0.00e+00** sobre las 4449 celdas

La reconstruccion es exacta, de modo que `model_dis_*` no era una aproximacion: es la fraccion de fallo real de esos 20 canales.

## 2. Que significan las tres cifras

| cifra | unidad | que es exactamente |
|---|---|---|
| **757** | label-cell | celdas (tweet, etiqueta) con `pred != y_true` sobre 1483 filas x 3 etiquetas. **No** son tweets con error |
| **147** | label-cell | celdas de error de V03 que acierta **una sola variante**, `C_meta36_downweight` con umbrales 0.5. Es la `cobertura_de_errores_v03` de V17-00 |
| **715** | label-cell | celdas de error de V03 que acierta **al menos uno de los 20 canales** de la fase v2, binarizados a 0.5 |

Las dos no son la misma magnitud y nunca debieron presentarse como si lo fueran: 147 es el rescate de **un** modelo, 715 es el de la **union de veinte**.

## 3. El 147 se reproduce exactamente

| magnitud | registrado | reproducido |
|---|---:|---:|
| rescates de `C_meta36_downweight` | 147 | **147** |
| cobertura | 0.194187582562748 | **0.194187582563** |
| danos | 90 | 90 |

Reproduce: **True**.

## 4. La prueba decisiva: senal o azar

Si los 20 canales se equivocasen de forma independiente, con la probabilidad marginal observada, el numero de celdas de error de V03 en las que **ninguno** acierta seria:

- probabilidad media de acierto de un canal en una celda de error: **0.2756** (5.51 de 20 canales)
- celdas irrecuperables **esperadas** bajo independencia: **1.2**
- celdas irrecuperables **observadas**: **42**

**Es azar: False.**

Si las irrecuperables observadas son del orden de las esperadas bajo independencia, la cifra de rescatables es un artefacto de tener muchos canales y no senal. Si las observadas superan con claridad a las esperadas, los canales fallan juntos con V03 y la cifra es senal de un hueco de representacion compartido

## 5. Matriz de rescate y damage por canal

| canal | rescate | damage | net | cobertura errores V03 |
|---|---:|---:|---:|---:|
| `base:M1_word_lr` | 123 | 185 | -62 | 0.1625 |
| `base:M2_union_lr_c2` | 71 | 60 | +11 | 0.0938 |
| `base:M3_union_svc` | 98 | 105 | -7 | 0.1295 |
| `base:M4_char_lr` | 153 | 256 | -103 | 0.2021 |
| `synt:SYN-POS` | 309 | 1014 | -705 | 0.4082 |
| `synt:SYN-DEPS` | 302 | 1077 | -775 | 0.3989 |
| `synt:SYN-MORPH` | 297 | 1134 | -837 | 0.3923 |
| `synt:SYN-3` | 295 | 936 | -641 | 0.3897 |
| `cita:CITA` | 186 | 285 | -99 | 0.2457 |
| `cita:NEG` | 337 | 2043 | -1706 | 0.4452 |
| `emoc:EMO-EST@C=2.0` | 300 | 875 | -575 | 0.3963 |
| `emoc:EMO-LEX@C=1.0` | 306 | 899 | -593 | 0.4042 |
| `emoc:EMO-ALL@C=1.0` | 305 | 857 | -552 | 0.4029 |
| `emoc:NB-WORD-LR@C=1.0` | 168 | 214 | -46 | 0.2219 |
| `emoc:NB-CHAR-LR@C=1.0` | 184 | 276 | -92 | 0.2431 |
| `emoc:NB-UNION-LR@C=1.0` | 191 | 202 | -11 | 0.2523 |
| `emoc:NB-UNION-SVC@C=1.0` | 217 | 274 | -57 | 0.2867 |
| `emoc:ORT-A@C=1.0` | 123 | 185 | -62 | 0.1625 |
| `emoc:ORT-ABCD@C=2.0` | 106 | 168 | -62 | 0.1400 |
| `emoc:ORT-AE@C=2.0` | 102 | 186 | -84 | 0.1347 |

## 6. Margen sobre el rescate OBSERVADO

| etiqueta | bin de margen | errores | rescatables | irrecuperables | tasa |
|---|---|---:|---:|---:|---:|
| HS | [0,0.10) | 121 | 121 | 0 | 1.0000 |
| HS | [0.10,0.20) | 98 | 96 | 2 | 0.9796 |
| HS | [0.20,1] | 97 | 90 | 7 | 0.9278 |
| TR | [0,0.10) | 54 | 54 | 0 | 1.0000 |
| TR | [0.10,0.20) | 38 | 37 | 1 | 0.9737 |
| TR | [0.20,1] | 72 | 61 | 11 | 0.8472 |
| AG | [0,0.10) | 107 | 105 | 2 | 0.9813 |
| AG | [0.10,0.20) | 71 | 70 | 1 | 0.9859 |
| AG | [0.20,1] | 99 | 81 | 18 | 0.8182 |

## 7. ERROR-DENSE frente a RESCUE-DENSE

| slice | lift error | lift rescate | lift irrecuperable | clase |
|---|---:|---:|---:|---|
| negacion | 1.279 | 1.241 | 1.921 | RESCUE_DENSE |
| longitud_corta | 0.697 | 0.657 | 1.388 | NEUTRAL |
| segunda_persona | 0.950 | 0.935 | 1.208 | NEUTRAL |
| colectivo | 1.187 | 1.188 | 1.167 | RESCUE_DENSE |
| menciones | 0.875 | 0.864 | 1.058 | NEUTRAL |
| atribucion | 1.191 | 1.200 | 1.038 | RESCUE_DENSE |
| leet | 1.012 | 1.015 | 0.972 | NEUTRAL |
| puntuacion_interna | 1.031 | 1.037 | 0.923 | NEUTRAL |
| elongacion | 1.211 | 1.229 | 0.899 | RESCUE_DENSE |
| risas | 0.863 | 0.864 | 0.841 | NEUTRAL |
| margen_bajo | 1.959 | 2.026 | 0.825 | RESCUE_DENSE |
| urls | 1.082 | 1.105 | 0.694 | NEUTRAL |
| cita | 1.445 | 1.496 | 0.579 | RESCUE_DENSE |
| hashtags | 1.075 | 1.123 | 0.245 | NEUTRAL |
| rt | 0.000 | 0.000 | 0.000 | NEUTRAL |

## 8. Decision

**CASE = A.** Ver el detalle en `reports/p49_rescue_reconciliation.csv`, donde cada cifra lleva su unidad, su denominador, su fuente, y si es observacion directa o cota.

