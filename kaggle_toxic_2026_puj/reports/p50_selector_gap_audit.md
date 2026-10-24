# P50 - Auditoria de brecha de la familia selectora

Generado por `experiments/v50_00_brecha_y_oracle.py` (V50-00 (auditoria de brecha y techo del oracle)).

Fecha 2026-10-20T02:50:51+00:00. **0 modelos, 0 submissions, 0 consultas a Kaggle.**

## 1. Las cinco categorias, y donde esta cada una

| categoria | ?es por celda? | estado | evidencia |
|---|---|---|---|
| A. ensemble / blending | no | **CERRADO** | reports/v17_00_inventario_complementariedad.json (ENS-04, proba_ens, macro_f1 0,7865) y experiments/23_candidatos_finales.py |
| B. ponderacion global | no | **CERRADO** | experiments/17_stacking_v2.py (late fusion de segunda generacion), variantes meta_shift / B_meta36 / A_cost_sensitive_downweight / C_meta36_downweight, y la funcion ranking_canales de experiments/23_candidatos_finales.py |
| C. selector de umbral | no | **CERRADO** | umbrales V03 0,44/0,51/0,46 y umbrales optimos por variante en reports/v17_00_inventario_complementariedad.json |
| D. correccion selectiva | parcialmente, pero un challenger cada vez | **CERRADO CON VEREDICTO NEGATIVO** | reports/v18_01_cambio_y_senal.json: 5 challengers, 0 con senal tras Bonferroni, mejor AUC 0,6181, mejor mejora fuera de muestra 2,4 celdas frente a 26,75 necesarias |
| E. seleccion por label-cell | si | **NO IMPLEMENTADO** | no existe. Ningun script entrena sobre las salidas de los 20 canales a la vez tomando una decision discreta; los scripts que los consumen (14, 17, 19, 20, 23, 28, 35) los usan para fusion, para ranking o para medida de dureza |

**`SELECTOR_FAMILY = GENUINELY_NEW`.** La categoria E no esta implementada en ninguna parte del repositorio.

## 2. La categoria D ya se midio y dio veredicto negativo

V18-01 es el caso degenerado de esta hipotesis: en vez de elegir entre veinte canales, elige entre V03 y **un** challenger, y en vez de un selector multivariable usa una feature con un corte. Su resultado es el mejor punto de partida que existe para la categoria E.

| magnitud | valor |
|---|---:|
| estado | **NO SELECTIVE CORRECTION SIGNAL** |
| challengers medidos | 5 |
| con senal nominal (p < 0,05) | 2 |
| con senal tras Bonferroni | 0 |
| mejor AUC | 0.6181 |
| mejor p global | 0.0200 (umbral 0.0100) |
| mejor mejora fuera de muestra | 2.4 celdas |
| mejora necesaria para la escala | 26.75 celdas |

Un AUC de 0,62 sobre 202 a 287 celdas de desacuerdo es indistinguible del azar. La mejora fuera de muestra mas alta fue de 2,4 celdas, cuando hacen falta 26,75.

**Consecuencia para P50.** La familia es formalmente nueva, pero hereda un veredicto muy negativo. Solo puede distinguirse de la categoria D si el poder de identificacion esta en la multidimensionalidad o en considerar el pool conjunto. Ese es exactamente el test que P50 tiene que superar.

## 3. Techo del oracle

> USA y_true. NO es modelo, NO es evaluacion valida, NO puede usarse para seleccion. Es solo un techo superior.

El espacio de acciones tiene **21 acciones**: `KEEP_V03` mas los 20 canales.

| magnitud | valor |
|---|---:|
| macro-F1 de V03 | 0.7772114525 |
| macro-F1 del oracle | 0.9870178614 |
| **delta del oracle** | **+0.2098064089** |
| celdas de error de V03 | 757 |
| celdas con alguna accion correcta | 4407 |
| celdas sin ninguna accion correcta | 42 |
| rescate / damage / net | 715 / 0 / 715 |

Por etiqueta:

| etiqueta | F1 de V03 | F1 del oracle | delta | rescate |
|---|---:|---:|---:|---:|
| HS | 0.7616892911 | 0.9930715935 | +0.2313823024 | 307 |
| TR | 0.8157303371 | 0.9865168539 | +0.1707865169 | 152 |
| AG | 0.7542147294 | 0.9814651368 | +0.2272504074 | 256 |

El techo es **69.9 veces** el umbral de decision de 0,003 que la especificacion fija. El margen teorico no es pequeno: es enorme.

Pero ese techo esta concentrado: 42 celdas no tienen ninguna accion correcta entre las 21, luego el oracle no puede llegar a 1,0. En esas celdas ningun selector, por bueno que sea, puede acertar.

## 4. Fiabilidad de los canales, condicional al desacuerdo

La columna que importa no es `accuracy_global`, sino `accuracy_dado_desacuerdo`: que acierta el canal **precisamente en las celdas donde V03 se equivoca**, que es donde un selector lo usaria.

| canal | acc. global | acc. dado desacuerdo | rescate | damage | net |
|---|---:|---:|---:|---:|---:|
| `base:M2_union_lr_c2` | 0.8323 | **0.5420** | 71 | 60 | +11 |
| `emoc:NB-UNION-LR@C=1.0` | 0.8274 | **0.4860** | 191 | 202 | -11 |
| `base:M3_union_svc` | 0.8283 | **0.4828** | 98 | 105 | -7 |
| `emoc:NB-UNION-SVC@C=1.0` | 0.8170 | **0.4420** | 217 | 274 | -57 |
| `emoc:NB-WORD-LR@C=1.0` | 0.8195 | **0.4398** | 168 | 214 | -46 |
| `emoc:NB-CHAR-LR@C=1.0` | 0.8092 | **0.4000** | 184 | 276 | -92 |
| `base:M1_word_lr` | 0.8159 | **0.3994** | 123 | 185 | -62 |
| `emoc:ORT-A@C=1.0` | 0.8159 | **0.3994** | 123 | 185 | -62 |
| `cita:CITA` | 0.8076 | **0.3949** | 186 | 285 | -99 |
| `emoc:ORT-ABCD@C=2.0` | 0.8159 | **0.3869** | 106 | 168 | -62 |
| `base:M4_char_lr` | 0.8067 | **0.3741** | 153 | 256 | -103 |
| `emoc:ORT-AE@C=2.0` | 0.8110 | **0.3542** | 102 | 186 | -84 |
| `emoc:EMO-ALL@C=1.0` | 0.7058 | **0.2625** | 305 | 857 | -552 |
| `emoc:EMO-EST@C=2.0` | 0.7006 | **0.2553** | 300 | 875 | -575 |
| `emoc:EMO-LEX@C=1.0` | 0.6966 | **0.2539** | 306 | 899 | -593 |
| `synt:SYN-3` | 0.6858 | **0.2396** | 295 | 936 | -641 |
| `synt:SYN-POS` | 0.6714 | **0.2336** | 309 | 1014 | -705 |
| `synt:SYN-DEPS` | 0.6557 | **0.2190** | 302 | 1077 | -775 |
| `synt:SYN-MORPH` | 0.6417 | **0.2075** | 297 | 1134 | -837 |
| `cita:NEG` | 0.4464 | **0.1416** | 337 | 2043 | -1706 |

La mejor accuracy dado el desacuerdo es **0.5420**. El valor de referencia es 0,5000.

## 5. Decision de la auditoria

- `SELECTOR_FAMILY = GENUINELY_NEW`: la categoria E no existe todavia.
- `ORACLE_DELTA = +0.209806`: el techo es amplio, luego la especificacion obliga a continuar.
- La categoria D ya dio veredicto negativo con AUC 0,62, luego la carga de la prueba la carga E.

