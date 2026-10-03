# P50 - Analisis del selector de canal por label-cell

Generado por `experiments/v50_01_selector.py` (V50-01 (baselines y selector contextual cross-fitted)).

Fecha 2026-09-28T01:26:13+00:00. **0 submissions, 0 consultas a Kaggle.**

> dado un techo del oracle de +0,2098, ?se puede identificar por celda que canal escuchar?

## 1. Configuracion

- acciones: **21**, con `KEEP_V03` siempre disponible, luego el selector puede abstenerse
- modelo: `LogisticRegression(C=1.0, liblinear)`, 15 variables por accion
- cross-fitting: 5 pliegues canonicos de TOXIC, agrupados
- ninguna variable es `y_true`, la correccion de V03 o el canal correcto

## 2. Baselines de diagnostico

| selector | corrected_cv | delta | cambiadas | rescate | damage | net |
|---|---:|---:|---:|---:|---:|---:|
| B0_V03 | 0.7772114525 | +0.0000000000 | 0 | 0 | 0 | +0 |
| B1_mayoria_canales | 0.3502747884 | -0.4269366641 | 2006 | 361 | 1645 | -1284 |
| B2_canal_mas_fiable_base:M1_word_lr | 0.3616799365 | -0.4155315160 | 2028 | 373 | 1655 | -1282 |
| B3_V03_salvo_mayoria | 0.3502747884 | -0.4269366641 | 2006 | 361 | 1645 | -1284 |

## 3. S1, selector de correccion

| magnitud | valor |
|---|---:|
| corrected_cv | **0.7657442265** |
| delta vs V03 | **-0.0114672260** |
| celdas cambiadas | 291 |
| tasa de abstencion | 93.46 pct |
| rescate | 48 |
| damage | 67 |
| net | -19 |
| accuracy de la accion elegida | 0.8256 |
| **AUC maxima de los 21 clasificadores** | **0.7672** |

Correlacion de Spearman entre el AUC de cada canal y su net real: **+0.5559** (p = 0.0109, n = 20 canales). El AUC mide lo predecible de la salida de un canal, no el valor de actuar sobre ella.
El AUC importa mas que el delta: mide si la accion correcta es **identificable** con variables observables. Un AUC de 0,50 significa que elegir es indistinguible de tirar una moneda.

### AUC por accion

| accion | AUC |
|---|---:|
| `0` | 0.7672 |
| `cita:NEG` | 0.7237 |
| `emoc:NB-UNION-SVC@C=1.0` | 0.6315 |
| `emoc:NB-UNION-LR@C=1.0` | 0.6270 |
| `base:M1_word_lr` | 0.6172 |
| `emoc:ORT-A@C=1.0` | 0.6172 |
| `emoc:NB-CHAR-LR@C=1.0` | 0.6165 |
| `emoc:NB-WORD-LR@C=1.0` | 0.6157 |
| `base:M3_union_svc` | 0.6148 |
| `base:M2_union_lr_c2` | 0.6147 |
| `emoc:ORT-ABCD@C=2.0` | 0.6099 |
| `emoc:EMO-EST@C=2.0` | 0.6076 |
| `base:M4_char_lr` | 0.6048 |
| `cita:CITA` | 0.6034 |
| `emoc:ORT-AE@C=2.0` | 0.5968 |
| `synt:SYN-3` | 0.5960 |
| `synt:SYN-POS` | 0.5828 |
| `emoc:EMO-ALL@C=1.0` | 0.5711 |
| `synt:SYN-DEPS` | 0.5698 |
| `emoc:EMO-LEX@C=1.0` | 0.5524 |
| `synt:SYN-MORPH` | 0.5518 |

## 4. Pliegues

| pliegue | filas | F1 de V03 | F1 del selector | delta |
|---:|---:|---:|---:|---:|
| 0 | 297 | 0.7960101143 | 0.7786242237 | -0.0173858906 |
| 1 | 297 | 0.7940339498 | 0.7815857737 | -0.0124481761 |
| 2 | 296 | 0.7788600289 | 0.7765841654 | -0.0022758634 |
| 3 | 296 | 0.7385980131 | 0.7272043962 | -0.0113936169 |
| 4 | 297 | 0.7790564997 | 0.7643558985 | -0.0147006012 |

Pliegues con delta positivo: **0 de 5**.

## 5. Interseccion de errores

| categoria | celdas |
|---|---:|
| ambos correctos | 3625 |
| ambos incorrectos | 709 |
| V03 mal, selector bien (rescate) | 48 |
| V03 bien, selector mal (damage) | 67 |

## 6. Por margen de V03

| bin | celdas | errores | cambiadas | rescate | damage | net |
|---|---:|---:|---:|---:|---:|---:|
| [0,0.10) | 650 | 282 | 112 | 45 | 67 | -22 |
| [0.10,0.20) | 748 | 207 | 3 | 3 | 0 | +3 |
| [0.20,1] | 3051 | 268 | 0 | 0 | 0 | +0 |

## 7. Por confianza de la accion elegida

| bin | n | fraccion accionada | accuracy de la accion |
|---|---:|---:|---:|
| [0.45,0.55) | 168 | 0.7798 | 0.5119 |
| [0.55,0.70) | 606 | 0.2541 | 0.5726 |
| [0.70,1] | 3675 | 0.0016 | 0.8816 |

## 8. Gate y decision

- delta obtenido -0.011467 frente a 0.003 exigido
- pliegues positivos 0 de 4 exigidos
- gate pasado: **False**

**CASE = B.** Ver `reports/p50_selector_gap_audit.md` para la auditoria de brecha y el techo del oracle.

