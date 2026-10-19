# 07 - Checklist de verificacion de numeros clave

**Objetivo:** tabla maestra de los numeros que el notebook independiente (`notebooks/kaggle_final_pipeline.ipynb`) debe reproducir (recomputar) o citar (documentar). Cada valor fue verificado contra su fuente el 2026-10-17.
**Fuentes:** `experiments.csv`, `CHAMPION.md`, `DECISIONS.md`, `RUNLOG.md`, `PROJECT_STATE.md`, `FINAL_REPORT.md`, `reports/`, `notebooks/99_entrega_consolidado.ipynb`, `plan/kaggle_notebook/paquetes/04_neuronales.md`.

**Codigos de verificacion:**
- `A` = assert exacto (string/int/float congelado; en floats, tolerancia <= 1e-9).
- `T` = tolerancia (comparacion numerica con tolerancia declarada; sugerido 1e-4 para macro-F1 y deltas, 1e-6 para recomputos deterministas).
- `I` = informativo (se cita como documentado; no se recomputa: puntajes LB, refs Kaggle, cifras historicas sin artefacto reconstruible).

## 0. Correcciones a la lista recibida (verificadas contra fuente)

1. **"suelo uniforme 0.04854" esta mal etiquetado.** `0.04854` (exacto `0.04854212180687087`) es el **suelo estricto/trivial de clase mayoritaria (B0)**; el **azar uniforme** es `0.1666667` (1/6). Son dos anclas distintas y el notebook debe citar ambas.
2. **"ADV-01: CV +0.0031, holdout +0.0052, LB +0.0043"**: valores correctos, pero la fila `ADV-01` de `experiments.csv` es el gap analysis sin metricas. Los numeros son del candidato **AGE_ADVANCED_V01**: CV +0.0031 = `ADV-02` (0.58967 vs 0.5866); holdout +0.0052 = `ADV-03` (0.5966 vs 0.5914); LB +0.0043 = v09 (0.60807 sobre 0.60376 de v08), en `RUNLOG.md:290-294` e `INFORME_EXPERIMENTOS_Y_LEADERBOARD.md:114`.
3. **"Plug-in CV 0.6159732487657944"**: es el valor **global congelado** (V30 + delta +0.0025261) y el baseline de V38+ (`f1_v30_baseline`). No confundir con la estimacion anidada de la particion canonica (**0.6154643**, delta +0.0020172) ni con la alternativa (**0.6164700**). Los tres estan documentados.
4. **"R2 0.998"**: los coeficientes `LB = 0.00223 + 1.02825 x CV` son reproducibles de los **4 pares canonicos** (v01, v07, v08, v12) y su R2 recomputado es **0.996776 (~0.9968)**, no 0.998. El `0.998` solo aparece en prosa posterior (`INFORME...md:66`, notebook seccion 43+); no es reconstruible de los pares documentados.
5. **"media +0.0177 +- 0.0023"**: la media +0.0177 es reproducible; la desviacion recomputada es **~0.0020** (11 envios historicos) o **~0.0010** (12 modernos). El `+-0.0023` es cifra de prosa del informe actual: tratar como informativa.
6. **"D-090 (200 filas = 151 + 26 + 12 + 10 + 1)"**: el acta D-090 habla de **195 filas** (151 + **21** documentales + 12 + 10 + 1). El conteo de **200 filas / 26 documentales** es el inventario actualizado (195 + las 5 filas `AUD-*`), en `plan/contexto.md:294` e `INFORME...md:12`. Citar ambos con su alcance.
7. **"D-091 (inner split)"**: correcto, pero `DECISIONS.md` tiene **dos actas numeradas D-091** (inner split 2026-10-01 y V38 TabularNet 2026-10-16). Citar por titulo ademas del numero.
8. **"V48 vs V43"**: hay dos referencias de V43: **0.6184783** (CV publicado) y **0.6184252** (OOF recomputado). Delta headline +0.0001668 vs 0.6184783 (`CHAMPION.md:33`); +0.0002199 vs 0.6184252 (`RUNLOG.md`/D-101). Los deltas por fold usan la referencia recomputada.
9. **"V39: 0.6168017 / 0.63588"**: correcto; su ganancia LB es **-0.00002** (V38 0.63590 > V39 0.63588), no una mejora.
10. **"techo oracle 0.6935"**: el valor exacto documentado es **0.693547** (V45/D-097); 0.6935 es su redondeo. Medido sobre los 6 miembros del ensamble V43.

## 1. Anclas del corpus y la particion

| Valor esperado | Que es | Exp./version | Fuente documental | Verif. |
|---|---|---|---|---|
| 169,336 filas train / 112,892 filas test | tamanos del corpus | corpus | `notebooks/99...ipynb:19828`; `04_neuronales.md`; `CHAMPION.md:37` | A |
| 164,018 clusters (`dup_clusters.parquet`) | agrupacion near-dup | V20/V30/V38+ | `experiments.csv` (V38); `04_neuronales.md` | A |
| 5 folds: 33868/33868/33867/33866/33867; seed 42 | particion canonica | `artifacts/p33_scores/todos` | `04_neuronales.md:22-23` | A |
| 6 clases `["13-17","18-24","25-34","35-49","50-64","65-xx"]` | orden canonico | `src/config.py:62` | A |
| balance 16.04%-17.04% (razon 1.062) | balance de clases | V18-P0-AUDITORIA | `experiments.csv` | T |
| suelo estricto 0.04854212180687087 | ancla trivial (mayoria) | B0; V33/V34 (gate G3) | `scripts/v33_00_preregistro.py:105`; `experiments.csv` (B0=0.04854) | A |
| azar uniforme 0.1666667 | azar 1/6 | V37-DIAG-POS (D-089) | `PROJECT_STATE.md:69` | A |
| E-001 CV 0.53444 +- 0.00502 | primer baseline | E-001 | `experiments.csv` (fila E-001) | T |
| E-001 permutacion 0.1672 (=azar) | control anti-leakage | E-001 | `experiments.csv` | T |
| V12 pooled 0.6005017473 vs media-folds 0.6004103057 | efecto Jensen; canonico pooled | V17-P0-REPRODUCCION | `experiments.csv`; `reports/v17_p0_reproduccion.md:7` | A/T |
| v01 LB 0.55235 | primer envio | v01 | `INFORME...md:36` | I |
| mejora acumulada +0.08489 (0.55235 -> 0.63724) | ganancia neta | v01/V48 | `INFORME...md:58`; `notebooks/99...ipynb:21` | I |

## 2. Progresion CV / LB del arco campeon

| Version | CV local | LB Kaggle | Ganancia LB vs previo | Fuente | Verif. |
|---|---|---|---|---|---|
| v09 (AGE_ADVANCED_V01) | 0.5897 | 0.60807 | +0.00431 | `INFORME...md:44`; `RUNLOG.md:290` | CV=T; LB=I |
| V12 | 0.6005017473 | 0.62182 | +0.00438 | `CHAMPION.md:22`; `INFORME...md:46` | CV=T; LB=I |
| V19 | 0.6079388 | 0.62578 | +0.00396 | `CHAMPION.md:21` | T/I |
| V20 | 0.6097782 | 0.62689 | +0.00111 | `CHAMPION.md:20` | T/I |
| V30 | 0.6134471 | 0.63302 | +0.00613 | `CHAMPION.md:19` | T/I |
| V30 + Plug-in | 0.6159732487657944 | 0.63575 | +0.00273 | `CHAMPION.md:18`; D-090/D-091 | T/I |
| V38 | 0.6166579 | 0.63590 | +0.00015 | `CHAMPION.md:17`; `experiments.csv` | T/I |
| V39 | 0.6168017 | 0.63588 | -0.00002 | `CHAMPION.md:16` | T/I |
| V40 | 0.6175912 | 0.63685 | +0.00097 | `CHAMPION.md:15` | T/I |
| V41 | 0.6181119 | 0.63722 | +0.00037 | `CHAMPION.md:14` | T/I |
| V42 | 0.6184162 | 0.63716 | -0.00006 | `CHAMPION.md:13` | T/I |
| V43 | 0.6184783 | 0.63722 | +0.00006 | `CHAMPION.md:12` | T/I |
| V48 | 0.6186451 | 0.63724 | +0.00002 | `CHAMPION.md:11` | T/I |
| V12 replica (segundo puntaje) | - | 0.62188 | - | `experiments.csv` (AGE-DECISION-EJECUTIVA) | I |
| conteo de envios: 22 versiones / 23 puntajes (1 degenerado v08 0.39563) | registro | auditoria | `INFORME...md:13,62-64` | I |
| Refs Kaggle: V48 57283487; V43 57277515; V41 57275353; V40 57274929; V39 57274330; V38 57273347 | referencias | V38-V48 | `experiments.csv` | I |

## 3. Deltas, folds y controles del arco V19 -> V48

| Delta / control | Valor esperado | Exp./version | Fuente | Verif. |
|---|---|---|---|---|
| V19 delta CV (5 folds / 4 folds) | +0.0074 / +0.0085 | V19-B1 | `experiments.csv` | T |
| V19 IC95, filas, neto | IC95 [+0.0064,+0.0085]; 11,392 filas (10.09%); neto +1311 | V19-B1 | `experiments.csv` | T |
| V20 delta / folds / IC95 / neto | +0.0018; 5/5; [+0.0012,+0.0025]; +294 (5,674 cambios) | V20-F3 | `experiments.csv` | T |
| V30 recomputado vs publicado | 0.6134471 vs 0.6130051 (desviacion 0.000442); LB pred 0.63301 vs real 0.63302 (error +0.00001) | V31-00 / D-066 | `experiments.csv`; `DECISIONS.md:2061-2064` | A/T |
| Plug-in canonico | +0.0020172; 5/5; IC95 [+0.0010143,+0.0029480] | AGE-PLUGIN-CANON | `experiments.csv`; `DECISIONS.md:2451` | T |
| Plug-in alternativo | +0.0020580; 5/5; IC95 [+0.0011204,+0.0030622]; dif. entre particiones 0.0000408 | AGE-PLUGIN-ALT | `experiments.csv` | T |
| Plug-in global congelado | +0.0025261 (=> 0.6159732487657944) | AGE-PLUGIN-SUBMISSION | `experiments.csv` | T |
| Plug-in submission | 7408 filas cambian (6.56%); 0 cambios en alta confianza; confianza media 0.3361 vs 0.6247 | AGE-PLUGIN-SUBMISSION | `experiments.csv` | A/T |
| V38 delta / folds / IC95 | +0.0006847; 5/5 (+0.00044,+0.00073,+0.00020,+0.00103,+0.00057); [+0.0002189,+0.0011523] | V38 | `experiments.csv`; D-091 (V38) | T |
| V39 delta | +0.0008285 vs V30; 5/5 | V39 | `experiments.csv` | T |
| V40 delta / IC95 | +0.0016180 vs V30; 5/5; [+0.0008660,+0.0023367]; p<1e-6 | V40 / D-092 | `experiments.csv` | T |
| V41 delta / IC95 | +0.0021387 vs V30; +0.0005207 vs V40; 5/5; [+0.0012525,+0.0029622] | V41 / D-093 | `experiments.csv` | T |
| V42 delta / IC95 | +0.0024430 vs V30; +0.0003043 vs V41; 5/5; [+0.0014994,+0.0033831] | V42 | `experiments.csv` | T |
| V43 delta / IC95 | +0.0025050 vs V30; +0.0000621 vs V42; 5/5; [+0.0016135,+0.0034260] | V43 | `experiments.csv` | T |
| V48 delta / folds / bootstrap | +0.0002199 vs 0.6184252; 4/5; media +0.0002151; IC95 [+0.0000007,+0.0004095]; p=0.0250; headline +0.0001668 vs 0.6184783; vs plug-in +0.0026719 | V48 / D-101 | `CHAMPION.md:33-36`; `RUNLOG.md:2743-2747` | T |
| V48 deltas por fold y cambios | +0.000266, -0.000008, +0.000312, +0.000286, +0.000240; 290 filas vs V43 (0.26%) | V48 | `RUNLOG.md:2738-2742`; `CHAMPION.md:35,39` | T |
| V48 pesos | `w=[0.5628,0.0832,0.1724,0.0754,0.0513,0.0549]`; plugin `[1.0000,1.1749,1.2721,1.1630,1.0010,0.8128]` | V48 | `CHAMPION.md:48,63`; `04_neuronales.md:246-251` | A |
| V48 submission | 112,892 filas; 8/8 checks; SHA256 `bf2554571ca6676195af66a053e28411fc1cfc9e3946fd95bac1cae0ce71df62` | V48 | `CHAMPION.md:37-38`; D-101 | A/I |
| standalones neuronales (contexto) | V42 0.613276; V43 0.612840; V44 0.612903; V46 0.605551; V47 0.611082 | V42-V47 | `04_neuronales.md` | T |

## 4. Falsaciones V44-V52 (deltas vs campeon vigente)

| Falsacion | Delta CV | Folds+ / p / IC95 | Detalle adicional | Fuente | Verif. |
|---|---|---|---|---|---|
| V44 (EMD L1) | -0.0001720 | 1/5; p=0.8550 | POS fold0 0.620946; Mixup 0.621222; stacking 0.617191; XGB 0.621319 | `experiments.csv`; `RUNLOG.md` | T |
| V45 (oracle + 21 hipotesis) | n/a (ratifica V43) | n/a | Oracle 0.693547; 65,271 errores; 61.94% d=1; 23.59% d=2; 85.5% <=2 | D-097; `RUNLOG.md:2683` | I |
| V45 ablaciones | n/a | n/a | 1D 0.580048; Focal-Cramer 0.621256; GLU 0.621326; inyeccion V30 0.621073; Nystroem 0.621488; reopt anidada 0.618141 (0/5) | `RUNLOG.md:2685-2697` | I |
| V46 (RankGauss) | -0.0000327 | 2/5 | Standalone 0.6055514 | `experiments.csv`; `RUNLOG.md:2707` | T |
| V47 (Weighted-Cramer + clip) | +0.0000743 | 3/5; p=0.2620; [-0.0001503,+0.0002838] | Standalone 0.6110824; geometrica 0.6185761 (p=0.0930); rank-avg 0.5958594 | `experiments.csv`; `RUNLOG.md:2723-2725` | T |
| V48 (PoE geometrico) | +0.0002199 | 4/5; p=0.0250 | **Adoptado campeon** (ver tablas 2 y 3) | D-101 | T |
| V49 (CatBoost oblivious) | -0.0000816 | 2/5; p=0.7480; [-0.0003022,+0.0001361] | Standalone 0.6066473; segmentos 0.6760801 (con mencion) vs 0.5043177 (sin) | `experiments.csv`; D-102 | T |
| V50 (MHSA + W_6x6) | +0.0000474 nominal | p=0.1940; [-0.0000633,+0.0001577] | MHSA fold0 0.6073432; blend -0.0001058 | `experiments.csv`; D-103 | T |
| V51 (Consistent Smoothed Cramer) | -0.0000302 | 3/5; p=0.6900; [-0.0001337,+0.0000721] | Standalone 0.6128371 | `experiments.csv`; D-104 | T |
| V52 (Power-Mean + Dual-Transport + SNN) | -0.0000610 | 3/5; p=0.7040; [-0.0002945,+0.0001803] | SNN fold0 0.608477; paradoja de Simpson -0.0013763 | `experiments.csv`; D-105 | T |

## 5. Auditoria D-090 / D-091

| Valor esperado | Que es | Fuente documental | Verif. |
|---|---|---|---|
| 195 filas = 151 EJECUTADO + 21 DOCUMENTAL + 12 PREREGISTRO + 10 ENMIENDA + 1 BLOQUEADO | inventario del acta D-090 | `DECISIONS.md:3445-3446`; `PROJECT_STATE.md:12` | A/I |
| 200 filas = 151 + 26 + 12 + 10 + 1 | inventario actualizado (195 + 5 filas `AUD-*`) | `plan/contexto.md:294`; `INFORME...md:12` | A/I |
| 20 defectos (4 media-alta + 9 media + 7 baja); 0 rechazos del campeon | auditoria V1-V37 | `experiments.csv` (AUD-REV); D-090 | I |
| techo LB 0.70 refutado (K-01): rejilla `TAUS=(0.30..0.50)`; en tau=0.2930 el oraculo da +0.0646335 > +0.0626 | prueba de imposibilidad refutada | `DECISIONS.md:3462-3478`; `PROJECT_STATE.md:20-27` | T |
| IC95 [0.635,0.653] del techo: declarado, no estimado (tasa de captura 4%-9% a mano) | etiquetado A-05 | D-091 (2); `experiments.csv` (AUD-IC95-TECHO) | I |
| multiplicidad: SE~0.00188; gate +0.0015 ~32 FP y +0.0030 ~8 FP; Bonferroni 151 +0.0147 | defecto de marco A-01 | `DECISIONS.md:3484-3494` | I |
| inner split: inner ingenuo 3.907% vs inner agrupado 0.000%; outer 0.000%; dif. 0.039071; folds 3.787%-4.152% | reserva A-02 medida | `DECISIONS.md:3563-3570`; `experiments.csv` (AUD-INNER-SPLIT) | A/T |
| correccion de la medicion erronea de D-090: 0.963% vs 1.030% (texto exacto) | autoevaluacion corregida | `DECISIONS.md:3572-3577` | I |
| holdout de V12 quemado (K-02): criterio +0.003 sobre su propio holdout | defecto de razonamiento | `DECISIONS.md:3479-3482` | I |
| campeon intacto: 0.6159732487657944; SHA256 `29cbe6cca4312a65f5176bad25c7d4ae4624d348f3e55586dfa36767a3abd6c0`; 112,892 filas | verificado bit a bit | D-090/D-091; `FINAL_REPORT.md:871` | A/I |
| D-091 duplicado: mismo numero para inner split (2026-10-01) y V38 (2026-10-16) | colision de numeracion | `DECISIONS.md:3534,3608` | I |

## 6. Calibracion CV <-> LB

| Valor esperado | Que es | Fuente documental | Verif. |
|---|---|---|---|
| 4 pares canonicos: v01 0.5344444/0.55235; v07 0.586/0.60372; v08 0.5866/0.60376; v12 0.6005017473/0.62182 | base de la regresion | `experiments/age_cierre_auditoria.json:58-101`; `reports/age_lb_calibracion.csv` | A |
| pendiente 1.02825; intercepto 0.00223 | `LB = 0.00223 + 1.02825 x CV` | D-050; `age_cierre_auditoria.json:60-61` | T |
| R2 recomputado 0.996776 (~0.9968) | ajuste de los 4 pares | `age_cierre_auditoria.json:62` | T |
| R2 0.998 (claim de prosa posterior; no reconstruible de los pares) | discrepancia declarada | `INFORME...md:66`; `notebooks/99...ipynb:17751` | I (flag) |
| brecha media +0.0177 | 11 envios historicos: media +0.01768, sd 0.00204; los informes citan +-0.0023 (prosa) | `notebooks/99...ipynb:1120,1286-1287`; `INFORME...md:68`; `FINAL_REPORT.md:176` | T/I |
| proyeccion de la ley (ejemplo V40) | 0.00223 + 1.02825 x 0.61759 = 0.63732 (error <0.07%) | D-092 (6) | T |
| punto degenerado excluido | v08 segundo puntaje 0.39563; primera regresion: pendiente 0.231, R2 0.0043 | `INFORME...md:62-66` | I |
| gaps por version | v07 +0.0177; v09 +0.0184; v12 +0.0213; V12 replica 0.62188 | `FINAL_REPORT.md:163-174`; `experiments.csv` | I |

## 7. Notas de uso para el notebook

1. Elegir y declarar UNA referencia canonica por version (V43: 0.6184783 publicado vs 0.6184252 recomputado; V30: 0.6134471 recomputado vs 0.6130051 publicado). El notebook debe decir cual usa en cada delta.
2. Los valores `A` deben fallar ruidosamente si no coinciden; los `T` con la tolerancia declarada en Fase 0; los `I` se imprimen como contexto citado con su fuente.
3. `29cbe6cc...` y `bf2554...` solo pueden verificarse como assert si el fichero de submission existe en el entorno; en caso contrario, citar como informativo.
4. La unica cifra de la lista recibida que resulto incorrecta en su etiqueta es "suelo uniforme 0.04854" (es el suelo estricto). Las demas son correctas; las matizaciones estan en la seccion 0.
