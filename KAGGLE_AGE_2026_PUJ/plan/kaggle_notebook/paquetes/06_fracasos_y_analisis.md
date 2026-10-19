# Paquete 06 - Fracasos seleccionados y analisis matematicos/estadisticos

Material para portar al notebook independiente `notebooks/kaggle_final_pipeline.ipynb` (paquete 06 del
plan `plan/kaggle_notebook/00_plan_maestro.md`, Fase 1). Solo lectura del repo; cero experimentos nuevos.

Convenciones de cita:
- `experiments.csv:NN` = linea fisica NN del CSV (la linea 1 es la cabecera; cada linea es una fila).
- `DECISIONS.md:NNNN` = linea del acta; se acompana el id (D-0XX) porque hay numeros repetidos (ver B.4).
- `RUNLOG.md:NNNN` = linea del hito. `99 ipynb seccion S` = celda del consolidado (fuente, no se modifica).
- Los fragmentos citados de codigo/registro se copian tal cual (ASCII del proyecto).

## A. FRACASOS SELECCIONADOS

### A.1 Tabla maestra (que se probo, resultado exacto, por que se cerro)

| # | Fracaso | Que se probo | Resultado exacto | Cierre / por que | Re-ej. | Fuente |
|---|---|---|---|---|---|---|
| A1 | MLP denso (E-019/E-020; miembro del blend en la era V19) | MLP(128,64) sobre 35 densas; MLP(256,128) sobre densas + SVD(64); luego como miembro del ensamble | 0.3381 +- 0.0040 (peor que LightGBM denso 0.3567); 0.4071 +- 0.0060 (la SVD explica solo 7.8% de la varianza); como miembro: 0.5784 -> 0.5851 (+0.0066) | D-015: el DL no se usa como modelo principal; entra al ensamble solo por decorrelacion de errores. En la era V19 el MLP(64) es un tercio del blend (V19-B1, LogReg + MLP + arbol) | barato (~5-6 min CPU) si se reconstruye el bloque de 35 densas; si no, documentable | `experiments.csv:29-31`; `DECISIONS.md:265-281` (D-015); `RUNLOG.md:209-210`; blend en `experiments.csv:111` y `RUNLOG.md:1208` |
| A2 | CatBoost (V32-E2) | CatBoost MultiClass sobre la matriz 279 congelada (3000 it cap, lr 0.03, depth 6, l2 3.0, ES 150) vs gemelo LightGBM (A2) y barajado (A3) | A1 comb 0.6148474, delta -0.0011258 (1/5 folds, IC95 [-0.0019378, -0.0003467]); A2 -0.0012871; A3 -0.0012138; A1 vs A2 +0.00017 (IC95 contiene 0, p+=0.76); A1 rescata 9 errores de alta confianza vs 9.780 del campeon | D-082: CLOSED por el arbol pre-registrado; sin segunda particion. Despues V49 (oblivious trees) tampoco: delta -0.0000816, 2/5, p=0.7480 | documentable (1.149 s CPU y exige la matriz 279 congelada) | `experiments.csv:173` y `:213`; `DECISIONS.md:2823-2849` (D-082), `:3820-3835` (D-102); `RUNLOG.md:2213`, `:2763` |
| A3 | FTRL-Proximal (V32-V35; cierre por mecanismo D-085) | Cuatro ciclos pre-registrados: OvR 10 pasadas (E3), softmax 1 pasada con alpha derivado (V33), multi-pasada con regla de margen (V34), lambda1>0 (V35) | E3: A1 0.6147759, delta -0.0011973 (1/5, A3 empatado bit a bit; standalone 0.040169 < trivial 0.048542). V33: puerta G1-G6 cerrada en fold 0 (ll_train 1.8048 vs techo 1.6126; colapso de frecuencias a una clase). V34: `LINEA_ONLINE_CERRADA_POR_MECANISMO` (G0 D_acum 12.1->23.2 vs banda [0.805,3.219]; G1 ll_train 10.09->19.17 vs techo 1.7816; gemelo SGD sano 1.0622->0.4042). V35: lambda1=0.23993793862904017; la norma L2 de w se duplica (razon 2.1304); cierre definitivo | D-083/D-084/D-085/D-086: cerrada por MECANISMO, no por gates de delta; integracion y bootstrap no se ejecutan por pre-registro. Evidencia previa: 27 celdas de alpha, 0/27 por debajo de H(prior)=1.7914537 | documentable (nucleo FTRL propio + TF-IDF por fold; fail-fast en fold 0) | `experiments.csv:175,178,182,190`; `DECISIONS.md:2930` (D-083), `:2852` (D-084), `:2980` (D-085), `:3072` (D-086); `RUNLOG.md:2232,2272,2315,2404` |
| A4 | CORAL / cumulative-link (V23) | Un unico w compartido + 5 umbrales ordenados por construccion (`src/ordinal_coral.py`, numpy+scipy) vs control B de 5 binarios lineales independientes | CORAL 0.4837017, delta -0.126077 (0/5 folds; rescate 24.468 / dano 47.087 / neto -22.619); control B 0.4690055 (-0.140773); compartir pesos vale +0.014696; monotonia 0 violaciones vs 18.927 de V20; 36/36 celdas de confusion cambian (L1 57.650) | D-058: `ORDINAL_SHARED_WEIGHTS = NO_SIGNAL` y `CORAL = CLOSED`; la consistencia ordinal no era el cuello de botella (2a confirmacion de D-057). Sin submission ni bootstrap por pre-registro | documentable (2.470 s; depende de `src/ordinal_coral.py`) | `experiments.csv:128-129`; `DECISIONS.md:1683-1757` (D-058); `RUNLOG.md:1729`; `99 ipynb seccion 32` (linea 5147) |
| A5 | Modelado ordinal (V22) | Compartir representacion (un multiclase con 5 CDF por suma) y coste de error proporcional a la distancia ordinal | Compartido 0.6096150, delta -0.000163 (3/5, IC95 [-0.00078, +0.00045], neto -13, 4.451 cambios; 0 violaciones vs 18.927). Coste ordinal: alfa1 0.5949198 (-0.0149, neto -3.995) y alfa2 0.5392168 (-0.0706, neto -13.838) | D-057: `WRONG_OBJECTIVE`. Resolver la monotonia gratis no compra metrica; el coste ordinal destruye el macro-F1. Cierra la familia ordinal en 4 subfamilias (con D-054, D-055, D-056) | documentable | `experiments.csv:124,127`; `DECISIONS.md:1633-1682` (D-057) |
| A6 | Especialistas de frontera (V13-P3a y V22-B1) | Binarios por par de fronteras; sustitucion del primer discriminador B1 de V20 (LinearSVC C 0.2/1.0 y LightGBM binario) | P3a: 18-24 / 25-34: 0.4963 -> 0.4944 (LogReg) y 0.4946 (MLP), neto -109/-96; 25-34 / 35-49: 0.4974 -> 0.4942/0.4945, neto -180/-163; global 0.6005 -> 0.5993 (6.944 filas). E-028a: 0.6864 y 0.7218 en las dos fronteras. V22-B1: C0.2 0.5992098 (-0.0106, B1 0.6782); C1.0 0.5992058 (B1 0.6784); boosting 0.6097377 (B1 0.7606 < 0.7632 de V20) | P3a CERRADA (el especialista dedicado es peor que V12 restringida al par). V22: `B1_SPECIALIST = NO_SIGNAL`; razon estructural (las 5 CDF comparten features y modelo). Correccion D-048: sin compuerta el especialista sobre las 279 da 0.7136/0.7493: la senal existe, el problema es de identificabilidad. Auditoria V22-00: 14 familias cerradas, 1 parcial, 5 sin explorar | documentable | `experiments.csv:86` (P3a), `:96` (V17-P1), `:100` (P9, delta -0.0001, 8.634 filas activadas, 3 rescates / 22 danos), `:103` (D-048), `:125-126` (V22); `DECISIONS.md:1142` (D-048) |
| A7 | Canales linguisticos V31 (URL, ortografia, composicion, discurso) | Banco de representaciones con modelo/folds/semilla fijos (LinearSVC C=1.0); integracion como 4o miembro (N1) y como 48 features del meta-espacio (N2) | Ver tabla A.2. Integracion: N1 +0.0000642 sobre el control reoptimizado (los +0.0004119 vs V30 congelado son en su mayoria el efecto de reoptimizar pesos); N2 +0.0002948 sobre el control barajado | D-067: senal real y consistente pero ya absorbida; 0 finalistas, 0 submissions. URL CLOSED; posicion y forma_palabra CLOSED; ortografia/composicion/discurso MARGINAL (senal real, sin sitio donde V30 no la exprese) | documentable (banco: 3.802 s) | `experiments.csv:142-148`; `DECISIONS.md:2086-2152` (D-067) |
| A8 | Seleccion supervisada de features (V25) | Top-50% por `mutual_info_classif` in-fold (K=139 de 279), estandarizado dentro del training fold | T0 0.6097782 vs T1 0.6091492: delta -0.000629 (1/5 folds, neto -101, 3.545 filas). Frecuencia 1.000: las 78 probabilidades del ensamble y las 6 de NB-SVM; nunca sobreviven 74 de 87 estructuradas ni 30 de 35 densas; 137 features estables en los 5 folds | D-060: `SUPERVISED_FEATURE_SELECTION = CLOSED`; la seleccion no elimina ruido, elimina senal (lo que ordena son las probabilidades de los miembros). Sin otro K ni otro selector | documentable (953 s; exige la matriz 279) | `experiments.csv:130-131`; `DECISIONS.md:1816-1882` (D-060); `99 ipynb seccion 33` (linea 6695) |
| A9 | Poda de variables redundantes (V15) | Un representante por cluster de abs(rho)>=0.9, elegido por mayor eta2 in-fold: 197 de 312 variables | 0.5965 vs 0.6005: delta -0.0040 (0/5 folds); 2a particion 0/4 con media -0.0065; IC95 [-0.0054, -0.0027]. Diagnostico previo: 402 pares con abs(rho)>=0.80, 245 >=0.90, 15 duplicados exactos, 58 clusters (171/310), Jaccard entre folds 1.0000; VIF mediano 18.87 y 170 variables con VIF>=10 | D-045: NO ADOPTAR la poda; la redundancia es funcional (las casi identicas actuan como conjunto de estimadores debiles que promedia varianza). La criba de redundancia queda obligatoria antes de fusionar bloques, no como poda | documentable (860 s) | `experiments.csv:75` (V13-P0B), `:91-92` (V15a/V15b); `DECISIONS.md:1015-1049` (D-045) |

### A.2 Canales linguisticos V31 en detalle

Base del banco: `raw` = 0.4979941 (LinearSVC, folds canonicos de V20). Fuente de las filas:
`reports/v31_representation_benchmark.csv` (lineas 2-16); deltas replicados en `experiments.csv:143-144`.

| Banco | Representacion | macro-F1 | delta vs raw | folds + | Lectura |
|---|---|---|---|---|---|
| v31_02 URL | url_canal | 0.4987321 | +0.000738 | 3/5 | marginal; no llega al gate +0.0015 |
| v31_02 URL | url_struct | 0.4926034 | -0.005391 | 0/5 | sustituir la URL por token estructural cuesta mas que dejarla |
| v31_02 URL | url_mask | 0.4910249 | -0.006969 | 0/5 | enmascarar la URL cuesta: la URL no es ruido |
| v31_02 URL | solo_url | 0.1563043 | -0.341690 | 0/5 | los tweets solo-URL son azar (sin semantica de dominio) |
| v31_03 canales | ortografia | 0.4999868 | +0.0019927 | 5/5 | senal real y consistente |
| v31_03 canales | composicion | 0.5016523 | +0.0036582 | 5/5 | senal real y consistente |
| v31_03 canales | discurso | 0.5030002 | +0.0050061 | 5/5 | la familia mas fuerte |
| v31_03 canales | ortografia+composicion+discurso | 0.5048142 | +0.0068201 | 5/5 | mejor combinacion del banco |
| v31_03 canales | url (como canal) | 0.4998877 | +0.0018936 | 5/5 | coincide con D-067 ("el canal URL explicito solo aporta 0,001894") |
| v31_03 canales | posicion | 0.4956907 | -0.0023034 | 0/5 | CERRADA |
| v31_03 canales | forma_palabra | 0.4950099 | -0.0029842 | 0/5 | CERRADA |
| v31_03 canales | todos | 0.4961184 | -0.0018757 | 1/5 | sumar los canales malos diluye a los buenos |

Hallazgos complementarios:
- Taxonomia URL (`experiments.csv:142`): prevalencia global 40.20%, monotona por edad (13-17: 33.4% -> 65-xx: 50.0%), pero 99% son `t.co` con token opaco de 10 caracteres, 0 status URLs, 1.917 URLs externas y 54 acortadores; enriquecimiento 0.745-1.244. No hay semantica de dominio que extraer.
- Complementariedad (`experiments.csv:145`): acuerdo 0.75228; rescate 11.761 / dano 30.187 / neto -18.426; error compartido 31.7%; bootstrap del rescate neto IC95 [-0.11103, -0.10683]. El neto negativo es tautologico para un clasificador mas debil; el criterio correcto es la tasa de error compartido (D-067, `DECISIONS.md:2116-2122`).
- Mineria por errores (`experiments.csv:148`): 65.403 errores (38.62%); 9.780 de alta confianza (15.0%); el 90% de las diez confusiones principales son adyacentes (35-49 -> 25-34: 0.1334; 35-49 -> 50-64: 0.1313). Diagnostico: resolucion de escala, no informacion ausente.

## B. ANALISIS MATEMATICOS / ESTADISTICOS

### B.1 Techo oracle V45 = 0.693547 (como se calculo)

Receta exacta (codigo en `notebooks/99_entrega_consolidado.ipynb`, seccion 44.1, lineas 18852-18948;
codigo 18907-18947, salida 18892-18903):
1. 169.336 filas OOF; 6 miembros: V30, V43, V42, V40, V39_Ord, V39_HGB (arrays `artifacts/p33_scores/todos/` y `features/v*_oof_*.npy`).
2. `p_blend = 0.555*V30 + 0.085*V43 + 0.175*V42 + 0.080*V40 + 0.050*V39_ord + 0.055*V39_hgb`; `pred = argmax(p_blend * w)` con el plug-in congelado `w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]`.
3. Errores del blend: 65.271 (38.55%). Para cada error, si algun miembro cumple `argmax(p_m[i] * w) == y_true[i]`, la prediccion se sustituye por `y_true`; macro-F1 del vector corregido = 0.693547.
4. Descomposicion ordinal de los errores: d=1: 40.429 (61.94%); d=2: 15.395 (23.59%); d>=3: 9.447 (14.47%) -> 85.5% a distancia <= 2.
Caveats: es un oraculo de SELECCION por fila sobre los 6 miembros ya entrenados (cota superior de un selector perfecto con la misma evidencia), no un modelo; el intento de realizarlo (arbitro V17-P9) fallo con 3 rescates / 22 danos (`experiments.csv:100`). No presentarlo como objetivo alcanzable.
Fuentes: `99 ipynb seccion 44.1`; `DECISIONS.md:3720-3759` (D-097); `RUNLOG.md:2681-2701`.
Re-ej.: segundos si se adjuntan los 6 npy OOF + `y_true.npy`; en notebook independiente sin artefactos = documentable.
Antecedente: el oraculo restringido de D-087 dio +0.0567955 < +0.0626 exigido (`experiments.csv:191`; `DECISIONS.md:3160-3231`); su prueba quedo refutada por K-01 (ver B.3).

### B.2 Calibracion CV <-> LB

| Cantidad | Valor | Fuente |
|---|---|---|
| Regresion final | `LB = 0.00223 + 1.02825 x CV` (R2 = 0.998) | `reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md:66`; `99 ipynb seccion 38.1` (lineas 17749-17752) |
| Primera regresion (incluia el v08 degenerado, LB 0.39563) | pendiente 0.231 y R2 0.0043; excluido el punto se obtiene la linea final | `reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md:62-66` |
| Brecha media CV -> LB | +0.0177 +- 0.0023 (10 mediciones) | `reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md:68` |
| Causa medida | el test contiene near-duplicados del train que el CV agrupado elimina por diseno; el LB los cobra | `reports/INFORME_EXPERIMENTOS_Y_LEADERBOARD.md:68` |
| Dato persistido | `reports/age_lb_calibracion.csv` (4 filas; p.ej. CV 0.6005017473 -> LB 0.62182, gap 0.02132) | `reports/age_lb_calibracion.csv` |
| Nota de discrepancia | `experiments.csv:109` (AGE-CIERRE-AUDITORIA) cita R2 0.9968; la cifra vigente del informe y del notebook es 0.998 | `experiments.csv:109` vs informe:66 |

Re-ej.: barato (regresion lineal sobre la tabla de pares CV/LB del informe, seccion 3).

### B.3 Auditoria D-090 (2026-10-01)

- 20 defectos confirmados; ninguno invalida una adopcion. Desglose del registro: 4 media-alta, 9 media,
  7 baja (`experiments.csv:198`). Revisiones por especialidad (nlp, arquitectura, DL, kaggle, reviewer y
  QA) mas verificacion propia del auditor, en `plan/auditoria/02a` (N-01..N-08),
  `02b` (A-01..A-07) y `02c` (K-01..K-04 + QA).
- K-01 (ALTA): el techo de LB 0.70 queda REFUTADO como prueba de imposibilidad. La rejilla de
  `scripts/v35_00g_techo_analisis.py:69` es `TAUS = (0.30, 0.35, 0.40, 0.45, 0.50)` y el "techo" se tomo
  como su maximo, pero la curva del oraculo es monotona creciente al bajar tau. Reproduccion propia sobre
  `artifacts/p33_scores/todos/` (base 0.6159732, 65.683 errores):

| tau | filas corregibles | ganancia macro-F1 | supera +0.0626? |
|---|---|---|---|
| 0.30 (maximo publicado) | 10.384 | +0.0593677 | no |
| 0.2930 (punto que el propio D-087 usa como objetivo) | 11.298 | +0.0646335 | SI |
| 0.28 | 13.199 | +0.0755813 | SI |
| 0.25 | 18.345 | +0.1054505 | SI |

  Conclusion correcta: el objetivo queda exactamente en el borde de lo que un oraculo sobre la salida de
  V30 podria entregar (acertar ~16% de los errores con `p(y_true) >= 0.293` = conocimiento
  casi-oracular). "LB 0.70 sigue siendo inalcanzable en la practica; la prueba publicada no lo demuestra."
- A-01 (ALTA, defecto de marco analitico): la multiplicidad de las ~151 corridas no esta cuantificada.
  Con `macro_f1_std` mediano 0.0042 (76 filas que lo declaran), el SE de la media de 5 folds es ~0.00188:
  gate Delta >= +0.0015 -> z ~ 0.80 -> ~32 falsos positivos esperados; Delta >= +0.0030 -> z ~ 1.60 -> ~8;
  un Bonferroni de familia 151 exigiria Delta >= 0.0147. Lo que salva las conclusiones: las 4 adopciones
  reales (V19-B1, V20, V30, plug-in) tienen IC95 pareado excluyendo 0 y confirmacion externa; el unico
  control de multiplicidad previo era el Bonferroni interno de V37 (18 pruebas por clase).
- Otros defectos corregidos o registrados: fila `V37-DIAG-POS` escrita con 21 de 22 campos; SHA256 del
  pre-registro V37 mal transcrito (30d5d3cf -> 1d6d6573); holdout de V12 quemado (K-02, era puerta de
  seleccion); NER `N1b` con indices desalineados en ~55.8% de los documentos (N-01, sin impacto porque
  todos los brazos quedan bajo el gate); `es_core_news_lg` sin fijar en requirements (N-02, corregido);
  sin `torch.manual_seed` (A-06: V18-P3/P4/P5 no reproducibles); `max_iter=40` del MLP miembro del
  campeon (A-03, su JSON de reoptimizacion no existe); registro de envios obsoleto (K-03).
- Acciones de cierre: `AUD-ENV-LG` (pin del wheel `es_core_news_lg-3.8.0`, sha256 7c6c2127...),
  `AUD-IC95-TECHO` (el IC95 [0.635, 0.653] de D-087 se etiqueta como DECLARADO, no estimado) y
  `AUD-INNER-SPLIT` (-> D-091, ver B.4).
Fuentes: `DECISIONS.md:3434-3532` (D-090); `plan/auditoria/05_sintesis_y_cierre.md:13-56`;
`reports/auditoria_experimentos_v1_v37.md:19-21, 119-136, 168-183`; `experiments.csv:198-200`.
Re-ej.: la aritmetica de multiplicidad es barata (segundos, desde la tabla maestra); la refutacion K-01 es
barata con `artifacts/p33_scores/todos` (segundos); sin artefactos = documentable.

### B.4 D-091 (sesgo del inner split de Optuna)

| Nivel | solape de cluster en la validacion interior |
|---|---|
| outer agrupado (canonico) | 0.000 % |
| inner ingenuo (codigo real) | 3.907 % |
| inner agrupado (propuesto) | 0.000 % |

- Reconstruye las dos particiones interiores exactamente como las construye
  `v30_01_optuna_refinamiento.py:220-226`, sobre los 5 folds canonicos (169.336 filas, 164.018 clusters).
  Criterio declarado ANTES de correr: diferencia > 0.02 = material; medido 0.039071 (casi el doble);
  consistente en los 5 folds (3.787%-4.152%).
- Corrige a D-090: la frase "sin impacto medible (0.963% vs 1.030%)" estaba mal por (a) medir
  solapamiento de texto exacto cuando los clusters agrupan por Jaccard, y (b) comparar el inner contra
  cero teorico en vez del inner agrupado. Con la medicion correcta el leakage es del orden de 4%, no de 1%.
- Alcance: el leakage esta en la SELECCION de hiperparametros de 5 scripts del camino al campeon
  (v29_04, v29_06, v30_01, v30_02, v31_05); la estimacion externa uso el outer agrupado y es valida ->
  la cifra publicada del campeon NO cambia. Re-optimizar quedo pendiente de decision y no se ejecuto.
- Nota de numeracion: `DECISIONS.md` tiene DOS actas "D-091": esta medicion (linea 3534, 2026-10-01) y la
  adopcion de V38 (linea 3608, 2026-10-16). Al citar, usar linea y contexto.
Fuentes: `DECISIONS.md:3534-3606` (D-091 inner split); `experiments.csv:201` (AUD-INNER-SPLIT);
`plan/auditoria/06_accion_inner_split.md`; `scripts/auditoria_medir_inner_split_optuna.py`.
Re-ej.: 170 s si se adjuntan `dup_clusters.parquet` + folds canonicos; si no, documentable.

### B.5 Falsaciones V44-V52 (una linea por version)

| Version | Falsacion (mecanismo) | Delta | Folds | p | Fuente |
|---|---|---|---|---|---|
| V44 | Boundary-EMD L1 + ablaciones (POS 305, Multi-Sample Dropout, Mixup, Stacking logistico, XGBoost) | -0.0001720 vs V43 (POS -0.000776; Mixup -0.000500; Stacking -0.001288; XGBoost -0.000403) | 1/5 | 0.8550 (brazo principal) | `experiments.csv:208`; `DECISIONS.md:3696-3718` (D-096); `RUNLOG.md:2664-2679` |
| V45 | Bateria de 21 hipotesis (discretizacion 1D, compuertas, Nystroem, meta-arbitro, reoptimizacion anidada del plug-in, ...) | ninguna supera V43; peor: discretizacion -0.04167; mejor marginal: compuertas +0.000029 (3/5); reopt. plug-in -0.0003374 (0/5) | n/d | n/d (gate por CV) | `experiments.csv:209`; `DECISIONS.md:3720-3759` (D-097); `RUNLOG.md:2681-2701` |
| V46 | RankGauss (QuantileTransformer normal) | -0.0000327 vs V43 (standalone -0.00729 vs StandardScaler); simplex reoptimizado -0.0000942 | 2/5 | n/d (no reportado; gate por folds) | `experiments.csv:210`; `DECISIONS.md:3761-3772` (D-098); `RUNLOG.md:2703-2716` |
| V47 | Weighted-Cramer + recorte [-6, +6]; enlace acumulativo (falsacion #24) | +0.0000743 vs V43 (enlace acumulativo 0.5810757, -0.033) | 3/5 | 0.2620 (IC95 [-0.0001503, +0.0002838]) | `experiments.csv:211`; `DECISIONS.md:3774-3785` (D-099); `RUNLOG.md:2718-2732` |
| V48 | Geometric Product-of-Experts (ADOPTADO, no es fracaso) | +0.0002199 vs V43 (media bootstrap +0.0002151) | 4/5 | 0.0250 (IC95 [+0.0000007, +0.0004095]); LB 0.63724 | `experiments.csv:212`; `DECISIONS.md:3797-3818` (D-101); `RUNLOG.md:2734-2762` |
| V49 | CatBoost oblivious trees | -0.0000816 vs V48 | 2/5 | 0.7480 (IC95 [-0.0003022, +0.0001361]) | `experiments.csv:213`; `DECISIONS.md:3820-3835` (D-102); `RUNLOG.md:2763-2777` |
| V50 | TabTransformer MHSA; matriz W 6x6 desacoplada | MHSA fold 0 -0.0001058 (sustitucion -0.0005204); W +0.0000474 nominal | 4/5 (W) | 0.1940 (W; IC95 contiene 0) | `experiments.csv:214`; `DECISIONS.md:3837-3847` (D-103); `RUNLOG.md:2778-2787` |
| V51 | Consistent Smoothed Cramer-von Mises | -0.0000302 vs V48 (standalone -0.0000031) | 3/5 | 0.6900 (IC95 [-0.0001337, +0.0000721]) | `experiments.csv:215`; `DECISIONS.md:3849-3866` (D-104); `RUNLOG.md:2788-2803` |
| V52 | Power-Mean; Dual-Transport 7-model PoE; SNN/SELU; paradoja de Simpson | Dual-Transport -0.0000610 vs V48; SNN fold 0 -0.00913; regla por segmento global -0.0013763 | 3/5 (Dual) | 0.7040 (Dual; IC95 [-0.0002945, +0.0001803]) | `experiments.csv:216`; `DECISIONS.md:3868-3885` (D-105); `RUNLOG.md:2804-2817` |

Detalles de apoyo de V52 (`DECISIONS.md:3878-3884`): Power-Mean confirma a PoE (alpha -> 0) como optimo
global; alpha=-0.30 da 0.618721 nominal pero solo 3/5 folds; alpha=0.50 -0.000242 (0/5) y alpha=1.00
-0.000208 (0/5). La paradoja de Simpson: subir el F1 del segmento sin menciones (+0.000185) hunde el
global (-0.0013763) porque los FP/FN de ambos segmentos se suman en el denominador armonico.

## C. Reglas de uso para el constructor del notebook

1. Estas tablas se insertan como markdown; no re-ejecutar lo que dependa de artefactos del repo (el plan
   maestro exige notebook independiente que reconstruye desde los CSV crudos).
2. Re-ejecutables baratos de verdad: B.2 (regresion CV/LB con la tabla del informe) y la aritmetica de
   B.3 (SE, z y falsos positivos esperados desde la columna `macro_f1_std`). B.1 es de segundos SOLO si
   se adjuntan los 6 npy OOF; sin ellos, documentable.
3. Todo lo demas (A1-A9 y B.5) va como documentacion con la cifra y su fuente; no se re-entrena.
4. No citar "D-091" sin aclarar cual (colision de numeracion, B.4). No citar R2 0.9968 como cifra final
   (usar 0.998, B.2). Para el techo de 0.6935 citar siempre el caveat de oraculo (B.1).
