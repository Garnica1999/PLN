# PROJECT_STATE - KAGGLE_TOXIC_2026_PUJ

Ultima actualizacion: 2026-09-26 (fases **v6** a **v13** CERRADAS; v13 cierra la linea de
inferencia estructurada de etiquetas, D-100 a D-104). La seccion de detalle mas reciente que
habia aqui era la de v5; las de v6 a v12 se registran en DECISIONS.md, RUNLOG.md y en los
informes de fase de reports/.

## Fase v13 "Estimacion de HS e inferencia estructurada de etiquetas" (CERRADA el 2026-09-26)

Rama `proyecto-kaggle-toxic-2026-v13` (worktree `/home/carlos/Git/PLN-v13`, base v12 `ce774bc`).
**Dos experimentos, V13-00 y V13-01, mas el informe final V13-02.** Decisiones D-100 a D-104.

**Resultado: 0 candidatas adoptables.** El mejor delta es **+0.000534** frente a un gate de
+0.003. 0 submissions, 0 de 3 evaluaciones de holdout2, segunda particion no ejecutada, test no
utilizado, V03 congelada e intacta.

**Lo que la fase deja establecido, y reencuadra dos fases anteriores**:

1. **El techo de +0.046271 NO esta en la estimacion de HS.** El oraculo que lo produce no cambia ni
   una sola celda de la columna HS: pone TR y AG a cero en las 835 filas donde el HS verdadero es
   cero. Es un efecto aguas abajo de la condicion de HS, y ningun metodo que mejore HS puede
   alcanzarlo.
2. **Corregir la etiqueta de HS deja TR y AG con delta exactamente cero.** El meta produce una
   probabilidad para HS y no altera lo que el sistema decide sobre las otras dos etiquetas.
3. **TR y AG SI llevan informacion sobre los errores de HS, y no es utilizable.** Con el score propio
   de HS excluido, el meta alcanza 0.8433 de AUC frente a 0.8677 de V03 con el suyo, luego el score
   propio solo aporta +0.0244. Pero ese mismo brazo pierde 0.007541 en el macro-F1, porque el umbral
   esta congelado y un AUC alto no es una buena decision en el punto de corte.
4. **La invariante de TR=1 implica HS=1 es dura, con cero violaciones, y la de AG=1 con HS=0 no lo
   es, con 22 filas reales que la rompen.** El proyecto la tenia registrada como uniforme.
5. **La formulacion prohibida por el enunciado, p(TR|X) = p(HS|X) * p(TR|HS=1,X), es exactamente
   V6-B**, ya evaluada y cerrada. No se ha repetido.

**Decision de gobierno: la linea de inferencia estructurada de etiquetas queda CERRADA FORMALMENTE.**
No se afirma que TOXIC no pueda mejorar; se afirma que esa via no es rentable con los mecanismos
permitidos. La unica pregunta que sigue abierta, y que **no** es una hipotesis de esta linea, es
conseguir que la condicion de HS se estime mejor, porque de ella dependen los tres caminos y el
proyecto ya cerro las familias que podrian atacarla.

## Fase actual: v5 «Embedding from Scratch» (CERRADA el 2026-09-25)

Rama `proyecto-kaggle-toxic-2026-v5` (worktree `/home/carlos/Git/PLN-v5`, base
v4 `5d06ce7`; PR #28 de v4 sigue abierto y sin mergear). Spec transcrita en
`docs/embedding_from_scratch_v5_spec.md`; plan en `tasks/plan_v5.md`; checklist en
`tasks/todo_v5.md`; compliance en `reports/embedding_compliance.md`. Decision de
gobierno: **D-030**.

**Alcance autorizado**: corpus externos **no etiquetados** solo para aprender la
representacion distribuida desde cero. Prohibido: entrenamiento supervisado con
datos externos, generacion de labels, pseudo-labeling, ajuste de umbrales y
cualquier otro aprendizaje sobre HS/TR/AG. No existe evidencia documental
archivada de la autorizacion; se registra solo lo comunicado por el usuario.

| Campo | Valor |
|---|---|
| CURRENT_BEST_MODEL | **V03** (intacto; v5 no lo modifica) |
| CURRENT_BEST_CORRECTED_CV | **0.777211** |
| CURRENT_BEST_HOLDOUT | 0.8285 (holdout **agotado 5/5**: v5 no lo usa) |
| CURRENT_BEST_KAGGLE | 0.78306 (0 sondas nuevas en v5) |
| HIPOTESIS | H-EMB (el embedding desde cero aporta senal complementaria) / H0-EMB (no aporta) |
| IMPLEMENTACION | fastText C++ compilado localmente (commit `1142dc4c4ecb`, MIT); control Word2Vec con `minn=0,maxn=0`; PyTorch solo de respaldo; **gensim descartado** (sin wheel para 3.14) |
| FUENTES | Wikipedia ES (principal) -> mC4-es (web) -> mixto -> HPLT/CEREAL/OSCAR solo con margen real; OSCAR esta gated (no se fuerza) |
| ESCALA | 10M -> 50M -> 100M -> decidir -> 250M -> decidir -> 500M; cada expansion exige `RATIONALE_FOR_CORPUS_EXPANSION` |
| P0 | **GO**: fastText entrena (~930k tok/s con 16 hilos), `corrected_cv` de V03 reproduce 0.777211, 0 violaciones de recursos |
| P1 | cerrado: corpus 10M de Wikipedia ES (2.847 docs, 271.209 tipos), cobertura 0.6839 por tipo / 0.8833 por token, 0 fugas |
| P2 | cerrado: embedding solo 0.6603 (-0.1169 vs V03, IC95 negativo); subpalabras +0.0138; suelo aleatorio 0.5404 |
| P3 (complementariedad) | cerrado: `FUSION-HSTACK-L2` **0.7818** (+0.0046, IC95 [-0.0032, +0.0124]) = **CANDIDATE**; meta anidado 0.7678 (cerrado) |
| P4-P5 (corpus web) | mC4-es 10M + mixto 20M: cobertura por tipo 0.6839 -> **0.8413**; la fusion NO mejora (+0.0040 web, +0.0004 mixto) |
| CIERRE | **D-032**: la palanca de corpus (registro) agotada; fase cerrada **sin adoptar**. Auditoria final **GO** |
| P8 (curva 10M->50M) | 50M tokens: cobertura por tipo 0.7703, embedding solo **0.6687** (+0.0157), fusion **0.7803** (+0.0031) |
| D-033 | **No se escala a 100M/500M**: 5x tokens no mueven la aportacion complementaria. El techo no es la representacion |
| P9-P10 (tramo inferior 5M) | 5M tokens: la fusion **deja de aportar** (delta -0.0015) -> **10M es el suelo operativo**; curva del embedding plano |
| D-034 | Corrige a D-033: la ventaja de las subpalabras no es monotone (joroba con maximo en 10M) |
| v5.1 (arquitectura) | En curso. Baseline B0 = Wikipedia 10M / 200d / w5 / minC3 / sub3-6 / neg10 (ya entrenado, no se reentrena) |
| D-035 | Identidad de entrenamiento por firma con corpus procesado y bucket; presupuesto 9+1 abierto, 0 colisiones |
| P1 vector_size | CERRADA por inercia: 100d 0.7823 (+0.0050), 200d 0.7818, 300d 0.7805; dispersion 0.0017 |
| D-036 | Gate de dos niveles: adoptable exige delta >= +0.005 **y** IC95 que excluya el 0 |
| P2 window | CERRADA por inercia: ws3 0.7830 (+0.0057), ws5 0.7818, ws7 0.7806; mejor embedding aislado de v5 (0.6694) |
| D-037 | La ventana no es la palanca y el orden es 3 > 5 > 7: menos contexto rinde mas con 10M tokens |
| P3 min_count | CERRADA por inercia: mc2 0.7771, mc3 0.7818 (el mejor), mc5 0.7792; 6.059/6.059 tipos con vector en los tres |
| D-038 | El mecanismo de min_count ya esta ocupado por las subpalabras: con 3-6, todo tipo consultado tiene vector |
| P4 subwords | CERRADA: 3-5 0.7791, 3-6 0.7818 (el optimo), 3-7 0.7767; orden monotono |
| D-039 | 3-6 ya es el optimo; regla de dispersion revisada cuando el baseline es el maximo de la familia |
| P5 negative | CERRADA por inercia: neg5 0.7811 (+0.0039), neg10 0.7818; dispersion 0.0006 |
| D-040 | Presupuesto de v5.1 agotado: 9/9 exploratorias, reserva 0/1 intacta. La arquitectura cambia el embedding pero no aporta a V03 |
| P6 comparacion final | 14 configuraciones, 0 adoptables | embedding aislado rango 0.0297 vs fusion 0.0063 |
| D-041 | Conclusion v5.1: la arquitectura mejora la representacion pero no aporta a V03. No se escala el corpus |
| P7 complementariedad | WS-3 arregla 8 filas y rompe 2 (el baseline 8 y 5): la ventaja esta en el damage, no en el salvage |
| D-042 | El embedding aporta la misma informacion con mas ruido; correccion de vocabulario: la unidad son filas, no celdas |
| P8-P11 cierre | Notebook 63 celdas validado GO | FINAL_REPORT seccion 28 | auditoria final 11/11 GO |
| D-043 | **Cierre v5.1**: V03 se mantiene, embedding CANDIDATE no adoptado, auditoria GO, rama sin push |
| v6 (mecanismos) | Abierta. Gap analysis hecho: 11 lineas ejecutables, 3 cerradas por duplicacion |
| V6 P0 | C2, C3, C4 cerradas (V4-D, E2-10). C se reduce a C1. J no es duplicado de PRE |
| V6 P0-bis | Interseccion de solo 9-12 filas (techo +0.0027 < gate): **D y E se cierran por presupuesto** |
| D-044 | Gate de dos niveles (delta + IC95 + folds, luego segunda particion); TR implica HS es determinista |
| V6 Track B | **Cerrado negativo**: B1 -0.0107, B2 -0.0008; 0/5 pasan el Gate 1 |
| D-045 | El control B0 es obligatorio; la invariante es cierta pero solo 1.35 % de las predicciones la violan |
| V6 Track A | **Cerrado negativo**: A1 +0.0006, A2 +0.0002, A3 -0.0149; 0/4 pasan el Gate 1 |
| D-046 | Solo el 7.1 % de filas es repetible y penalizarlo empeora: la repeticion es señal |
| V6 Track I | **Cerrado negativo**: optimo interior en 1.5 pero +0.0016, mitad del gate |
| D-047 | El peso heredado de hstack no era optimo, pero la escala no era el cuello de botella |
| V6 Track C1 | **Cerrado negativo y neutro**: -0.0001 con IC95 que incluye el 0; **C queda cerrada entera** |
| D-048 | El control de C1 detecto un bug de tokenizacion que hacia los deltas inatribuibles |
| V6 Track H | **Cerrado por protocolo**: -0.0105 con diversidad real (corr. 0.957); los umbrales de V03 no son validos para un promedio |
| D-049 | La hipotesis de techo del track se refuto a si misma midiendo; el cierre es de protocolo, no de mecanismo |
| V6 Track F | **Cerrado negativo**: +0.0001 con las cuatro senales; la deteccion encuentra dificultad, no ruido |
| D-050 | V03 acierta 10.5 % en las filas sospechosas frente a 28.3 %: marcar es estar donde hay que estar |
| V6 Track J | **Cerrado negativo**: -0.0061 y -0.0063; doblar copias no empeora mas (es sesgo, no cantidad) |
| D-051 | **v6 sin lineas ejecutables**: B, A, C, I, H, F, J negativas; D, E por techo; G no ejecutar |
| V6 Track J | **Cerrado negativo**: -0.0061 y -0.0063; doblar copias no empeora mas (es sesgo, no cantidad) |
| V6 entrega | 25 variantes, 7 controles, **0 adopciones**; 7 de 24 experimentos gastados |
| D-052 | **Fase v6 CERRADA**: no queda ninguna linea ejecutable de la especificacion |
| variable abierta | El tamano de los datos etiquetados (1.483), unica palanca que deja la evidencia |
| GATE DE CIERRE | §29: A (+0.005 con estabilidad) / B (complementariedad robusta) / C (senal representacional fuerte) |

## Fase anterior: v4 «Advanced Research v4» (CERRADA el 2026-09-25)

Rama `proyecto-kaggle-toxic-2026-v4`. Spec archivada en
`docs/advanced_research_v4_spec.md`; plan en `tasks/plan_v4.md`; backlog
pre-registrado en `reports/research_backlog_v4.md`; informe de cierre en
`reports/advanced_research_v4.md`; matriz de cumplimiento en
`reports/spec_compliance_v4.md`. Decisiones D-025 a D-029.

**Resultado: la fase se cierra con la condicion B de la spec (§102): hipotesis
nuevas investigadas y cerradas con evidencia, sin mejora transferible.** El
unico candidato que supero los tres gates de D-016 (C, `meta36_downweight`,
corrected_cv 0.7882 = +0.0110, 4/5 pliegues, IC95 [+0.0054, +0.0200]) **fue
rechazado por el holdout congelado: 0.8163 frente a 0.8285 de V03 (delta
-0.0122)**. Su submission (v12) queda como artefacto reproducible marcado
`DESCARTADO`; no se envio a Kaggle.

| Campo (§104) | Valor |
|---|---|
| CURRENT_BEST_MODEL | **V03** (sin cambios; verificada byte a byte en P0 y en la auditoria final) |
| CURRENT_BEST_CORRECTED_CV | **0.777211** (protocolo v2, desarrollo n=1.483) |
| CURRENT_BEST_HOLDOUT | 0.8285 (V03, P7-01). **Holdout agotado: 5/5 evaluaciones usadas** |
| CURRENT_BEST_KAGGLE | **0.78306** (v03, mejor de 12 sondas; 0 sondas nuevas en v4) |
| NEW_SIGNALS | ninguna adoptada. Señales nuevas caracterizadas y **cerradas**: rareza del ruido de etiqueta (2.8 % de contradicciones, no explica el error); dureza util = distancia a la frontera (la BCE agregada no preserva el orden); reweighting con signo negativo; verosimilitud class-conditional aporta +0.0041 en fusion; SVC RBF sobre SVD 300 = 0.7904 con IC fragil |
| CLOSED_TRACKS | A, B, C, D, E, F y H (TRACK G queda cerrado salvo el aporte documentado de la verosimilitud) |
| OPEN_TRACKS | ninguno |
| SUBMISSION_CANDIDATES | ninguno. `toxic_submission_v12_descartado_meta36_downweight.csv` conservado como artefacto, marcado DESCARTADO |
| NEXT_STEP | integrar la seccion v4 en `FINAL_REPORT.md` y en el notebook, y abrir el PR de la rama v4 a `main` |

**Lectura metodologica que cierra la fase**: con 1.731 ejemplos de train, 1.154
de test y sin modelos preentrenados, el corrected_cv de este protocolo **no es
un selector fiable**: es el cuarto caso consecutivo de mejora local que no
transfiere (Optuna, meta v2, candidatos v09-v11 y ahora C). El error estandar
del test (~0.013) es del mismo orden que las mejoras que el protocolo selecciona.

## Fase anterior: v2 cerrada + fase final de candidatos

**FASE v2 (Advanced Research) CONCLUIDA el 2026-09-25 sin mejora transferible.
BACKLOG v2 AL 100 % (24/24 lineas evaluadas y cerradas, D-021).**
Modelo final: **V03** (LB **0.78306**), congelado en
`experiments/baseline_v03_frozen/` y reconstruible byte a byte.

Resumen v2 (detalle en `reports/advanced_research_v2.md`): se evaluaron 24 lineas
de investigacion con un protocolo corregido (grupos near-duplicado UNION
pseudo-autor, holdout congelado de 248 tweets, bootstrap por tweet). El meta de
dependencia de etiquetas mejoro en desarrollo (+0.0081) y en el holdout (+0.0180
pareado) pero **no transfirio**: LB v08 = 0.78163 vs 0.78306 de v03 (dentro del
error estandar ~0.013). El paquete P8 cerro las cinco lineas restantes
(calibracion, encoder de entidades por sparsity 0.5 %, umbrales por rebanada,
especialistas y espacios por etiqueta) sin mejora. Lectura: con 12 sondas,
0.78306 esta sesgado al alza por seleccion (winner's curse); el nivel esperado de
la familia es ~0.776-0.780.

Entregables (actualizados y verificados el 2026-09-25):
- `notebooks/10_final_pipeline.ipynb`: 42 celdas, 2.4 s, 0 errores, **26 checks
  [OK] / 0 fallas**, idempotente; 16 secciones con fase 1 + fase v2.
- `FINAL_REPORT.md`: 26 secciones numeradas + resumen + referencias; seccion 26
  dedicada a la fase v2 (incluye el cierre del backlog).
- `reports/advanced_research_v2.md`: informe de cierre de la fase v2.
- `docs/advanced_research_v2_spec.md` + `reports/spec_compliance_v2.md`: spec
  maestra archivada y matriz de cumplimiento FASE 0-99 con evidencia, mas
  `reports/rule_audit_v2.md` (0 violaciones) y
  `reports/target_encoding_audit_v2.md` (sin leakage de aprendizaje).

## Fase final de candidatos y submissions (2026-09-25)

- CURRENT BEST LOCAL (corrected_cv): **0.7874** (meta de dependencia,
  META-BASES-RIDGE; reproducido en `experiments/23_candidatos_finales.py`).
- CURRENT BEST CORRECTED CV nuevo: **C-EMO 0.7809** (+0.0037 vs V03, 3/5 pliegues).
- CURRENT BEST HOLDOUT: **0.8465** (meta v08, P7-02, no transferido al LB).
  Entre los candidatos nuevos: C-EMO 0.8364 (IC95 del delta [-0.0075, +0.0228]).
- CURRENT BEST SUBMISSION: **V03** (`submissions/toxic_submission_v03_lr_union_thr.csv`,
  LB **0.78306**, hash congelado `7d7478f2...`).
- CANDIDATE FILES: `submissions/toxic_submission_v09_candidate_a.csv` (C-EMO),
  `v10_candidate_b.csv` (C-SYN), `v11_candidate_c.csv` (C-EMO-SYN); manifest en
  `submissions/submission_manifest.csv`; validacion `reports/submission_validation_v2.md`;
  hashes `reports/submission_hashes_v2.json` (reproducibilidad verificada).
- **LB DE LOS CANDIDATOS (2026-09-25)**: v09 **0.77895** | v10 **0.77978** | v11
  **0.78150**; los tres por debajo de V03 (el mejor, v11, a -0.00156, dentro del
  error estandar ~0.013); el ranking local no predijo el del LB (D-024).
- DECISION (spec 39, cerrada): **ningun candidato supero a V03; V03 sigue siendo
  la referencia y la submission final** (12 sondas registradas; D-024).
- NEXT ACTION: (1) push de la rama `proyecto-kaggle-toxic-2026-v2`; (2) entrega
  del curso (deadline 2026-10-19); (3) opcional: presentacion.

## Entregables finales

| Entregable | Ruta | Estado |
|---|---|---|
| Notebook entregable (30 % de la nota) | `notebooks/10_final_pipeline.ipynb` | ejecutado, 0 errores |
| Informe final | `FINAL_REPORT.md` (27 secciones, 42 citas) | completo |
| Auditoria final (QA) | `reports/auditoria_final.md` | GO con condiciones cumplidas |
| Auditoria de rendimiento | informe de `ml-performance-auditor` | P1-P3 documentadas |
| Modelo final | `models/mod03b_lr_union.joblib` | reconstruye v03 exacto |
| Submission de respaldo | `submissions/toxic_submission_v03_lr_union_thr.csv` | LB 0.78306 |
| Memoria | `PROJECT_STATE.md`, `RUNLOG.md`, `DECISIONS.md` (D-001..D-023), `experiments.csv` (128 filas) | al dia |
| Evidencia | 26 scripts, 32 figuras, 67 tablas CSV, 12 submissions | en disco |

## Fases 6-8 (features densas) - resultado negativo documentado

B5-DENSE 0.6607 (46 features); hibridos sobre la union: HIB-H12 0.7836
(-0.0091), HIB-H11 0.7713, HIB-H14 0.7630, HIB-UNION 0.7610 (-0.0316).
Control con las densas a cero reproduce 0.792659 exacto -> la perdida es de las
features, no del hstack. H11/H12/H14 no se validan como palanca; quedan como
evidencia negativa para el informe. Rebanadas: solo mejoran referencia colectiva
y Q4 con HIB-H12. Artefactos: `reports/features_densas_resultados.csv/.json`,
`reports/eda/features_densas_slices.csv`, filas B5-DENSE/HIB-* en
`experiments.csv`.

## Sondas de robustez pendientes de envio (D-013, max 2)

- `submissions/toxic_submission_v06_robustez_c1.csv` (C=1.0, min_df=2):
  OOF 0.7929, gap +0.1635 -> candidato debil.
- `submissions/toxic_submission_v07_robustez_mindf5.csv` (C=1.0, min_df=5,
  10.515 features): OOF 0.7993, gap +0.1272 (la mitad que MOD-03b) -> candidato
  fuerte si la hipotesis de capacidad es correcta.
Regla: se adopta la que supere 0.78306 (v03); si ninguna, se mantiene MOD-03b.

`submissions/toxic_submission_v04_ens04_mediana.csv`: par M2+M3 con umbrales
medianos (0.44/0.47/0.44), F1 anidado 0.7985. Predice mas positivos que la
prevalencia de train (HS 46.5 % vs 43.7 %), asi que si el LB decepciona el
sospechoso es el umbral, no la representacion.

**RESULTADO (2026-09-23)**: LB 0.77510 -> el ensemble NO se traslada (peor que
v02 y v03). El sospechoso era el umbral y se confirmo. Descartada.

## Ensembles (Fase 13b) - resultado

4 bases OOF con el mismo split: M1 word LR 0.7587, M2 union LR 0.7927,
M3 union LinearSVC 0.7849, M4 char LR 0.7567. Diversidad real (desacuerdo
4-11 %, correlacion de errores 0.63-0.84) pero ningun ensemble supera al single
de forma convincente: ENS-02 0.7949 anidado (+0.0023), ENS-04 con umbrales
medianos 0.7985 (+0.0058, maximo de 12 estimaciones). Decision: NO se gasta
envio; se mantienen v02/v03.

## Analisis de error (Fase 12) - hallazgos accionables

Rebanadas mas debiles (F1, n>=40): hashtags 0.586 (-0.213 vs global),
preguntas 0.633, referencia colectiva 0.639, textos largos Q4 0.645,
sin mencion 0.697.
Patrones dominantes: (1) insulto interpersonal en registro de broma/afiliacion
(40.5 % de los textos con insulto + mencion + sin colectivo son (0,0,0));
(2) odio implicito contra grupos sin insulto del lexico (45.5 % de los FN de AG
tiene referencia colectiva vs 13.9 % de los TP); (3) regimen noticia/atribucion
a terceros con URL.
Hipotesis nuevas: H11 registro bromista/afectivo, H12 bloque de objetivo
individual vs colectivo para TR, H13 features de cita/atribucion, H14 lexico de
deshumanizacion inducido, H15 calibracion por rebanada (se espera sin aporte).
Detalle en `reports/error_analysis.md`.

## Submissions: estado

- v00 (sonda E00): LB 0.51494. v01 (B1): 0.75677. v02 (MOD-03b, 0.5): 0.77993.
  **v03 (MOD-03b + umbrales): 0.78306 = MEJOR ACTUAL.**
- v04 (ensemble): 0.77510 descartada. v05 (OPT-01 de Optuna): 0.77602 descartada.
- v06/v07 (sondas de robustez, D-013): pendientes de envio por Carlos.

## Hallazgos del EDA (Fase 2, resumen)

- 0 emojis Unicode en el corpus (normalizacion ASCII); 6 bytes no ASCII en test.
- Senales mas fuertes (V de Cramer, train): `tiene_insulto` (HS/AG 0.355,
  TR 0.339), `tiene_mencion_inicial` (TR 0.325), `tiene_insulto_dirigido`
  (TR 0.266), `tiene_referencia_colectiva` (TR 0.215, direccion negativa),
  `tiene_hashtag` (asociado a HS=0).
- Dentro de HS=1 (n=756): TR=1 tiene mencion inicial 64.9 % vs 9.7 % (TR=0) y
  segunda persona/insulto dirigido altos; TR=0 concentra referencias
  colectivas explicitas (57.0 % vs 10.0 %) y URLs.
- Desbalanceo maximo 2.34:1: no requiere remuestreo agresivo.
- Deduplicar por near-duplicado elimina solo 4 filas de train y no cambia el
  orden ni la magnitud de los efectos.
- Hipotesis con apoyo inicial: H1 (indirecta), H2 (parcial), H3 (con reservas),
  H6, H8 y H9. Sin evaluar: H4, H5, H7, H10.

## Revision de codigo (Fase C)

- `ml-reviewer`: APROBADO CON CAMBIOS MENORES (0 criticos, 1 mayor, 5 menores).
- HALLAZGO MAYOR a implementar en modelado: los lexicos inducidos con
  etiquetas de train (`n_insultos`, `tiene_insulto`, `tiene_colectivo_lexico`)
  estan sesgados in-sample; en los modelos deben inducirse DENTRO de cada fold
  de `StratifiedGroupKFold` y reportar el efecto out-of-fold.
- Cambios menores aplicados y re-verificados (ruff + idempotencia).

## Decisiones pendientes de Carlos

1. **D-005 (semantica de TR)**: revisar
   `reports/eda/tr_ejemplos_para_revision.csv` (40 TR=1 + 40 TR=0 anotados) y
   `reports/eda/fase2_tr0_con_referencia_colectiva.csv` (casos limite) para
   fijar si TR = rango del objetivo (individuo vs grupo) o dirigido vs no
   dirigido.
2. **E00 (metrica)**: enviar `submissions/toxic_submission_v00_plantilla.csv`
   y reportar el score publico para identificar la implementacion de F1.

## Estado del arte (Fase 1, resumen)

- 45 referencias verificadas (`reports/estado_arte.md`, `.csv`, `references.bib`).
- Hallazgo central: MineriaUNAM (Argota Vega et al. 2019, DOI 10.18653/v1/S19-2079)
  gano la Subtask A en espanol de HatEval SIN transformers ni embeddings
  preentrenados: SVM lineal + char n-gramas, n-gramas de palabras funcionales
  2-4, n-gramas de puntuacion 2-4, POS, patrones morfologicos y series de
  palabras agresivas (macro-F1 0.73 en espanol).
- Evidencia util: Nina-Alcocer 2019 (TFIDF+EMOTIONS EMR 0.711 vs 0.629 base;
  TFIDF+STRUCT mejor F1 0.780); contraevidencia: combinar 3 bloques de features
  degrada y el ensemble por votacion no supera al SVM lineal (SINAI 0.707 vs 0.701).
- Hipotesis H1-H10 y experimentos E01-E10 en `reports/hipotesis.md`.
- Regla de decision pre-registrada: una mejora se adopta si es >= +0.005 absoluto
  en la metrica primaria y en >= 3 de 5 folds.

## Competicion

- Kaggle `tweet-toxic-2026-puj` (id 167384). Organizacion JarvisLab_PUJ.
- Metrica oficial: F1 Score. Deadline: 2026-10-19 23:00 UTC. Max 10 envios/dia.
- Nota: 70 % ranking Kaggle (LB privado) + 30 % calidad del notebook.
- Restricciones: sin transformers/LLM; sin embeddings preentrenados; todo
  vectorizador/embedding entrenado desde cero; test nunca para entrenar.

## Datos (verificados)

- `data/raw/toxic_train.csv` (205.417 B, sha256 3d40e734...): 1.731 filas,
  sep `;`, UTF-8. HS 756 (43.67 %), TR 519 (29.98 %), AG 662 (38.24 %).
- `data/raw/toxic_test.csv` (127.006 B, sha256 e91ff825...): 1.154 filas,
  sep `;`, **cp1252** (6 bytes 0xBF). Solo `id,text`.
- `data/raw/submission.csv` (13.860 B): plantilla `id,HS,TR,AG` separada por
  coma, todo 1,1,1. Ids identicos al test.
- Combinaciones HS x TR x AG: (0,0,0)=949, (1,1,1)=491, (1,0,1)=145,
  (1,0,0)=92, (1,1,0)=28, (0,0,1)=26. **No existe (0,1,\*)**.
- Texto ASCII sin tildes/enye (normalizacion de los organizadores).
- Longitud: 6-306 chars, mediana 91.

## Hallazgos de la auditoria (Fase 0)

- Parser correcto: quoting estandar CSV (el quoting QUOTE_NONE rompe 13 filas).
  1 fila con comilla anomala resuelta por el parser estandar.
- **10 textos de test estan en grupos de near-duplicados con train**
  (umbral 0.85; 4 grupos; 7 filas de train y 10 de test afectadas).
  Guardado en `data/processed/grupos_near_duplicados.csv`.
- 5 duplicados exactos dentro de test; 0 entre train y test.
- `id` no asociado a etiquetas (Spearman p>0.49; chi2 p>0.20): sin fuga por orden.
- 17 pseudo-autores (primera mencion @usuario) con >=3 tweets, 77 tweets
  (4.4 %); 9 grupos homogeneos en HS.
- Riesgo externo: el corpus sigue el esquema HatEval (SemEval-2019 T5 ES);
  prohibido usar etiquetas/textos externos (D-002).

## Leaderboard (registro en `submissions/leaderboard_log.csv`)

| Envio | Local (OOF) | Kaggle | Gap | Que es |
|---|---|---|---|---|
| v00 plantilla | - | 0.51494 | - | sonda E00 (todo 1,1,1) |
| v01 baseline TF-IDF+LR | 0.75778 | 0.75677 | -0.0010 | B1 word(1,2) |
| v02 union word+char LR | 0.79270 | 0.77993 | -0.0128 | MOD-03b, umbral 0.5 |
| **v03 union + umbrales** | 0.79890 | **0.78306** | -0.0158 | **MEJOR / modelo final** |
| v04 ensemble M2+M3 | 0.79850 | 0.77510 | -0.0234 | descartada |
| v05 OPT-01 Optuna | 0.80390 | 0.77602 | -0.0279 | descartada |
| v06 robustez C=1.0 | 0.79290 | 0.77227 | -0.0206 | descartada |
| v07 robustez min_df=5 | 0.79930 | 0.77034 | -0.0290 | descartada |

Conclusiones: metrica = F1 macro sobre 3 columnas (D-009); los umbrales por
etiqueta aportan +0.0031 pareado (D-012); **plateau de la familia union: el
rango 0.77034-0.78306 equivale a 0.99 errores estandar (n=1154) -> las
diferencias entre variantes son ruido** (D-014). La cota local es optimista entre
-0.013 y -0.029; el gap train-OOF NO predice el gap local-LB (D-013 refutada).

## Mejor modelo

**MODELO FINAL (D-014): MOD-03b_BEST - union TF-IDF word(1,2) min_df=2
sublinear + char_wb(3,5) min_df=2 sublinear + LogisticRegression C=2.0 balanced
x3, con umbrales por etiqueta 0.44/0.51/0.46.**
Script `experiments/04_refit_submissions_v02_v03.py`; modelo
`models/mod03b_lr_union.joblib`; submission
`submissions/toxic_submission_v03_lr_union_thr.csv` (LB **0.78306**, el mejor).
OOF 0.7927 (0.5) / 0.7989 (umbrales). Dimension 27.431.

**Plateau documentado (D-014)**: 7 submissions utiles; toda la familia union cae
en la banda 0.77034-0.78306 del LB, un rango de 0.0127 = **0.99 errores
estandar** (n=1154). Las diferencias entre variantes son ruido; el unico
contraste pareado fiable (v02 vs v03, mismo modelo) confirma +0.0031 por los
umbrales OOF. Se detiene el sondeo del leaderboard.

Alternativas evaluadas y descartadas con evidencia: ensembles (v04 0.77510),
Optuna OPT-01 (v05 0.77602), control de capacidad (v06 0.77227, v07 0.77034),
post-proceso jerarquico (-0.0056 local), features densas H11/H12/H14
(-0.009 a -0.032 local), todas las variantes de preprocesamiento (parte A).

## Mejores features

Pendiente. Hipotesis de partida por validar en Fases 2-8:
n-gramas de caracter (evasion ortografica), lexico de agresividad,
2a persona/imperativos (targeting), estilometria, sentimiento/emocion.

## Problemas conocidos / riesgos

1. Metrica exacta ambigua -> sonda E00 con `submission.csv` (todo 1) que envia
   Carlos; los valores esperados difieren por definicion (0.541 vs 0.543 vs
   0.270 vs 0.284 con distribucion de train).
2. Semantica de TR ambigua entre la definicion oficial y HatEval -> Fase 2.
3. Near-duplicados train-test -> validacion con `StratifiedGroupKFold`
   (grupos = near-duplicados, estratificado por combinacion).
4. Corpus sin diacriticos -> no se deben "inventar" tildes; documentar.
5. Fuga potencial por pseudo-autor -> control de robustez adicional.

## Hipotesis abiertas (a refinar en Fase 1)

- H1: char n-grams 3-6 capturan evasion ortografica mejor que word n-grams.
- H2: lexico agresivo + 2a persona separa AG.
- H3: features de targeting separan TR.
- H4: word+char > cada representacion sola.
- H5: sentimiento/emocion aporta informacion complementaria.
- H6: estilometria ayuda en tweets cortos.
- H7: lematizacion no aporta en tweets.
- H8: post-proceso jerarquico HS=0 => TR=0, AG=0 mejora F1.

## Proximos experimentos

- E00: identificacion de la metrica oficial (sonda de submission).
- E01: char n-grams vs word n-grams (Fase 5/9).
- Fases 2-4: EDA, textometria, variantes de preprocesamiento.

## Rutas clave

- Datos: `data/raw/`; grupos near-dup: `data/processed/`.
- Reportes: `reports/` (auditoria, data dictionary, eda/).
- Memoria: `PROJECT_STATE.md`, `RUNLOG.md`, `DECISIONS.md`, `experiments.csv`.
- Reglas verbatim: `docs/competition_rules.md`.
- Codigo: `src/data.py`; auditoria: `experiments/00_auditoria_datos.py`.

## Entorno

`/home/carlos/venv` (Python 3.14.7) - sklearn 1.8, lightgbm 4.6, xgboost 3.2,
catboost 1.2.10, nltk 3.9.4, spacy 3.8.13 + es_core_news_md, torch 2.14+rocm7.2,
kaggle 2.2.4, emoji 2.16.0, tabulate. GPU RX 9070 XT (ROCm) con protocolo de
chequeo previo (D-008).

| v7 (Advanced Research) | **Cerrada sin adopciones**: P9, P8 y P0.1 negativos con causa medida; 8 tracks cerrados por duplicacion en la auditoria |
| V7 auditoria | 11 tracks candidatos auditados: **8 cerrados con cifra**, 2 nuevos ejecutados, 1 aplazado por no medible en local |
| D-056 | **V03 mantiene sus umbrales por evidencia medida**: desviacion de 0.04 a 0.065, recalibrarlos empeora |

## Fase v8 (Transfer Gap + Supervised Sparse Reweighting) - en curso

- **Aislamiento**: worktree `/home/carlos/Git/PLN-v8`, rama
  `proyecto-kaggle-toxic-2026-v8`, desde el cierre de v7 (`2f8fac8`). `data/` enlazado,
  `tools/` copiado. v7 intacta.
- **Prohibicion de modelos preentrenados ratificada** (D-057), aunque las reglas de la
  competencia los permitan. Decision consciente, no olvido.
- **Freno de transferencia heredado**: por encima de +0.005 local, un candidato solo llega a
  finales si no aumenta dimension ni brecha entre ajuste y validacion.
- **P0 cerrado** (`v8_00_gap_audit.py`): de diez pistas, cinco sin probar y cinco con
  precedente; del material de clase, 22 tecnicas, **9 nuevas** y 13 cerradas. Tres falsos
  positivos propios corregidos. P2 llega con precedente adverso (es el estimador de
  Mosteller y Wallace, ya medido en E2-15). P3 tiene respaldo doble.
- **P1 ejecutado, ficha, en espera de sonda** (`v8_01_ensamble_folds.py`): identidad de
  pipeline verificada celda a celda frente al OOF canonico (**0.000e+00**). El cambio sobre la
  prediccion de V03 es de **100 celdas de 3462 (2.9 %)**. Fichero
  `submissions/toxic_submission_v08_p1_fold_ensemble.csv` escrito y verificado con nueve
  comprobaciones. **No enviado. Parada obligatoria** hasta conocer el numero (D-058).
- **Techo esperado de P1, declarado antes del envio**: orden de milesimas, no de centesimas.

## Fase v8: estado tras la sonda P1

- **P1 CERRADO, negativo** (D-059). Externo 0.77470 frente a 0.78306 de V03, diferencia
  -0.00836. Sin variantes: ni umbral, ni pesos, ni semilla.
- **Lectura correcta del numero**: el ruido del leaderboard es 0.01524 para la F1, que equivale
  a **19.2 aciertos netos**. La diferencia de P1 son -10.5 netos, un doceavo del ruido. **No hay
  evidencia de que el ensemble sea peor, hay evidencia de que no es mejor.** V03 sigue como
  referencia por no tener evidencia en contra, no por tener evidencia a favor.
- **Se cae una premisa propia** (D-059): el paso de v02 a v03, citado como el unico
  movimiento positivo y como justificacion del freno de transferencia, son +3.9 netos frente a
  un ruido de 19.2. Fue ruido. El freno sigue valiendo por otra razon.
- **Filtro de detectabilidad adoptado**: una candidata que cambie C celdas solo es resoluble
  por el externo si su tasa de acierto entre ellas supera `0,5 + 19,2/C`. Con 100 celdas hace
  falta 0.60; con 200, 0.55; con 800, 0.51. **Ninguna candidata por debajo del minimo se
  envia**, y el minimo se estima en el desarrollo antes de gastar la sonda.
- **De once sondas reales, diez son indistinguibles entre si y ninguna supera a V03.** Ni la
  capa de decision, ni los hiperparametros, ni la representacion producen una diferencia que
  este leaderboard pueda resolver.
- **Reordenacion**: P5, P6 y P7 bajan de prioridad. **P3, la estabilidad de features, sube a la
  primera posicion** y es la siguiente linea a ejecutar.

### Tras P3 (V8-03)

- **P3 CERRADO, negativo** (D-060). Tres umbrales de estabilidad, los tres monotonos a peor:
  -0,003308, -0,004628 y -0,018890, este ultimo con el IC95 excluyendo cero.
- **El filtro de detectabilidad fallo en los tres** (0,4402, 0,4436 y 0,3930 frente a
  minimos de 0,55, 0,54 y 0,53). **No se genero submission y no se gasto sonda.** El filtro
  de V8-02 cumple su funcion en su primer uso.
- **Como pierde es el hallazgo**: acierta del 39 % al 44 % en las celdas que toca, por debajo
  del azar. Conserva rasgos de magnitud grande y repetida, que son mas probablemente
  frecuentes y neutros que discriminantes, y con ello pierde la senal minoritaria.
- **Conclusion que sobrevive**: la estabilidad de features es el mecanismo correcto para
  atribuir autoria y el equivocado para clasificacion de tres etiquetas. El criterio mide forma
  del texto, no pertenencia a una clase.
- **Siguiente linea: P2**, la ponderacion por razon logaritmica de recuentos, que conserva su
  precedente adverso de E2-15 (-0.016 a -0.039).

### Tras P2 (V8-04)

- **P2 CERRADO, negativo** (D-061). Con suavizado 1e-6 el ratio sobre la TF-IDF da -0,058496;
  con 1e-3 da +0,004745. El control de conteos con constante 2.0 da -0,075027.
- **Precedente adverso confirmado.** Ninguna variante es admisible; el filtro de detectabilidad
  falla en las tres (0,3266, 0,5281 frente a 0,5317, y 0,3217). Sin submission, sin sonda.
- **Prema propia corregida antes de ejecutar**: la auditoria V8-00 decia que el ratio por
  etiqueta no se habia probado, y era falso. E2-15 ya lo calculaba por etiqueta. Lo no probado
  era el ratio sobre la TF-IDF de V03 con la constante 2.0.
- **El suavizado sostiene el metodo entero**: salto de +0,063241 por un factor de mil.
- **No se barre el suavizado**: que la linea pase de -0,058 a +0,005 con un parametro sin
  caracterizar significa que el resultado positivo es un punto de una superficie sin
  explorar, no una senal.
- **Dato colateral del control**: la constante y el ratio no se pueden subir a la vez
  (-0,075 con 2.0 frente a -0,016 con 1.0).
- **Estado de la ordenacion**: P1 cerrada, P3 cerrada, P2 cerrada. Quedan P4, P5, P6, P7 y las
  lineas de reserva, **todas con la prioridad baja que les asigna el filtro de
  detectabilidad**. P4 es la unica con mecanismo propio, y su techo lo acotan tres
  predecesores fallidos.

### Tras P4 (V8-05)

- **P4 CERRADO, negativo** (D-062). Peso conjunto 0,5 da -0,001471 y peso 2,0 da -0,006781.
  Sin submission y sin sonda.
- **El mecanismo funciona y empeora**: el coseno entre las tres cabeceras sube de 0,7380 a
  0,8265 a 0,9216. Compartir estructura entre las tres etiquetas es posible, y resta.
- **Identidad confirmada por el metodo correcto**: el objetivo propio (2,928125) es menor que
  el de la biblioteca (3,359072) y la superficie es estrictamente convexa, luego la solucion
  propia es la optima. El criterio de identidad paso del coseno al valor del objetivo, porque
  exigir el coseno pide que las dos implementaciones convergieran igual de bien.
- **El patron que unifica P1, P3 y P4**: la tasa de acierto entre las celdas modificadas es
  de 0,46, de 0,39 a 0,44 y de 0,42 a 0,39, respectivamente. **Tres mecanismos distintos y el
  mismo sintoma: en las filas que tocan, todas mueven la decision hacia peor.** El limite no es
  de mecanismo sino de representacion.
- **Cinco errores de escala detectados antes de obtener cifra**, dos de ellos por comparar la
  norma de los pesos con la biblioteca, comparacion que se anade al repertorio porque la
  metrica sola no detecta un error de escala.
- **Estado de la ordenacion**: P1, P3, P2 y P4 cerradas. Quedan **P5, P6 y P7**, las tres
  solo de capa de decision, con la prioridad baja del filtro de detectabilidad y **sin
  razon para esperar que alguna pase** a la luz del patron.

### Tras la fase v10 completa (V10-00 a V10-10)

- **V03 SIGUE SIENDO LA SUBMISSION RECOMENDADA**, con 0.7772114525165721 local y 0.78306 en el
  externo, umbrales 0.44, 0.51 y 0.46, y `corrected_cv`. No se ha tocado ninguno de sus
  artefactos y su referencia se verifica exacta en la entrada de cada sub-via.
- **0 submissions escritas, 0 enviadas, 0 recomendadas** en toda la fase. v17 y v18 no las
  respalda ningun gate, y v19 no se genero porque ninguna candidata llego a la mesa de finales.
- **El hallazgo que reencuadra todo el proyecto**: la metrica corregida tiene una **banda de
  ruido de mas de 0.017 entre particiones equivalentes**, con desviacion tipica de 0.0053, medida
  en V10-02 por una via independiente del ruido de 0.01524 que V8-02 medico en el externo. **El
  proyecto opera por debajo de su propio suelo de medicion.** El objetivo de superar 0.80 esta a
  1.1 veces el ruido del leaderboard, y la mayor ganancia anidada de toda la fase es de +0.000148.
- **El protocolo cambio por esa medicion, no por los resultados**: el addendum de `tasks/plan_v10.md`
  se escribio **antes** de ejecutar la via A y exige tres cifras para toda candidata, con el
  **recuento de particiones que mejoran** como criterio decisivo. Los gates por sub-linea no se
  relajaron, solo se dijo con que instrumento se miden.
- **Las tres sub-vias de la via A cerradas**: calibracion **no accionable** (mejora el ECE un
  29.94 por ciento y mueve la F1 en +0.000148), regla F1 optima **cerrada** (-0.003921 con prior
  simulado, porque los umbrales de V03 ya estan por encima del optimo plug-in natural), umbrales
  estables **cerrados** (la holgura es irreducible con estos datos).
- **La via de shift de features se CERRO porque el discriminador esta por debajo del azar**, con AUC
  de 0.487717. No hay shift de covariables; el unico desplazamiento medido es el de prevalencia de
  factor 0.94, y V10-05 mostro que adaptarle la capa de decision empeora.
- **holdout2 creado, congelado y con su gate NO superado**: la referencia de dev2 quedo en
  0.763436, desviacion -0.013776, explicada despues como ruido de tamano y de particion. Se creo
  **antes** de la via A para que el juez existiera, y las tres evaluaciones disponibles siguen
  **sin gastar**. No es virginal, y eso quedo escrito antes de ver resultados.
- **Correlacion: la conclusion de la v8 era correcta y su prueba no la sostenia.** Colapsaba el
  99 por ciento de las caracteristicas y usaba un criterio de magnitud, no de redundancia. Ahora
  esta medido y con el control que faltaba: las caracteristicas estan **poco correlacionadas**
  (mediana 0.1704 de correlacion maxima), **casi todas aportan poco por si solas** (AUC mediano
  0.51 a 0.52, y el modelo combina miles de rasgos debiles), y **eliminar una de cada par empeora
  en los tres umbrales con 0 de 5 particiones**. La regla de conservacion es **irrelevante**: la
  aleatoria es la mejor de las cuatro y la brecha es menor que la banda de ruido.
- **Correccion de documentacion, D-072**: la cifra de **+0.0036 no es el efecto de optimizar
  umbrales**. Era un error de comparacion entre brazos con distinto conjunto de ajuste, ya
  demostrado por la fase v7. El efecto aislado es -0.0019 y la medicion propia de V10-06 da
  +0.000131. Las tres coinciden en que el efecto del umbral es pequeno.
- **Lo que queda abierto y es deliberado**: que V03 sea el maximo de 18 sondas puede ser que sea
  genuinamente el mejor o el maximo de ruido de una serie de comparaciones, y con este tamano de
  conjunto no se puede distinguir. La fase v10 no ha anadido el dato que lo resolveria.
- **P4 ejecutado y H1 cerrada** (D-077): el afinamiento de embeddings **funciona** y es la mayor
  mejora entre dos variantes de la misma arquitectura del proyecto, de **+0.017645** sobre la
  congelada. Pero el control con tasa de aprendizaje unica, que el plan no habia pedido, da solo
  **+0.005092**, asi que **+0.012553 de los +0.017645 los aporta la tasa diferencial** y no el
  hecho de mover el embebido. Sigue a **-0.065572** de V03, de modo que **la via de representacion
  secuencial queda cerrada** y no se abre con mas capas, mas unidades o mas epocas.
- **Notebook re-ejecutado con la seccion 20**: 67 celdas, 34 de codigo, **0 errores**, 4.1 s.
  Registro en `reports/notebook_ejecucion_v10.json`.
- **La fase v10 esta completa**: no queda ninguna tarea abierta del plan.

### Tras la fase v11 completa (V11-00 a V11-08)

- **V03 SIGUE SIENDO LA SUBMISSION RECOMENDADA**, con 0.7772114525165721 local y 0.78306 en el
  externo. **Ninguna submission generada, ninguna enviada, ninguna recomendada** en la v11.
  **holdout2 intacto, con 0 de 3 evaluaciones gastadas.**
- **La respuesta a la pregunta central de la v11 es NO**: el desplazamiento de prevalencia **no es
  una fuente real de mejora** bajo las restricciones actuales. Se ha podido demostrar con un
  factor **conocido por construccion**, que es el caso en el que el mecanismo se puede comprobar.
- **El mecanismo esta cerrado y la causa esta medida.** La correccion de prior da -0.001032 en HS
  con 1 de 5 particiones, +0.000000 en TR con 0 de 5, y +0.000154 en AG con 1 de 5; macro medio
  **-0.000293** y los tres intervalos incluyen el cero. La direccion del efecto es sistematica:
  empeora cuando la prevalencia baja y ayuda cuando sube, y el shift medido es hacia abajo. La
  causa de fondo la midio V10-05: los umbrales de V03 estan ya **por encima** de los optimos de
  F1, luego la capa de decision ya es conservadora y la correccion corrige dos veces.
- **Con el cierre caen P5 y P7**, no por decision de gesto sino por consecuencia logica: ambas
  presuponen que el shift es explotable. **P10 no tiene objeto** por la auditoria de legalidad.
  Las tres lineas que no dependen del shift, **P4, P6 y P8**, ya estaban medidas y cerraron
  negativas en v10. **No queda ninguna linea abierta con expectativa razonable.**
- **El uso de la prueba para estimar prevalencia queda PROHIBIDO.** La auditoria de legalidad
  clasifico once pasos contra el texto verbatim de las reglas: 5 permitidos, 1 ambiguo, 5
  prohibidos. El ambiguo era precisamente ese uso, porque la regla prohibe entrenamiento y aqui no
  hay entrenamiento, pero la intencion de la regla es que el test no participe del desarrollo, y
  las reglas **no mencionan la transduccion ni para permitirla ni para prohibirla**. Con la decision
  de NIVEL 2 pasa a prohibido y **v17 transductiva queda en cuarentena**.
- **La consecuencia de alcance esta escrita sin adornos**: la pregunta de si el mecanismo
  transfiere a esta prueba concreta **no tiene respuesta verificable dentro de las reglas**. Se
  respondio la version metodologica, que es si el mecanismo recupera un desplazamiento conocido.
  La v11 no termina en una submission, y no por falta de calculo sino por las reglas.
- **Lo que si se aprende en la v11**: los estimadores de prevalencia funcionan, y el error
  exactamente cero de clasificar-y-contar se explica como un **teorema** bajo shift puro, no como
  una propiedad favorable. El punto ciego se declara: el resultado vale dentro del supuesto de
  shift puro y nada mas.
- **Se arrastro una ambiguedad de un artefacto historico**: el campo `factor_vs_train` de V9-02
  usa como denominador la prevalencia del **desarrollo** y no la del train completo. V9-02 es
  internamente consistente y su reproduccion es exacta, con nivel R3.
- **Dos correcciones al instrumento de auditoria**: la politica de caracteres de la seccion 17 es
  **mas estricta** que `verificar_caracteres.py`, que busca alfabetos no latinos mientras que la
  seccion 17 pide ASCII puro; y un escaner de cumplimiento por palabras es **inservible**, porque
  marca sus propias citas de prohibicion, por lo que se sustituyo por analisis estructural de
  imports.
- **Pendiente**: no queda nada abierto de la v11. La v10 tiene tres ficheros con acentos que no
  se corrigen, por decision, porque modificarlos rompe la correspondencia entre codigo y resultado
  historico.

### Tras la fase v12 completa (V12-00 a V12-05)

- **V03 SIGUE SIENDO LA SUBMISSION RECOMENDADA**, con 0.7772114525165721 local y 0.78306 en el
  externo. **Ninguna submission generada, ninguna enviada, ninguna recomendada** en la v12.
  **holdout2 intacto, con 0 de 3 evaluaciones gastadas.** Segunda particion no ejecutada.
- **La asimetria que reencuadro la fase**: de 24 familias de mecanismo, **17 de las 24 cerradas estan
  en la representacion y el clasificador, y los 3 huecos que quedaban estan todos en el objetivo de
  entrenamiento.** No se habia tocado nunca la funcion que se minimiza. La auditoria lo encontro con
  una **sonda estructural sobre el arbol sintactico**, no con busqueda textual, y se autovalido con
  un positivo conocido.
- **Los tres mecanismos del objetivo de entrenamiento se cierran negativos o marginales**: H1
  +0.002738 sostenido por un solo pliegue con el maximo de su curva **sobre el valor neutro**; H2a
  +0.001616 con el intervalo incluyendo el cero; H3 **-0.002469** con deteccion 0.4915, peores que el
  azar. Es **rendimientos decrecientes** en ese eje, no una succession de ideas malas.
- **El techo con HS verdadero es de +0.046271** y sigue sin capturar. La premisa se midio por
  primera vez: el **80,44 por ciento** de los negativos que ve TR en entrenamiento **no estan
  definidos**, porque TR y AG solo existen cuando HS=1.
- **El cuello de botella NO es la falta de capacidad para TR y AG**, cuya F1 donde la etiqueta esta
  definida es de 0.8941: es **la inferencia de la condicion de HS**. Las tres vias fallan por
  razones distintas: la llave dura porque su llave se equivoca en 143 filas; la llave suave, que es
  V6-B y ya estaba cerrada, porque solo alcanza 20 de las 78 filas del techo; y el acoplamiento en
  el objetivo porque un objetivo de orden no restringe el nivel absoluto que los umbrales fijos
  necesitan.
- **La hipotesis candidata de la siguiente fase NO se abre, porque no es nueva: es V6-B.** Leida del
  codigo, la cabecera declara sin routing duro y hay 0 condicionales que anulen una etiqueta a
  partir de HS.
- **La correccion conceptual que la fase conserva**: H2 y H3 **no demuestran que la funcion de
  perdida no sea el problema en sentido absoluto**. Un unico mecanismo positivo no autoriza a
  descartar la familia. Lo que si se dice es que **las variaciones de objetivo evaluadas tienen un
  impacto pequeno frente al techo que introduce el error de la condicion**, del 3,5 por ciento en el
  mejor caso.
- **Queda abierto un solo problema, que no es de esta fase**: conseguir que la condicion de HS se
  estime mejor.
- **Instrumento**: todo objetivo a medida verifica su derivada contra diferencias finitas antes de
  mirar un resultado. En H3 eso detecto cuatro fallos propios que habrian dado cifras plausibles y
  falsas, mas un fallo de diseno en el que el termino por pares estaba en otra escala que la
  entropia cruzada y lambda era irreconciliable.
- **Pendiente**: no queda nada abierto de la v12. La v10 conserva tres ficheros con acentos que no
  se corrigen, por decision, porque modificarlos rompe la correspondencia entre codigo y resultado
  historico.
