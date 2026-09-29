# PROJECT_STATE - KAGGLE_TOXIC_2026_PUJ

Ultima actualizacion: 2026-09-26 (fases **v6** a **v15** CERRADAS; v15 cierra el
proyecto TOXIC en V03, D-110 a D-114). La seccion de detalle mas reciente que habia aqui era
la de v5; las de v6 a v14 se registran en DECISIONS.md, RUNLOG.md y en los informes de fase
de reports/.

## Fase v15 "Cierre de la frontera final" (CERRADA el 2026-09-26)

Rama `proyecto-kaggle-toxic-2026-v15` (worktree `/home/carlos/Git/PLN-v15`, base v14 `c5be6b9`).
**Tres experimentos, V15-00 a V15-02. Decisiones D-110 a D-114.**

**ESTADO DEL PROYECTO: CLOSED. TOXIC se cierra en V03.** V03 continua siendo la mejor solucion
reproducible bajo las restricciones actuales, con 0.78306 en el externo y 0.7772114525165721 en el
desarrollo. 0 adoptables, 0 marginales, 3 rechazadas, 0 submissions, 0 de 3 evaluaciones de
holdout2, segunda particion no ejecutada, test no utilizado.

**Lo que la fase establece**:

1. **La categoria de identidad y metadata no existe.** El input es `id;text;HS;TR;AG` y nada mas.
   No hay user ID, ni fecha, ni perfil. La via queda cerrada por inexistencia, no por resultado.
2. **V03 tiene dos propiedades demostrablemente incompletas**, y de la primera se sigue la unica
   clase de senal que no puede expresar: la posicion en el documento, porque `TfidfVectorizer`
   extrae n-gramas sin registrar donde esta cada palabra.
3. **De doce clases de senal inventariadas, tres no existen, cuatro ya estaban medidas y cinco
   estan en familias cerradas.** Los rasgos estructurales de E2-18 son negativos en dos
   estimaciones independientes, -0.0052 y -0.0051.
4. **La senal residual existe y es complementaria, y es inutilizable.** El AUC estratificado por el
   margen de V03 da 0.5547 con valor p empirico por debajo de 1 entre 500, es decir el rasgo ve
   algo que V03 no ve. Y anadido a V03 cambia entre 14 y 57 celdas con **acierto CERO** entre las
   modificadas, frente al umbral de detectabilidad de 0.5166.
5. **El mejor resultado local de la historia, 0.8465, perdio en el externo** con 0.78163 frente a
   0.78306. V03 sigue siendo la mejor solucion por evidencia en contra medida, no por falta de
   intentos.
6. **El techo de +0.046271 no es un techo de HS.** Es un efecto aguas abajo de conocer la condicion
   semantica, y V13 demostro que corregir HS deja TR y AG con delta exactamente cero.

**Decision de gobierno: no se abre una V16.** La regla metodologica final prohibe la dinamica de
experimento negativo seguido de reformulacion, y las tres adiciones de V15 son negativas con acierto
cero, que es el caso mas limpio de ese patron. Cualquier trabajo futuro requiere una hipotesis que
demuestre una diferencia **informacional** antes de gastar un experimento.

## Fase v14 "Auditoria de entrenamiento partially-label" (CERRADA el 2026-09-26)

Rama `proyecto-kaggle-toxic-2026-v14` (worktree `/home/carlos/Git/PLN-v14`, base v13 `b07e410`).
**Un experimento, V14-00, mas el informe final V14-01. Decisiones D-105 a D-109.**

**Resultado: 0 ejecuciones y 0 candidatas.** La auditoria cumplio la condicion de parada numero 1 del
enunciado, que permite cerrar la fase sin ejecutar nada. 0 submissions, 0 de 3 evaluaciones de
holdout2, segunda particion no ejecutada, test no utilizado, V03 congelada e intacta.

**Lo que la fase deja establecido**:

1. **La perdida enmascarada de P1 ES H2-M, con diferencia exactamente cero** en coeficientes,
   probabilidades y macro-F1, y reproduce el -0.116581 ya registrado hasta el ultimo decimal. La
   razon: una fila con peso cero no aporta gradiente, luego enmascarar y eliminar dejan el mismo
   subconjunto de filas aportando senal.
2. **No se puede concluir que los negativos ordinarios sean correctos.** La alternativa no fallo:
   resulto ser el mismo estimador. No hay evidencia ni a favor ni en contra de la semantica de
   observacion parcialmente definida, y las preguntas 1, 2 y 5 del enunciado se declaran sin
   respuesta por evidencia.
3. **Lo que si queda restringido es lo construible**: cualquier mecanismo que se implemente
   enmascarando la perdida de clasificacion sobre las filas HS=0 va a reproducir H2-M. Para
   apartarse haria falta una senal distinta de cero y sin etiqueta en esas filas, lo que exigiria
   datos externos, una representacion nueva o informacion transductiva, los tres prohibidos.
4. **Los cuatro mecanismos que la fase propone como nuevos por su terminologia tienen cero
   menciones historicas**, una vez descontada la autorreferencia de la propia consulta. Y ninguna de
   las 13 llamadas con `sample_weight` del codebase condiciona los pesos a la definicion de la
   etiqueta.

**Decision de gobierno: la linea de entrenamiento partially-label de TR y AG en el regimen HS=0
queda CERRADA.** No se afirma que haya que mejorar HS, que v13 ya desmentio con delta exactamente
cero en TR y AG. La unica linea realmente abierta, y que no es de esta familia, es una formulacion
que aporte senal sin etiqueta a las filas HS=0, y requiere una decision de jerarquia distinta, no un
experimento mas.

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

## Fase actual: v5 "Embedding from Scratch" (CERRADA el 2026-09-25)

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
| D-046 | Solo el 7.1 % de filas es repetible y penalizarlo empeora: la repeticion es senal |
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
| GATE DE CIERRE | parrafo 29: A (+0.005 con estabilidad) / B (complementariedad robusta) / C (senal representacional fuerte) |

## Fase anterior: v4 "Advanced Research v4" (CERRADA el 2026-09-25)

Rama `proyecto-kaggle-toxic-2026-v4`. Spec archivada en
`docs/advanced_research_v4_spec.md`; plan en `tasks/plan_v4.md`; backlog
pre-registrado en `reports/research_backlog_v4.md`; informe de cierre en
`reports/advanced_research_v4.md`; matriz de cumplimiento en
`reports/spec_compliance_v4.md`. Decisiones D-025 a D-029.

**Resultado: la fase se cierra con la condicion B de la spec (parrafo 102): hipotesis
nuevas investigadas y cerradas con evidencia, sin mejora transferible.** El
unico candidato que supero los tres gates de D-016 (C, `meta36_downweight`,
corrected_cv 0.7882 = +0.0110, 4/5 pliegues, IC95 [+0.0054, +0.0200]) **fue
rechazado por el holdout congelado: 0.8163 frente a 0.8285 de V03 (delta
-0.0122)**. Su submission (v12) queda como artefacto reproducible marcado
`DESCARTADO`; no se envio a Kaggle.

| Campo (parrafo 104) | Valor |
|---|---|
| CURRENT_BEST_MODEL | **V03** (sin cambios; verificada byte a byte en P0 y en la auditoria final) |
| CURRENT_BEST_CORRECTED_CV | **0.777211** (protocolo v2, desarrollo n=1.483) |
| CURRENT_BEST_HOLDOUT | 0.8285 (V03, P7-01). **Holdout agotado: 5/5 evaluaciones usadas** |
| CURRENT_BEST_KAGGLE | **0.78306** (v03, mejor de 12 sondas; 0 sondas nuevas en v4) |
| NEW_SIGNALS | ninguna adoptada. Senales nuevas caracterizadas y **cerradas**: rareza del ruido de etiqueta (2.8 % de contradicciones, no explica el error); dureza util = distancia a la frontera (la BCE agregada no preserva el orden); reweighting con signo negativo; verosimilitud class-conditional aporta +0.0041 en fusion; SVC RBF sobre SVD 300 = 0.7904 con IC fragil |
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


---

## Fase P45 (2026-09-27) - METRIC_MATCH = TRUE, JOINT_LABEL_STRUCTURE = CLOSED

**Que se respondio.**

1. La metrica local `corrected_cv` **si** es exactamente
   `f1_score(average="macro")` sobre entrada multilabel: diferencia `0.000e+00` y 0
   discrepancias en 600 casos aleatorios.
2. `corrected_cv = 0.7772114525165721` se reproduce exactamente desde el OOF canonico.
3. La media de pliegues (`0.7773117212`) y el global (`0.7772114525`) no son intercambiables;
   el proyecto reporta el global.
4. El parametro `average` del host sigue sin ser legible, pero `samples` queda **excluido**
   por el discriminante de escala del leaderboard. Las tres hipotesis restantes estan a menos
   de 0.0036 entre si y ordenan los candidatos igual (spearman >= 0.98).

5. La estructura conjunta `HS x TR x AG` es **real y medible**: los errores de las tres
   etiquetas no son independientes dado el estado verdadero (chi2 343.156, gl 20, p < 1e-6).
6. Pero **no es explotable**: la correccion estructural dura es negativa en las dos formas
   (-0.000646 y -0.003542), y todos los mecanismos de aprendizaje conjunto estan ya
   registrados con resultado negativo o sin transferencia.

**Lo que queda abierto.** Solo una resolucion menor: distinguir `macro` de `micro` y
`weighted`. Es inmaterial para el ranking de candidatos, asi que no bloquea nada.

**Lo que queda cerrado.** `JOINT_LABEL_STRUCTURE = CLOSED`. No se abre ninguna familia
conjunta nueva.

**Estado del campeon.** Sin cambios. V03 sigue con `corrected_cv = 0.7772114525165721`,
LB 0.78306, sha256 `7d7478f2`, intacta. 0 submissions en esta fase. 3 submissions reservadas.

**Reconciliacion documental de V44 (misma fecha).** Antes de P45 se reconcilio V44 contra
sus artefactos. Se corrigieron tres filas de `experiments.csv` (V44-01 declaraba una config
de VW que el script nunca tuvo; V44-02 decia "CANDIDATO UNICO / REPENDIENTE DE CORRER"
cuando ya estaba medido y era negativo; V44-03 declaraba 173->176 celdas cuando el paso real
fue 177->181), una celda del notebook cuya tabla no producia ningun script, y un fallback
copy-pegado en `v44_02_pls_correcto.py` que devolvia la decision de V44-01. Ninguna cifra
experimental cambio. Commit `2285423`.


---

## Fase P46 (2026-09-28) - Champion reconciliation

**Que se resolvio.** El verdadero mejor resultado de leaderboard del proyecto.

```text
leaderboard_max_verified = 0.78306   (v03_lr_union_thr = V03)
baseline_operativo       = V03
sha256                    = 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
CHAMPION_STATUS           = INCONSISTENTE_AUTOCORREGIDO
CASE                      = B
```

**Por que los 0.78535 y 0.78650 no son Scores.** La columna que los contiene se titula
`LB combinado` y el propio CHAMPION.md declara la formula que la produce. Recalculada, los
reproduce con desviaciones de 7.9e-06 y 5.2e-06. **Son predicciones del modelo causal de la
fase v16.** Las mediciones reales de `v16_ch1` y `v16_ch5` son 0.77828 y 0.77495, ambas
REJECTED, y estan en el mismo fichero.

**Integridad del log.** 22 filas, 0 duplicados, 0 nulos, 0 fuera de rango, 22/22 con artefacto
en disco. Ninguna supera 0.78306. **0 de 22 submissions superaron V03**, y ahora eso esta
respaldado por artefacto y hash, no por lectura de una tabla.

**Las cuatro magnitudes separadas.** mejor corrected_cv utilizable 0.7772114525165721 (OOF
canonico de 1483 filas); mejor local del log 0.80390 (`v05_optuna`, LB 0.77602); mejor
submission por LB 0.78306 (V03); mejor modelo congelado V03. **El mejor local no es el mejor
LB**, y la unica magnitud que decide el campeon es la tercera.

**Defecto del registro detectado.** La columna `macro_f1` de `experiments.csv` mezcla tres
bases de medida (holdout2, split D-004 supersedido, OOF canonico). Ordenarla sin clasificar la
base produce rankings invalidos. Documentado en D-117; no se reescriben las filas historicas.

**Correccion de P45 aplicada.** Las decisiones que P45 escribio como D-030 y D-031 estanban ya
ocupadas por la fase v5. Renumeradas a D-115 y D-116. Ademas, `METRIC_MATCH = TRUE` se
desdobla: `LOCAL_METRIC_REPRODUCED = TRUE` y `KAGGLE_SCORER_DIRECTLY_VERIFIED = FALSE`,
porque el codigo del scorer del host no esta en el repositorio.

**Estado del campeon.** Sin cambios. V03 intacto. 0 submissions, 0 consultas a Kaggle,
0 modelos, holdout y test sin usar. 3 submissions reservadas.


---

## Fase P47 (2026-09-28) - Data Availability Audit

**Que se resolvio.** Si existe alguna fuente de datos etiquetados legalmente utilizable que V03
no haya usado.

```text
CASE                = A + D
DATA_EXPANSION      = CLOSED
TRAIN_AUGMENTATION  = CLOSED
experimento nuevo   = False
```

**El train esta completamente asignado.** 1731 filas = 1483 en dev + 248 en holdout. Cero
filas sin asignar. La recomendacion de "mas datos etiquetados" que P46 dejo abierta **no es
operacionalmente posible** con este corpus, y queda cerrada por evidencia.

**Dos cifras de la documentacion reciente eran falsas.**

- **519 no es el holdout.** Es el numero de positivos TR en train. El holdout tiene **248** filas.
- **No existen ~44k filas sin etiquetar.** Los corpus externos se miden en tokens (5M a 50M), no
  en filas, y suman 42808 documentos. El "44" que circulaba son las 44 features del
  meta-modelo `E2-02-META`.

**Lo que si hay, y para que sirve.** 950.114.116 tokens externos repartidos en cinco corpus
(wikipedia_es CC BY-SA 4.0, m_c4_es ODC-BY 1.0, y sus mezclas). Legales **solo** para aprender
la representacion distribuida segun D-030, y la familia de embeddings esta cerrada. Ademas hay
19 modelos fastText ya entrenados en disco, que son salidas derivadas, no fuentes de filas.

**La augmentation ya existia.** V6-08 implemento las tres transformaciones ortograficas que la
directiva proponia, con la colocacion en el pipeline que la directiva exigia y los grupos
preservados. J1 = **-0.0061**. `TRAIN_AUGMENTATION = CLOSED`. No se creo ninguna fila
experimental ficticia.

**Estado del campeon.** Sin cambios. V03 intacto, `corrected_cv = 0.7772114525165721`, LB
0.78306. 0 modelos, 0 submissions, 0 consultas a Kaggle, 3 submissions reservadas.


---

## Fase P48 (2026-09-28) - EDA-Driven Residual Research

**Que se respondio.** Si el EDA original se convertirio en hipotesis experimentales, y si quedan
patrones de error residual con mecanismo sin probar.

```text
EDA_RESEARCH_GAP   = NONE
hipotesis nuevas   = 0
experimentos      = 0 de un presupuesto de 3
```

**Trazabilidad: 14 de 15 hallazgos del EDA cerrados, ninguno sin experimento.** La conversion
del EDA en mecanismos fue completa. El unico hallazgo abierto es la rareza de etiqueta, con un
unico experimento asociado. Ademas, el informe EDA de origen solo declara hipotesis en 3 de sus
6 hallazgos clave.

**Atlas de residuos: no hay poblacion rescatable escondida.** De las 757 celdas de error de V03,
**715 (94,45 por ciento)** las acierta al menos un canal OOF historico y **42 (5,55 por ciento)**
ninguno. La tasa de error global es 0,2974; la de rescate, 0,9445.

**El hallazgo que decide la fase.** Hay slices con error muy por encima de la media, con lift de
1,21 a 1,60 y p < 0,01. **Pero ninguna se enriquece en rescate**: su tasa de rescate es igual o
inferior a la global, y varias quedan significativamente por debajo. Elevar la tasa de error no es
lo mismo que concentrar informacion recuperable, y la prueba que lo separa es el enriquecimiento
en rescate.

**Sobre el margen.** El rescate cae de forma monotona con el margen de V03: 0,9929 por debajo de
0,10 frente a 0,8657 desde 0,20. El **85,7 por ciento de las celdas irrecuperables estan en
margen alto** (36 de 42). Es la firma de un hueco de representacion, pero son 42 celdas de 757.

**La respuesta a la pregunta de la directiva.** Seguimos sin mejorar **porque ya no existe senal
utilizable**, no porque el EDA quedara sin convertir. Las dos ramas se resuelven a la vez.

**Limitaciones que quedan declaradas.** Los tensores OOF por canal ya no estan en disco, asi que
la categoria C es una cota superior del rescate y no identifica el modelo concreto. Y la matriz
de rescate literal de la directiva no es computable con los artefactos presentes.

**Estado del campeon.** Sin cambios. V03 intacto. 0 modelos, 0 submissions, 0 consultas a
Kaggle, 0 uso de holdout o test.


---

## Fase P49 (2026-09-28) - OOF Rescue Reconciliation

**Que se resolvio.** La contradiccion entre 715/757 y 147/757.

```text
OBSERVED_RESCUE        = 715 de 757  (94,45 pct) - observacion directa
UPPER_BOUND_RESCUE     = no aplica a 715
147                    = rescate de UNA sola variante, no de la union
irrecuperables         = 42 de 757   (5,55 pct)
esperado si azar       = 1,20
CASE                   = A
```

**Los tensores SI eran recuperables**, en los worktrees `PLN` y `PLN-v8`. La reconstruccion del
pool de 20 canales coincide con `reports/hardness_v4.csv` en las 4449 celdas con diferencia
`0.00e+00`. La limitacion que P48 declaro era falsa.

**Las dos cifras no son comparables y nunca debieron serlo.** 147 es el rescate de una variante
concreta (`C_meta36_downweight`, umbral 0.5) y **se reproduce exactamente**: 147 rescates, 90
danos, cobertura 0.194187582562748. 715 es el de la union de veinte canales. La confusion venia
de presentar ambas sobre el mismo denominador sin decir que el numerador significa cosas
distintas.

**No es azar, y ese es el hallazgo.** Con independencia entre canales se esperarian 1,20
celdas irrecuperables; se observan **42**. Un canal acierta una celda de error de V03 solo el
27,56 por ciento de las veces. Los canales fallan juntos con V03.

**Pero no se puede cobrar.** **19 de los 20 canales tienen net negativo.** El unico positivo
es el propio V03 con umbral 0.5, que no es un modelo alternativo. El canal con mas rescates
(`cita:NEG`) tiene **net -1706**: rescata 337 y dania 2043.

**Correccion de P48.** Dos elementos superados: la limitacion sobre los tensores y el criterio
de enrichment. El resto verificado, y `EDA_RESEARCH_GAP = NONE` **se mantiene** por un motivo mas
fuerte: no es que no haya rescate, es que es difuso e identificable.

**Estado del campeon.** Sin cambios. V03 intacto. 0 modelos, 0 submissions, 0 consultas a
Kaggle, 0 uso de holdout o test.


---

## Fase P50 (2026-09-28) - Contextual Channel Selection

**Que se res respondio.** Si el techo de V03 se explica por falta de senal o por incapacidad de
identificacion.

```text
SELECTOR_FAMILY            = GENUINELY_NEW (la categoria E no existia)
ORACLE_DELTA               = +0,209806 macro-F1  (69,9x el margen de 0,003)
S1_CORRECTED_CV            = 0,7657442265
S1_DELTA                   = -0,0114672260
PLIEGUES_POSITIVOS         = 0 de 5
RESCATE / DAMAGE / NET     = 48 / 67 / -19
ACCION ELEGIDA             = 82,56 pct de aciertos
NO CAMBIAR NUNCA           = 82,98 pct de aciertos
CASE                       = B
```

**El techo es enorme y la captura es negativa.** El hipotesis planteo que macro-F1 podria ofrecer
poca ganancia practica: **queda refutado**. Los 715 rescates valen **+0,2098 de macro-F1**, setenta
veces el margen exigido.

**Y sin embargo el selector pierde 0,0115 y queda 19 celdas por debajo de no hacer nada.** Todos los
canales que eligio tienen net negativo, y los dos mas elegidos son los dos peores del pool. El
argmax sobre P(acierta) premia la **predecibilidad**, no la **utilidad**.

**19 de los 20 canales tienen accuracy inferior a 0,5** sobre el conjunto de desacuerdo, es decir
sobre exactamente las celdas donde un selector los usaria. Solo V03 con otro umbral llega a 0,5420.

**Estado del campeon.** Sin cambios. V03 intacto. 0 modelos, 0 submissions, 0 consultas a Kaggle, 0
uso de holdout o test.


---

## Fase P51 (2026-09-28) - Utility-Aware Contextual Gating

**Que se respondio.** Si los 715 rescates potenciales se convierten en mejora real
cuando el selector aprende la utilidad del cambio en vez de la correccion del canal.

```text
SELECTOR_UTILITY_ALREADY_TESTED = FALSE
EQUILIBRIO_ANALITICO            = 0,3892 en las DOS direcciones (no 0,5)
AUC del target de utilidad      = 0,5507 a 0,9249, media 0,8091
AUC del target de correccion    = 0,55 a 0,77  (P50)
elevacion necesaria al subir    = 2,18x
elevacion necesaria al bajar    = 1,03x
S1-U  delta = -0,7477491670   0/5 pliegues   net -916
S2-U  delta = -0,0101586431   1/5 pliegues   net  -62
permutacion: observado -62 vs nula -115  -> 53 celdas MEJOR que el azar, p = 0,0000
solo direccion 1->0: net +5
CASE = A
```

**Lo que P51 corrige de P50.** P50 concluyo que la informacion no era identificable.
P51 demuestra que **si lo es**: el AUC sube a 0,81 de media y el selector es 53
celdas mejor que colocar los cambios al azar, con p de una cola 0,0000. El problema
es otro y es mas duro: **el conjunto de cambios disponibles tiene expectativa
negativa**, y la mejor regla conocida recupera solo una parte de esa perdida.

**La asimetria de direccion es el hallazgo mecanico.** El equilibrio no es 0,5 sino
0,3892, y sale de la forma cerrada de V19-01, no de un grid. Las dos direcciones no
son intercambiables: subir necesita 2,18x de elevacion y bajar solo 1,03x. Y al
medir, bajar de 1 a 0 sale **neta positiva** (+5) mientras subir sale muy negativa
(-67).

**Estado del campeon.** Sin cambios. V03 intacto. 0 modelos, 0 submissions, 0
consultas a Kaggle, 0 uso de holdout o test.


---

## Fase P52 (2026-09-28) - Policy Oracle, decision boundary y complementariedad

**Que se respondio.** Si existe una politica de decision basada solo en senales observables
con un techo materialmente superior a V03.

```text
V03                          0,7772114525
estrato_out_of_fold          0,7549322355   (-0,0222792170)
STRATUM_ORACLE               0,7753804469   (-0,0018310056)  <- techo rigido
TRUE_ORACLE                  0,9870178614   (+0,2098064089)
hueco TRUE -> STRATUM        +0,211637
hueco STRATUM -> V03         -0,001831
CASE = C
```

**El techo es negativo.** La mejor politica constante dentro de estratos observables,
con el permiso de mirar `y_true`, pierde frente a no hacer nada. Ningun selector
condicional puede superar a V03, y eso es una cota y no un fallo de modelo.

**El +0,2098 del oracle esta casi todo en la granularidad de celda.** Coarsar la politica
a 18 estratos observables cuesta +0,211637. Exige una accion distinta en cada celda, y
los estratos no dicen cual.

**Rescate exclusivo: la hipotesis queda refutada con evidencia.** 78 de 715 celdas
rescatadas tienen un unico rescuer, HHI del reparto 0,0020 (casi uniforme, por debajo del
minimo teorico de 0,05 porque el solapamiento es enorme), 2 de 20 canales con
`net_exclusivo` positivo, y **0 de 190 pares con exclusividad bilateral**. El dato duro:
donde dos canales difieren de V03, **nunca** difieren entre si.

**Correcciones a P51.** (1) La elevacion necesaria al bajar era **1,62x**, no 1,03x: P51
dividio `(1 - eq)` por la tasa base cuando el rescate al bajar es `P(y=0)` y su equilibrio
es el propio `eq`. (2) P51 aplico el umbral **0,5** donde la teoria da 0,3892 al subir y
0,6108 al bajar, luego sobre-cambio en la direccion que mas dana. (3) P51 mezclaba las dos
direcciones en un solo clasificador por canal.

**Estado del campeon.** Sin cambios. V03 intacto. 0 modelos, 0 submissions, 0 consultas a
Kaggle, 0 uso de holdout o test. P52-01 no se ejecuta: el techo no lo justifica.


---

## Fase P53 (2026-09-28) - Politica observable de grano fino

**Que se respondio.** Si la identidad exacta de los canales que discrepan aporta algo que
el conteo no aporta.

```text
ESPACIO DE ACCIONES REAL      = 2   (no 21)
celdas con valor distinto     = 0
celdas con dos contradictorios = 0
P0 (no hacer nada)             0,7772114525   +0,0000000000
P1 margen + conteo            0,7623696498   -0,0148418027
P2 + mascara exacta           0,7566874656   -0,0205239869
P3 + etiqueta                 0,7588669459   -0,0183445066
P4 + probabilidades           0,7712584603   -0,0059529922
P52 STRATUM_ORACLE            0,7753804469   -0,0018310056
TRUE_ORACLE                   0,9870178614   +0,2098064089
CASE = C
```

**El colapso estructural es el hallazgo central.** En las 3896 celdas con discrepancia,
todos los canales que discrepan proponen exactamente `1 - v03_pred`. Cero celdas donde
dos se contradigan. Cambiar al canal 3 o al canal 17 da **la misma prediccion**, luego la
mascara no puede decir el valor propuesto. El problema entero es **una decision binaria
por celda**.

**Ninguna particion observable supera V03.** El mejor techo es P0, que es no hacer nada.
Y el diagnostico de soporte lo explica: con soporte minimo 20 los cambios caen a cero,
porque los patrones con expectativa positiva en retrospectiva son los pequenos.

**Tres correcciones a P52.** (1) El `STRATUM_ORACLE` no estaba contaminado, la pregunta
estaba mal planteada: elegir la accion con `y_true` es lo que hace que sea un techo. (2) La
exclusividad bilateral de P52 estaba mal calculada:|reportaba 0 pares y son **189 de 190**,
porque restringia la medicion a un conjunto degenerado. (3) El HHI de P52 estaba mal
definido, con proporciones que no sumaban 1.

**Estado del campeon.** Sin cambios. V03 intacto. 0 modelos, 0 submissions, 0 busqueda de
umbrales, 0 consultas a Kaggle, 0 uso de holdout o test. No se entrena modelo.


---

## Fase P54 (2026-09-28) - Atlas semantico de los errores no rescatables

**Que se respondio.** Que son los errores que ningun canal historico puede corregir, y si
eso es representacion, decision o informacion ausente.

```text
A correcto        3692
B rescatable       715
C no rescatable     42   de 33 tweets distintos
errores            757   verificado contra P49
FP 22  FN 20
HS 2,85 pct | TR 7,32 pct | AG 7,58 pct de los errores son no rescatables
margen alto       85,7 pct de C
alta confianza    36 de 42
P54_RESEARCH_GAP  = NONE
hipotesis nuevas  = 0
```

**El soporte efectivo es 33, no 42.** Las 42 celdas proceden de 33 tweets, y 9 de ellas
vienen de solo 3 tweets con las tres etiquetas mal.

**El residuo no es falta de informacion.** El 66,67 por ciento de C ya contiene el lexico
de insulto, y solo el 2,85 por ciento de los errores de HS son no rescatables frente al
7,58 por ciento de AG.

**Es frontera de etiqueta.** Mitad falsos positivos con insulto lexico y etiqueta 0, mitad
falsos negativos con narrativa sobre grupo y etiqueta 1. Texto inequivoco, convencion del
anotador.

**El mayor candidato, la negacion, no es hipotesis.** 17 celdas, enrichment 1,81 con
p=0,0064, pero su mecanismo queda **sin explicar**: la version concedente se refuta con
**una sola celda** de 42 y enrichment 0,76. Y la negacion tambien marca dificultad en
general, no un modo de fallo.

**El valor absoluto cierra el resto.** `mi + insulto` da 7,33x con 4 celdas; el lexico de
grupo marginal da 2,68x con 9 celdas y no aparece en ningun script del proyecto. Ninguno
alcanza el margen: 26,75 celdas valen ~0,003 de macro-F1.

**Estado del campeon.** Sin cambios. V03 intacto. 0 modelos, 0 submissions, 0 consultas a
Kaggle, 0 uso de holdout o test. 0 filas nuevas en `experiments.csv`.


---

## Fase P55 (2026-09-28) - Auditoria estrategica del pipeline final de V03

**Que se respondio.** Si existe un defecto o perdida de informacion en el procedimiento de
V03 que explique parte de la brecha OOF a leaderboard.

```text
P55_PROCEDURAL_GAP              = B  (BUG_FOUND, no mejora)
ORIGINAL_RESULT  corrected_cv  = 0,7772114525   umbrales 0,44/0,51/0,46 de D-004
CORRECTED_RESULT corrected_cv  = 0,7781367700   umbrales anidados de D-016
delta                          = +0,000925        por debajo del margen de 0,003
experimentos nuevos            = 0
submissions nuevas             = 0
```

**La discrepancia.** Los dos numeros de la referencia de V03 vienen de **protocolos de
agrupacion distintos**: los umbrales se optimizaron con grupos D-004 (solo near-duplicate,
que deja cruzar pliegues a un pseudo-autor) y el `corrected_cv` se reporta con D-016
(near-dup UNION pseudo-autor). Al corregir el protocolo en D-016 no se reoptimizaron los
umbrales.

**El sesgo es sistematico.** Los umbrales anidados por pliegue son 0,40-0,41 en HS y
0,48-0,50 en TR, frente a los 0,44 y 0,51 de V03. **V03 queda fuera del rango en los cinco
pliegues sin excepcion**, en ambas etiquetas.

**La magnitud honesta esta bajo el margen.** El delta anidado es **+0,000925**, y el
in-sample +0,003629 no es una mejora esperable porque el umbral se eligio sobre el mismo OOF
que se puntua. Luego la correccion es legal pero **no justifica una submission**.

**La consecuencia operativa.** Los umbrales de V03 son demasiado altos: el OOF produce
0,4572 de positivos en HS frente a 0,4970 que piden los umbrales del propio protocolo. Es
consistente en direccion con un LB 0,0058 por encima del OOF, aunque no lo demuestra.

**Lo que la auditoria no encuentra.** Vectorizador dentro de cada pliegue, verificado en el
codigo. Prevalencias homogeneas entre pliegues. Y el mecanismo no es el tamano de
entrenenamiento: con 80, 90 y 100 por ciento de grupos los umbrales son practicamente
identicos.

**Integridad.** `features/oof_mod03b.npz`, que produjo los umbrales adoptados, **no existe en
disco**, luego los umbrales de V03 no son auditables. El sha256 de la submission si
coincide, y el `corrected_cv` se reproduce exactamente.

**Estado del campeon.** Sin cambios. V03 intacto, sha256 verificado. 0 modelos, 0
submissions, 0 consultas a Kaggle, 0 uso de holdout o test.


---

## Fase P56 (2026-09-28) - Transferencia de umbrales corregida

**Que se respondio.** Si el sesgo de transferencia de umbrales de P55 es un mecanismo real y
reproducible que merece formar parte del pipeline final.

```text
P55_CORRECTED_RESULT_REPRODUCED = TRUE   (0,7781367700 exacto)
T0 historico                          0,7772114525   +0,0000000000
T1 mediana aplicada GLOBALMENTE        0,7798514633   +0,0026400107   4/5 pliegues
umbral por pliegue exterior (HONESTO)   0,7745478730   -0,0026635795
optimismo de la cifra global de T1                  +0,0053035902
net a nivel de celda                                 -11
bootstrap del delta de T1             IC95 [-0,003456, +0,008989]  INCLUYE 0
umbrales de V03 fuera del rango interior            NINGUNA de las tres
THRESHOLD_TRANSFER_BIAS                = NOT_CONFIRMED
STATUS                                 = CLOSED
CASE                                   = B
```

**El hallazgo central es una correccion de P55.** Los umbrales de V03 **caen dentro** del
rango de las cinco estimaciones interiores, en las tres etiquetas. El supuesto sesgo
sistematico de +0,03 fuera de rango en los cinco pliegues era un **artefacto del
procedimiento mas simple de P55**, que tomaba el umbral de probabilidades out-of-fold de
modelos ya entrenados y de ahi una varianza artificialmente baja.

**La cifra honesta es negativa.** El optimismo de aplicar una sola mediana a los cinco
pliegues, usando datos de los cinco para definirla, vale **+0,0053**, el doble de la mejora
aparente de +0,0026. La cifra honesta, con el umbral de cada pliegue exterior aplicado a ese
mismo pliegue, da **-0,0027**.

**Tres senales lo confirman.** El net de celdas es **-11** (43 rescates, 54 danos) y es
negativo en las tres etiquetas. El IC95 del delta **incluye 0**. Y de los cinco pliegues,
cuatro son positivos y el pliegue 3 se lleva **-0,0310**.

**El defecto procedimental de P55 sigue siendo un hecho real**: los umbrales si se adoptaron
bajo el protocolo D-004, mas debil. Lo que se corrige es su **magnitud**, no su existencia.

**Estado del campeon.** Sin cambios. V03 intacto. 1 experimento real (V56-01), **0
submissions**, 0 consultas a Kaggle, 0 uso de holdout o test. No se crea candidato local.


---

## Fase P57 (2026-09-28) - Cierre estrategico y plan final

**Naturaleza: auditoria y sintesis. 0 modelos, 0 entrenamientos, 0 submissions, 0 consultas a
Kaggle, 0 uso de holdout, 0 uso de test.** No se registro ninguna fila nueva en
`experiments.csv`. Decision **D-128**.

**Baseline congelado, verificado al arrancar la fase:** V03, `corrected_cv`
0.7772114525165721, LB 0.78306, sha256
`7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e`, intacto.

**Lo que P57 encontro y que las fases anteriores no tenian formulado:**

1. **Las columnas `f1` y `macro_f1` de `experiments.csv` no son una metrica cada una.** Mezclan
   al menos tres metricas y tres protocolos. El maximo absoluto de `macro_f1` es 0.8743, que es
   macro-6 sobre holdout, no una medida de la competencia. La diferencia macro-6 contra metrica
   del proyecto es +0.04101 sobre el mismo OOF de V03, y el `local_f1` del log para V03 es
   +0.021689 por encima del `corrected_cv` canonico. Cualquier tabla de "mejor CV" construida
   tomando el maximo de esas columnas es falsa. La clasificacion `metrica` / `protocolo` /
   `poblacion` / `anidado` se anade en `reports/p57_final_experiment_inventory.csv`.
2. **Un envio real a Kaggle no estaba en el log.** V33-03 documento el envio ID **56608788** del
   2026-09-27, fichero `toxic_submission_v32_05_url_length.csv`, LB 0.78306, delta
   **exactamente 0** porque el fichero es identico a V03. El total historico de envios es
   **23**, no 22. El log no se modifica.
3. **No hay candidato.** Ningun artefacto cumple a la vez legal, reproducible, sin leakage, no
   rechazado y con mecanismo claro. Los tres con mejor evidencia interna tienen veredicto
   externo medido y contrario.
4. **El patron de fondo: 10 envios de candidatos con mecanismo nuevo, los 10 a la baja.** Y el
   mejor CV comparable del proyecto no es el campeon.
5. **PLS es cierre de tipo B, no A.** Sus numeros nunca fueron evidencia, porque el
   experimento se invalido. El resultado verdadero llego despues como verificacion (V43-02,
   -0.0137423671).
6. **Dos lineas de estructura de etiquetas nunca se ejecutaron** (V13-02, V14-00, 0 de 4
   experimentos): se cerraron por condicion de parada. Ahi no cabe decir "no funciono".

**Jerarquia de magnitudes, sin mezclar metricas:**

```text
best_architectural_CV (macro-6, diagnostica)   0.8743      S08            NO comparable
best_holdout_CV                                0.8465      S08            NO comparable
best_CV protocolo D-004                        0.8039      S05 / v05      NO comparable
best comparable D-016                          0.7904      svd300_rbf     SI, con SELECCION
best anidado D-016                              0.7885      E2-13          SI, sin seleccion
V03 corrected_cv                               0.7772114525165721            SI
best verified LB                               0.78306      v03_lr_union_thr
current champion                               V03          sha256 7d7478f2...
remaining submission budget                    limite diario exacto desconocido; 0 hoy; reserva 3
```

**Submissions:** 37 ficheros en disco, 23 envios historicos, 22 con LB en el log, 1 envio real no
registrado, 14 generados y nunca enviados. 0 de 23 superaron 0.78306.

**Estado del campeon:** sin cambios. V03 intacto.

**Artefactos de la fase:** `reports/p57_final_experiment_inventory.csv` (323 filas),
`p57_family_status.csv` (22 familias), `p57_candidate_matrix.csv` (7 artefactos),
`p57_open_questions.csv` (5 preguntas con mecanismo), `p57_strategic_closure.{md,json}`.
Notebook 208 -> 210 celdas, seccion 53, returncode 0.

**La decision competitiva NO se toma en P57.** Quedan tres salidas, ninguna ejecutada
automaticamente: cerrar con V03, usar un candidato historico, o una ultima investigacion
dirigida. Y la opcion de candidato **no es ejecutable** sin una excepcion explicita, porque no
existe ninguno que cumpla los cinco criterios.


---

## Fase P58 (2026-09-28) - Auditoria final de candidatos existentes

**Naturaleza: auditoria. 0 modelos, 0 entrenamientos, 0 submissions, 0 consultas a Kaggle, 0
uso de holdout, 0 uso de test labels.** **NO se registro ninguna fila en `experiments.csv`**
(siguen siendo 323). Decision **D-129**.

**Pregunta:** existe ya, dentro de los artefactos que hemos producido, algun candidato
suficientemente defendible como para justificar una unica sonda final.

**Respuesta: NO. `FINALIST = NONE`. `CASE A`.**

**El hallazgo que corrige a P57.** P57 escribio que `E2-13-META-ENT-RIDGE` era el unico
candidato con delta honesto anidado positivo y sin veredicto externo, y que solo le faltaba una
verificacion externa. **Es incorrecto.** `v2_meta_oof.npz` declara
`variante = META-BASES-RIDGE`, ridge alpha 3, 44 features, que es exactamente el modelo
enviado como `v08_meta_v2` con LB 0,78163. Luego E2-13 es `v08_meta_v2` mas un canal de
cobertura informativa **0,52 por ciento** frente a un umbral de cierre del 20 por ciento. Su
delta propio es **+0,0011 con 3 de 5 pliegues** y complementariedad negativa. El +0,0113 que
P57 le atribuia es la ganancia de la familia META, refutada 5 de 5 veces. **El cierre por
sparsity es metodologico y esta bien justificado**, y el caso limite es `B_meta36`.

**Descomposicion celular recomputada sin reentrenar**, desde OOF ya en disco, reproduciendo
`reports/v4_candidates.json` al sexto decimal:

```text
candidato                              ccv       delta  plieg  cambian    r    d  neto  acierto  LB
META-BASES-RIDGE (= v08_meta_v2)  0.787357  +0.010146   4/5     246  132  114   +18   0.5366  0.78163
A_cost_sensitive_downweight       0.785416  +0.008205   5/5     173   99   74   +25   0.5723  sin veredicto
B_meta36                          0.782774  +0.005562   4/5     186  115   71   +44   0.6183  sin veredicto
C_meta36_downweight               0.788173  +0.010962   4/5     212  134   78   +56   0.6321  holdout
D_svd300_rbf                      0.790410  +0.013198   5/5     240  136  104   +32   0.5667  0.77938
```

**Los siete criterios.** El que decide es el **criterio 6, artefacto de prediccion completo**:
solo tres de los cinco candidatos tienen CSV, y los tres ya fueron enviados o rechazados por
el holdout.

| candidato | 6 artefacto | 7 no rechazado por LB | FINALISTA |
|---|:-:|:-:|:-:|
| META-BASES-RIDGE | si | **NO** | **NO** |
| A_cost_sensitive_downweight | **NO** | si | **NO** |
| B_meta36 | **NO** | si | **NO** |
| C_meta36_downweight | si | **NO** | **NO** |
| D_svd300_rbf | si | **NO** | **NO** |

**El caso limite: B_meta36**, el unico que pasa el gate D-016 pre-registrado (IC95
[+0,003147, +0,016633], 4 de 5 pliegues, acierto 0,6183, el mas alto de los cinco) y no tiene
**ningun** veredicto externo. No es finalista por tres razones independientes: no existe
artefacto de prediccion y la directiva 12 prohibe crearlo; el holdout esta agotado 5 de 5; y su
hermano `C` es estadisticamente el mismo modelo que `v08_meta_v2`
(`delta_vs_meta_v08 = -4,25e-05`). **Pasa cinco de siete y falla en el sexto por ausencia de
artefacto, no por defecto del modelo.** Es el `quasi-finalista` del proyecto.

**Panorama de los 31 candidatos con corrected_cv D-016 > V03:** 0 con LB positivo, 5 con LB
negativo, 1 rechazado por holdout, 2 sin veredicto externo y sin artefacto, 10 de familias de
embeddings cerradas por inercia con 9 de 9 unidades agotadas, 3 por debajo de su gate, y 1 que
es una medicion de protocolo y no un modelo.

**D_svd300_rbf = CLOSED.** El mejor corrected_cv del proyecto, 5 de 5 pliegues, McNemar
p=0,0452, perdio 0,00368 en el externo. La firma esta en el soporte: 60 a 76 por ciento de
vectores de soporte, la marca de un kernel RBF con gamma=scale sobre 300 dimensiones que
memoriza. El LB ya aporto la evidencia negativa suficiente.

**Integridad:** los 37 CSV de `submissions/` pasan la validacion formal (1154 filas, columnas
`id,HS,TR,AG`, ids y orden identicos a `toxic_test.csv`, binarios, sin nulos, sha256).
`v20_03_reproduccion_v03.csv` es bit a bit identico a V03. **Ningun artefacto se modifica.**

**Presupuesto:** `local_reservation = 3`, `competition_limit = UNKNOWN`, 23 envios
historicos, 0 hoy. No se consulta Kaggle.

**Artefactos:** `reports/p58_final_candidate_audit.{md,csv,json}`. Notebook 210 -> 212 celdas,
seccion 54, returncode 0.

**Formulacion final:** > V03 remains the best verified leaderboard result.

```text
best verified result  !=  proof of global optimum
```

Lo que se SOSTIENE: que **el conjunto de artefactos que hemos producido no contiene** un
candidato que justifique una sonda final. Lo que NO se SOSTIENE: que no exista una mejora
defendible.

**P58 se detiene. No se ha enviado nada, no se ha modificado ningun artefacto, y no se abre
P59.**


---

## Fase P59 (2026-09-28) - Frontier research: geometria local, estructura y data-centric

**Naturaleza: investigacion de frontera. 0 submissions en toda la fase, 0 consultas a
Kaggle, 0 uso de holdout, 0 uso de test labels. V03 congelado e intacto.**
Presupuesto: 12 experimentos principales + 6 de reserva. Decision **D-130** (track A).

**Pregunta de la fase:** ?existe una estructura local, geometrica o data-centric del
train que V03 no esta explotando y que pueda producir una mejora reproducible?

### Gate de novedad ejecutado (P59-00), y su consecuencia

El registro de hipotesis **verifica** sus precedentes contra `experiments.csv` y
aborta si la cifra citada no esta en la fila real. 16 precedentes verificados.
Eso cazo dos transcripciones erroneas propias durante la redaccion.

```text
P59-G  DUPLICATE            cerrado SIN consumir experimento (V8-03 ya lo hizo)
P59-A  PARTIALLY_DUPLICATE  abierto, alcance reducido a lo no visto
P59-B  NOVEL                V03 nunca se ha usado como puntuador local
P59-C  NOVEL                V30-00 cerro por identificadores de dificultad, no por
                            un estimador formal de etiqueta latente
P59-D  NOVEL                con precedente adyacente declarado (V6-03)
P59-E  NOVEL                la propagacion dura ya tiene techo nulo
P59-F  NARROW               solo sobrevive con contexto secuencial real
P59-H  NOVEL                0 coincidencias en 323 filas
P59-I  NOVEL                Huber no es redescendiente y V35-01 cambio el resolutor
```

### P59-A Voronoi: CLOSED con 1 experimento

```text
AUC del score contra error de V03          0,725194   IC95 [0,7061, 0,7446]
AUC del score contra celda no rescatable    0,471798   IC95 [0,3954, 0,5469]
```

La geometria local **predice los errores de V03** con IC95 que excluye 0,5. **No
identifica las 42 celdas no rescatables de P54**: el IC95 incluye 0,5, luego no hay
senal en ninguna direccion. El objetivo fuerte del parrafo 60 queda refutado.

El bloque complementario, que el parrafo 14 autorizaba porque la puerta del parrafo
15 si se supero:

```text
A  V03 crudo                     0,7772114525
B  LR([p_v03])                   0,7743581357   -0,0028533168
C  LR([p_v03, 8 rasgos])         0,7796406441   +0,0024291916
C menos B                        +0,0052825084   con 5 de 5 pliegues
IC95 de C-A                      [-0,008991, +0,000674]   incluye cero
acierto en las celdas que cambian 0,4297   por debajo de 0,5
rescate 55 | damage 73 | neto -18
de las 42 no rescatables, rescata 1
```

H1 y H2 fallan, luego `CLOSED`. El mecanismo del fallo es que el acierto en las
celdas modificadas esta **por debajo del azar**: el delta de metrica es positivo y
el neto es negativo a la vez. Y el detalle que no se pierde: **la geometria aporta
+0,005283 sobre el brazo B con 5 de 5 pliegues**; lo que no hace es mejorar el
punto de operacion de V03, cuya puntuacion cruda ya es mejor que cualquier reajuste.

**Experimentos registrados:** 1 (`P59-02`). `experiments.csv` pasa de 323 a 324.
Notebook 212 -> 216 celdas, secciones 55 y 56, ambas con `returncode = 0` aisladas.

**Formulacion:** > V03 remains the best verified leaderboard result.
<!-- P60 2026-09-28 -->
<!-- P60 2026-09-28 -->
## Estado tras P60

- Rama: proyecto-kaggle-toxic-2026-p60, creada desde v25 que contiene P59.
- Experimentos registrados: 336 en experiments.csv, 336 ids unicos.
- Campeon: V03, corrected_cv 0.7772114525165721, LB 0.78306, sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e, sin cambios.
- P60: 0 candidatos, 1 marginal (pista L), 7 cerradas, 4 cerradas sin experimento.
- Submissions nuevas en P60: 0. Consultas al leaderboard: 0.
- Artefactos de P60: 9 JSON de pista o auditoria, 9 informes .md, 9 CSV, 3 informes consolidados.

<!-- P61 2026-09-28 -->
## P61 (2026-09-28) - Capacidad, diversidad y distribucion de probabilidad

Tres experimentos: `P61-01` LightGBM con Optuna anidado, `P61-02` ensemble con pertenencia
y pesos anidados, `P61-03` segunda particion con control negativo. Ademas la
reconciliacion de los nueve problemas de instrumentacion de P60, la EDA de las
probabilidades OOF de V03 y el inventario con diversidad de los veinte canales del pool de
P50.

```text
P61-01  LightGBM + Optuna anidado    0.7513651502
         familia -0.0302397191
         busqueda +0.0043934168
         total -0.0258463023   2/5   CLOSED
P61-02  ensemble anidado             0.7835159596  +0.0063045070  gate superado
         control de atribucion: el modelo nuevo SUBTRAIA -0,000850
P61-03  segunda particion            test 1 -0.0011527969
         test 2 holdback medio -0.0015908404 con 7/10
         test 3 control negativo +0.0059889238   REJECTED
```

**El hallazgo que mas importa.** El control negativo NO colapsa. Con los tres peores
canales del pool, entre 0,04 y 0,24 por debajo de V03, la ganancia del ensemble sigue
siendo +0.0059889238. Un promediado ponderado que contiene a V03 mejora el OOF casi con
seguridad porque los pesos se eligen para exploitear su residuo sobre las mismas filas, y
el bootstrap no lo detecta porque mide ruido de muestreo y no sesgo de seleccion. Esto
explica de una sola causa el fracaso historico de ENS-01 a ENS-04.

**El diagnostico que queda.** Entre el 60 y el 68 por ciento de los errores de V03 son
errores de alta confianza, con la distancia al umbral POR ENCIMA de la de los aciertos.
El limite de V03 no es de decision ni de calibracion: es de representacion.

```text
no evidence of exploitable remaining model capacity/diversity
V03 remains the best verified leaderboard result.
```
<!-- /P61 2026-09-28 -->

<!-- P62 2026-09-29 -->
## P62 (2026-09-29) - Representacion, estructura linguistica y error de alta confianza

Cinco experimentos, tres pistas cerradas sin gastar nada, y una correccion de lectura.

```text
P62-01  L palabra__POS     -0,009572  1/5  CLOSED
P62-02  M triple lexico   -0,011236  0/5  CLOSED
P62-03  curva de presupuesto   ningun K mejora
P62-04  POS contra lemas   ambos pierden 8/8, el POS ayuda 6/8
P62-05  cobertura min_df=1  -0.0033508936  CLOSED
```

**La interaccion existe y es medible.** La synergy palabra-POS es +0.0657, el
17.41 por ciento de la senal de la palabra. El problema no
es que falte informacion, es que no cabe: repaga el
17.4 por ciento del coste de anadir el canal.

**La cobertura tampoco es la via.** `min_df=2` conserva solo el
28.3 por ciento de las formas superficiales
presentes, y `min_df=1` lo sube al 99.5 por
ciento y resta.

**La correccion.** Yo lei mal mi propia EDA: el perfil de longitud se llamaba
`estructura_corta` pero su cuartil superior es lo largo, luego la delta negativa decia que
V03 es MEJOR en los largos, no peor en los cortos. La comprobacion independiente lo
desmiente: V03 alcanza 0.9639 de AUC en los cortos y su F1
es 0,94 contra 0,72 en los largos. El limite esta en el ORDEN de los textos largos, donde
V28-01 ya habia localizado el techo estructural.

```text
no evidence of exploitable remaining model capacity in the linguistic representation
V03 remains the best verified leaderboard result.
```
<!-- /P62 2026-09-29 -->

<!-- P63 2026-09-29 -->
## P63 (2026-09-29) - Representacion a nivel de registro, estructura del dataset y composicion

Seis experimentos, dos auditorias y un dataset auxiliar. Ninguno supera a V03, y lo
interesante no son los negativos sino que cada uno lleva un mecanismo MEDIDO.

```text
P63-00  ID forensics      D. STRUCTURED_IDENTIFIER, confianza alta, 0 experimentos
P63-02  id predictivo     0,419136 contra 0,424038 de un control de igual dimension
P63-03  relacional        el bloque esta 92,92 por ciento vacio
P63-04  composicion       brecha especifica +0,0083528268, y +0,1055 en solitario
P63-05  texto largo       la brecha en MUY_LARGA es -0,009437
P63-06  integracion       brecha +0,0085891400, MARGINAL
P63-07  residuo           los bloques restan al margen, luego son redundantes con el
                         margen y no con los n-gramas
```

**El id no es un contador.** V15-00 lo afirmo sin medirlo. Train y test son dos muestras
disjuntas del mismo bloque contiguo, con interseccion 0 y union
2885 sobre 4999 posiciones. Es un indice de
coleccion, y su contenido es la posicion en un ensamblado, que no correlaciona con la
etiqueta.

**La composicion en ratios porta senal y no cabe.** Es un resultado distinto del de P62, no
uno contrario: un canal disperso de varios cientos de columnas compite por el presupuesto
L2 de un espacio de 21000 dimensiones, y 42 proporciones acotadas no compiten. Y aun asi
no mejora a V03, porque lo que aportan ya esta implicito en la confianza de V03.

**La hipotesis mas antigua del repositorio queda refutada.** V30-01 escribio que el efecto de
la URL y la mencion cambia de signo con la longitud, y nunca gasto un experimento. El bloque
de posiciones es PEOR que un bloque aleatorio del mismo ancho precisamente en la region
muy larga, con -0.009437. Refutada en su prediccion, no solo
no confirmada.

**El limite que queda es de orden, no de decision.** El margen hasta el techo moviendo solo
el corte va de +0.0000 en la region corta a +0.0890 en la
muy larga, y en la region CORTA es exactamente +0.0000: el umbral de V03 ya
es el optimo ahi sobre las tres etiquetas.

**Dos correcciones cambiaron la direccion de una conclusion** y tienen fila propia en el
registro. El confound de escala de P63-04 paso el delta de -0,1899987838 a
-0,0102687019 y cambio el signo de la brecha. El techo de P63-06 paso el margen de MUY_LARGA
de +0,1589 a +0,0890.

```text
no evidence of exploitable remaining model capacity in the record-level and
aggregate-composition dimensions of this representation

V03 remains the best verified leaderboard result.
```
<!-- /P63 2026-09-29 -->

<!-- P64 2026-09-29 -->
## P64 (2026-09-29) - Representacion del tweet largo y distribucion de la evidencia interna

Cuatro experimentos, dos auditorias, trece figuras. Ninguno supera a V03, y la respuesta
tiene dos mitades que no se pueden separar.

**La estructura interna es REAL.** P59-07 habia explicado su fracaso por cobertura, no por
concepto: con ventana de 32 tokens solo el
9.6 por ciento de las filas tiene mas
de una unidad, mientras que la segmentacion por puntuacion llega al
50.6 por ciento. Un factor de
5.29 en las celdas alcanzables, que no es una variacion
de parametro.

**Y el leave-one-out, que P59 nunca computo, muestra el mecanismo.** El 80,88 por ciento de
los documentos multi segmento de la region MUY_LARGA tienen segmentos que empujan la decision
en signos opuestos, la entropia interna sube de 0,1048 a 0,4565 con la longitud, y la
concentracion cae de 0,7554 a 0,6170. En los tweets largos la decision no descansa en un
segmento dominante sino en un reparto contradictorio. Eso es exactamente lo que una bolsa de
n-gramas aplana.

**Y no aporta.** El bloque de evidencia queda -0.0140959650 frente al
margen de V03, y el leave-one-out -0.0004650203, que es casi nula.
Tampoco gana en la region larga, que es donde P63 midio el margen, y el margen tampoco mejora
ahi, luego el problema de MUY_LARGA no es de representacion ni de confianza.

**El perfil segmental es longitud disfrazada.** El mejor rasgo da
0.589847 de AUC frente a
0.600506 de contar tokens, luego queda
cerrado como rasgo.

**La razon de fondo.** El margen de V03 solo ya explica sus propios errores con AUC 0,770875
contra el acierto, y el mejor rasgo de evidencia interna llega a 0,605302. La estructura no
falta: esta dentro de la confianza de V03.

```text
the internal evidence distribution of long tweets is real, measured and reproducible,
and V03 already carries it inside its own confidence

V03 remains the best verified leaderboard result.
```
<!-- /P64 2026-09-29 -->

<!-- P65 2026-09-29 -->
## P65 (2026-09-29) - Representacion de largo alcance, alcance de operadores y jerarquia

Cuatro experimentos y una auditoria, trece figuras. Ninguno supera a V03, y la respuesta tiene
una forma que no estaba prevista.

**LA RESPUESTA: la senal existe y la adicion la rechaza.** Cinco candidatos de este proyecto
baten a los dos controles, el de capacidad y el de la confianza de V03, y ninguno mejora:

| candidato | contra el control de anchura | contra el margen de V03 | delta vs V03 |
|---|---:|---:|---:|
| P64-07 alcance de negacion | +0,0201566146 | +0,0088975231 | -0,0021826774 |
| P65-01 word (1,3) | +0.0596671770 | +0.0071281317 | -0,0039520688 |
| P65-03 cohesion documental | +0.0090835614 | +0.0083711943 | -0,0027090062 |

Lo que falta no es informacion: es sitio. Unas cientos de columnas compitiendo por el
presupuesto L2 de un espacio de veinte mil, y lo que la columna nueva necesita decir es mas
pequeno que lo que cuesta decirlo. Eso es una afirmacion sobre la ARQUITECTURA de la adicion,
y es falsable: predice que la misma evidencia fundida en el espacio de probabilidad, y no
concatenada en el de caracteristicas, sobrevive.

**EL GAP ANALYSIS CERRO MAS DE LO QUE ABRIO.** 6 abiertas y
10 cerradas, leidas en el codigo y no en el registro. Los skip-grams ya los
habia medido V6-05, con un vocabulario un 83 por ciento mayor y un resultado neutro. La
negacion y la cita las habian medido E2-07 y E2-08, y despues P64-06 y P64-07.

**EL CONTROL DE IMPLEMENTACION REPRODUCE A V6-05.** El rango de hueco 1 a 2, que es
exactamente el suyo, da
-0.0010515908 frente al -0,0001 que el
registro. Luego la implementacion de los pares con hueco es la misma, y los rangos que V6-05 no
alcanzo, de tres a cinco y de seis a diez tokens, tampoco aportan.

**Y LA LONGITUD DEL TWEET NO ES EL MECANISMO.** Un especialista ajustado en LARGA y MUY_LARGA
es BIT A BIT el mismo numero que un reentrenado sobre las mismas
714 filas sin ningun criterio de longitud:
+0.0000000000. Un negativo asi no es *faltan datos*,
que seria una excusa valida: dice que la especializacion no es el mecanismo, punto. Y el
especialista gana
+0.0212 en MUY_LARGA y
pierde -0.0257 en LARGA, que tiene
el triple de filas, luego hay una mejora local real que la metrica primaria rechaza.

```text
the information about long-range relations is present, measurable and not what V03 already
knows, and none of it fits through concatenation

the length of a tweet is not the mechanism either: a model retrained on the long rows with no
length criterion is bit for bit the specialist

V03 remains the best verified leaderboard result.
```
<!-- /P65 2026-09-29 -->

<!-- P66 2026-09-29 -->
## P66 (2026-09-29) - Late fusion: la evidencia de largo alcance en el espacio de scores

Dos experimentos y dos auditorias, trece figuras. Y es la primera vez que la senal de largo
alcance mueve el numero.

**LA RESPUESTA A LA PREGUNTA DE LA FASE: SI, Y ES ATRIBUIBLE AL CANDIDATO.**

| fusion | delta vs V03 | pliegues | brecha de atribucion |
|---|---:|---:|---:|
| MI_arbol + residual anidada | +0.0026743167 | 4/5 | +0.0041133588 |
| MI_arbol + residual con score barajado | -0.0014390421 | | |

La brecha de atribucion es la cifra que manda: el mismo mecanismo con el score del candidato
**barajado** pierde casi todo.

**Y MI_arbol es el unico que funciona porque es NO LINEAL.** Las correlaciones entre z_V03 y
z_C de los siete candidatos van de +0,9294 a +0,9988: son casi la misma puntuacion. El AUC del
residuo deja a cinco de siete por debajo de 0,5, luego su parte unica destruye informacion en
vez de anadirla. El de MI_arbol es el unico por encima del azar, y su modelo es el unico no
lineal de la lista. Si el error de V03 fuera una funcion lineal del error de MI_arbol, la
regresion residual no dejaria nada.

**EL MATIZ, Y VA PEGADO AL RESULTADO.** De las
53 celdas que la fusion rescata, solo
6 son errores de
alta confianza y las demas estan pegadas al umbral, luego es sobre todo **recalibracion**. Y el
bootstrap por 1415 grupos da IC95
[-0.0044311634, +0.0102108745], que
**incluye el cero**. Veredicto PROMISING, no FINALIST.

**EL INVENTARIO FUE EL PROBLEMA REAL.** Ningun script de P64 y P65 escribio sus
probabilidades a disco, y `features/` esta en `.gitignore`. Los scores se regeneraron y se
verificaron contra el numero de cada fase: 5 de 7 reproducen con diferencia de 1e-11, uno es
no verificable y uno tiene una diferencia declarada. El regenerador **se paro** cuando dos no
cuadraron.

**Y UNA CORRECCION QUE CAMBIO EL VEREDICTO DEL MEJOR RESULTADO.** `blend_prob` usaba siempre
la probabilidad REAL del candidato y nunca la barajada, luego su control de barajado salia
identico al real POR CONSTRUCCION, la brecha daba cero falso, y el mejor resultado de la fase
queda INVALIDATED por un motivo que no era el suyo. Corregido, ese mismo resultado pasa a
+0,0033 con brecha +0,0093.

```text
adding columns to a twenty thousand dimension space does not fit, and correcting a three
column score does, for the SAME evidence and with the SAME base model

the only measured exception is MI_arbol, and its distinguishing feature is that the model
producing it is NON LINEAR

and the caveat: it is mostly recalibration, because 6 of 53 rescues are high confidence and
the IC95 includes zero

V03 remains the best verified leaderboard result.
```
<!-- /P66 2026-09-29 -->

<!-- P67 2026-09-29 -->
## P67 (2026-09-29): el residuo de MI_arbol es MIXED, no una senal transferible

- Categoria del mecanismo: **MIXED_CON_ATRIBUCION_PARCIAL**.
- Reentreno anidado de un escalar por etiqueta: `-0.0029731513`, es decir la calibracion **no** explica el efecto.
- AUC del residuo de MI_arbol contra la etiqueta **fuera** de la banda: `0.579597`, sobre 0,5. Los otros seis candidatos caen por debajo.
- Sustitucion cruzada: `4 de 7` candidatos dan positivo con la misma operacion.
- Enriquecimiento de los rescates: `1.338302`.
- Efecto en celdas: `4.01` de 1483 filas.
- 1 de 12 tracks principales ejecutados, 0 de 4 reservas.
- 0 nuevas representaciones, 0 submissions, 0 Kaggle, 0 holdout, 0 test.
- Figuras: 6 en `figuras/p67/`.

> V03 remains the best verified leaderboard result.
<!-- /P67 2026-09-29 -->
\n<!-- POST-P67 2026-09-29 -->\n## Post-P67 (2026-09-29): el gap analysis y una linea nueva cerrada

- Poblacion **B = 215 celdas** donde V03 falla y alguien acierta. El 28,4 por ciento de sus
  errores. **122 exclusivas** de un solo candidato.
- Once familias clasificadas: 6 `CLOSED`, 2 `INVALIDATED`, 1 `DUPLICATE`, 1 `COMPLEMENTARY`,
  1 `PARTIALLY_OPEN` (MI_arbol).
- `P68-H1` (enrutado por estructura del ejemplo): **INVALIDATED**, delta
  -0.0018992302, 1 de 5 pliegues.
- El techo era `+0,0390871336`. El enrutador no lo captura porque la estructura predice
  dificultad (AUC 0,6080), no ganador (AUC 0,5491).
- 1 experimento principal, 2 controles, 1 ablacion. Sin grid.

> V03 remains the best verified leaderboard result.
<!-- /POST-P67 2026-09-29 -->\n
<!-- CIERRE 2026-09-29 -->
### P68-H2 y el cierre (2026-09-29)

- `P68-H2 (rescue re-weighting)`: **DUPLICATE**. El mecanismo (sample weighting) esta medido
  nueve veces; el indicador propuesto es un subconjunto estricto del de V4-B03, y en la
  direccion que salio mal. **Coste: 0 experimentos.**
- Los ocho ejes del encargo tienen cita historica. **0 sin recorrer.**
- **Fase de investigacion cerrada**, con la condicion del encargo cumplida.
- 122 rescates exclusivos, 77 de MI_arbol (46 en LARGA). Techo oracle de las celdas
  delegables: +0,0391, no alcanzado por cuatro mecanismos distintos.
<!-- /CIERRE 2026-09-29 -->

<!-- DOWNWEIGHT-AUDIT 2026-09-29 -->
### Auditoria de A_cost_sensitive_downweight (2026-09-29)

- **REPRODUCIBLE, nivel R3.** `0,7854160387` reproducido exacto a `3,2e-11`; delta
  `+0,0082045862`; cinco deltas por pliegue identicos; 99 rescates y 74 danos identicos.
- **Criterio 6 = `A_falta_de_artefacto`**: faltaba el CSV de submission, y la directiva 12
  prohibia crearlo. Fallo procedimental, no experimental.
- **Segunda particion CONFIRMED**: delta `+0,0105253130` con 5 de 5 pliegues.
- **ESTADO FINAL: MARGINAL.** Bootstrap por grupos `[-0,000691, +0,011818]` incluye el cero.
  Con 1415 grupos sobre 1483 filas, el remuestreo por filas subestimaba la incertidumbre.
- Correccion documental: el mecanismo **no es un meta-modelo**; P58 lo etiquetaba mal.
<!-- /DOWNWEIGHT-AUDIT 2026-09-29 -->
