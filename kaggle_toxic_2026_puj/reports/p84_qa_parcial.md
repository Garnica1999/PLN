# QA parcial P84: P84-01, P84-02, P84-03 y P84-05

- Fecha: 2026-10-02
- Alcance: experimentos cerrados P84-01 (DL-01 corregido), P84-02 (fastText
  supervisado), P84-03 (ROS equivalencia) y P84-05 (EM + shrinkage).
- Fuera de alcance: P84-04 (SGNS) sigue en ejecucion en segundo plano. No se audita y
  no se interrumpe.
- Metodo: verificacion Prove-It sobre artefactos existentes. No se re-ejecutan
  entrenamientos ni se modifica codigo o datos.
- Evidencia: reports/p84_0*.json/.md, features/p84_0*.npz y
  features/oof_v03_final.npz, reports/p84_0*_parciales/, experiments.csv,
  docs/p84_preregistro.md, RUNLOG.md, DECISIONS.md,
  /home/carlos/.opencode/plan/kaggle_toxic_exploracion_modelos_f1_080.md y
  tools/verificar_caracteres.py.

## 1. Veredicto por experimento

| Experimento | Artefactos | Numeros | Sello | Veredicto |
|---|---|---|---|---|
| P84-01 | json+md (reports/p84_01_dl01.*), npz 65865 B, 15 parciales, 15 checkpoints mlp en modelos/p84/, fila 452 del CSV una sola vez | F1 0.758301; delta -0.0189104006; 1/5 pliegues; IC95 [-0.033910,-0.004150]; cierre CERRAR DL-01 COMO ARTEFACTO DE PROTOCOLO; CSV==JSON | Gate0 identidad V03 delta 0.0 (tol 1e-12); dev 1483 / holdout 0 / test 0; hashes Gate0 OK; sin lectura de toxic_test | PASA |
| P84-02 | json+md (reports/p84_02_fasttext.*), npz 47690 B, 10 parciales, fila 453 una sola vez | principal anidada 0.583398, delta -0.1938133867, 0/5, IC95 [-0.217172,-0.171220]; control 0.737771, delta -0.039441, 0/5; ensemble_gain 0.0 (vacio); cierre CERRAR fastText; CSV==JSON | Gate0 identidad V03 delta 0.0 (tol 1e-12); dev 1483 / holdout 0 / test 0; hashes Gate0 + binario fastText 59c51cc5 OK | PASA |
| P84-03 | json+md (reports/p84_03_ros.*), npz 142080 B (A/B1/B2/B3), 5 parciales + humo, fila 454 una sola vez | B1 delta congelado +0.0002656445; refutacion parcial en probabilidad max 7.188e-02 > 1e-3, \|dF1\| 2.664e-4 <= 1e-3; CLOSED BY EQUIVALENCE; A=V03 delta 0.0 | Gate0 identidad V03 completa 0.0 (tol 1e-12); dev 1483 / holdout 0 / test 0; hashes Gate0 OK | PASA |
| P84-05 | json+md (reports/p84_05_em.*), sin npz propio (simula sobre features/oof_v03_final.npz), 5 parciales + humo, fila 455 una sola vez | delta medio -0.008394471035915544 (umbral +0.002); 2/5 pliegues; IC95 [-0.015762757886,-0.001231010501]; guardarrail AG 0.014835 > 0.005; cierre familia EM/threshold; CSV==JSON en delta y gate | Gate0 identidad V03 corrected_cv 0.7772114525165721 exacta; dev 1483 / holdout 0 / test 0; hashes Gate0 OK | PASA CON AVISOS |

## 2. Checks ejecutados (evidencia)

- CSV: cabecera de 23 columnas; 454 filas de datos; 0 filas con != 23 columnas; sin
  ningun experiment_id duplicado; P84-01/02/03/05 en lineas 452/453/454/455, una vez
  cada una.
- npz: features/p84_01_oof_mlp.npz, p84_02_oof_fasttext.npz y p84_03_oof_ros.npz con
  1483 filas y 3 etiquetas; sus ids, y_true y pliegues son identicos a
  features/oof_v03_final.npz (1483 ids unicos) -> mismas filas de desarrollo que V03,
  sin holdout.
- Numeros cruzados CSV vs JSON: P84-01 delta_global_0_5 -0.0189104006 == gate.delta_primario;
  P84-02 delta_global anidada -0.1938133867 y control -0.039441 (ruta
  lecturas.control.anidada.comparacion.delta_global); P84-03 B1
  equivalencia_B1_A.delta_f1_B1_A_frozen 0.0002656445 y etiqueta_cierre; P84-05
  metricas.delta_medio -0.008394471035915544 y gate.pasa False. Todos coinciden.
- Sellos: los 4 JSON declaran contadores dev 1483 / holdout 0 / test 0 /
  toxic_test_abierto False e identidad V03 exacta (max 0.0, tolerancia 1e-12).
- Hashes Gate 0 re-verificados con sha256sum real: toxic_train.csv 3d40e734...,
  oof_v03_final.npz 559265816..., grupos_near_duplicados.csv afabf9ae..., binario
  fastText 59c51cc5...; coinciden con docs/p84_preregistro.md lineas 11-15.
- Hashes Enmiendas 2-5 (docs/p84_preregistro.md lineas 103, 129, 149 y 173) vs sha256
  real: p84_02 bd06478f94a4...802d55, p84_03 13e500f907...f84e481, p84_04
  26ca470f...66e5975 y p84_05 9c6d62c7...f7694251. Los 4 coinciden.
- toxic_test.csv: grep en experiments/p84_0*.py; las menciones son solo comentarios y
  la clave toxic_test_abierto. Ninguna ruta de apertura o lectura.
- RUNLOG.md lineas 5057-5083: entradas 2026-10-02 para P84-01 (corrida 2, con
  previo_fix declarado), P84-02, P84-03 y P84-05.
- DECISIONS.md lineas 10044-10130: D-281, D-282, D-283 y D-284 completas.
- Plan lineas 226-305: bitacora EXP-01, EXP-02, EXP-03 y EXP-05 como COMPLETADAS;
  EXP-04 EN CURSO (coherente con el alcance de este QA).
- tools/verificar_caracteres.py: exit 0; salida "OK: 0 caracteres no permitidos".
- Sin submissions: find submissions -newermt 2026-10-02 solo devuelve
  leaderboard_log.csv (05:26, anterior al inicio de P84 a las 12:25); ningun script
  P84 escribe en submissions/.
- Scripts nuevos y cierres: p84_0*.py figuran como no rastreados en git (nuevos);
  mtime de cada script < mtime de su informe (12:25<12:32, 12:55<13:01, 13:29<13:31,
  16:36<16:38); ningun script de experimento previo fue modificado en la ventana P84
  (12:25-16:38). Los cambios de p69/p82/holdout_evaluaciones_v2.csv son de 05:26-05:52,
  fase de auditoria previa, con diffs basicos de 46 y 49 lineas.

## 3. Discrepancias encontradas

1. [Menor] P84-05: el F1 0.772169 de la columna f1 del CSV (result.f1_candidato_rho1 =
   0.7721687479249395) no aparece como escalar en reports/p84_05_em.json; el JSON si
   trae F1 por etiqueta y rho, precision/recall por regla y las componentes EM. La
   trazabilidad CSV -> JSON es parcial solo en ese campo.
2. [Menor, por diseno] P84-05 no genera npz propio: simula sobre
   features/oof_v03_final.npz (oof_npz = None). Declarado en JSON y MD.
3. [Menor, global] La cadena "toxic_test.csv" aparece literal en comentarios/notas y en
   la clave toxic_test_abierto de los 4 scripts. No hay ninguna apertura o lectura,
   pero el requisito de cero menciones literales no se cumple al pie de la letra.
4. [Menor] La fila P84-05 deja vacias las columnas macro_f1, weighted_f1, precision y
   recall del CSV; el JSON si incluye precision/recall por regla. Es consistente con
   la metrica F1 de la fase, pero queda incompleta frente al resto de filas P84.
5. [Informativo] git status no esta limpio (p69/p82, reportes y
   holdout_evaluaciones_v2.csv modificados) por la fase de auditoria previa; no hay
   ediciones de experimentos cerrados dentro de la ventana P84.
6. [Informativo] P84-01 conserva previo_fix (reports/p84_01_dl01_previo_fix.json y
   features/p84_01_oof_mlp_previo_fix.npz) como evidencia pre-fix declarada en RUNLOG
   5057-5063; no sustituyen a los finales.
7. [Informativo] P84-02: ensemble_gain principal +0.000000 con IC [0,0] es un intervalo
   degenerado por el brazo constante en 0.5; RUNLOG lo marca como "vacio" y no debe
   leerse como no-complementariedad demostrada.

## 4. Definition of Done de la fase

| Criterio | Resultado |
|---|---|
| Resultado (positivo o negativo) registrado | PASA: 4/4 en CSV, JSON/MD, RUNLOG y DECISIONS |
| Sin fuga | PASA (base): ajustes sobre outer_train / OOF congelado; identidad V03 exacta re-verificada |
| Sin uso de test ni holdout | PASA: contadores 1483/0/0, npz solo filas de desarrollo, toxic_test nunca abierto |
| Sin submissions | PASA |
| Script nuevo por experimento sin editar cerrados | PASA |

## 5. Veredicto global

PASA CON AVISOS. Los cuatro experimentos cerrados tienen artefactos completos,
numeros coherentes entre informes y CSV, sellos Gate 0 exactos, hashes de enmiendas
correctos y trazabilidad completa. Los avisos son de forma (menciones literales,
campos CSV vacios en P84-05, F1 escalar de P84-05 solo en CSV) y no comprometen los
resultados ni el no-uso de test/holdout. P84-04 queda pendiente de QA al cerrar.
