# Revision adversarial del núcleo metodológico - P67

Fecha: 2026-10-02. Revisor: ml-reviewer (subagente, sesgo a desprobar).
Entorno: `/home/carlos/venv/bin/python` (ruff 0.16.3). Solo lectura salvo este informe.
Alcance: `src/evaluation.py`, `src/splits.py`, `src/data.py`, `experiments/14_oof_corregido_v2.py`,
`experiments/04_refit_submissions_v02_v03.py`, `config/holdout_v2.json`,
`docs/competition_rules.md` y los call sites de `metricas_multietiqueta` en `src/` y
`experiments/`. No se leyeron filas del holdout ni del test (solo ids, cabeceras y agregados).
No se toco V03 ni sus artefactos. git status: arbol limpio.

## A1. "corrected_cv = 0.7772114525165721 se recomputa desde features/oof_v03_final.npz con los umbrales congelados 0.44/0.51/0.46 y es f1_media_columnas"

Veredicto: **CONFIRMADO**.

- Recomputo ejecutado con el venv: `proba >= [0.44, 0.51, 0.46]` -> `metricas_multietiqueta`
  da `f1_media_columnas = 0.7772114525165721`, identico bit a bit al guardado (`corrected_cv` del npz)
  y a la afirmacion (diferencia 0.0).
- `f1_por_etiqueta` guardado = `[0.7616892911010558, 0.8157303370786517, 0.7542147293700089]`
  (HS, TR, AG) coincide con el recomputo.
- `pred` guardado es exactamente `proba >= umbrales` (0 discrepancias); `umbrales` guardados =
  `[0.44, 0.51, 0.46]`; `etiquetas` = HS, TR, AG.
- Contexto: a 0.5 el mismo OOF da 0.771778476256122, consistente con D-016 (0.7772 con umbrales).

Evidencia: `features/oof_v03_final.npz`; `experiments/14_oof_corregido_v2.py:70,236-244`;
`CHAMPION.md` (CV local 0.7772114525165721).

## A2. "Los pliegues son StratifiedGroupKFold con grupos fusionados (near-duplicado UNION pseudo-autor) y ningun grupo cruza pliegues; el holdout de 248 filas queda fuera del desarrollo"

Veredicto: **CONFIRMADO**.

- Hash del CSV de particion: `e5b62e5f22c4c63a8c43df4019969504d44c0baa5b503c19ba3560032f5ee6cd`,
  identico al de `config/holdout_v2.json:32`.
- La particion recomputada con `src.splits.holdout_v2` reproduce exactamente los ids guardados:
  dev = 1483, holdout = 248, solape = 0; distribuciones por combinacion identicas a las del config
  (dev: 813/22/79/124/24/421; holdout: 136/4/13/21/4/70).
- Grupos: 1645 en total (igual a `n_grupos_total`). Fusion real comprobada sin excepciones:
  0 violaciones de "mismo near-duplicado => mismo grupo fusionado" (1727 grupos near-dup) y
  0 de "mismo pseudo-autor => mismo grupo" (740 autores). Grupo mayor: 14 filas.
- Frontera dev/holdout: 1415 grupos dev, 230 holdout, interseccion 0; 0 ids del holdout dentro de
  `train_dev`.
- 5 pliegues corregidos (StratifiedGroupKFold(5, shuffle=True, seed=42), `src/splits.py:153-177`):
  interseccion de grupos train/validacion = 0 en los 5; el holdout nunca aparece como validación.

Evidencia: `src/splits.py:71-113,130-150,153-196`; `config/holdout_v2.json`; ejecución propia.

## A3. "metricas_multietiqueta recibe y_pred BINARIO en todos los call sites reales; no hay ningun sitio que le pase probabilidades. El caso histórico de P83 (línea 6674) ya se corrigio"

Veredicto: **CONFIRMADO** (clasificacion estatica; ver limitacion).

- Analisis AST de `src/`, `experiments/` y `audit/`: 282 llamadas a `metricas_multietiqueta`
  (excluida la definicion de `src/evaluation.py` y la copia historica de `02_baseline`).
  Clasificacion: 240 por nombre de variable tipo `pred*`, 26 con `astype(int)`, 28 con expresion
  binaria/`np.column_stack`, 26 via helper, 2 indirectas y 14 ambiguas.
- Las 14 ambiguas se inspeccionaron una a una y todas reciben binario:
  `12_control_pseudo_autores.py:175,207,211` (`oof` = `np.zeros_like` rellenado con
  `predict_proba >= 0.5`), `06_error_analysis.py:396,608-610` (`Escenario.prediccion` =
  `(proba >= umbrales).astype(int)`), `v45_01_joint_label_structure.py:151,290-302,418-419`
  (`p_oof = datos["pred"]` int64), `v8_01_ensamble_folds.py:179-188` (`aplicar_umbrales` umbraliza),
  `v8_08_corrector_margenes.py:307` (`oof_promedio = aplicar_umbrales(...)`),
  `v10_03_arnes_evaluacion.py:287` y `v10_05:292` (`predicciones_de_referencia` = umbrales V03),
  `19_sintaxis_cita_negacion_v2.py:1257` (`P2.predecir` = `16_targeting_v2.py:1550`, umbraliza).
- Helpers usados como `y_pred` verificados en su cuerpo: `predecir_con_umbrales`,
  `predecir_con_umbral`, `predecir_binario`, `aplicar_umbrales`, `binarizar_*` todos aplican
  comparación o `astype(int)`.
- P83: el caso histórico (línea 6674 del script anterior) esta corregido. El script actual
  (`experiments/p83_01_bayesiano.py:7202-7205`) llama a `binarizar_con_umbrales_congelados`
  (definida en 2801-2813) con los umbrales leidos del artefacto congelado, y
  `docs/p83_AVISO_NO_EJECUTABLE.md:23-25,126-127` lo declara "CORREGIDO y verificado" reproduciendo
  0.7772114525165721. La llamada actual a `metricas_multietiqueta` (línea 7205) recibe binario.
- Limitacion: 159 sitios se clasificaron por nombre (`pred`, `prediccion`, ...), no por ejecución.
  El modo de fallo conocido (matriz de probabilidades) haria que sklearn lanzara
  `ValueError: Classification metrics can't handle a mix of binary and continuous targets`
  (citado en el propio P83), por lo que no podria inflar métricas en silencio.

Evidencia: análisis AST propio (282 sitios) + inspeccion manual de los 14 ambiguos;
`docs/p83_AVISO_NO_EJECUTABLE.md`.

## A4. "Los umbrales de V03 están congelados; optimizar umbrales sobre el mismo OOF infla la metrica en +0.0036 y ningun script vigente presenta esa reoptimizacion como mejora"

Veredicto: **CONFIRMADO con matiz documentado**.

- Rejilla propia 0.20-0.80 paso 0.01 sobre el OOF canónico completo: optimos HS 0.41, TR 0.48,
  AG 0.45 -> `f1_media_columnas = 0.7808400797718141`; delta contra los congelados
  0.44/0.51/0.46 = **+0.0036286** (contra 0.5 = +0.0090616). La cifra +0.0036 se reproduce.
- Ningun script vigente la presenta como mejora: `04_refit_submissions_v02_v03.py:572-577` la
  etiqueta como "cota optimista" y añade verificacion anidada (+0.0014, 2/5 pliegues);
  `09_optuna_tfidf_lr.py:1128` avisa que umbral y metrica comparten OOF; `v10_04_calibracion_fuerte.py:6`
  evita explicitamente esa comparación; `v10_06_umbrales_estables.py:6-11` mide +0.000131 y corrige
  la cifra; `reports/v7_umbral_anidado.md:42-52` documenta que +0.0036 mezcla coste del protocolo
  (control congelado -0.0081) y que el efecto honesto del umbral es -0.0019; `v57_00_sintesis_estrategica.py:563`
  y D-072 lo corrigen por escrito.
- Matiz: +0.0036 es el delta crudo pedido, pero el propio proyecto documenta que no es el efecto
  causal del umbral. La afirmacion es cierta como número y ya esta matizada en los informes.

Evidencia: recomputo propio; `experiments/04:572-577`; `v10_06:6-11,428-481`;
`reports/v7_umbral_anidado.md`; `v10_04:6`; `v57_00:563`.

## A5. "El holdout v2 esta agotado (5 de 5 evaluaciones) y ningun script lo usa para entrenar o seleccionar"

Veredicto: **REFUTADO (parcial)**: el registro tiene 5 filas, pero hay al menos un uso adicional
no registrado y usos literales para entrenar/gatear submissions.

- El registro `reports/holdout_evaluaciones_v2.csv` tiene exactamente 5 evaluaciones
  (P7-01-V03, P7-02-META, FIN-02, FIN-04, P8-C) -> la parte "5 de 5" es correcta para el registro.
- Contraejemplo 1 (severidad ALTA): `experiments/v33_01_validacion.py:307-316` evalua el MISMO
  holdout v2 (248 filas de `holdout_v2_ids.csv`) con V03 y V32-05 y lo usa para
  "validación completa de V32-05 antes de la submission". `reports/v33_01_validacion.json`
  (fecha 2026-09-27) da V03 = 0.8284684313523535, identico a P7-01, y V32-05 = 0.8279491147252713.
  No aparece en el registro ni en `target_encoding_audit_v2.md` (que afirma "5 de 5") ni en RUNLOG.
  Cronologia: las 5 evaluaciones se cerraron el 2026-09-25 y el 2026-09-26 V10-01 ya declaro el
  holdout agotado al crear holdout2 (`config/holdout2.json`); v33 corre después (2026-09-27).
- Contraejemplo 2 (severidad MEDIA): `experiments/18_p7_holdout_y_submission_v08.py:1107-1111`
  reajusta las bases del meta "con el train completo (1731 filas)" = desarrollo + holdout para la
  submission v08; `25_submissions_candidatos.py:35,915-919` reproduce el criterio. Esta documentado
  y no reevalua el holdout con esos modelos, pero es literalmente entrenar con las 248 filas.
- La seleccion del candidato final v4 si fue en desarrollo (criterios D-016 de nivel, pliegues e
  IC95; `38_refit_final_v4.py:5-9`); el gate GO/NO-GO de P7 también se decidio con el holdout de
  forma pre-registrada (evaluación 2 de las 5). Esa parte del protocolo es deliberada.

Evidencia: `reports/holdout_evaluaciones_v2.csv`; `reports/v33_01_validacion.json`;
`reports/p7_holdout_resultados.json`; `config/holdout2.json:4`; `experiments/18:1107-1111`;
`experiments/25:35`; `experiments/38:5-9`.

## A6. "El parser lee train utf-8 con separador ';', test cp1252 con ';' y submission con ','"

Veredicto: **CONFIRMADO**.

- Ejecucion con el venv: `toxic_train.csv` 205.417 bytes, decodifica utf-8, cabecera
  `id;text;HS;TR;AG` (5 campos); `toxic_test.csv` 127.006 bytes, NO decodifica utf-8 (byte 0xBF en
  posicion 191) y cae a cp1252, cabecera `id;text` (2 campos, ';'); `submission.csv` 13.860 bytes,
  utf-8, cabecera `id,HS,TR,AG` (4 campos, ',').
- `read_train`/`read_test`/`read_submission` usan exactamente ';', ';' y ',' (`src/data.py:140,155,167`)
  y `decode_bytes` (`src/data.py:81-93`) implementa la deteccion utf-8 -> cp1252.
- Lecturas reales sin error: train (1731, 5), test (1154, 2), submission (1154, 4).

Evidencia: salida de ejecución propia; `src/data.py:9-18,81-93,140,155,167`.

## A7. "El escaner estático no encontro aserciones vacias (D2=0) en 376 scripts"

Veredicto: **CONFIRMADO** (con alcance acotado).

- `auditoria/escaner_resumen.md:15-16` declara 376 archivos `.py` barridos y
  `D2_asercion_vacia = 0` en todas las severidades; el único `D2` del CSV de hallazgos es una
  subcadena (`D3_rama_muerta`), no un hallazgo D2.
- Comprobacion independiente propia (AST, patrones mas amplios que el escaner: `assert True`,
  `assert not False`, `x == x`, `x <= x`, `x >= x`, `x is x`, literales iguales/desiguales):
  64 asserts en `experiments/` + `src/` + `audit/`, **0 tautologicos**. Grep directo de los
  patrones obvios: 0 coincidencias.
- Alcance: D2 solo cubre tautologias sintacticas. Otra clase de "chequeo vacio" existe y esta
  documentada: P83 reconoce siete afirmaciones que "pasan en vacio" fabricando sus propios números
  (`docs/p83_AVISO_NO_EJECUTABLE.md:149-155`). D2=0 no debe leerse como ausencia de chequeos vacios.

Evidencia: `auditoria/escaner_resumen.md`; `auditoria/escaner_estatico.py:541-578`;
escaneo AST independiente.

## Hallazgos nuevos (no documentados en lo revisado)

1. **ALTA - Evaluacion no registrada del holdout v2 en v33.** Uso del 2026-09-27, después de que
   las 5 evaluaciones se cerraran (2026-09-25) y de que V10-01 declarara el holdout agotado
   (2026-09-26). Se uso para gatear V32-05 antes de una submission. Rompe el presupuesto maximo de
   5 y la afirmacion de que ningun script lo usa para seleccionar. Evidencia en A5.
2. **MEDIA - "5 de 5" es el conteo del registro, no el total de lecturas del holdout v2.** El
   conteo real conocido es 6+ (v33). Conviene anotarlo en `target_encoding_audit_v2.md` y RUNLOG.
3. **MEDIA - Refits de submission entrenan con las 1731 filas (incluye holdout)** en
   `18_p7:1107-1111` y `25:915-919`. Documentado y posterior al gasto, sin reevaluacion, pero
   contradice la letra de "ningun script lo usa para entrenar".
4. **INFO - La cifra +0.0036 es el delta crudo, no el efecto del umbral.** El proyecto ya lo
   corrige (v7 -0.0019; v10_06 +0.000131; D-072/V57); al citarla conviene mantener el matiz.
5. **INFO - D2=0 no cubre las "afirmaciones que pasan en vacio" de P83** (fabricacion de números),
   documentadas en el propio aviso de no ejecutable.
6. **BAJA - `submissions/leaderboard_log.csv` no es UTF-8** (byte 0xf1 = "n con virgulilla" en
   posicion 2305); se lee con cp1252 en las comprobaciones. No afecta al parser oficial
   (train/test/submission), pero rompe la convencion utf-8 si algun script lo leyera como tal.

## What's Done Well

- `src/evaluation.py` define la metrica oficial con `zero_division=0`, documenta su trazabilidad y
  `metrica_macro` se valida contra la definicion oficial dentro de `bootstrap_diferencia_pareada`.
- `src/splits.py` centraliza el protocolo v2 y añade `_verificar_sin_fuga` en la construccion de
  cada particion: la independencia de grupos no depende del llamador.
- El caso P83-A3 se corrigio binarizando con los umbrales leidos del artefacto (no constantes
  escritas a mano) y se verifico reproduciendo 0.7772114525165721.
- Honestidad metodologica repetida y verificable sobre el sesgo de umbrales en 04, 09, v10_04,
  v10_06 y V57, con regla de adopcion pre-registrada y control anidado.
- Idempotencia y validación estricta de submissions en `04` (columnas, 1154 filas, orden de ids,
  valores 0/1, relectura del archivo escrito).

## Verificaciones del checklist

- Lint: ruff 0.16.3, `All checks passed!` sobre `src/` y los dos scripts del alcance; `py_compile` OK.
- Venv: todas las ejecuciones con `/home/carlos/venv/bin/python`; sin paquetes ausentes.
- Leakage/split 3 vias: OK según A2; excepcion de presupuesto del holdout en A5 (v33).
- Anti-overfitting: 14 y 04 reportan métricas OOF por pliegue; sin early stopping (ML clasico);
  no se auditaron curvas train/val en este alcance.
- Trazabilidad: git status limpio; no se modifico ningun script previo; unica escritura este informe.
- Emojis en código: 0 en los archivos del alcance.
- GPU/hilos: no aplica (sklearn/liblinear; sin DL en el alcance).

## No evaluable

- Nada de las 7 afirmaciones quedo NO_EVALUABLE. Limitaciones declaradas: la clasificacion de
  call sites de A3 es estatica (159 por nombre) y la ejecución de v33 en el arbol P67 frente a su
  worktree de origen no puede determinarse con lo unificado; el reporte con fecha 2026-09-27 y los
  valores identicos a P7-01 sostienen el hallazgo.
