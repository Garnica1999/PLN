# Revision de codigo - notebook final de Kaggle (AGE 2026 PUJ)

**Alcance**: `scripts/generar_notebook_kaggle_age.py`, `repro/cadena/*.py` (12 modulos), `repro/spike_cadena.py`, `repro/verifica_ejecucion.py`, y contraste de fidelidad contra los originales citados en los docstrings (`src/cv.py`, `src/textometry.py`, `src/advanced_features.py`, `src/pipeline.py`, `src/preprocessing.py`, `src/config.py`, `scripts/experimento_e009/e013/e014/e016/e019/e021/e028`, `scripts/advanced_02/05`, `scripts/phase2_ponderacion.py`, `scripts/phase3b_recorregido.py`, `scripts/phase7/phase8/phase25`, `scripts/v13_p1b_fusion_tokens.py`, `scripts/v19_b1_boosting.py`, `scripts/v20_f3_umbrales.py`, `scripts/v29_04_optuna_individual.py`, `scripts/v30_01/02/03/05`, `scripts/age_plugin_submission.py`, `scripts/v38_01..v43_01`).

**Fecha**: 2026-10-17. Revision estatica (solo lectura + ruff del venv). No se modifico ningun archivo del repo salvo este informe.

## Estado: APROBADO CON OBSERVACIONES

No hay hallazgos criticos. No hay leakage: todos los ajustes son in-fold y el test solo se transforma/predice; el protocolo de folds esta congelado y embebido; no hay emojis en el codigo; ningun script previo fue modificado; y el generador pasa sus 17 comprobaciones con **0 fallos**. Las observaciones son una desviacion de fidelidad en la reconstruccion documental de los OOF de V29/V30 (importante, no bloqueante para la submission) y varios avisos de estilo/robustez (menores).

### Criticos (bloqueantes)

- Ninguno.

### Importantes

1. **`repro/cadena/v30.py:189-202` (`lgbm_refit_oof`) no reproduce la semantica del OOF original de V29/V30 (early stopping).**
   - Evidencia: los originales reajustan la mejor configuracion con `lgb.early_stopping(60)` monitorizado sobre el fold exterior (`scripts/v29_04_optuna_individual.py:243-260` y `scripts/v30_01_optuna_refinamiento.py:258-260`), y los diagnosticos guardados confirman que el corte **si** ocurrio en los 10 reajustes: V29 iteraciones 499/620/698/731/638 sobre techos 816/1372/912/863/1084 (`experiments/v29_04_optuna_lightgbm.json` -> `folds[*].diagnostico_refit`); V30 999/1173/644/1129/398 sobre 1011/1486/1023/1477/1000 (`experiments/v30_02_ensemble.json` -> `manifest.params_por_fold_v29[*].diagnostico_refit`). El porte entrena el techo completo sin early stopping, de modo que los OOF reconstruidos en la seccion `04_fase4.md` diferiran de los artefactos `v29_lgbm_optuna_oof.npy` / `v30_lgbm_refinado_oof.npy`, y los prints "V29 individual / V30 individual / V30 mezcla / V30 + plug-in" pueden caer fuera de la tolerancia 1e-4 del checklist de `repro/verifica_ejecucion.py`.
   - **Correccion propuesta (fidelidad)**: en `lgbm_refit_oof`, entrenar con `eval_set=[(matriz[va], clase[va])]`, `eval_metric="multi_logloss"` y `callbacks=[lgb.early_stopping(60, verbose=False)]`, pasando el techo congelado como `n_estimators` (quitandolo de `_normaliza` en esa ruta). Es exactamente lo que hacia el original; el early stopping mira el fold retenido del propio OOF (nunca el test) y no toca la submission, porque el lado test ya usa `lgbm_refit_test` con el techo congelado, identico a `v30_03_submission.py:139-146`.
   - **Alternativa (si se prefiere no mirar el fold exterior ni para el diagnostico)**: mantener el porte actual y actualizar los valores documentados y/o la tolerancia de las entradas V29/V30 del checklist, declarando la diferencia (reconstruccion sin early stopping).

### Menores

1. `repro/cadena/structured.py:246-250`: la closure `positions()` captura `length`/`value` del bucle (ruff B023). Es un porte literal de `src/advanced_features.py:240-244` y se invoca dentro de la misma iteracion, luego **no hay bug**; se documenta como fidelidad intencional. Opcional: ligar los valores por parametro o `functools.partial` para silenciar el lint.
2. `repro/cadena/neuronales.py:229`: `DataLoader(..., shuffle=True)` sin `num_workers`/`pin_memory`; fiel a los originales (`v38_01_tabularnet_5folds.py:104` y equivalentes). Sugerencia de rendimiento en GPU (opcional, no altera resultados): `num_workers` y `pin_memory=True`.
3. `repro/cadena/v30.py:98`: `int(round(float(x)))` con `round` que ya devuelve `int` (ruff RUF046). Porte literal de `v30_03_submission.py:146`; cosmetico.
4. `repro/spike_cadena.py:184` y `repro/spike_cadena.py:279`: variables asignadas y no usadas (`densa_te`, `clase`) (ruff F841); no afectan la verificacion del spike.
5. `repro/spike_cadena.py:35-42` y `repro/01_dup_clusters_folds.py:26-28`: directivas `# noqa` no usadas (ruff RUF100); cosmetico.
6. `repro/cadena/miembros_oof_b.py:19-34`: bloque de imports no ordenado (ruff I001) por el import multilinea `from cadena.comun import (...)`; el generador elimina ese bloque al embeber el modulo, asi que no llega al notebook.
7. `repro/diag_structured.py:26`: `cadena.dense` importado sin uso (F401); script de diagnostico, fuera de la cadena embebida.
8. `scripts/generar_notebook_kaggle_age.py:158-160`: rama muerta `if tipo == "cadena": continue` (no hay entradas `("cadena", ...)` en `CONTENIDO`); inofensiva.
9. `scripts/generar_notebook_kaggle_age.py:105-112` (robustez del transformador): el salto de imports multilinea termina en la primera linea que contenga `)`; un comentario o string con `)` en una linea de continuacion cortaria el salto antes de tiempo. No ocurre en los 12 modulos actuales (el check F821 del generador pasa y todos los bloques multilinea cierran con `)`), pero conviene endurecerlo (parentesis balanceados) si se anaden modulos.
10. `repro/verifica_ejecucion.py:11-12`: los ejemplos de uso del docstring dicen `python repro/...`; en este proyecto corresponde `/home/carlos/venv/bin/python`.

### What's Done Well

- **Fidelidad verificada modulo a modulo** contra los originales citados: mismos modelos, hiperparametros, ordenes de columnas (78 miembros + 35 densas + 87 estructuradas + 73 supervisadas = 273; 273 + 6 ADV07 = 279) y transformaciones `log`/`raw` segun `MEMBER_FILES` (`scripts/phase2_ponderacion.py:49-55`). `clusters.py` es linea a linea `src/cv.py:33-229`; `dense.py` reproduce `src/textometry.py:83-173` (incluidas las regex con escapes unicode equivalentes y los patrones copiados de `src/preprocessing.py:18,25-32`); `structured.py` porta las 6 familias mas la correccion `ALNUM_RE` de `scripts/phase3b_recorregido.py:42-51`; `supervised.py`, `nb_svm.py` y los 14 miembros OOF son portes literales (E-013 con C=0.2 = `mejor_C` del barrido de `experiments/barrido_c_resultados.json`; E-016/E-014/E-015/E-019b/E-021a/E-021b/E-028a/ADV05 identicos); `v30.py` reproduce la receta V20/V30 con pesos congelados (0.3548/0.3354/0.3098, iguales al manifest de `v30_02_ensemble.json`) y la regla plug-in (1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128, iguales a `age_plugin_submission.py:79`); `neuronales.py` reproduce arquitecturas, perdidas, epocas, SWA (`swa_start` 8 y 6), semillas (42+fold, 42+fold*100), y estrategias de test (full-train vs media/5).
- **Anti-leakage**: todos los ajustes son in-fold (`fit_transform` solo con `textos.iloc[train_idx]`; SVD, `StandardScaler` y `LabelEncoder` ajustados con el train del fold o con todo el train solo en el camino de submission); el test unicamente se `transform`/`predict`; los pesos y las configuraciones estan congelados antes de tocar el test.
- **Determinismo**: semillas portadas tal cual (SEED=42; 42+fold; 42+fold*100) y folds canonicos embebidos comprimidos (zlib + base85, 169336 valores) con asserts de forma y clases, inmunizando el notebook frente a la version de scikit-learn.
- **El generador**: parser de fences correcto (bloques ```python -> celdas de codigo, resto -> markdown); `transforma_modulo` elimina los imports `cadena.*` (incluidos multilinea) y los prefijos cualificados `comun.`/`miembros_oof_a.`/`miembros_oof_b.`, conservando `from __future__ import annotations` (necesario para anotaciones auto-referentes como `-> EncodedMLP`); los payloads (folds y submission sellada) son de una sola linea y el alfabeto base85 no contiene comillas ni backslashes, luego la inyeccion en literales es segura; orden de modulos coherente con las dependencias (comun -> clusters -> dense -> structured -> supervised -> nb_svm -> miembros_oof_a -> miembros_oof_b -> meta -> v30 -> neuronales -> test_side); el verificador cubre 17 comprobaciones (ASCII, compilacion por celda, F821 sobre el codigo concatenado, fases, payloads, `plt.show`, imports prohibidos).

### Verificaciones ejecutadas

- `/home/carlos/venv/bin/python scripts/generar_notebook_kaggle_age.py --verificar` -> **0 fallos**: 85 celdas (42 markdown / 43 codigo), codigo 100% ASCII, sin imports del proyecto, FASE 1..6 presentes, `FOLD_IDX_B85` embebido, `SUBMISSION_SELLADA_B85` inyectado (marcador ausente), `plt.show()` presente, imports de cadena eliminados, 12 modulos presentes, todas las celdas compilan, ruff F821 sin nombres indefinidos.
- `/home/carlos/venv/bin/python -m ruff check scripts/generar_notebook_kaggle_age.py repro/` -> generador limpio; en `repro/` 23 avisos de estilo (I001, RUF100 x11, F841 x2, PIE808, RUF007, B023 x2, RUF046, F401); ninguno es error de logica.
- Contraste de fidelidad (lectura dirigida): modulo por modulo contra los originales del alcance, y contra los artefactos/JSON del proyecto (`barrido_c_resultados.json` -> `mejor_C` 0.2; `v30_02_ensemble.json` -> pesos y `diagnostico_refit`; `v29_04_optuna_lightgbm.json` -> `diagnostico_refit`).
- Anti-leakage (grep dirigido sobre `repro/cadena/`): ningun `fit`/`fit_transform` con `valid_idx` ni con test; los unicos `transform` del test estan en `test_side.py` y en `nb_svm.nb_svm_margins` (con vectorizadores ajustados con el train del fold o con todo el train en el camino de submission).
- Trazabilidad: revision solo lectura; no se modifico ni borro ningun script previo (solo se creo este informe).

### Veredicto sobre `repro/verifica_ejecucion.py`: CORRECTO (con observaciones menores)

- Extrae `stream` y `execute_result` de todas las celdas y aplica regex con tolerancia; los 8 patrones del checklist coinciden con los `print` reales de las secciones, verificado uno a uno: `02_fase2.md:172` (folds), `04_fase4.md:40-57` (V20/V29/V30/plug-in), `05_fase5.md:54` (V48), `06_fase6.md:37,58` (SHA y coincidencia con la sellada), incluidos los formatos `{:20}`, `:.7f`, `:.10f`, `{:,}` y el `%`.
- Observaciones: (a) en `--comparar`, una clave ausente en ambos notebooks se reporta como `DIFIERE` (conservador, no bug); (b) el ejemplo de uso del docstring usa `python` generico (deberia ser el venv); (c) `sha_reconstruida` compara el SHA de `submission.csv` entre corridas, correcto para idempotencia, recordando la nota ya presente en `06_fase6.md` sobre posibles diferencias de los componentes neuronales entre entornos.
- Dependencia: hasta resolver el hallazgo importante 1, el checklist puede fallar en las entradas V29/V30 (tolerancia 1e-4).

## Conclusion

La cadena es apta para el notebook final: fidelidad, anti-leakage y determinismo verificados, generador con 17/17 comprobaciones en verde y sin hallazgos criticos. Antes de sellar la ejecucion final conviene decidir el hallazgo importante 1 (portar el early stopping del OOF de V29/V30, recomendado, o actualizar los valores/tolerancias del checklist) y, opcionalmente, limpiar los avisos de ruff de `repro/`.
