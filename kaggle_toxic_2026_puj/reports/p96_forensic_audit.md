# P96 - Auditoria forense integral: validez experimental, codigo, metrica y reproducibilidad

Fecha: 2026-10-13. Fase exclusivamente forense: sin entrenamiento, sin
predicciones nuevas, sin submissions, sin leaderboard, sin abrir los contenidos
del test, del holdout historico ni de holdout2. Las 14 candidatas de etiquetas
siguen PRESERVED_NOT_CONFIRMED y no se tocaron.

**Principio**: la pregunta no es cuantos experimentos salieron negativos, sino
cuantos de esos resultados constituyen evidencia fiable, comparable y
reproducible de que una tecnica no mejora V03.

---

## 0. Veredicto

```text
AUDIT_CLEAN_WITHIN_VERIFIED_SCOPE
```

- La metrica canonica `corrected_cv` de V03 se reproduce **exactamente**
  (0.7772114525165721, delta 0.000e+00; por etiqueta y por pliegue identicos).
- La particion dev/holdout, los grupos fusionados y los pliegues corregidos se
  recomputan **exactamente** (biyeccion de numeracion 0..4 -> 1..5).
- 20/20 comparaciones curadas contra valores documentados reproducen
  (de 1e-13 a 1e-6): candidatos V4 A/B/C/D, meta_shift x4 en dos protocolos,
  P84-01 semillas 42/43/44, P84-02 control (dos protocolos), P84-03 control=V03,
  cadena completa de V03.
- 159 recomputaciones adicionales de variantes guardadas (v2, v4, v6-era stores,
  P59, P66, P84, P85) sin discrepancia contra el flujo documentado.
- 17 hallazgos: **0 P0, 0 P1**, 4 P2 (comparabilidad/estadistica), 5 P3
  (provenance), 8 P4 (documentales). Dos notas metodologicas relevantes para
  gates futuros (F96-06 y F96-17, ver seccion 9 y 12).
- Ninguna conclusion historica de P94/P95 se suspende; el benchmark interno
  queda **reforzado** en el alcance auditado.
- Reproducciones futuras justificadas por un defecto concreto: **ninguna**.

---

## 1. P0 - Estado inicial y preservacion

- Git real: rama `main`, HEAD `e43c594`, 187 entradas sin rastrear y 1
  modificada (solo el notebook consolidado); sin commit, push ni limpieza.
- Se preservaron todos los cambios locales de P90-P95. No se re-ejecuto ningun
  experimento historico.
- Rutas verificadas antes de leer; 8 entregables P96 generados en rutas nuevas
  (ninguna ruta p96* existia).

## 2. P1 - Inventario completo del registro

Recomputado desde `experiments.csv` (no se confio en la cifra historica):

| Metrica | Valor |
|---|---|
| total_rows | 459 (confirmado) |
| unique_experiment_ids | 459 (0 duplicados, 0 ids vacios) |
| campos vacios | f1 253; validation 111; result 32; script 7; resto menor |
| con script existente en disco | 434/459 |
| sin script real | 25 = 7 vacias + 18 con referencias textuales (auditorias/envios/demostraciones; ninguna fila de modelado) |
| con metrica declarada | 206 |
| con pliegues declarados | 230 |
| con artefacto referenciado existente | 124 |
| configuraciones repr+modelo+hiper repetidas | 7 grupos (PRE-00..09 = 9 transformaciones deterministas con base comun; V31-00..04 features estructurales; integraciones; controles) - no son el mismo pipeline duplicado |

Trazabilidad por niveles: A+ 52, A 152 (codigo y metrica), B 247 (codigo sin
metrica: mayoritariamente auditorias y decisiones documentales), C 2, D 6.

Distincion experiment/run: el registro no separa runs de experimentos; el
campo `script` es el discriminante estable. Varias filas comparten script y
configuracion base (p. ej. 03_variantes_representaciones.py con distinto
preprocessing). Documentado como limite estructural del registro (no un bug).

## 3. P2 - Codigo realmente ejecutado

**Cadena canonica de V03 verificada de punta a punta:**

1. `experiments/04_refit_submissions_v02_v03.py` produjo el modelo y las
   submissions v02/v03 (umbrales 0.44/0.51/0.46 por rejilla sobre el OOF).
2. `experiments/14_oof_corregido_v2.py` produjo `features/v2_oof_base_corrected.npz`
   (4 modelos, ajuste de vectorizador y modelo DENTRO de cada pliegue; grupos =
   near-duplicado UNION pseudo-autor).
3. `experiments/23_candidatos_finales.py` escribio `features/oof_v03_final.npz`
   con una **autoverificacion que aborta si no reproduce 0.7772** y con la
   fuente grabada dentro del artefacto (`fuente`, `nota_mapeo`).
4. Evidencia forense: la proba M2 del paso 2 es identica bit a bit a la del
   paso 4; las etiquetas del artefacto son identicas a las originales del
   train; los hashes del paquete congelado coinciden con los originales
   (`models/mod03b_lr_union.joblib` == `baseline_v03_frozen/model_v03.joblib`;
   `submissions/toxic_submission_v03_lr_union_thr.csv` == copia congelada).

**Nota de mtimes**: la ola del 2026-10-02 (auditoria previa + `audit_fixes/`)
reseteo mtimes de muchos scripts; para V03 la secuencia del 2026-09-25 es
coherente (script 08:40:19 -> npz 08:40:58 -> oof_final 13:11:25). No se
encontro ningun caso de `SOURCE_ARTIFACT_MISMATCH` demostrado.

**Muestreo de lectura de codigo** (fits fuera de la porcion de entrenamiento):
29_reweighting_v4 (fit por pliegue con indices de entrenamiento; refit final
declarado), v6_02 (fits por pliegue; el modelo de TR solo ve HS=1 de
entrenamiento), v7_01 (control BCE con el mismo codigo, deltas atribuidos al
control), p84_02 (binario fastText con sha256 verificado; control reproduce),
p84_03 (brazo A = V03 exacto). El inventario crudo de `.fit`/`fit_transform`
queda en `reports/p96_static_scans.txt`.

Clasificacion dominante: `SOURCE_IDENTIFIED_RUN_NOT_FULLY_VERIFIABLE` para la
mayoria de las 459 filas (no hay historia git de scripts, todos sin rastrear);
`SOURCE_AND_RUN_MATCH_VERIFIED` para la cadena V03 y los artefactos
recomputados numericamente en esta fase.

## 4. P3 - La metrica canonica

- Definicion real: media column-wise de la F1 binaria de la clase positiva
  (HS, TR, AG), `zero_division=0`, prediccion `proba >= umbral`; implementada en
  `src/evaluation.py` e historicamente duplicada en el script 02 (equivalencia
  declarada y verificada por inspeccion).
- Rango de `corrected_cv` = OOF del desarrollo (1483 filas, umbrales congelados).
- Recomputo V03: **0.7772114525165721** identico al guardado y al documentado;
  f1 por etiqueta y por pliegue identicos; `y_true` alineado con el train.
- El OOF historico de 1731 filas (incluye el holdout) quedo **excluido por
  politica**: no se abrio su contenido.
- Convenciones distintas conviven (D-128): la columna `f1` del registro mezcla
  0.5 / umbrales congelados / postproceso / protocolo historico vs corregido;
  muestreo confirma el caveat. Fuente de verdad: reportes y JSON por
  experimento.
- **Optimismo de umbrales (F96-05)**: los umbrales de V03 se optimizaron sobre
  el mismo OOF en que se reporta la metrica historica. Re-optimizados in-sample
  sobre el dev corregido dan 0.7808400798 (+0.003629 sobre 0.7772114525165721).
  Caveat ya documentado (D-126); todas las comparaciones posteriores usan
  umbrales congelados, y el valor externo (LB 0.78306) es la unica confirmacion
  independiente disponible.

## 5. P4 - Validacion, folds y leakage

- Particion: 1731 = 1483 (dev) + 248 (holdout); completa y disjunta.
- Grupos fusionados recomputados desde `src/splits.py`: **identicos etiqueta a
  etiqueta** a los guardados en `v2_oof_base_corrected.npz` (1415 componentes
  en dev).
- Pliegues corregidos recomputados (StratifiedGroupKFold(5, seed 42)):
  tamanos [297, 297, 296, 296, 297]; **particion identica** a la guardada en
  `oof_v03_final.npz`; unica diferencia: numeracion (0..4 vs 1..5), biyeccion
  exacta. Ojo de trazabilidad F96-04.
- Sin fuga de grupos: 0 pliegues con grupos compartidos entre entrenamiento y
  validacion (verificacion directa, no solo la del constructor).
- Escaneos estaticos: 0 vecindades de test con `.fit` cercano; 54 scripts
  mencionan test (generacion de submissions/candidatos, uso permitido); 51
  menciones de holdout2 (gobernanza v10+; 0/3 evaluaciones gastadas).
- Artefactos que cubren el holdout quedaron EXCLUIDOS del recomputo:
  `label_consistency_v4.npz`, `oof_bases.npz`, `oof_ens04_v04.npz` (1731
  filas), `oof_mod03b.npz` y `baseline_v03_frozen/oof_v03_historical.npz`.
- Clasificacion: `NO_LEAKAGE_FOUND_IN_AUDITED_SCOPE` en el alcance declarado
  (ver el informe companion). No se declara cero leakage de todo el proyecto:
  el alcance es el codigo y los artefactos revisados.

## 6. P5 - Comparabilidad

- Politicas de umbral declaradas: congelados-v03 95, fijo-0.5 74, nested 33,
  mencionado 21, no declarado 236. Las comparaciones centrales usan congelados.
- Casos revisados:
  - **V32-05**: submission byte-identica a V03 (sha256 igual);
    el candidato no cambio ninguna decision en el test. Confirmado.
  - **R1-C1 / R2-MINDF5 / OPT-01**: deltas locales +0.0157/+0.0221/+0.0240
    que no transfirieron (-0.0108/-0.0127/-0.0070). Comparaciones pareadas
    por submission; son la evidencia base del freno de transferencia.
  - **D-016 en V4 (F96-17)**: el criterio IC95 pareado > 0 se aplico sobre un
    proxy (acuerdo por ejemplo), no sobre la metrica primaria. Recomputo P96
    con la metrica primaria (filas / clusters): A [+0.000192,+0.016189] /
    [+0.000347,+0.016311]; B [-0.004135,+0.014312] / [-0.003715,+0.014602];
    C [+0.000726,+0.021142] / [+0.000466,+0.021360]; D [+0.003164,+0.023957]
    / [+0.002303,+0.023833]. Bajo la metrica primaria A, C y D superan
    nominalmente el IC y B no: la frase "el unico que superaba los tres
    criterios era C" es dependiente del estimando. La adopcion final no
    cambia (ninguno adoptado; C rechazado por el holdout; A por su sonda
    externa). Nota para gates futuros: IC de la metrica primaria, por clusters.

## 7. P6 - Preprocesamiento

- El inventario de ajustes (`p96_static_scans.txt`) no muestra ningun caso de
  ajuste sobre test.
- Muestras verificadas: vectorizadores y modelos ajustados por pliegue en la
  cadena v2/v4/v6; escaladores densos ajustados con la porcion de entrenamiento
  del pliegue (08_features_densas); refits finales sobre todo el train solo
  para producir artefactos de produccion (declarado).
- Las familias de embeddings (v5, P69-P85) tienen auditorias dedicadas previas
  (P80/P81/P84/P85) y artefactos que recomputan los valores documentados
  (p84_04 agregado, p85_01). No se reabre ninguna conclusion sin un defecto
  concreto; no se encontro ninguno.

## 8. P7 - Optimizacion y entrenamiento

- Convergencia/warnings solo verificables donde hay logs: `.log` de sondas de
  robustez y backlog v2 presentes; para el resto no hay logs de ejecucion y se
  clasifica como no verificable (sin evidencia de fallo).
- Sin evidencia de: modelos cargados de rutas incorrectas (hashes identicos en
  los paquetes congelados), ejecuciones parciales promovidas como resultados
  (los p84_04 parciales estan fuera del registro, F96-11), o parametros
  ignorados silenciosamente (los parametros efectivos coinciden con lo
  declarado en las muestras leidas).
- Si se quisiera verificar convergencia de un entrenamiento concreto, la via es
  la reproduccion minima con preregistro (seccion 14).

## 9. P8 - Estadistica y gates

- Origenes: banda de particion ~0.017 = D-070 (8 semillas: rango 0.017184,
  std 0.005302; 8 subconjuntos: rango 0.020248) -> derivada de datos. Gate de
  tres cifras (delta, IC, 4/5 particiones) = D-071 (acta pre-registrada).
  `dOOF > +0.0376` = regresion dLB~dOOF sobre 19 observaciones (intercepto
  -0.0092), reexpresada como gate por gobernanza (F96-13): derivada de datos +
  decision, NO constante universal.
- **Incertidumbre pareada (F96-06)**: `src.evaluation.bootstrap_diferencia_pareada`
  remuestrea filas, no clusters. Existen implementaciones por clusters en
  fases P73/P74 (y en la propia P96). Recomputo de los cuatro candidatos V4:
  los IC por clusters son casi identicos a los de filas (diferencias < 0.0015)
  y no cambian ninguna direccion. Conclusion: el punto es de higiene
  metodologica para gates futuros, no un defecto que invalide cierres.
- No se detecto conversion de intervalos no significativos en pruebas de
  equivalencia en los cierres clave (los cierres fueron direccionales y con
  evidencia externa adicional).

## 10. P9 - Artefactos, hashes y notebook

- Hashes canonicos intactos: train `3d40e7345b588860`, OOF `559265816ce4f631`,
  submission V03 `7d7478f271e6edfd`, modelo V03 `263ed5bd0fc0de7c`, particion
  `e5b62e5f22c4c63a`.
- Notebook consolidado: 487 celdas, 220 de codigo; 27 celdas ejecutadas con
  salidas de una corrida en el worktree `/home/carlos/Git/PLN-v10` (ola
  v9/v10; evidencia `reports/v10_00_verificacion.json`); el resto de secciones
  no tiene salidas. No se detecto conflicto numerico en el muestreo; se declara
  como nota de provenance (F96-10). Los generadores de secciones son
  idempotentes (marca + reemplazo).
- Sin rutas absolutas que impidan reproducir por script (los scripts usan
  PROJECT_ROOT relativo).

## 11. P10 - Reproduccion numerica (desde artefactos guardados)

Matriz completa: `reports/p96_reproducibility_matrix.csv` (200 filas).

| Estado | Filas |
|---|---|
| Cadena canonica V03 (exacta) | 2 |
| Comparaciones curadas vs documentado (exactas dentro de tolerancia) | 20/20 |
| Recomputed EXACT (barrido con claim escalar) | 3 |
| Within tolerance (M2 vs 0.7772 a 4 decimales) | 1 |
| Recomputadas sin claim escalar guardado (valores registrados) | 159 |
| No recomputables (staging p84_04_x por semilla; sin y_true completo) | 12 |
| Excluidas por politica (cubren holdout) | 5 |

Ninguna recomputacion produjo un NUMERIC_MISMATCH.

## 12. P11/P12 - Hallazgos y cuanto cambia la conclusion

Registro completo: `reports/p96_code_and_metric_findings.csv` (17 hallazgos).
Resumen por impacto:

- A (metadatos/documentacion): F96-01, F96-02, F96-03, F96-07, F96-10, F96-11,
  F96-12, F96-14, F96-15.
- B (resultado aislado): F96-17 (frase "unico C" dependiente del estimando).
- C (comparabilidad): F96-05 (optimismo de umbrales, documentado), F96-06
  (bootstrap filas vs clusters), F96-13 (origen del gate +0.0376), F96-17.
- D/E/F: **ninguna** familia afectada; la validez del benchmark queda
  reforzada; "no existe otra mejora viable" (P94) no cambia.

Conclusiones que se suspenden: **ninguna**. Conclusiones que se matizan:
la interpretacion del criterio D-016 de la fase V4 (estimando) y el uso de
bootstrap por filas en gates antiguos. Ambas direcciones se preservan.

## 13. P13 - Reproducciones futuras

`reports/p96_reproduction_candidates.csv`: 2 candidatas, ambas condicionales
y ninguna justificada por un defecto confirmado:

- REP-96-01 (V6): cerrar el hueco de provenance de V6 (no hay matrices OOF V6);
  re-ejecucion minima de v6_02 con pliegues canonicos. NO ejecutar ahora.
- REP-96-02 (IC por clusters): homogeneizar gates futuros; recomputable sin
  entrenamiento (P96 lo hizo para los cuatro candidatos V4).

## 14. P14 - Cierre y alcance

```text
AUDIT_CLEAN_WITHIN_VERIFIED_SCOPE
MODEL_EXPLORATION_REMAINS_PAUSED
V03 = FROZEN
NO_TRAINING_AUTHORIZED
NO_SUBMISSIONS_AUTHORIZED
```

Alcance verificado: la cadena de V03 completa; la particion y los folds; la
metrica canonica; 20 comparaciones curadas; 159 variantes recomputadas; los
scans estaticos del codigo; hashes de todos los artefactos de la fase.

Fuera de alcance (no auditado en profundidad): convergencia interna de
entrenamientos sin logs; contenidos del test/holdout/holdout2 (prohibido);
cada uno de los ~480 scripts individualmente (muestreo dirigido); semantica
fina de cada JSON de fase (se usaron sus valores documentados).

Limitaciones declaradas: la ausencia de historia git de scripts impide probar
"que version exacta se ejecuto" para la mayoria de filas; la evidencia fuerte
esta en la reproduccion numerica de los artefactos (que si demuestra
consistencia entre codigo actual y resultados guardados).

## 15. Entregables y hashes

| Artefacto | sha256 (16) |
|---|---|
| reports/p96_forensic_audit.md | ver manifest |
| reports/p96_experiment_integrity_matrix.csv | ver manifest |
| reports/p96_code_and_metric_findings.csv | ver manifest |
| reports/p96_reproducibility_matrix.csv | ver manifest |
| reports/p96_validation_and_leakage_audit.md | ver manifest |
| reports/p96_historical_conclusions_review.md | ver manifest |
| reports/p96_reproduction_candidates.csv | ver manifest |
| reports/p96_manifest.json | ver manifest |

Evidencia de soporte: `p96_folds_evidence.json`, `p96_static_scans.txt`
(escaneos crudos), scripts `p96_00..p96_05`. Hashes completos en
`reports/p96_manifest.json`.
