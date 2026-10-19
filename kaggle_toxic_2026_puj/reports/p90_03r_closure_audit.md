# P90.03-R - Cierre auditado y trazabilidad

Fecha: 2026-10-10. Fase exclusivamente documental: auditoria de artefactos, conciliacion de
inventario, verificacion de hashes y controles, y cierre administrativo. **Sin entrenamiento,
sin experimentos, sin test/holdout, sin submissions, sin commits.**

## 1. Resumen ejecutivo

P90.03-R queda **CERRADA** administrativamente: todos los criterios de cierre se cumplen y se
verifican contra artefactos. La reconstruccion fold-specific reproduce el OOF canonico de forma
exacta (error 0.0 en 4449 probabilidades; 0 discordancias binarias; `corrected_cv` identico;
residuo de descomposicion maximo 4.17e-14). Se resolvieron tres incidencias documentales
(recuento "13", limitaciones ausentes en la subseccion del notebook, celda duplicada por un bug
del marcador del generador) y se documento una nota de alcance del manifiesto. V03, OOF canonico,
holdouts y submissions permanecen intactos.

## 2. Auditoria inicial de Git

```text
git status --short: 48 archivos sin seguimiento + 1 modificado (notebooks/10_final_pipeline.ipynb)
git branch --show-current: main
git log -1 --oneline: e43c594 TOXIC: P90-00 builds the V03 failure casebook (descriptive, no model execution)
git diff --stat: notebooks/10_final_pipeline.ipynb | 273 inserciones (secciones 135/136/137/137R)
```

Los valores historicos reportados (HEAD e43c594, main, 1 modificado, 48 sin seguimiento) se
confirman. No se ejecuto ninguna operacion destructiva; los cambios locales de P90.02, P90.03 y
P90.03-R quedan preservados.

## 3. Conciliacion del inventario

Desglose verificado de los 48 archivos sin seguimiento previos a este cierre (inventario completo
por archivo en `reports/p90_03r_artifact_inventory.csv`, 49 filas: 48 + notebook):

| fase | informe_md | tabla_csv | json | script | total |
|---|---|---|---|---|---|
| P90.02 | 4 | 15 | 2 | 3 | 24 |
| P90.03 | 2 | 8 | 2 | 3 | 15 |
| P90.03-R | 1 | 3 | 2 | 3 | 9 |
| notebook (modificado) | - | - | - | - | 1 |

Total: 24 + 15 + 9 = 48 sin seguimiento + 1 modificado.

**Resolucion de la discrepancia "13 vs 9"**: la respuesta final de P90.03-R afirmo "los 13
artefactos nuevos"; la verificacion establece que el conjunto de 13 corresponde al **escaneo
byte a byte** realizado al cierre de la fase (`reports/p90_03r*` + `experiments/p90_0[3-5]*`),
que contiene 6 informes nuevos de P90.03-R y 7 scripts de tres fases distintas. Los scripts de
fases anteriores incluidos en ese escaneo son: `p90_03_integrar_notebook.py` (P90.02, seccion
136), `p90_03_diagnostico_contribuciones.py` (P90.03), `p90_03b_informe_handoff.py` (P90.03) y
`p90_04_integrar_notebook.py` (P90.03, seccion 137). **Los artefactos nuevos reales de P90.03-R
son 9** (6 informes + 3 scripts: `p90_03r_reconstruccion_oof.py`, `p90_03rb_informe.py`,
`p90_05_integrar_notebook.py`), coincidentes con la lista enumerada en la respuesta. La cifra
"13" fue una imprecision de redaccion de la respuesta de chat; ningun artefacto del repositorio
contiene esa cifra. **Correccion complementaria registrada aqui** (no se sobrescribio ningun
informe historico).

Nota de clasificacion: `experiments/p90_03_integrar_notebook.py` lleva prefijo "p90_03" pero
pertenece a P90.02 (integrador de la seccion 136); se clasifico por evidencia de contenido, no
por el nombre (fila correspondiente en el inventario).

### Alcance del inventario (aclaracion P90.04-R, 2026-10-10)

- El inventario de 49 filas incluye intencionalmente los **48 artefactos previos sin seguimiento**
  (24 de P90.02 + 15 de P90.03 + 9 de P90.03-R) y el **notebook modificado**: documenta el estado
  auditado **al inicio de este cierre**, no el estado final.
- Los **cuatro entregables del cierre** se identifican por separado y **no** forman parte del
  inventario (se crearon despues de la fotografia que este documenta):
  `reports/p90_03r_closure_audit.md`, `reports/p90_03r_artifact_inventory.csv`,
  `reports/p90_04_hypothesis_gap_review.md`, `reports/p90_04_hypothesis_gap_review.csv`.
  Sus hashes quedan registrados en la respuesta final de P90.04-R; el inventario no registra su
  propio hash ni el de los entregables del cierre porque la auto-validacion recursiva no es
  fijable.
- Por tanto, **el inventario de 49 filas no representa la totalidad** de los archivos modificados
  o creados al finalizar P90.04; el estado final completo es el de la seccion 9 (1 modificado +
  52 sin seguimiento, incluidos estos 4 entregables).

## 4. Verificacion del manifiesto y de las cifras

Manifiesto `reports/p90_03r_manifest.json`: las **7 rutas** de `artifact_paths` existen y sus
hashes coinciden (recomputados; 7/7 OK). Cifras verificadas contra artefactos (no contra
Markdown):

| comprobacion | resultado | fuente | clasificacion |
|---|---|---|---|
| 5 folds reconstruidos | folds 1..5; 297/297/296/296/297 filas | p90_03r_oof_reproduction.csv | VERIFIED_FROM_ARTIFACT |
| probabilidades OOF reproducidas exactamente | 4449/4449 identicas; allclose True; max 0.0 | CSV + gate | VERIFIED_FROM_ARTIFACT |
| error absoluto maximo / medio / p99 | 0.000e+00 / 0.000e+00 / 0.000e+00 | CSV (recomputado) | VERIFIED_FROM_ARTIFACT |
| decisiones binarias discordantes | 0 | CSV (recomputado) | VERIFIED_FROM_ARTIFACT |
| corrected_cv | 0.7772114525165722 recomputado desde CSV (dif 1.1e-16 del canonico); gate del script 0.0 con la funcion oficial | CSV + audit_results.json | VERIFIED_FROM_ARTIFACT |
| residuo maximo del logit | 4.170e-14 (<= 1e-9) | p90_03r_feature_contributions.csv | VERIFIED_FROM_ARTIFACT |
| recuentos | 4449 filas-etiqueta; 757 errores (FP 390 + FN 367) | CSVs | VERIFIED_FROM_ARTIFACT |
| manifiesto (rutas y hashes) | 7/7 OK | manifiesto + recomputo | VERIFIED_FROM_ARTIFACT |

Nota de alcance: `p90_05_integrar_notebook.py` no figura en `artifact_sha256` del manifiesto
(se creo despues del manifiesto); su hash queda registrado en el inventario de este cierre. El
propio `p90_03r_manifest.json` no se auto-hashea (patron normal del proyecto). Ambas notas son de
alcance, no errores de contenido.

## 5. Conservacion del baseline

Hashes actuales verificados contra las referencias historicas:

```text
experiments/baseline_v03_frozen/model_v03.joblib        263ed5bd... OK
features/oof_v03_final.npz                              55926581... OK
submissions/toxic_submission_v03_lr_union_thr.csv       7d7478f2... OK
```

V03 conserva sus invariantes (pipeline_v03.json + joblib): word TF-IDF (1,2); char_wb (3,5);
min_df=2; sublinear_tf; lowercase; LogisticRegression class_weight=balanced, C=2.0,
solver=liblinear, random_state=42; umbrales HS 0.44 / TR 0.51 / AG 0.46; corrected_cv canonico
0.7772114525165721. Sin discrepancias.

## 6. Notebook consolidado

- Estructura: 451 celdas; secciones 135 (P90-00), 136 (P90.02), 137 (P90.03) y **137R (P90.03-R)
  unica**, en las celdas 449 (markdown) y 450 (codigo de verificacion, sin ejecutar, 0 outputs:
  no presenta salidas antiguas como ejecucion nueva).
- Fuente de verdad: el notebook consolidado es un artefacto ensamblado por integradores
  especificos por fase; para la subseccion 137R el generador es
  `experiments/p90_05_integrar_notebook.py`, que carga cifras de `reports/p90_03r_*` (artefactos
  verificados). No es el unico generador del notebook (existen integradores historicos por
  seccion).
- Contenido 137R verificado: ausencia de artefactos fold-specific, procedimiento de
  reconstruccion, control de reproduccion, descomposicion del logit, distincion
  exacta/auxiliar, **limitaciones de interpretacion** (anadidas en este cierre) y decision final.
- Correcciones aplicadas en este cierre (solo a la subseccion 137R, via su generador; ninguna
  seccion historica tocada): (a) se anadio el parrafo de limitaciones de interpretacion;
  (b) se corrigio el marcador de limpieza idempotente del generador (incluia un parentesis de
  cierre que nunca coincidia) que habia dejado una celda de verificacion duplicada; el notebook
  quedo limpio (451 celdas, una sola subseccion 137R).

## 7. Auditoria de caracteres (ASCII-only)

- `tools/verificar_caracteres.py` (verificador oficial): **0 caracteres no permitidos**.
- Escaneo independiente: los 4 entregables de este cierre y todos los artefactos de P90.03-R
  tienen **0 bytes > 127**.
- Los datos originales (textos) no fueron modificados.

## 8. Discrepancias no resueltas

**Ninguna.** Incidencias encontradas y resueltas/documentadas: (1) recuento "13 vs 9" (corregido
en la seccion 3); (2) limitaciones ausentes en la subseccion 137R (anadidas); (3) celda
duplicada por bug del marcador del generador (corregida); (4) alcance del manifiesto sobre
p90_05 y auto-hash (documentado). No hay MISMATCH ni MISSING en el inventario ni en los hashes.

## 9. Estado de Git final

```text
Rama: main | HEAD: e43c594 (sin cambios) | 0 commits en esta fase | sin push
git status --short: 1 modificado (notebooks/10_final_pipeline.ipynb, 273 inserciones)
                  + 52 sin seguimiento (48 previos + 4 entregables de este cierre)
Sin operaciones destructivas; cambios locales preservados.
```

## 10. Conclusion administrativa

**CLOSED.** Los resultados de P90.03-R estan vinculados a artefactos verificables (manifiesto
7/7 y cifras recomputadas); el inventario y los recuentos quedan reconciliados; V03 y los
artefactos historicos permanecen intactos; el notebook refleja correctamente la fase; la
auditoria de caracteres se ejecuto; y el estado final de Git queda documentado. La revision
documental de hipotesis futuras (`p90_04_hypothesis_gap_review.md`) concluye
NO_NEW_HYPOTHESIS_IDENTIFIED.
