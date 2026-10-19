# P91-R - Auditoria final de consistencia metodologica

Fecha: 2026-10-09 (fecha del proyecto). Revision documental de cuatro detalles de P91.
No entrena, no modifica etiquetas, no genera predicciones, no usa test/holdouts, no usa el
leaderboard, no ejecuta la revision humana ni genera el paquete de anotacion.

## 1. Precision agregada frente a precision por etiqueta

Las tres cifras estiman magnitudes DISTINTAS y no deben intercambiarse. Formulas
(prevalencia p, nivel 95 por ciento, correccion por poblacion finita FPC y efecto de
diseno DEFF declarado): `SE = sqrt(p(1-p)/n) * sqrt((N-n)/(N-1)) * sqrt(DEFF)`;
`margen = 1.96 * SE`.

| magnitud | unidad | N | n | p | DEFF | FPC | margen 95 por ciento | estado |
|---|---|---|---|---|---|---|---|---|
| Tasa AGREGADA de celdas (3 etiquetas juntas) | celdas etiqueta-ejemplo | 4449 | 300 | 0.05 | 1.2 | 0.9658 | **+-2.61 pp** | texto original correcto ("+-2.6") |
| Tasa POR ETIQUETA (nivel tweet) | tweets | 1483 | 100 | 0.05 | 1.2 | 0.9660 | **+-4.52 pp** | no explicitado en el original; documentado aqui |
| Tasa a nivel tweet | tweets | 1483 | 100 | 0.10 | 1.0 | 0.9660 | +-5.68 pp | el "+-5.7" original OMITIA el DEFF |
| Tasa a nivel tweet (con DEFF) | tweets | 1483 | 100 | 0.10 | 1.2 | 0.9660 | **+-6.22 pp** | valor corregido con efecto de diseno |

Conclusion: el "+-2.6" del protocolo es correcto **solo** para la tasa agregada de celdas
con DEFF 1.2 (recalculo: SE=0.01331). La precision por etiqueta a p=0.05 es +-4.52 pp; la
cifra de tweets a p=0.10 debe reportarse con DEFF (+-6.22 pp). Correccion aplicada de forma
**aditiva** al protocolo (seccion 10 "Nota de consistencia P91-R"); el texto original se
conserva integro.

## 2. Reconciliacion de motivos y candidatos

Los motivos NO son mutuamente excluyentes: cada celda puede portar varios motivos y el CSV
los une con "; ". Verificado sobre `reports/p91_label_review_candidates.csv`:

```text
unique_candidate_cells         882   (ids unicos; pares (example_id, label) unicos)
total_reason_assignments       953   (suma de motivos por celda)
cells_with_multiple_reasons     69   (de ellas, 2 con 3 motivos)
celdas con un solo motivo      813
```

`reason_overlap_counts` (pares de motivos que coocurren):

| par | celdas |
|---|---|
| HIGH_CONFIDENCE_OOF_DISAGREEMENT + RANDOM_REFERENCE_SAMPLE | 24 |
| HIGH_CONFIDENCE_OOF_DISAGREEMENT + POSSIBLE_INSTANCE_LABEL_INCONSISTENCY | 19 |
| POSSIBLE_INSTANCE_LABEL_INCONSISTENCY + RANDOM_REFERENCE_SAMPLE | 9 |
| LABEL_SPECIFIC_MARGIN + RANDOM_REFERENCE_SAMPLE | 8 |
| DOCUMENTED_RULE_CONFLICT + HIGH_CONFIDENCE_OOF_DISAGREEMENT | 5 |
| LABEL_SPECIFIC_MARGIN + POSSIBLE_INSTANCE_LABEL_INCONSISTENCY | 4 |
| HIGH_CONFIDENCE_OOF_DISAGREEMENT + LABEL_SPECIFIC_MARGIN | 2 |
| DOCUMENTED_RULE_CONFLICT + LABEL_SPECIFIC_MARGIN | 2 |

La diferencia 953 - 882 = 71 es exactamente el numero de asignaciones extra de las 69
celdas multi-motivo (67 con 2 motivos + 2 con 3). **El numero de motivos no equivale al
numero de candidatos unicos**; el CSV ya usa una fila por celda unica (no un motivo
exclusivo por fila). No se cambio la muestra ni se re-seleccionaron candidatos.

## 3. Estado de los 22 DOCUMENTED_RULE_CONFLICT

Los 22 registros son celdas con gold AG=1 y HS=0 (ids: 21041, 22203, 22528, 22691, 24588,
20707, 21251, 21374, 21522, 21582, 22034, 22242, 22265, 22278, 22621, 22671, 23985, 24039,
24441, 24533, 24596, 24638; 15 con motivo unico, 5 con desacuerdo 4-modelos, 2 en frontera).

**Fuente normativa que supuestamente contradicen: NINGUNA.** El motivo NO se deriva de la
relacion empirica TR=1 => HS=1 (que ademas es sobre TR, no AG) y P91 concluyo
DOCUMENTATION_INSUFFICIENT: no existe regla oficial que exija HS=1 para AG=1. Clasificacion
correcta de los 22 registros:

```text
OFFICIAL_RULE_CONFLICT          0 registros (ninguna regla oficial aplica)
EMPIRICAL_STRUCTURE_ANOMALY    22 registros (excepciones a la lectura condicionada D-001)
POSSIBLE_ANNOTATION_ISSUE      no afirmado (sin evidencia; requiere revision humana autorizada)
```

`DOCUMENTED_RULE_CONFLICT` queda documentado como **codigo de seleccion**, no como
violacion confirmada (nota aditiva en el protocolo, seccion 10). Los candidatos, textos y
etiquetas originales se conservan sin cambios.

## 4. Fecha de consulta bibliografica

`reports/p91_label_state_of_art.md` registraba "acceso 2026-10-10" (cabecera y referencia
cleanlab). La fecha del proyecto es 2026-10-09: la fecha registrada es posterior y **no pudo
verificarse** contra un registro independiente. La consulta a la documentacion de cleanlab
SI se realizo durante la sesion de trabajo de P91 (evidencia: resultados de busqueda de esa
sesion), pero la fecha exacta no es confirmable. Correccion **aditiva** aplicada al estado
del arte (seccion 7: "fecha no verificada"); el registro original se conserva. No se
realizo ninguna busqueda nueva ni se invento una fecha. La misma nota se aplico a la
cabecera del informe principal (fecha no verificada).

## 5. Archivos modificados (solo correcciones aditivas)

| archivo | sha256 antes | sha256 despues |
|---|---|---|
| reports/p91_label_review_protocol.md | ffbe1113622b42b5... | 73475f1e13dc9153... |
| reports/p91_label_state_of_art.md | 776f00a7a2a3a9a6... | c3e88f49b8853573... |
| reports/p91_label_quality_audit.md | 09f3ff4517b28e52... | db85e42783b7ef23... |

Nuevo: `reports/p91r_methodology_consistency.md` (este informe). Los CSV de la cola y los
diagnosticos NO se modificaron. Ninguna etiqueta, texto ni resultado experimental fue
alterado.

## 6. SHA256 de los artefactos revisados (sin cambios)

```text
reports/p91_label_review_candidates.csv     074675225df0bae453fc759fe51aa6c8024e7c01da8875e58795f258395759dc
reports/p91_label_quality_diagnostics.csv   3ee4f0c0211ee1d5431426570e9733d97c586d45d1d979b34e3b307a83552442
reports/p91_annotation_gap_analysis.csv     652449f9c0673639c7f0e03ccb37d4fbfc29e4ca34cc59fcdf5721ec43fe6fa5
reports/p91_audit_results.json              cff7f1d75ef8d893f3ad4c4fcbf44551dbb5b0e52e4b7c94101baedc948e83ab
```

## 7. Auditoria de caracteres y estado de Git

- Verificador oficial `tools/verificar_caracteres.py`: **0 caracteres no permitidos**
  (resultado real de la ejecucion en esta fase). Los tres documentos corregidos y este
  informe: 0 bytes > 127.
- Git (al cierre de esta revision): rama `main`, HEAD `e43c594` sin cambios, **0 commits**,
  sin push; 63 archivos sin seguimiento (62 previos + este informe) + 1 modificado
  (notebook). El arbol NO esta limpio: todos los cambios locales se conservan.

## 8. Cierre

La decision cientifica de P91 permanece sin cambios:

```text
NO_CONFIRMED_LABEL_ISSUE
ANNOTATION_POLICY_UNRESOLVED
GOVERNANCE_REVIEW_REQUIRED
V03 = FROZEN
NO NEW MODELING PHASE
```

Las cuatro verificaciones quedan cerradas: precision agregada correcta y por etiqueta
documentada (+-4.52 pp a p=0.05; tweets con DEFF +-6.22 pp a p=0.10); motivos y candidatos
reconciliados (882 unicos / 953 asignaciones / 69 multi-motivo); los 22 casos reclasificados
como EMPIRICAL_STRUCTURE_ANOMALY (ninguna violacion de regla confirmada); y la fecha
bibliografica registrada como no verificada. La muestra, sus motivos y las afirmaciones
metodologicas quedan correctamente especificados antes de considerar una revision humana
autorizada.
