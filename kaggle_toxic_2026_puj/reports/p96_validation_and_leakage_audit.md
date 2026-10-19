# P96 - Auditoria de validacion, folds y leakage (companion del informe forense)

Fecha: 2026-10-13. Solo lectura y computo determinista. Evidencia cruda:
`reports/p96_folds_evidence.json` y `reports/p96_static_scans.txt`.

## 1. Particion congelada dev/holdout

| Comprobacion | Resultado |
|---|---|
| filas de train | 1731 |
| dev | 1483 |
| holdout | 248 |
| dev y holdout particionan train | SI (union exacta, sin sobrantes) |
| dev y holdout disjuntos | SI |
| orden de desarrollo replicado | SI (orden de train menos holdout, 1483 filas) |

Nota: se leyo unicamente la membresia (`id`, `particion`) del archivo de
particion; no se abrieron etiquetas ni scores del holdout. El holdout no se
uso para ninguna decision de esta fase. La matriz de integridad translitera a
ASCII los campos verbatim copiados del registro (convencion de artefactos
generados de P96).

## 2. Grupos y pliegues (recomputacion determinista)

- `grupos_fusionados` (near-duplicado UNION pseudo-autor) recomputado desde
  `src/splits.py`: **iguales etiqueta a etiqueta** a los guardados en
  `features/v2_oof_base_corrected.npz` (mismo orden de ids, 1415 componentes
  en el desarrollo).
- `pliegues_corregidos` (StratifiedGroupKFold(5, shuffle, seed 42)):
  tamanos [297, 297, 296, 296, 297]; todas las filas asignadas; sin filas
  huerfanas.
- Comparacion con `features/oof_v03_final.npz`: **misma particion**; la
  numeracion difiere (recomputada 0..4, guardada 1..5) con biyeccion exacta
  fila a fila. Traduccion: 0->1, 1->2, 2->3, 3->4, 4->5.
- Fuga de grupos por pliegue: 0 casos (comprobacion directa por interseccion de
  conjuntos de grupos entre entrenamiento y validacion, en los 5 pliegues).
- `y_true` de los artefactos == etiquetas originales del train (identicos).

## 3. Escaneos estaticos (evidencia cruda en p96_static_scans.txt)

| Escaneo | Resultado |
|---|---|
| scripts que mencionan test | 54 (uso permitido: transform/predict para submissions y candidatos) |
| vecindades de test con `.fit`/`fit_transform` cerca | **0** |
| scripts que mencionan holdout | 354 (evaluaciones gobernadas, particion) |
| scripts que mencionan holdout2 | 51 (gobernanza v10+: creacion, presupuesto, auditorias; 0/3 evaluaciones gastadas por registro) |
| ocurrencias `.fit`/`fit_transform` inventariadas | ver archivo (inventario completo) |

## 4. Artefactos excluidos del recomputo (politica P96)

No se abrieron (cubren filas del holdout historico):

- `experiments/baseline_v03_frozen/oof_v03_historical.npz` (1731 filas)
- `features/oof_mod03b.npz` (fase 13a sobre train completo)
- `features/oof_bases.npz` (1731; detectado por membresia de ids)
- `features/oof_ens04_v04.npz` (1731)
- `features/label_consistency_v4.npz` (1731)

## 5. Clasificacion de leakage por ruta auditada

| Ruta | Clase |
|---|---|
| Cadena V03 (04_refit, 14_oof, 23_candidatos, oof_v03_final) | NO_LEAKAGE_FOUND_IN_AUDITED_SCOPE |
| Pliegues/grupos (src/splits + artefactos) | NO_LEAKAGE_FOUND_IN_AUDITED_SCOPE |
| Fits de muestras v2/v4/v6/P84 (lectura dirigida) | NO_LEAKAGE_FOUND_IN_AUDITED_SCOPE |
| Uso de test en scripts (escaneo) | NO_LEAKAGE_FOUND_IN_AUDITED_SCOPE (0 vecindades fit+test) |
| Holdout/holdout2 | Gobernanza vigente; contenidos no consultados; NOT_AUDITABLE por politica |
| Convergencia de entrenamientos sin logs | NOT_AUDITABLE (sin evidencia de fallo) |

No se declara cero leakage de todo el proyecto: el alcance es el listado
anterior. Ninguna ruta auditada produjo `LEAKAGE_CONFIRMED` ni
`LEAKAGE_POSSIBLE`.

## 6. Checklist de la seccion 17 de P96 (resultados reales)

```text
V03 model unchanged              OK (hash identico)
V03 submission unchanged         OK (hash identico)
canonical OOF unchanged          OK (hash identico)
original labels unchanged        OK (y_true == train)
human responses unchanged        OK (hashes P93 intactos)
holdouts untouched               OK (sin accesos)
test content not accessed        OK (sin accesos)
leaderboard unused               OK (valores solo de registros)
no training                      OK
no new predictions               OK
no submissions                   OK
no data corrections              OK
no external labeled resources    OK
no secrets exposed               OK
ASCII-only generated artifacts   OK (verificador oficial + scan propio)
all new artifacts hashed         OK (manifest P96)
Git changes preserved            OK (187 ??, 1 M; sin commit/push)
```
