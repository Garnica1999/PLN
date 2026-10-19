# P101 - Cumplimiento y limitaciones de procedencia

Fecha: 2026-10-14. Cierre documental; sin anotacion ejecutada; sin
entrenamiento; sin submissions.

## 1. Estado de la rama

- Sin dos revisores disponibles (decision del propietario 2026-10-14): la
  anotacion humana NO se ejecuta; NO se simula; P6 no se ejecuta.
- P101 = CLOSED_BLOCKED (sin evidencia humana admisible); reabrible solo con
  revisores reales y gobernanza resuelta.

## 2. Hueco de trazabilidad P91-P93 (completado con addendum)

- P91-P93 no constan en DECISIONS.md, RUNLOG.md, PROJECT_STATE.md ni
  experiments.csv; su registro vive en reports/p9x_* y artifacts/.
- Se anade un addendum retrospectivo marcado en RUNLOG.md y DECISIONS.md
  (sin reescribir entradas, sin cambiar fechas, sin borrar nada), que
  identifica los artefactos revisados y separa hechos verificados de
  inferencias.

## 3. Discrepancias y eventos de procedencia (registrados)

- Cuarentena v2: 38 ids de holdout2 en el paquete original (35 principales
  + 3 calibracion); reemplazado por v2.1 holdout-safe (0 reservados,
  verificado en P101-00); aviso de cuarentena preservado.
- Calibracion v2.1: handoff preparado; sin respuestas; exposicion incierta
  -> excluida de elegibilidad por prudencia.
- Desviaciones de esquema en respuestas A/B (A sin columna text; B doble
  archivo) ya documentadas en reports/p92z_*.
- Ronda 2 (C1/C2): dependencia de las respuestas A/B (visibilidad en ronda
  2); registrada.
- Muestras internas P90/P91: incluyeron filas de holdout2 (35 y 69 ids);
  nunca distribuidas a anotadores; sin revision humana.
- Identidades de revisores: solo IDs opacos; sin registro nominal.

## 4. Acceso a particiones reservadas

- Solo columnas de id (nunca texto, labels ni scores): holdout_v2 (columna
  particion), holdout2_ids.csv y toxic_test.csv (columna id; 1154 ids).
- Sin uso de leaderboard; sin submissions nuevas.

## 5. Sin evidencia de contaminacion por gold

- Decisiones humanas bloqueadas antes de revelar el gold (P93-R4); la
  comparacion con el gold (P93-R5) fue interna y autorizada; sin evidencia
  de feedback a los anotadores.

## 6. Preservacion

- Artefactos historicos intactos (inventario con hashes:
  reports/p101_materials_inventory.csv); V03/P98/P99/P100 con hashes
  canonicos verificados en P101-00; sin commits; sin limpieza del worktree.
