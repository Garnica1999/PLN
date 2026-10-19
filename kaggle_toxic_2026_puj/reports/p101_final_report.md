# P101 - Informe final (auditoria de supervision y objetivos de entrenamiento)

Fecha: 2026-10-14.
Estado: CLOSED_BLOCKED - rama humana bloqueada por ausencia de revisores;
sin evidencia humana admisible; P6 no ejecutado; labels originales intactas.

## 1. Las diez preguntas (respuestas registradas)

| # | Pregunta | Respuesta |
|---|---|---|
| 1 | Datos de supervision permitidos disponibles | Train/dev + pool historico de decisiones; sin fuente supervisada NUEVA admisible (sin revisores) |
| 2 | Evidencia de errores de la auditoria anterior | No suficiente: 0 confirmados; 14 posibles (solo componente B enriquecido, P93-R5) |
| 3 | Nueva muestra humana realmente independiente | No - bloqueada por ausencia de revisores (decision del propietario) |
| 4 | Errores confirmados | Ninguno; los 14 siguen "posibles" |
| 5 | Proporcion probabilistica vs enriquecida | Historico: A (aleatoria) sin candidatas; nueva A no ejecutada; B aplazado |
| 6 | Gobernanza para objetivos alternativos | AMBIGUOUS (P1); consulta al profesor redactada, no enviada |
| 7 | Mejora con targets alternativos | No evaluable (sin evidencia humana + sin autorizacion) |
| 8 | Complementariedad con V03 | No evaluable |
| 9 | Riesgos y limitaciones | Ruido de evaluacion no medido; sesgo del enriquecido; dependencia A/B en ronda 2; pool de revisores historico; exposicion holdout2 en muestras internas |
| 10 | Seguir o cerrar | CERRAR la rama humana por bloqueo; reabrible solo con revisores reales |

## 2. Lo ejecutado (P0/P1)

- P0 (experiments/p101_00_auditoria_procedencia.py): inventario (229
  archivos con hashes), tabla de procedencia (1.327 filas: dev2 completo +
  universo historico), elegibles 971/1236, reservados en v2.1: 0, canonicos
  verificados. Entregables: reports/p101_provenance_audit.md,
  reports/p101_provenance_table.csv, reports/p101_materials_inventory.csv.
- P1 (reports/p101_governance_note.md): veredicto AMBIGUOUS; plantilla P95
  completada; consulta al profesor redactada (no enviada).

## 3. Artefactos

- reports/p101_provenance_audit.md, p101_provenance_table.csv,
  p101_materials_inventory.csv, p101_governance_note.md, p101_compliance.md,
  p101_final_report.md (este), p101_reproducibility.md.
- experiments/p101_00_auditoria_procedencia.py, p101_90_registrar.py,
  p101_95_integrar_notebook.py, p101_99_verificacion.py.
- experiments.csv: fila P101-00; notebook: seccion 164.

## 4. Que haria falta para reabrir

- Dos revisores reales (bajo las condiciones del propietario 2026-10-14);
- Freeze P2 (poblacion, elegibilidad, exclusiones, n=300, semilla,
  algoritmo, version de guia, plan de analisis) antes de extraer;
- Ejecucion P3-P4; gate de materialidad P5; P6 solo con evidencia humana
  suficiente y gobernanza resuelta (profesor si P1 sigue ambiguo).
