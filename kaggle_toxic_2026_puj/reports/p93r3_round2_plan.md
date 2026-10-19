# P93-R3 - Plan de la segunda ronda de adjudicacion independiente

Fecha: 2026-10-13. Preparacion; NO distribuida. El gold no se compara en
esta fase (GOLD_COMPARISON_NOT_YET_AUTHORIZED).

## 1. Diseno

- Cada revisor recibe una vista PROPIA de las mismas 176 celdas (mismos
  textos y respuestas A/B; IDs y orden distintos; sin la respuesta ni las
  notas del otro revisor).
- Cada fila muestra: texto, target_label, respuesta inicial propia y su
  nota, respuestas A y B de esa etiqueta (sin sus notas), guia v2.1, y las
  columnas final_response / final_rationale / remaining_uncertainty vacias.
- Justificacion obligatoria para cada cambio y para cada UTD final.
- No se muestra: gold, V03, OOF, umbrales, folds, motivos de seleccion,
  componente, grupos, el otro revisor, estadisticas de acuerdo, ni que la
  celda difiere del gold. Las respuestas A/B se presentan solo como
  evidencia de la ronda de revision.

## 2. Paquetes

- artifacts/p93_adjudication_v2/annotator_C1/ y annotator_C2/ (CSV de 176
  celdas + guia v2.1 + README + manifiesto).
- Semillas: {'C1_ids': 20261028, 'C1_orden': 20261029, 'C2_ids': 20261030, 'C2_orden': 20261031}. Asignacion interna de IDs:
  reports/p93r3_round2_id_assignment.csv.
- Un mismo tweet puede aparecer en varias filas (una por etiqueta); no son
  tweets distintos (explicado en el README).

## 3. Post-recepcion (P3)

Cuando lleguen las decisiones finales de C1 y C2, se preparara la tabla
interna de reconciliacion (plantilla lista: p93r3_final_reconciliation_template.csv)
con categorias: FINAL_AGREEMENT_BINARY, FINAL_AGREEMENT_UTD, FINAL_DISAGREEMENT,
POLICY_AMBIGUITY, MISSING_OR_INVALID. Para FINAL_DISAGREEMENT y POLICY_AMBIGUITY
se propone una tercera revision independiente (sin gold ni V03; solo texto,
guia y evidencia minima). Si no hay tercer revisor disponible, los casos quedan
sin resolver: no hay desempate automatico.

## 4. P4 (no autorizado)

La comparacion con el gold se realizara solo tras bloquear las decisiones
finales, en una fase separada; sin CONFIRMED_ANNOTATION_ERROR y sin corregir
etiquetas en P93-R3.

## 5. Limitaciones vigentes

- ROUND1_BLINDING_NOT_FULLY_VERIFIABLE (ver p93r3_round1_blinding_audit.md).
- La coincidencia A/B (y de C1/C2 con ellos) es evidencia para investigar,
  no autorizacion para modificar etiquetas.
- Componente enriquecido no representativo: sin estimaciones poblacionales.
