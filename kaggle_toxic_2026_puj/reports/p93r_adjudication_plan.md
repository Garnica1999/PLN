# P93-R - Plan de adjudicacion ciega de candidatos

Fecha: 2026-10-13. Fase de preparacion; sin entrenamiento, sin
predicciones, sin cambios de etiquetas. NO se distribuye automaticamente.

## 1. Alcance

- Celdas candidatas: 176 (HS 23, TR 87, AG 66).
- Tweets unicos a revisar: 127 (agrupados por tweet para no
  repetir textos; un tweet puede aportar varias celdas).
- Candidato = A y B coinciden en una etiqueta binaria distinta del gold
  original. NO son errores confirmados.
- Los candidatos provienen de un componente aleatorio (A) y uno enriquecido
  (B): no estimar la tasa poblacional de errores con esta muestra.

## 2. Diseno en dos rondas

1. Ronda 1 (independiente): C anota las TRES etiquetas de cada tweet sin
   ver las respuestas de A/B ni el gold (paquete independent_review).
2. Ronda 2 (comparacion): despues de recibir y validar las respuestas de C,
   se prepara la comparacion reviewer_C_initial vs annotator_A vs
   annotator_B (plantilla interna lista; sin decisiones de C por ahora).
   C conserva su decision inicial y, cuando sea necesario, registra una
   decision final razonada; el gold permanece oculto a C en ambas rondas.

## 3. Estados de auditoria (se aplicaran cuando existan respuestas de C)

THIRD_REVIEW_MATCHES_A_B | THIRD_REVIEW_SUPPORTS_ORIGINAL_GOLD |
THIRD_REVIEW_DIFFERS_FROM_BOTH | UNABLE_TO_DETERMINE |
NEEDS_POLICY_CLARIFICATION. Describen resultados de adjudicacion, no
errores confirmados. La comparacion con el gold sera una fase separada y
posterior (HUMAN_DISAGREEMENT / ADJUDICATED_DIFFERENCE / POSSIBLE_LABEL_ISSUE /
CONFIRMED_ANNOTATION_ERROR, con justificacion documentada).

## 4. Cegamiento

- El paquete de la ronda 1 no contiene: ids de origen, gold, predicciones V03,
  OOF, umbrales, folds, respuestas o notas de A/B, etiquetas candidatas,
  motivos de seleccion, componente de muestreo ni grupos.
- No se revela que los tweets fueron seleccionados por coincidencia A/B
  distinta del gold. Auditoria completa: p93r_blinding_audit.md.

## 5. Priorizacion opcional (si el presupuesto humano lo requiere)

Alternativas transparentes (NO aplicadas; la cola preparada es completa):
- Lote completo (recomendado): los 127 tweets unicos.
- Subconjunto por esfuerzo: primero los tweets con UNA sola celda candidata
  (revision mas rapida por tweet).
- Subconjunto por etiqueta: revisar primero HS (la etiqueta con mas peso
  normativo), luego TR y AG.
- No se descarta ningun caso automaticamente; cualquier recorte debe
  registrarse y justificarse.

## 6. Limitaciones

- La guia v2.1 es politica operativa interna; persisten ambiguedades oficiales.
- La coincidencia A/B + C no demuestra automaticamente un error del gold.
- La muestra adjudicada no debe usarse como validacion independiente si sus
  correcciones se usaran para entrenar.

## 7. Hashes

| archivo | sha256 |
|---|---|
| artifacts/p93_adjudication_v1/independent_review/README_adjudicator.md | `c043a2d72cffdb6a912104d729444c998578ad7d2f372e46eeb08aeba12a306d` |
| artifacts/p93_adjudication_v1/independent_review/adjudicator_C.csv | `84ca2179378fa0d88e1e8b99921402219b79493dcd9b8792b7cac7864f7f937d` |
| artifacts/p93_adjudication_v1/independent_review/annotation_guidelines_v2_1.md | `fd8d4c65ccd194fbce866321c7dbbbdc6870653f606e3111f147e41bc7fc9204` |
| artifacts/p93_adjudication_v1/independent_review/manifest.json | `1771f928f738483783eb51b08bc030dabc5350d0576e4921713f9fe770c26ae9` |
| reports/p93r_internal_adjudication_key.csv | `5f3f8eaeff229bcf72913a3d59e69b68581481edfef546ee6cdda61c49cf51e9` |
| reports/p93r_round2_comparison_template.csv | `14919ccdf6ba5befef5e328b71f8291ec3250fc6c67f6c110772c932f152fba8` |
