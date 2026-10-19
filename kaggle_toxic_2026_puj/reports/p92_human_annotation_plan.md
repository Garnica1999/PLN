# P92 - Plan de revision humana de etiquetas (preparacion del piloto)

Fecha: 2026-10-09 (fecha del proyecto). Fase de PREPARACION: no se ha anotado nada; la
anotacion humana comienza cuando los archivos se entreguen a personas reales. Sin
entrenamiento, sin modificacion de etiquetas, sin test/holdout, sin submissions.

## 1. Motivacion

P91 concluyo NO_CONFIRMED_LABEL_ISSUE con ANNOTATION_POLICY_UNRESOLVED y
GOVERNANCE_REVIEW_REQUIRED, y dejo listo un protocolo y una cola de candidatos. P92
construye el paquete ciego reproducible para dos anotadores independientes. El objetivo es
estimar el desacuerdo humano con las etiquetas originales y estudiar tipos de ambiguedad,
SIN modificar el dataset ni el proceso de validacion.

## 2. Procedencia y elegibilidad (P1)

| source_path | partition | row_count | eligible | test_status | reason |
|---|---|---|---|---|---|
| data/raw/toxic_train.csv | train (dev + holdout) | 1731 | 1483 | n/a | elegibilidad demostrada por id contra la particion congelada |
| data/processed/holdout_v2_ids.csv | dev=1483 / holdout=248 | 1731 | 1483 | n/a | manifiesto de particion congelado (hash verificado) |
| features/oof_v03_final.npz | dev | 1483 | 1483 | n/a | interseccion exacta con la particion dev |
| data/raw/toxic_test.csv | test | no inspeccionado | 0 | PROHIBIDO: no accedido | excluido por reglas |

Elegibilidad demostrada por id: la particion dev (1483) coincide EXACTAMENTE con los ids
del OOF canonico (verificado por interseccion); el holdout (248) queda excluido por
particion; el test no se accedio. La seleccion se construyo a partir de ids elegibles
verificados (no se copio ningun archivo completo al paquete).

## 3. Muestras

- **Componente A (referencia probabilistica)**: 100 tweets unicos (300 celdas),
  recuperados de P91 y verificados contra el RNG exacto (semilla 20261010, muestreo
  aleatorio simple sin reemplazo sobre 1483; probabilidad de inclusion 100/1483 conocida).
- **Componente B (enriquecido)**: 150 tweets unicos de 329 disponibles en la cola P91
  (semilla 20261011); seleccion por tiers: PILI 26 +
  DRC 22 + HCD 118 + LSM 11; cap 2 tweets por grupo
  near-dup/fusionado; A y B disjuntos. B NO es representativa de la poblacion (seleccion
  deliberada por motivos de revision): no se usa para estimar tasas sin ponderacion.
  Nota de codigos (P92-R): DRC es un identificador heredado del criterio de seleccion
  (tension estructural empirica AG=1/HS=0; P91-R la clasifico EMPIRICAL_STRUCTURE_ANOMALY);
  NO constituye una violacion normativa confirmada. Los 22 casos historicos DRC estan los 22
  en el componente B. Los motivos de seleccion no son visibles para los anotadores.
- **Calibracion**: 15 tweets fuera de A y B (semilla 20261012); ronda de familiarizacion; NO
  forma parte de las estimaciones de frecuencia.
- **Principal**: 250 tweets unicos (750 anotaciones de etiqueta por anotador; 3 respuestas
  posibles por anotacion: POSITIVE/NEGATIVE/UNABLE_TO_DETERMINE).

## 4. Paquete ciego

`artifacts/human_annotation_v1/`: `annotator_A.csv`, `annotator_B.csv` (mismos 250 casos,
orden aleatorio independiente; semillas 20261015 y 20261016), `calibration_sample.csv`,
`adjudication_template.csv` (750 filas sin decisiones), `README_annotators.md`.
Los `review_id` son opacos (`P92R-####`, asignados con semilla 20261013) y permiten alinear
respuestas sin exponer ids de origen. Oculto por completo: ids originales, etiquetas,
probabilidades OOF, predicciones, umbrales, folds, componente, motivos de seleccion y
grupos. Guia aplicable: `p92_annotation_guidelines_v1.md` (version v1).

## 5. Clave interna

`reports/p92_internal_sampling_key.csv` (FUERA del directorio del paquete): vincula cada
`review_id` con el id original, componente, motivos, estrato, etiquetas originales, grupo y
particion. Solo para auditoria y evaluacion posterior.

## 6. Acuerdo y adjudicacion (P6, al recibir respuestas)

Por etiqueta: acuerdo bruto, acuerdo positivo y negativo, Cohen kappa (2 anotadores) y
Krippendorff alpha cuando corresponda, matriz de confusion entre anotadores, tasa de
UNABLE_TO_DETERMINE y acuerdo exacto de la combinacion HS/TR/AG. Limitaciones: con
prevalencias extremas (TR positivo 30 por ciento; AG sin HS raro) kappa se degrada; nunca
se resume la calidad con una sola metrica. Adjudicacion: plantilla separada; conservar
ambas respuestas; sin reglas automaticas; sin mayoria artificial; tercer revisor
cualificado cuando sea posible; UNABLE_TO_DETERMINE se conserva si no hay decision.

## 7. Niveles de evidencia (P7)

NOT_ANNOTATED -> INDEPENDENT_ANNOTATIONS_COMPLETE -> DISAGREEMENT_REVIEW_PENDING ->
ADJUDICATED -> SILVER_REFERENCE_V1 -> HIGH_CONFIDENCE_ADJUDICATED_REFERENCE. No se
denomina "gold standard" al resultado; no se reemplazan las etiquetas originales; las
capas (original, anotador A, anotador B, adjudicado, estado) se conservan separadas.

## 8. Plan analitico posterior (P8, no ejecutado)

Componentes separados (A probabilistica; B enriquecida; C calibracion). Margenes P91-R
verificados: por etiqueta +-4.52 pp (p=0.05, DEFF=1.2); +-6.22 pp (p=0.10, DEFF=1.2);
agregado de 300 celdas +-2.61 pp. DEFF sujeto a sensibilidad; dependencia entre etiquetas
del mismo tweet considerada; numeradores y denominadores siempre; el conjunto enriquecido
no se generaliza a la poblacion; no se usa para seleccionar modelos o parametros.

## 9. Revision de etiquetas originales (P9)

Estados separados: MODEL_DISAGREEMENT, ANNOTATOR_DISAGREEMENT, POSSIBLE_LABEL_ISSUE,
ADJUDICATED_DIFFERENCE, CONFIRMED_ANNOTATION_ERROR, APPROVED_DATASET_CORRECTION. Una
correccion confirmada requiere regla aplicable + anotaciones independientes + adjudicacion
documentada + decision explicita + autorizacion de gobernanza. En P92 NO se modifica
ninguna etiqueta.

## 10. Estado del arte (P10)

Se reutiliza `reports/p91_label_state_of_art.md` (confident learning, deteccion de errores
de anotacion, acuerdo, cegamiento). No se repite la busqueda; no se instalan herramientas;
no se ejecuta confident learning.

## 11. Limitaciones del piloto

Guia provisional sin criterios operativos oficiales; B no representativa; dependencia por
grupos; dos anotadores como minimo (la adjudicacion requiere un tercero cualificado);
ninguna conclusion de "gold" ni de correccion sin autorizacion separada.
