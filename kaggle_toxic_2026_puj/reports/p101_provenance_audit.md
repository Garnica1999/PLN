# P101-00 - Auditoria de procedencia de los materiales de anotacion (P90-P93)

Fecha: 2026-10-14. Solo lectura sobre materiales historicos; sin entrenamiento; sin modificacion de labels; las particiones reservadas se consultaron SOLO por su columna de id (nunca texto, labels ni scores).

## 1. Controles canonicos

Verificados (sha256, prefijos): data/raw/toxic_train.csv 3d40e7345b588860, features/oof_v03_final.npz 559265816ce4f631, experiments/baseline_v03_frozen/submission_v03.csv 7d7478f271e6edfd, experiments/baseline_v03_frozen/model_v03.joblib 263ed5bd0fc0de7c, data/processed/holdout_v2_ids.csv e5b62e5f22c4c63a, data/processed/holdout2_ids.csv f6c6ed958abf728b.

## 2. Universo de ejemplos en materiales historicos

- Muestra principal v2.1 (anotada por A/B en ronda 1): 250 tweets (review_id P92W-0001..0250).
- Calibracion v2.1 (handoff preparado, sin respuestas): 15 tweets (P92W-C01..C15).
- Paquete v2 original: 265 ids, EN CUARENTENA y no distribuido (contaminacion holdout2: 38 ids; 35 principales + 3 calibracion segun el aviso de cuarentena).
- Muestra interna del casebook P90 (sin revision humana): 220 tweets.
- Candidatos P91 (nunca distribuidos; gobernanza pendiente): 429 ids.
- Ronda 2 C1/C2 (P93): 176 celdas sobre 127 tweets.
- Filas de la tabla de procedencia (dev2 completo + universo historico): 1327.

## 3. Particiones y elegibilidad

- dev (particion de oof_v03_final): 1483 | dev2 (dev menos holdout2): 1236 | holdout historico: 248 | holdout2: 247 | ids de test (solo columna id): 1154.
- Interseccion de la muestra principal v2.1 con particiones reservadas (holdout/holdout2/test): 0 (esperado 0). Calibracion: 0 (esperado 0).
- Poblacion elegible para una auditoria humana NUEVA: 971 = dev2 (1236) menos 250 revisados y menos 15 de calibracion excluidos por prudencia. La columna eligibility_new_audit de la tabla lo refleja por ejemplo.

## 4. Exposicion de revisores y del gold

- Ronda 1 (A/B): ciega; sin labels, predicciones ni respuestas cruzadas; guia v2; sin feedback del gold.
- Ronda 2 (C1/C2): decisiones bloqueadas ANTES de revelar el gold (P93-R4); los adjudicadores vieron las respuestas A/B de la ronda 1 (dependencia registrada).
- Gold: revelado solo en la comparacion P93-R5 (interna del proyecto, autorizada); sin evidencia de feedback a los anotadores.

## 5. Hueco de trazabilidad (P91-P93)

Las fases P91-P93 no constan en DECISIONS.md (salta D-295 a D-296), RUNLOG.md (P90-00 a P98), PROJECT_STATE.md ni experiments.csv; su registro vive en reports/p9x_* y artifacts/. Se completa con un addendum retrospectivo marcado (RUNLOG y DECISIONS) y se documentan las limitaciones en reports/p101_compliance.md.

## 6. Limitaciones

- Identidades de revisores: solo IDs opacos (A/B/C1/C2); sin registro nominal.
- Exposicion de la calibracion v2.1 incierta (handoff preparado sin respuestas) -> excluida de elegibilidad por prudencia.
- Los candidatos P91 y la muestra P90 nunca pasaron por revision humana; su elegibilidad se conserva pero se registra la seleccion previa (enriquecida/interna).
- Las muestras internas P90 (35 ids) y P91 (69 ids) incluyeron filas de holdout2 (seleccion interna sobre dev completo; nunca distribuidas; sin revision humana).
- La tabla cubre dev2 completo y el universo historico; no es una estimacion de la tasa de error del train.

Duracion: 0.0 s. Inventario: reports/p101_materials_inventory.csv (229 archivos).
