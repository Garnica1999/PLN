# V18 — Cumplimiento

## Restricciones del brief

| Restriccion | Estado | Evidencia |
|---|---|---|
| Sin embeddings ni vectorizadores aprendidos | **Cumplido** | las unicas representaciones son
  las 279 variables de V12 y, en P0', el TF-IDF disperso **preexistente** reconstruido con
  la configuracion de E-009, sin ajustarla ni ampliarla |
| Sin corpus, datasets ni etiquetas externas | **Cumplido** | no se ha descargado nada |
| Sin test | **Cumplido** | ningun script lee el test; toda medicion es sobre el train con
  folds agrupados |
| Sin holdout2 para explorar | **Cumplido** | no se ha tocado |
| Sin leaderboard para seleccionar | **Cumplido** | no se ha enviado ninguna submission |
| No podar V12 | **Cumplido** | V15 ya lo cerro y V18 no lo repite |
| No reabrir lo cerrado | **Cumplido** | P1 a P5 son objetivos de entrenamiento, no capas de
  decision, no poda, no meta-combinador global y no enrutamiento |
| Protocolo anidado | **Cumplido** | la cuadricula de P5 se selecciona en una particion
  interior del train exterior; el teacher se cross-fita en inner folds agrupados |
| Adoptar exige ganancia global | **Cumplido** | ningun candidato se adopto, y la mejora de
  frontera por si sola se reporto como evidencia mecanistica, nunca como admision |

## Destilacion, comprobaciones exigidas

| Comprobacion | Estado |
|---|---|
| Teacher entrenado solo con datos AGE permitidos | **si**, sobre las 279 variables de V12 |
| Targets del teacher cross-fitted | **si**, cinco inner folds agrupados; cada ejemplo del
  train exterior fue predicho por un teacher que no lo uso para entrenarse |
| Etiquetas verdadeiras como objetivo principal | **si**, la entropia cruzada manda y la
  senal del teacher es un termino aditivo con peso propio |
| Ninguna etiqueta generada por el teacher | **si**, la senal es una probabilidad continua |
| Predicciones del teacher sobre la validacion exterior | **solo diagnostico**, nunca entran en
  el entrenamiento del student |

## Idioma y caracteres

- Auditoria de idioma del proyecto: **0 errores** sobre 399 ficheros.
- `tools/verificar_caracteres.py`, que nombra el brief, **no existe en este proyecto**. El
  sustituto es `scripts/v13_p0a_auditoria_idioma.py`, mas un escaneo independiente.

## Git

- Rama `kaggle-age-2026-puj-v18` creada desde `46b32f9` con el arbol limpio.
- **Sin push, sin PR, sin merge, sin reescritura de historia y sin modificar remotes.**
- V12, V15, V16 y V17 intactas, con notas de correccion documentadas y sin tocar su codigo.

## Entregables

| Entregable | Estado |
|---|---|
| `v18_gap_audit.md` | entregado |
| `v18_p0_reproduccion.md` | entregado |
| `v18_p1_peso_clases.md` | entregado |
| `v18_p3_margen.md` | entregado |
| `v18_p4_multiobjetivo.md` | entregado |
| `v18_p5_destilacion.md` | entregado |
| `v18_master_results.csv` | entregado |
| `v18_hypothesis_registry.csv` | entregado |
| `v18_candidate_comparison.csv` | entregado |
| `v18_reproducibility.md` | entregado |
| `v18_compliance.md` | entregado |
| `v18_final_research.md` | entregado |
| `v18_distillation.csv` | entregado |
