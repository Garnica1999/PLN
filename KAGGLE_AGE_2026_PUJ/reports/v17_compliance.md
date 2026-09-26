# V17 — Cumplimiento

## Restricciones del brief

| Restriccion | Estado | Evidencia |
|---|---|---|
| Sin embeddings ni vectorizadores aprendidos | **Cumplido** | las unicas representaciones son el TF-IDF disperso y la matriz de 312 variables, ambos preexistentes. Las 47 entradas del arbitro y del combinador local son columnas OOF ya calculadas |
| Sin corpus, etiquetas ni modelos externos | **Cumplido** | no se ha descargado ni entrenado nada |
| Sin test | **Cumplido** | ningun script lee el conjunto de test; toda medicion es sobre el train con folds agrupados |
| Sin holdout para explorar | **Cumplido** | el holdout historico no se ha tocado |
| Sin leaderboard para seleccionar | **Cumplido** | no se ha enviado ninguna submission en V17 |
| No modificar V12, V15 ni V16 | **Cumplido** | solo lectura de sus artefactos |
| No modificar submissions historicas | **Cumplido** | ninguna escritura en `submissions/` |
| No podar V12 | **Cumplido** | V15 ya lo cerro y V17 no lo repite |
| Protocolo anidado | **Cumplido** | en P9 y P5 el umbral y la fuerza se eligen en un bloque interior agrupado que el modelo no ve; la validacion exterior no participa |
| Test no usado para seleccion | **Cumplido** | ninguna seleccion usa el test |

## Idioma y caracteres

- Auditoria de idioma del proyecto: **0 errores** sobre 0 ficheros.
- Segundo escaneo independiente sobre los ficheros de V17: sin caracteres de control, sin invisibles, sin emojis y sin alfabetos CJK.
- Sin emojis, sin emoticonos y sin caracteres decorativos en el codigo ni en los informes.
- `tools/verificar_caracteres.py`, que nombra el brief, **no existe en este proyecto**. El sustituto es `scripts/v13_p0a_auditoria_idioma.py`, mas el escaneo independiente.

## Git

- Rama `kaggle-age-2026-puj-v17` creada desde `a818349` con el arbol limpio.
- **Sin push, sin PR, sin merge, sin reescritura de historia y sin modificar remotes.**
- V12, V14, V15, V16, los lexicones y las submissions historicas siguen presentes.

## Entregables del brief: estado

| Entregable | Estado |
|---|---|
| `reports/v17_gap_audit.md` y `.csv` | entregado |
| `reports/v17_p0_reproduccion.md` | entregado |
| `reports/v17_p1_fronteras.md` | entregado |
| `reports/v17_p1b_curva_compuerta.md` | entregado |
| `reports/v17_p1c_regla_conjunta.md` | entregado |
| `reports/v17_p8_perfil_confianza.md` | entregado |
| `reports/v17_p9_arbitro.md` | entregado |
| `reports/v17_p5_combinador_local.md` | entregado |
| `reports/v17_master_results.csv` | entregado |
| `reports/v17_hypothesis_registry.csv` | entregado |
| `reports/v17_member_boundary_matrix.csv` | entregado |
| `reports/v17_boundary_analysis.csv` | entregado |
| `reports/v17_error_intersection.csv` | entregado |
| `reports/v17_candidate_comparison.csv` | entregado |
| `reports/v17_reproducibility.md` | entregado |
| `reports/v17_compliance.md` | este fichero |
| `reports/v17_final_research.md` | entregado |
| 10 figuras | entregado |
| Integracion en el notebook consolidado | **PENDIENTE**: los resultados estan en los informes y en las figuras, pero no se han volcado al notebook. Es una tarea de redaccion, no de medicion, y queda declarada como pendiente en vez de darse por hecha |

