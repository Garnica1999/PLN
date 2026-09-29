# P33 y P34 - auditoria de cumplimiento por AST

## Por que AST y no grep

Un `grep` de la palabra "embeddings" encuentra **el nombre de la variable que la declara**, y este propio archivo contiene la palabra. Recorrer el arbol de sintaxis encuentra el `import` y la llamada, que es lo unico que puede constituir un uso. Por eso los dos metodos darian resultados distintos y el AST es el que vale.

## Hallazgos

| Recurso | Hallazgos |
|---|---|
| embeddings | 0 |
| transformers | 0 |
| LLM | 0 |
| datos_externos | 0 |
| pseudo_labeling | 0 |
| test_labels | 0 |
| holdout | 0 |
| score_por_suma | 0 |

## Comprobaciones positivas

Una auditoria que solo busca ausencias pasa un proyecto vacio. Estos son los mecanismos permitidos que **si** tienen que estar presentes:

| Mecanismo | Scripts que lo contienen |
|---|---|
| validacion_agrupada | 1 |
| oos_estricto | 1 |
| inner_train_separado | 3 |
| bootstrap | 3 |
| control_barajado | 3 |
| control_aleatorio | 2 |
| score_type_declarado | 4 |

## Restricciones especificas de esta fase

| Restriccion | Estado | Como se comprueba |
|---|---|---|
| `decision_function` tratado como probabilidad | **0 usos** | busqueda en codigo del patron de division por suma, excluyendo los comentarios que la prohiben |
| `outer_val` usado para ajustar | **0 usos** | `a`, `b` en `inner_train`; `beta` en `inner_val`; `outer_val` solo recibe la correccion ya fijada |
| Comprobacion de `score_type` | **presente** | `reports/p34_score_inventory.csv` declara el tipo de cada candidato |
| Llamadas HTTP | **0** | AST sobre imports y llamadas |
| Test y holdout | **0** | AST; ninguna fase los abre |

## Resultado

**Cumplimiento global: APROBADO**.** 0 hallazgos y 7 de 7 comprobaciones positivas presentes.
