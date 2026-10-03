# P60 - Auditoria de cumplimiento

Generado por `experiments/p60_14_auditoria.py`. Fecha 2026-09-28T18:06:46+00:00.

## Como se verifico

Las comprobaciones NO son ceros escritos: cada una lee el codigo o el artefacto y dice
que encontro. Los identificadores prohibidos se buscan en el ARBOL DE SINTAXIS, no como
subcadena de texto, para que un nombre mencionado en un docstring como prohibition no
cuente como uso. Un chequeo que no se puede evaluar imprime SKIP con su motivo, que no es
lo mismo que pasar.

## Resultado por comprobacion

| comprobacion | estado | evidencia |
|---|---|---|
| identificadores prohibidos en p60_00_registro_hipotesis.py | **OK** | 0 de 33 identificadores prohibidos; 142 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_01_contrastive.py | **OK** | 0 de 33 identificadores prohibidos; 289 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_02_asl.py | **OK** | 0 de 33 identificadores prohibidos; 286 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_03_dro.py | **OK** | 0 de 33 identificadores prohibidos; 311 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_04_distribucion.py | **OK** | 0 de 33 identificadores prohibidos; 228 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_05_ruido_por_etiqueta.py | **OK** | 0 de 33 identificadores prohibidos; 218 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_06_intervencion_ag.py | **OK** | 0 de 33 identificadores prohibidos; 244 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_07_transporte.py | **OK** | 0 de 33 identificadores prohibidos; 316 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_09_polyloss.py | **OK** | 0 de 33 identificadores prohibidos; 254 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_10_metareweight.py | **OK** | 0 de 33 identificadores prohibidos; 239 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_11_cerrar.py | **OK** | 0 de 33 identificadores prohibidos; 198 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_12_total.py | **OK** | 0 de 33 identificadores prohibidos; 143 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_13_notebook.py | **OK** | 0 de 33 identificadores prohibidos; 97 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_14_auditoria.py | **OK** | 0 de 33 identificadores prohibidos; 183 identificadores distintos analizados via arbol de sintaxis |
| identificadores prohibidos en p60_registrar.py | **OK** | 0 de 33 identificadores prohibidos; 89 identificadores distintos analizados via arbol de sintaxis |
| patron holdout_como_etiqueta | **OK** | 0 coincidencias fuera de prosa que lo declara prohibido |
| patron test_como_etiqueta | **OK** | 0 coincidencias fuera de prosa que lo declara prohibido |
| patron url_de_kaggle | **OK** | 0 coincidencias fuera de prosa que lo declara prohibido |
| patron api_de_kaggle | **OK** | 0 coincidencias fuera de prosa que lo declara prohibido |
| patron gold_file | **OK** | 0 coincidencias fuera de prosa que lo declara prohibido |
| artefactos sin url de Kaggle, test ni holdout | **OK** | 0 coincidencias en 12 artefactos JSON; el JSON de esta auditoria queda fuera por autorreferencia |
| sha256 de la submission de V03 | **OK** | 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e | coincide con el congelado: True |
| corrected_cv del OOF canonico | **OK** | 0.7772114525165721 | exacto: True |
| grupos canonicos sin partidos entre pliegues | **OK** | 1415 grupos, 0 partidos |
| submission de V03 sin modificar | **OK** | fichero presente, 13860 bytes, sha256 verificado |
| LB de V03 leido de CHAMPION.md, sin transcribir | **OK** | leaderboard_max_verified = 0.78306 | no se ha modificado en P60 |
| ningun script lee etiquetas de holdout o de test | **OK** | 0 scripts usan y_holdout, labels_holdout, y_test o labels_test como variable |
| el fichero holdout_v2_ids se usa solo como fuente de grupos | **OK** | 10 scripts lo leen, y en todos casos unicamente la columna grupo_near_dup_autor, que es el agrupador canonico del CV |
| experiments.csv con ids unicos | **OK** | 336 filas, 336 ids unicos |
| los 8 experimentos de P60 registrados | **OK** | ['P60-01', 'P60-02', 'P60-03', 'P60-04', 'P60-05', 'P60-06', 'P60-07', 'P60-08'] |
| pistas cerradas sin experimento SIN fila en experiments.csv | **OK** | B, E, F y K no tienen fila, que es lo que deben tener |
| ningun bootstrap sin gate superado | **OK** | 0 bootstraps, 0 pistas con gate superado |
| ningun finalista declarado | **OK** | finalistas: 0 |
| cero submissions en P60 | **OK** | submissiones: 0 |
| cero consultas a Kaggle | **OK** | consultas: 0 |
| log de leaderboard sin filas nuevas de P60 | **OK** | 22 filas historicas, ninguna nueva; 7 mencionan v03 |
| secciones de P60 presentes en el notebook | **OK** | 14 secciones, celdas 264 -> 264 |
| celdas de codigo del notebook ejecutadas sin fallo | **OK** | 14 ejecutadas, 0 fallidas |
| integracion idempotente | **OK** | anadidas en la ultima pasada: 0 |
| ASCII en scripts, artefactos e informes de P60 | **OK** | 0 bytes no ASCII en 15 scripts, 13 JSON y 12 informes |
| rama de trabajo | **OK** | rama: proyecto-kaggle-toxic-2026-p60 |
| historial de commits | **OK** | 194 commits en la rama |
| la rama no tiene seguimiento remoto, luego no hay push posible desde aqui | **OK** | la rama proyecto-kaggle-toxic-2026-p60 no declara upstream; el repositorio tiene remoto origin, pero esta rama nunca se ha publicado |
| ultimos commits visibles | **OK** | 10 commits listados |
| diff del arbol ya consolidado | **OK** | cambios pendientes: 13 lineas de diff |

## Las prohibiciones, una por una, con como se verifico cada una

| prohibicion | como se verifico |
|---|---|
| transformers, BERT, RoBERTa, LLM | identificadores del arbol de sintaxis en los 15 scripts de P60 |
| embeddings preentrenados | identificadores `from_pretrained`, `hf_hub_download`, `sentence_transformers`, `fasttext`, `gensim`, `glove`, `spacy`, `flair` |
| clasificadores externos | identificadores `flair`, `textblob`, `openai`, `anthropic` |
| etiquetas externas y pseudo-etiquetas | identificadores `pseudo_label`, `pseudoLabel`, `self_training`, `selfTraining` |
| test labels | identificadores `y_test`, `labels_test`, `test_labels` y patron `gold.csv`, `gold_labels`, `solution.csv` en codigo y artefactos |
| holdout reuse | identificadores `y_holdout`, `labels_holdout` y el patron de uso de etiquetas del holdout |
| leaderboard tuning | patron `kaggle.com`, `api.kaggle`, `kaggle competitions` en codigo y en los 13 artefactos |
| corpus externos nuevos | la lista de identificadores anteriores cubre los unloaded publicos; ademas la estructura de datos de P60 no lee ningun directorio de corpus |

## La excepcion declarada: el fichero holdout_v2_ids.csv

Varios scripts de P60 leen `data/processed/holdout_v2_ids.csv`, y eso merece una
explicacion porque el nombre del fichero sugiere lo contrario. Se lee una sola columna,
`grupo_near_dup_autor`, que es el **agrupador canonico del cross-validation** y que
viene de la union de near-duplicate y pseudo-autor. Sin el, los pliegues de V03 no se
podrian reproducir. No se lee ninguna etiqueta de ese fichero, y la comprobacion
correspondiente busca identificadores de etiquetas de holdout y no encuentra ninguno.

## Conteo final

```text
comprobaciones 45 | OK 45 | FALLO 0 | SKIP 0
scripts de P60 analizados 15
artefactos JSON de P60 analizados 13
```

**No hay fallos de cumplimiento.**

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_14_auditoria.py`. Sin push.*