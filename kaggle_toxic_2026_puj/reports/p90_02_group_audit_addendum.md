# P90-02 - Addendum: auditoria de grupos (near-duplicate, pseudo-autor y union de validacion)

Fecha: 2026-10-10. Generado por la comprobacion final P90.02 (documental). Alcance: **conjunto
de desarrollo autorizado** (1483 filas). Reutiliza exclusivamente los metadatos y procedimientos
historicos. **No se crean grupos ni particiones nuevas; no se abre test ni holdout; no se
ejecutan modelos; no se modifica ningun artefacto existente.**

Motivo: el informe integral P90-02 (seccion 15) documento los grupos near-duplicate solo
parcialmente (1479 grupos en dev, 6 filas no unitarias, grupos 112 y 338). Este addendum completa
la documentacion exigida: near-duplicate, pseudo-autor, union usada en la validacion,
distribucion de tamanos, grupos unitarios y no unitarios, y divergencias de labels dentro de los
grupos. Todas las cifras dev se recalcularon con los metadatos historicos y el y_true del OOF
canonico (`features/oof_v03_final.npz`, sha256 55926581...); las cifras de train completo se
citan de los artefactos historicos sin recalcular.

## 1. Grupos near-duplicate (solo near-duplicate)

Fuente: `data/processed/grupos_near_duplicados.csv` (columnas id, split, grupo, tamano_grupo;
2885 filas: train 1731 + test 1154). Umbral 0.85, train+test; documentado en
`reports/leakage_audit.md` (seccion 1b) y usado como variable de agrupacion desde D-004.

- Grupos con mas de un texto en el archivo completo (historico, `leakage_audit.md` 1b): **4**
  (1899: 6 test + 0 train; 112: 2 test + 3 train; 338: 1 test + 3 train; 1272: 1 test + 1 train).
- Desarrollo (1483 filas):
  - grupos distintos: **1479**
  - distribucion de tamanos REAL en dev: **{1: 1477, 3: 2}**
  - filas en grupos no unitarios: **6** (grupos 112 y 338, 3 filas cada uno)
  - tamano declarado (`tamano_grupo`) de esas filas: 5 y 4 (incluye las copias en test);
    el tamano declarado en dev es {1: 1477, 4: 3, 5: 3}
- Divergencia de labels dentro de los grupos no unitarios (y_true del OOF dev):
  - grupo 112: **divergente en HS, TR y AG** (3 filas)
  - grupo 338: sin divergencia
  - grupos no unitarios con alguna etiqueta divergente: **1 de 2** (3 filas)
- Nota: el grupo 1272 tiene su unica fila train en la particion holdout (fuera de dev); no
  afecta al desarrollo.

## 2. Grupos pseudo-autor (primera mencion)

Definicion historica: primera mencion del tweet como pseudo-autor (D-004; control ejecutado en
D-015). El detalle por autor no esta serializado en los artefactos; se citan los agregados.

- `reports/leakage_audit.md` (seccion 3): **17 pseudo-autores con >= 3 tweets (77 tweets, 4.4 %
  del train)**; la tabla lista los 17 (p. ej. youtube 14, barbijaputa 7, sanchezcastejon 6,
  anaisbernal 5, leticiadolera 5, lortegadiaz 4, relatofeminista 4, zurine3 4, drodriguezven 4,
  lopezdoriga 3).
- `experiments/12_control_pseudo_autores.py` + `reports/control_pseudo_autores.json` + fila
  **PA-01** de `experiments.csv` (2026-09-23):
  - 1731 tweets; 822 con mencion; 740 autores distintos; 39 con mas de un tweet
  - **1645 grupos fusionados** (train completo)
  - F1 con grupos fusionados **0.7838245474334548** vs **0.7926588861781015** solo con
    near-duplicados (diferencia **-0.008834**; std de pliegues 0.01855536946824282)
- D-015: fusionar los grupos (near-duplicate UNION pseudo-autor) reduce el optimismo del split
  (F1 OOF 0.7927 -> 0.7838) y acerca el gap local-LB (-0.0128 -> -0.0039); el protocolo vigente
  usa los grupos fusionados.

## 3. Union usada en la validacion (near-duplicate UNION pseudo-autor)

Fuente: `data/processed/holdout_v2_ids.csv`, columna **grupo_near_dup_autor** (1731 filas:
dev 1483 + holdout 248; solo metadatos de id/particion/grupo). Protocolo: StratifiedGroupKFold(5)
con estratificacion por combinacion HS x TR x AG (D-004, corregido por D-015/D-016). El analisis
de este addendum se limita a las filas dev; el archivo se filtro por `particion == dev` antes de
usar cualquier campo.

- Desarrollo (1483 filas):
  - grupos distintos: **1415**
  - distribucion de tamanos: **{1: 1378, 2: 20, 3: 10, 4: 3, 5: 2, 6: 1, 7: 1}**
  - filas en grupos unitarios: **1378**; filas en grupos no unitarios: **105**
- Divergencia de labels en grupos no unitarios (dev): **13 grupos, 43 filas**:
  (18, 6), (77, 3), (109, 2), (112, 3), (120, 7), (194, 3), (218, 3), (246, 3), (477, 3),
  (530, 2), (907, 4), (1352, 2), (1439, 2). Formato: (grupo, n filas).
- Reconciliacion de la cifra 1415: coincide con (a) la unidad de remuestreo de
  `reports/p66_complementarity.md` ("1415 grupos, desde holdout_v2_ids.csv, columna
  grupo_near_dup_autor") y (b) el subconjunto "textos_unicos" n=1415 de V4-F02
  (`reports/discovery_v4.json`, robustez_duplicados).
- Relacion con los grupos near-duplicate solos: la numeracion coincide en 1419 de 1483 ids,
  pero la **particion de equivalencia difiere** (1479 -> 1415 grupos; las fusiones por
  pseudo-autor eliminan 64 grupos en dev). El train completo tiene 1645 grupos fusionados (PA-01).

## 4. Cluster de consistencia (contexto V4-F02)

`features/label_consistency_v4.npz` (generado por `experiments/27_label_consistency_v4.py`):
cuatro criterios independientes (exacto; normalizado; Jaccard >= 0.8; cosimilitud TF-IDF
word(1,1)+char_wb(3,5) >= 0.85 con union-find). 1731 filas; `en_cluster` total **84**;
**dev 68**.

- F1 de V03 por subconjunto (dev, `reports/discovery_v4.json`, robustez_duplicados):
  todos 0.7772 (n=1483); textos unicos **0.7568** (n=1415); en cluster near-duplicado
  **0.9769** (n=68).
- Propagacion por near-duplicate (V4-F04, `discovery_v4.json`): 0 tweets de test con texto
  identico a train; techo maximo de F1 **0.0 puntos**.

## 5. Reconciliacion de cifras de grupos (alcance y fuente)

| cifra | valor | alcance | fuente |
|---|---|---|---|
| grupos near-dup (solo) | 1479 | dev 1483 | grupos_near_duplicados.csv (recalculo dev) |
| distribucion real near-dup | {1: 1477, 3: 2} | dev 1483 | idem |
| tamano declarado near-dup | {1: 1477, 4: 3, 5: 3} | filas dev | idem |
| filas no unitarias near-dup | 6 (2 grupos) | dev 1483 | idem |
| grupos fusionados | 1415 | dev 1483 | holdout_v2_ids.csv (recalculo dev); p66_complementarity.md |
| distribucion fusionada | {1: 1378, 2: 20, 3: 10, 4: 3, 5: 2, 6: 1, 7: 1} | dev 1483 | holdout_v2_ids.csv |
| filas no unitarias fusionadas | 105 | dev 1483 | idem |
| grupos con labels divergentes | 13 (43 filas) | dev 1483 | idem + y_true OOF |
| grupos fusionados | 1645 | train 1731 | PA-01; control_pseudo_autores.json |
| F1 fusionado vs solo-near-dup | 0.7838 vs 0.7927 | train OOF | PA-01; D-015 |
| pseudo-autores >= 3 tweets | 17 (77 tweets, 4.4 %) | train | leakage_audit.md seccion 3 |
| autores distintos / con >1 tweet | 740 / 39 | train | PA-01 |
| en_cluster (consistencia) | 84 / 68 | train / dev | label_consistency_v4.npz |
| F1 unicos / en cluster | 0.7568 / 0.9769 | dev | discovery_v4.json |

## 6. Limitaciones del addendum

- El detalle por pseudo-autor (lista autor -> tweets) no esta serializado en los artefactos
  historicos; solo se dispone de los agregados citados (17/77/4.4 %; 740/39; 1645).
- Las divergencias de labels se miden con el y_true del OOF dev y son descriptivas: no
  demuestran causalidad ni explican el error de V03.
- Los grupos near-duplicate y fusionados son los historicos (D-004/D-015); no se crearon
  agrupaciones nuevas. La particion de test no se analizo.
- El addendum no modifica ningun artefacto previo; es documentacion nueva y separada.
