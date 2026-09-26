# v6: reproducibilidad (V6-09)

Generado por `experiments/v6_09_entregables.py` (V6-09 (Pcierre, ANALYSIS)).

## 1. Como ejecutar cada experimento

Todos con el interprete obligatorio del proyecto:

```bash
/home/carlos/venv/bin/python experiments/v6_00_gap_analysis.py
```

| experimento | script | artefactos | entrena | nota de determinismo |
|---|---|---|---|---|
| V6-00 | `experiments/experiments/v6_00_gap_analysis.py` | v6_hypothesis_gap_analysis.md, v6_hypothesis_gap_analysis.csv, v6_verificaciones_gap_v6.json | no | 15 patrones de implementacion en el codigo, excluyendo los scripts de la fase |
| V6-01 | `experiments/experiments/v6_01_estructura_errores.py` | v6_error_structure.md, v6_error_intersection.csv, v6_label_structure.csv | evaluacion downstream | reconstruye 3 candidatos con el protocolo v2; usa el OOF congelado de V03 |
| V6-02 | `experiments/experiments/v6_02_label_factorization.py` | v6_label_factorization.md, .csv, .json | si, 6 modelos por pliegue | el control B0 debe dar 0.777211 exacto; si no, el codigo esta mal |
| V6-03 | `experiments/experiments/v6_03_ponderacion_grupo.py` | v6_ponderacion_grupo.md, .csv, .json | si, 12 ajustes por esquema | los pesos se cuentan solo sobre el tren de cada pliegue |
| V6-04 | `experiments/experiments/v6_04_char_weight.py` | v6_char_weight.md, .csv, .json | si, 12 ajustes por peso | el control I0 debe dar 0.777211 exacto |
| V6-05 | `experiments/experiments/v6_05_skipgrams.py` | v6_skipgrams.md, .csv, .json | si, 9 ajustes por esquema | el control C1-0 debe dar 0.777211 exacto y 2744 caracteristicas de palabras |
| V6-06 | `experiments/experiments/v6_06_bagging_grupos.py` | v6_bagging_grupos.md, .csv, .json | si, 15 modelos por pliegue | el control H0 debe dar 0.777211 exacto; la semilla del bagging es 20260926 |
| V6-07 | `experiments/experiments/v6_07_ruido_par.py` | v6_ruido_par.md, .csv, .json | si, 12 ajustes por esquema | la deteccion usa el OOF congelado de V03 y NearestNeighbors con 5 vecinos |
| V6-08 | `experiments/experiments/v6_08_aumento.py` | v6_aumento.md, .csv, .json | si, hasta 3.558 filas de train por pliegue | el control J0 debe dar 0.777211 exacto; la transformacion se elige por indice |

## 2. Los controles de reproducibilidad de la fase

Cada track lleva un **control que debe reproducir V03 exactamente** (0.777211, delta
+0.0000, IC95 [0, 0], 0 salvage y 0 damage):

| track | control | resultado |
|---|---|---|
| B | B0 marginales reentrenados | 0.777211 exacto |
| A | A0 `w = 1` | 0.777211 exacto |
| I | I0 `char_weight = 1.0` | 0.777211 exacto |
| C1 | C1-0 sin skip-grams | 0.777211 exacto |
| H | H0 un modelo sin remuestreo | 0.777211 exacto |
| F | F0 sin ponderar | 0.777211 exacto |
| J | J0 sin copias | 0.777211 exacto |

**Los siete controles dieron el numero exacto.** Ese es el resultado mas util de la
fase desde el punto de vista de la confianza: cuando un control fallo (C1-0 en su
primera version, que emitia solo unigramas y daba 0.7554), el control lo detecto antes
de que se publicara ningun delta. Sin el, el resultado de C1 habria sido publicado
como evidencia y era falso.

## 3. Semillas fijadas

- bootstrap pareado: 10.000 repeticiones, semilla **20260926**
- bagging por grupos: semilla **20260926**
- aumento: semilla **20260926**, transformacion elegida por indice de fila
- `LogisticRegression` y `LinearSVC`: `random_state = 42`

## 4. Artefactos de entrada compartidos

| artefacto | quien lo usa |
|---|---|
| `features/oof_v03_final.npz` | V6-01, V6-02, V6-03, V6-04, V6-05, V6-06, V6-07, V6-08 |
| `src/splits.py` (`grupos_fusionados`, `holdout_v2`, `pliegues_corregidos`) | todos |
| `src/evaluation.py` (`metrica_macro`, `bootstrap_diferencia_pareada`) | todos |
| `data/external/embeddings/*.bin` (19 modelos) | no usados: D se cerro por techo |
