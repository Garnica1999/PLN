# P84-02: fastText supervisado desde cero (EXP-02)

- Fecha: 2026-10-02
- Script: experiments/p84_02_fasttext_supervisado.py
- Especificacion: docs/p84_spec.md seccion 3; preregistro docs/p84_preregistro.md
- Referencia V03: corrected_cv 0.7772114525165721; V03@0.5 0.771778476256122
- V03 remains the best verified leaderboard result.
- Modo: corrida completa

## Gate 0

- sha256 toxic_train.csv: 3d40e7345b588860... OK
- sha256 oof_v03_final.npz: 559265816ce4f631... OK
- sha256 grupos_near_duplicados.csv: afabf9ae1c295ea8... OK
- sha256 fasttext (binario oficial): 59c51cc53802bcad... OK
- Contadores: dev 1483 / holdout 0 / test 0
- Particiones: dev 1483 + holdout 248; pliegues [297, 297, 296, 296, 297] sin grupos compartidos
- Identidad V03: corrected_cv reconstruido 0.777211452517 (delta 0.000e+00); V03@0.5 0.771778476256 (delta 0.000e+00); tolerancia 1e-12

## Determinismo (tolerancia 0)

- (principal, HS, pliegue 1) con E*=5: max|delta proba| = 0.0; identicas = True

## Epocas y umbrales anidados

| pliegue | brazo | E_j (j=0,1,2) | E* | umbrales anidados |
|---|---|---|---|---|
| 1 | principal | [5, 5, 5] | 5 | {'HS': 0.2, 'TR': 0.5, 'AG': 0.5} |
| 2 | principal | [5, 5, 5] | 5 | {'HS': 0.2, 'TR': 0.5, 'AG': 0.5} |
| 3 | principal | [5, 5, 5] | 5 | {'HS': 0.5, 'TR': 0.5, 'AG': 0.5} |
| 4 | principal | [5, 5, 5] | 5 | {'HS': 0.5, 'TR': 0.5, 'AG': 0.5} |
| 5 | principal | [5, 5, 5] | 5 | {'HS': 0.2, 'TR': 0.5, 'AG': 0.5} |
| 1 | control | [20, 20, 20] | 20 | {'HS': 0.31, 'TR': 0.34, 'AG': 0.24} |
| 2 | control | [20, 20, 20] | 20 | {'HS': 0.4, 'TR': 0.46, 'AG': 0.29} |
| 3 | control | [20, 20, 20] | 20 | {'HS': 0.3, 'TR': 0.39, 'AG': 0.34} |
| 4 | control | [20, 20, 20] | 20 | {'HS': 0.31, 'TR': 0.38, 'AG': 0.4} |
| 5 | control | [20, 20, 20] | 20 | {'HS': 0.41, 'TR': 0.39, 'AG': 0.32} |

## Lecturas y comparacion con V03 (OOF congelado)

| brazo | lectura | F1 | delta global | delta medio/pliegue | pliegues | IC95 | rescate | dano |
|---|---|---|---|---|---|---|---|---|
| principal | anidada | 0.583398 | -0.193813 | -0.195036 | 0/5 | [-0.217172, -0.171220] | 319 | 1215 |
| principal | umbral_0_5 | 0.584529 | -0.192682 | -0.193910 | 0/5 | [-0.215951, -0.170063] | 318 | 1201 |
| principal | congelada | 0.387112 | -0.390099 | -0.390202 | 0/5 | [-0.412069, -0.367419] | 367 | 1806 |
| control | anidada | 0.737771 | -0.039441 | -0.039652 | 0/5 | [-0.054696, -0.024475] | 165 | 327 |
| control | umbral_0_5 | 0.717476 | -0.059736 | -0.059817 | 0/5 | [-0.079592, -0.040974] | 205 | 292 |
| control | congelada | 0.725733 | -0.051479 | -0.051700 | 0/5 | [-0.069803, -0.034102] | 176 | 270 |

## Principal vs control (aporte sublexico)

- Delta medio por pliegue (anidada): -0.155385; pliegues mejorados 0/5; IC95 [-0.177684, -0.131288]

## ensemble_gain (media de probabilidades a 0.5)

- principal: gain +0.000000 con IC95 [+0.000000, +0.000000]; F1 media 0.771778; V03@0.5 0.771778476256
- control: gain -0.028430 con IC95 [-0.041775, -0.015810]; F1 media 0.743349; V03@0.5 0.771778476256

## Correlaciones con V03

- principal global (r, rho) = (0.519432, 0.585028); HS: r=0.657631, rho=0.593223; TR: r=0.714526, rho=0.660225; AG: r=0.664866, rho=0.573275
- control global (r, rho) = (0.918673, 0.906276); HS: r=0.908829, rho=0.899752; TR: r=0.930666, rho=0.884972; AG: r=0.908555, rho=0.880602

## Matriz de solapamiento de errores (vs V03 congelado, lectura anidada)

| etiqueta | ambos correctos | solo candidato error | solo V03 error | ambos error | jaccard |
|---|---|---|---|---|---|
| HS | 513 | 654 | 142 | 174 | 0.179381 |
| TR | 1122 | 197 | 63 | 101 | 0.279778 |
| AG | 842 | 364 | 114 | 163 | 0.25429 |
- Global: ambos correctos 2477, solo candidato error 1215, solo V03 error 319, ambos error 438, jaccard 0.22211

## Gate y decision

- standalone: {'delta_global': -0.1938133867, 'delta_medio_pliegues': -0.195036466, 'umbral_delta': 0.005, 'pliegues_mejorados': 0, 'pliegues_minimos': 4, 'ic95': [-0.217172, -0.17122], 'pasa': False}
- complementariedad: {'ensemble_gain': 0.0, 'umbral_gain': 0.0015, 'ic95': [0.0, 0.0], 'pasa': False}
- Determinismo OK: True
- Elegible para EXP-06: False
- Decision: CERRAR fastText supervisado: sin rejilla nueva (resultado verificado)
- Motivo: Gate 1 standalone NO: delta global -0.193813 (min 0.005), delta medio/pliegue -0.195036 (diagnostico), pliegues 0/5 (min 4), IC95 [-0.217172, -0.171220]
- Motivo: complementariedad NO: ensemble_gain +0.000000 (min 0.0015), IC95 [+0.000000, +0.000000]

## Incidencias

- Ninguna.

## Trazabilidad

- Tiempo total 206.36 s; versiones {"python": "3.14.7", "numpy": "1.26.4", "pandas": "2.3.3", "sklearn": "1.8.0", "scipy": "1.17.1", "device": "cpu (fastText -thread 1; sin torch en esta corrida)"}
- Hiperparametros fijos: {"dim": 100, "lr": 0.1, "minCount": 2, "bucket": 2000000, "loss": "softmax", "wordNgrams": 2, "thread": 1, "seed": 20260926, "verbose": 0, "nota_verbose": "la spec citaba -verbose 2; se fija 0 (cosmetico, no altera el modelo)"}
- Bootstrap pareado por filas, 10000 remuestreos, semilla 20260926.
