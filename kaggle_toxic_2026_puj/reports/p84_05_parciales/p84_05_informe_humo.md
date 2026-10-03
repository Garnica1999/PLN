# P84-05: correccion de prevalencia EM con shrinkage (simulacion OOF)

- Fecha: 2026-10-02; script: `experiments/p84_05_em_shrinkage_simulado.py`.
- Modo: humo; corrida completa: False; pliegues: [1].
- Preregistro: docs/p84_spec.md seccion 6 + docs/p84_preregistro.md; sin test, sin holdout, sin submissions.

## 1. Supuesto y estimando

Pesos de importancia w_i = rho si y_i = 1 y w_i = 1 si y_i = 0 sobre las filas del OOF; estimando: F1 ponderada 2*sum(w*y*pred)/(sum(w*y)+sum(w*pred)) por etiqueta y su media. Simulacion analitica determinista que asume p(x|y) invariable.

## 2. Gate 0

- sha256 del npz verificado; corrected_cv recalculado 0.7772114525165721 (referencia 0.7772114525165721); V03@0.5 recalculado 0.771778476256122 (referencia 0.771778476256122).
- Contadores: dev 1483 / holdout 0 / test 0; toxic_test.csv nunca se abre.
- Pliegues: tamanos [297, 297, 296, 296, 297]; umbrales [0.44, 0.51, 0.46].

## 3. Seleccion LOFO (minimax con guardarrail)

| pliegue | etiqueta | m | t | t equiv | minimax | tasa rho=1 | tasa V03 | desv | factible |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HS | 0 | 0.52 | 0.4380 | 0.743306 | 0.460371 | 0.459528 | +0.000843 | True |
| 1 | TR | 50 | 0.66 | 0.5154 | 0.794061 | 0.300169 | 0.302698 | -0.002530 | True |
| 1 | AG | 50 | 0.61 | 0.4598 | 0.727248 | 0.383642 | 0.382799 | +0.000843 | True |

Optimo sin restriccion (diagnostico):

| pliegue | etiqueta | m | t | minimax | tasa rho=1 | desv vs V03 |
|---|---|---|---|---|---|---|
| 1 | HS | 25 | 0.44 | 0.746406 | 0.551433 | +0.091906 |
| 1 | TR | 0 | 0.61 | 0.803085 | 0.327993 | +0.025295 |
| 1 | AG | 25 | 0.58 | 0.734081 | 0.419056 | +0.036256 |

## 4. Resultados LOFO

| pliegue | delta (promedio de las 5 rho) |
|---|---|
| 1 | -0.025780 |

Delta medio: -0.025780 (umbral +0.002); delta medio por pliegue: -0.025780; pliegues mejorados: 0/5.

| rho | delta candidato vs V03 | delta EM (vs solo-umbral) | delta solo-umbral vs V03 |
|---|---|---|---|
| 0.85 | -0.042708 | +0.016859 | -0.059567 |
| 0.9 | -0.032062 | +0.029426 | -0.061489 |
| 0.94 | -0.021631 | +0.041271 | -0.062902 |
| 1.0 | -0.016487 | +0.048353 | -0.064840 |
| 1.05 | -0.016012 | +0.050298 | -0.066310 |

## 5. Controles

- Descomposicion: componente EM medio +0.037241 (media por pliegue +0.037241); componente umbral medio -0.063021.
- Sanidad sin shift rho=1.00: delta -0.016487 (p1 -0.016487).
| etiqueta | tasa cand rho=1 | tasa V03 | desviacion | cumple |
|---|---|---|---|---|
| HS | 0.404040 | 0.447811 | -0.043771 | False |
| TR | 0.259259 | 0.289562 | -0.030303 | False |
| AG | 0.323232 | 0.356902 | -0.033670 | False |

| etiqueta | rho | tasa candidato | tasa solo-umbral |
|---|---|---|---|
| HS | 0.85 | 0.367003 | 0.373737 |
| HS | 0.9 | 0.380471 | 0.373737 |
| HS | 0.94 | 0.380471 | 0.373737 |
| HS | 1.0 | 0.404040 | 0.373737 |
| HS | 1.05 | 0.414141 | 0.373737 |
| TR | 0.85 | 0.245791 | 0.218855 |
| TR | 0.9 | 0.249158 | 0.218855 |
| TR | 0.94 | 0.259259 | 0.218855 |
| TR | 1.0 | 0.259259 | 0.218855 |
| TR | 1.05 | 0.269360 | 0.218855 |
| AG | 0.85 | 0.282828 | 0.252525 |
| AG | 0.9 | 0.296296 | 0.252525 |
| AG | 0.94 | 0.303030 | 0.252525 |
| AG | 1.0 | 0.323232 | 0.252525 |
| AG | 1.05 | 0.333333 | 0.252525 |

- Rescate/dano a rho=1.00: 0 / 32 (neto -32).

## 6. Bootstrap pareado (filas, con pesos)

- No ejecutado: el bootstrap y el gate exigen la corrida completa de los 5 pliegues.

## 7. Gate y decision

- Gate no evaluable: corrida parcial o humo: el gate exige los 5 pliegues LOFO
- Decision provisional: PROVISIONAL: sin decision de cierre ni de promocion

## 8. n_eff de Kish (filas de seleccion por pliegue)

| pliegue | etiqueta | n_eff rho=0.85 | n_eff rho=0.9 | n_eff rho=0.94 | n_eff rho=1.0 | n_eff rho=1.05 |
|---|---|---|---|---|---|---|
| 1 | HS | 1178.52 | 1182.82 | 1184.89 | 1186.00 | 1185.30 |
| 1 | TR | 1179.87 | 1183.35 | 1185.07 | 1186.00 | 1185.39 |
| 1 | AG | 1178.95 | 1182.98 | 1184.95 | 1186.00 | 1185.33 |

n_eff de Kish sobre las 1483 filas de desarrollo (referencia):

| etiqueta | n_eff rho=0.85 | n_eff rho=0.9 | n_eff rho=0.94 | n_eff rho=1.0 | n_eff rho=1.05 |
|---|---|---|---|---|---|
| HS | 1473.66 | 1479.02 | 1481.62 | 1483.00 | 1482.13 |
| TR | 1475.36 | 1479.70 | 1481.84 | 1483.00 | 1482.24 |
| AG | 1474.18 | 1479.22 | 1481.68 | 1483.00 | 1482.16 |

## 9. Salidas y trazabilidad

- Parciales: /home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj/reports/p84_05_parciales/p84_05_p1_humo.json.
- Tiempo total: 0.1 s; device CPU (numpy puro); version numpy 1.26.4, sklearn 1.8.0.
- La superficie completa (m, t, rho) por pliegue y etiqueta esta en el JSON.
