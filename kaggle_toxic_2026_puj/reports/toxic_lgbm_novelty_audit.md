# TOXIC - P88-00: auditoria de novedad de LightGBM sobre la matriz TF-IDF de V03

Fecha: 2026-10-09. Script: `experiments/p88_00_auditoria_novedad_lgbm.py`.
Directiva auditada: arboles potenciados (LightGBM) directamente sobre la matriz
dispersa word+char de V03, tres clasificadores binarios, umbrales congelados.

## 1. Veredicto

**`DUPLICATE`.** El precedente metodologicamente equivalente existe y es **P61-01**
(`experiments/p61_03_lightgbm.py`, 2026-09-28): LightGBM de tres clasificadores
binarios sobre el TF-IDF exacto de V03, mismos pliegues y umbrales. La familia se
cerro por duplicacion entonces. **No se ejecuta nada.**

## 2. La propuesta frente al precedente, elemento por elemento

| elemento de la propuesta | P61-01 (ejecutado) |
|---|---|
| LightGBM, tres clasificadores binarios independientes | si: un `LGBMClassifier` por etiqueta |
| TF-IDF word (1,2) + char_wb (3,5) de V03, tal cual, CSR | si: mismos `TfidfVectorizer` (min_df=2, sublinear_tf) y `hstack(...).tocsr()`; 27.431 rasgos |
| folds agrupados y vectorizadores solo del train | si: pliegues canonicos (D-016) del OOF congelado; `fit` por pliegue sobre el outer_train |
| umbrales historicos congelados | si: `V03_UMBRALES = [0.44, 0.51, 0.46]` aplicados igual a los brazos |
| una sola configuracion fijada antes de ejecutar | P61 midio DOS puntos: LightGBM por defecto y LightGBM con Optuna anidado (30 trials por pliegue, 450 en total); la configuracion unica de la propuesta seria un subconjunto estricto |
| sin early stopping sobre el fold externo | P61 uso early stopping sobre la validacion INTERNA (100 rondas de paciencia), conforme a la propuesta |

## 3. El resultado medido (2026-09-28)

| brazo | corrected_cv | delta vs V03 |
|---|---:|---:|
| 0, V03 | 0.7772114525 | +0.0000000000 |
| 1, LightGBM por defecto | 0.7469717334 | -0.0302397191 |
| 2, LightGBM + Optuna anidado | 0.7513651502 | -0.0258463023 |

Descomposicion: cambio de familia -0.0302397191;
optimizacion +0.0043934168; total -0.0258463023.

Por etiqueta (brazo 2 vs V03): HS 0.7340 vs 0.7617;
TR 0.7779 vs 0.8157; AG 0.7422 vs 0.7542.

Deltas por pliegue del brazo 2 contra los pliegues canonicos de V03:

| pliegue | V03 | brazo 2 | delta |
|---:|---:|---:|---:|
| 1 | 0.7960101143 | 0.7541029547 | -0.0419071596 |
| 2 | 0.7940339498 | 0.7321504592 | -0.0618834906 |
| 3 | 0.7788600289 | 0.7775152111 | -0.0013448178 |
| 4 | 0.7385980131 | 0.7165190307 | -0.0220789824 |
| 5 | 0.7790564997 | 0.7775998204 | -0.0014566793 |

```text
pliegues que mejoran (criterio del precedente, vs media global de V03)   2/5
pliegues vs pliegues canonicos de V03 (tabla)                            0/5
rescate / dano / neto        225 / 253 / -28
gate superado                False
bootstrap                    no ejecutado (el gate no llego)
```

## 4. El cierre historico

- `reports/p61_lightgbm_capacidad.md`: decision **`CLOSED_BY_DUPLICATION`**. "La
  familia no aporta sobre V03 y el precedente OPT-01 ya cerro el patron de busqueda
  de capacidad por OOF".
- `DECISIONS.md` (D-143): la familia LightGBM era nueva en el registro (LightGBM,
  XGBoost y CatBoost: cero filas) y aun asi no llega; la busqueda Optuna FUNCIONA
  (recupera +0,004393, casi un 15 % del deficit) y lo que no llega es la familia.
- El riesgo se declaro antes de ejecutar: 27.431 rasgos frente a 1.186 filas de
  entrenamiento es un regimen donde los arboles no son naturalmente fuertes.
- `OPT-01` (transferencia): Optuna sobre el propio TF-IDF de V03 gano +0,008565 en
  OOF y se descarto porque no transfirio (0,77602 frente a 0,78306 de V03 en el
  leaderboard, -0,028). "La busqueda por OOF premia capacidad que no generaliza".

## 5. Familias vecinas de arboles, todas cerradas

| experimento | modelo | representacion | resultado | estado |
|---|---|---|---|---|
| P61-01 | LightGBM (3 binarios; default + Optuna) | TF-IDF de V03 tal cual (27.431, CSR) | -0.0302 / -0.0258; 2/5; neto -28 | CLOSED_BY_DUPLICATION |
| P61-05 | LightGBM como miembro de ensemble | scores del pool | supera el gate en la particion canonica; REJECTED en la segunda | cerrado |
| V8-13 | DecisionTree + RandomForest | TF-IDF de V03 reducido a subespacio discriminante (2.000 columnas) | ninguna alternativa supera al control | cerradas sin envio |
| V4-G01-G03 | HistGB 0.7297 / ExtraTrees 0.7033 | SVD-300 (denso) | muy por debajo de V03 | cerrado |
| P64-05 | HistGradientBoosting (MI_arbol) | bloque multiple-instance (canal denso) | unico residuo con componente no lineal; score regenerado en P66-01 | cerrado |
| P66-01 | HistGB para regenerar scores | espacio de probabilidad | REPRODUCEN 5, NO VERIFICABLE 1, DIFERENCIA NO RESUELTA 1 | cerrado |
| P68-01 | HistGB para enrutado | estructura del ejemplo (denso) | INVALIDATED, -0.0018992302, 1/5 pliegues | cerrado |

## 6. Contra el gate de la propuesta

1. Ningun punto medido se acerca: los dos brazos estan a -0.026 y -0.030, frente al
   +0.003 exigido; 2/5 pliegues; neto -28; bootstrap ni se ejecuto (el gate no llego).
2. El patron de la familia ya esta cerrado por partida doble: la busqueda de capacidad
   por OOF (OPT-01: +0.008565 OOF, -0.028 LB) y el cambio de familia (P61-01).
3. Ejecutar la configuracion unica de la propuesta seria re-medir un espacio ya
   cubierto por los dos brazos de P61-01, sin informacion nueva posible.

Conclusion: la hipotesis queda **cerrada sin ejecucion**. No se abre ninguna via y V03
permanece congelado.

## 7. Compliance

- Operaciones reales: solo lectura de `experiments/p61_03_lightgbm.py`,
  `reports/p61_lightgbm_capacidad.*`, `experiments/v8_13_clasificadores_alternativos.py`,
  `experiments.csv`, `DECISIONS.md` y el notebook. Cero entrenamientos; train/test/
  holdout no se abren para desarrollo; 0 submissions; 0 push; V03 y sus artefactos no
  se tocan.
- Escaneo byte a byte de lo generado: 0 bytes > 127.
- Hashes verificados: OOF `559265816ce4f631...`; submission `7d7478f271e6edfd...`.

## 8. Hashes de la evidencia

```text
p61_03_lightgbm.py              6b1b0ab204d6a24b8bf8942a83f6f6c39f2789fc06cbbe51a6ed171fd428da6d
p61_lightgbm_capacidad.json     8d25e1822139e86a07f639768cfd3658476b148cd4cbfb84c932a23a7b084288
p61_lightgbm_capacidad.md       f979c095b61f2dbd2e9568cb7e2d8a64101da89f05388487d182dcefa2215c30
```

> V03 remains the best verified leaderboard result.
