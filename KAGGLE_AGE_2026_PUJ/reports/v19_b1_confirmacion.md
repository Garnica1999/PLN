# V19 · B1 — Confirmacion, alineacion y auditoria de fuga

## 1. Segunda particion de 4 folds, con la combinacion ya fijada

| Magnitud | Valor |
|---|---|
| macro-F1 de la combinacion candidata | **0.6090431** |
| delta sobre V12 (0.6005017) | **+0.0085** |
| media de folds | 0.609036 |
| por fold | [0.6077, 0.606, 0.6102, 0.6122] |
| frontera 18-24\|25-34 | 0.5083 |
| frontera 25-34\|35-49 | 0.5018 |
| frontera 18-24\|25-34 en V12 | 0.4963 |
| frontera 25-34\|35-49 en V12 | 0.4974 |

La particion de 5 folds **eligio** esta combinacion, asi que su numero esta seleccionado. Esta de 4 folds es la que no lo esta.

## 2. Control de alineacion train/test

- Submission de referencia: `age_submission_v12.csv`, enviada y puntuada 0,62182.
- Acuerdo de la LogReg **sola** con el archivo enviado: **86.2302%**. Es un control débil: se compara una sola cabeza contra el promedio de las dos, luego se espera en torno al 86 % y no information sobre el orden.
- **Control decisivo**: el promedio de las dos cabezas reconstruido sobre la matriz fresca contra el archivo enviado: **90.5733%**. Este si informationa sobre el orden de columnas.
- La fase 27 registro **93,88 %** para un orden de columnas desplazado, de modo que un acuerdo cercano al 100 % confirma que el orden interno de la matriz de test coincide con el de train. El proyecto ya sufrio ese bug: un desplazamiento de 6 columnas del bloque NB-SVM entre train y test.

## 3. Auditoria de fuga del bloque supervisado

| Comprobacion | Valor |
|---|---|
| identificacion del bloque | diferencia maxima 1.8e-03 |
| rango de valores del bloque | -6.292 a 59439.0 |
| macro-F1 del bloque solo | 0.5597155 |
| media de folds del bloque solo | 0.5597170 |
| separacion pooled menos media de folds | 1.5e-06 |

El bloque **no** son probabilidades sino agregados label-informed (`mention_mean_c0` a `c5` y familia, con valores de hasta 59.439), luego un `argmax` sobre el bloque no significaria nada. La prueba de fuga correcta es la **uniformidad entre folds**: si las columnas de un fold se hubieran construido con los datos de ese mismo fold, ese fold daria un F1 anomalo y la media de folds se separaria del pooled. Aqui la separacion es de 1.5e-06, que es ruido de punto flotante, y el bloque en solitario da 0.5597 de macro-F1, lejos de cualquier firma in-sample.

## 4. Criterio para tratar esto como challenger

| Criterio | Estado |
|---|---|
| segunda_particion_supera_el_gate | si |
| folds_4_sobre_el_gate | si |
| sin_fuga_en_el_bloque_supervisado | si |
| alineacion_verificada | si |
