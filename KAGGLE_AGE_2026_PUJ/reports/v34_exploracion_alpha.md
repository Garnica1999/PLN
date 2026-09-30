# V34 - Exploracion metodologica previa de alpha (registro materializado)

- Fecha: 2026-09-30. Script: `scripts/v34_00b_exploracion_alpha.py`
- sha256 del script: `9d898ab8cf5908effd119fe78047cd0ee1375a2958539f8e4c4ab0468fe3e457`
- Semillas: subconjunto 7, ordenes 1000+pasada, gemelo SGD 20261010
- Presupuesto: 30000 filas de fold-train (fold 0) x 10 pasadas, vectorizador ajustado solo en fold-train (118409 terminos)
- Metrica unica: log_loss multiclase de train. NO se calcula macro-F1 aqui.
- Anclas del subconjunto: H(prior) = 1.7914537 (prior log), ln(K) = 1.7917595 (uniforme), ln(K-1) = 1.6094379 (ancla de margen de la regla de V34)

## No es un grid

Ningun valor se selecciono. La regla de margen que fija `alpha*` en V34 quedo derivada en
forma cerrada ANTES de estas mediciones y no usa ninguna de estas celdas. Ningun umbral de
la puerta G0-G4 se calibro con estas celdas: G1 se ancla en `H(prior) - 0.01` y H(prior) se
calcula del prior de las filas de fold-train, no de esta tabla. Estas 27 celdas NO se
repiten dentro del runner `v34_01`: el runner solo publica la trayectoria de su propia
configuracion congelada.

## Tabla de las 27 celdas

| # | alpha | lambda2 | power_t | L2 | ll_train | ll_train p1 | ll_train p10 | |w|max | s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.004075425 | 1 | 0 | deno | 1.82296 | 1.79402 | 1.82131 | 1.532e-01 | 16 |
| 2 | 0.01 | 1 | 0 | deno | 1.94667 | 1.79741 | 1.92381 | 6.414e-01 | 16 |
| 3 | 0.02 | 1 | 0 | deno | 8.48480 | 1.80372 | 8.14556 | 5.534e+00 | 16 |
| 4 | 0.04 | 1 | 0 | deno | 19.83041 | 1.93360 | 19.28132 | 1.279e+01 | 16 |
| 5 | 0.08 | 1 | 0 | deno | 28.86406 | 6.61964 | 28.86406 | 2.644e+01 | 16 |
| 6 | 0.0005 | 1 | 0 | deno | 1.79545 | 1.79203 | 1.79526 | 1.838e-02 | 16 |
| 7 | 0.001 | 1 | 0 | deno | 1.79918 | 1.79231 | 1.79879 | 3.688e-02 | 15 |
| 8 | 0.002 | 1 | 0 | deno | 1.80672 | 1.79286 | 1.80594 | 7.422e-02 | 16 |
| 9 | 0.004075425 | 1e-06 | 0 | deno | 1.82300 | 1.79403 | 1.82135 | 1.532e-01 | 15 |
| 10 | 0.05 | 1e-06 | 0 | deno | 24.91242 | 2.54828 | 24.25449 | 1.622e+01 | 16 |
| 11 | 0.176 | 1e-06 | 0 | deno | 28.86406 | 17.59127 | 28.86406 | 5.861e+01 | 16 |
| 12 | 0.03526 | 1 | 0 | deno | 17.28594 | 1.84096 | 16.77967 | 1.113e+01 | 15 |
| 13 | 0.054869 | 1 | 0.25 | deno | 1.80942 | 1.79453 | 1.80874 | 8.743e-02 | 16 |
| 14 | 0.004075425 | 1 | 0.25 | deno | 1.79304 | 1.79196 | 1.79300 | 6.390e-03 | 16 |
| 15 | 0.083634 | 1 | 0.5 | deno | 1.79289 | 1.79215 | 1.79286 | 5.603e-03 | 15 |
| 16 | 0.176316 | 1 | 1 | deno | 1.79176 | 1.79177 | 1.79176 | 8.571e-05 | 16 |
| 17 | 0.004075425 | 1e-06 | 0 | acum | 1.82300 | 1.79403 | 1.82135 | 1.532e-01 | 18 |
| 18 | 0.02 | 1e-06 | 0 | acum | 8.48983 | 1.80382 | 8.15063 | 5.535e+00 | 18 |
| 19 | 0.05 | 1e-06 | 0 | acum | 24.91177 | 2.54827 | 24.25387 | 1.622e+01 | 18 |
| 20 | 0.176 | 1e-06 | 0 | acum | 28.86406 | 17.59090 | 28.86406 | 5.861e+01 | 18 |
| 21 | 0.5 | 1e-06 | 0 | acum | 28.86406 | 27.28632 | 28.86406 | 1.667e+02 | 18 |
| 22 | 0.004075425 | 0.05 | 0 | acum | 1.82072 | 1.79395 | 1.81918 | 1.080e-01 | 18 |
| 23 | 0.004075425 | 0.2 | 0 | acum | 1.81770 | 1.79379 | 1.81631 | 8.287e-02 | 18 |
| 24 | 0.004075425 | 1 | 0 | acum | 1.81270 | 1.79341 | 1.81153 | 6.099e-02 | 18 |
| 25 | 0.05 | 0.2 | 0 | acum | 4.03888 | 1.81095 | 3.89645 | 3.012e+00 | 18 |
| 26 | 0.05 | 1 | 0 | acum | 1.92100 | 1.80385 | 1.91293 | 2.388e-01 | 18 |
| 27 | 0.1763158 | 1 | 0 | acum | 2.02240 | 1.81964 | 2.00523 | 4.297e-01 | 19 |

## Lectura

- Mejor celda: #16 (alpha 0.176316, lambda2 1, power_t 1, denominador) con ll_train 1.7917638, es decir el predictor uniforme: no hay aprendizaje.
- Peor celda: #5 (alpha 0.08, lambda2 1, power_t 0, denominador) con ll_train 28.8640554, es decir el colapso de la firma D-083.
- Celdas por debajo de H(prior) = 1.7914537: 0 de 27.
- La perdida DEGRADA con el numero de pasadas en toda la familia salvo en el regimen alpha -> 0 o power_t -> 1, donde el modelo se congela en el uniforme.
- Referencia no celda: gemelo SGD log_loss con el mismo presupuesto -> ll_train 0.15804 en 10 pasadas, con frecuencias de clase [0.1654, 0.1688, 0.1746, 0.1806, 0.1579, 0.1528].

## Conclusion de la exploracion

No existe un `alpha` sano: el punto de operacion optimo de FTRL-Proximal sobre esta TF-IDF
L2-normalizada es el predictor de prior, y el rango que si colapsa empieza justo por encima
del alpha derivado por la regla de margen. La causa no es el valor de alpha sino la norma no
acotada de `z / sqrt(n)`. Esto sustenta el cierre de la familia online por MECANISMO en
D-085 y NO se re-mide dentro del runner.
