# V24-03 (P2): sonda de kNN como predictor

Diagnostico de gate. Sin submission. Sin tocar el holdout.

## 0. Marco
- train 1731 | dev 1483 | holdout 248 (intacto)
- pliegues: [(1186, 297), (1186, 297), (1187, 296), (1187, 296), (1186, 297)]

## 1. V03 referencia (OOF corregido, umbrales congelados)
- corrected_cv = **0.7772** (por etiqueta 0.7617, 0.8157, 0.7542)
- celdas falladas = **757** de 4449
- reproduce el valor canonico 0.777211: True
- celdas que el post-proceso jerarquico cambiaria: 55 (protocolo V03 = umbrales solos, el post-proceso NO se aplica)

## 2. kNN en espacio `union` (OOF agrupado, sin fuga)
- corrected_cv kNN   p1 = 0.6831  (V03 = 0.7772, delta -0.0941)
- corrected_cv kNN   p3 = 0.7065  (V03 = 0.7772, delta -0.0707)
- corrected_cv kNN  p5w = 0.7197  (V03 = 0.7772, delta -0.0575)
- corrected_cv kNN  p25 = 0.7567  (V03 = 0.7772, delta -0.0206)
- similitud top-1: mediana 0.2953, p90 0.4779, max 0.8217
- filas con sim top-1 > 0.5: 121 (8.2 %)
- filas con sim top-1 > 0.7: 6 (0.4 %)

## 2. kNN en espacio `char` (OOF agrupado, sin fuga)
- corrected_cv kNN   p1 = 0.6119  (V03 = 0.7772, delta -0.1653)
- corrected_cv kNN   p3 = 0.6522  (V03 = 0.7772, delta -0.1250)
- corrected_cv kNN  p5w = 0.6721  (V03 = 0.7772, delta -0.1051)
- corrected_cv kNN  p25 = 0.7121  (V03 = 0.7772, delta -0.0651)
- similitud top-1: mediana 0.2719, p90 0.4137, max 0.7791
- filas con sim top-1 > 0.5: 47 (3.2 %)
- filas con sim top-1 > 0.7: 4 (0.3 %)

## 3. Puntos B-F en espacio `union`
- B. Vecinos confiables (sim top-1 >= p90 = 0.4779 y desacuerdo 5-vecinos <= 1): **138 de 1483 filas** (9.3 %)
   Celdas de error de V03 cubiertas por un vecino confiable: **12 de 757** = **1.6 %**
- C. Precision de kNN al contradecir a V03: **3 correctas / 20 volteos = 0.1500**
   McNemar: V03 gana 621 celdas donde discrepan, kNN gana 284, p = 0.0000
- D. Complementariedad: errors solo-V03 284, solo-kNN 621, ambos 473, Jaccard 0.343
- E. Volumen: red = 3 - 17 = -14 celdas. regla realizable -> corrected_cv 0.7717 (dOOF -0.0055)
   mezcla de probabilidades peso kNN 0.10 -> corrected_cv 0.7772 (dOOF +0.0000)
   mezcla de probabilidades peso kNN 0.20 -> corrected_cv 0.7774 (dOOF +0.0002)
   mezcla de probabilidades peso kNN 0.30 -> corrected_cv 0.7822 (dOOF +0.0050)
- ORACLE (cota superior, NO realizable): corrected_cv 0.8583 (dOOF +0.0811)

## 3. Puntos B-F en espacio `char`
- B. Vecinos confiables (sim top-1 >= p90 = 0.4137 y desacuerdo 5-vecinos <= 1): **127 de 1483 filas** (8.6 %)
   Celdas de error de V03 cubiertas por un vecino confiable: **23 de 757** = **3.0 %**
- C. Precision de kNN al contradecir a V03: **6 correctas / 56 volteos = 0.1071**
   McNemar: V03 gana 866 celdas donde discrepan, kNN gana 320, p = 0.0000
- D. Complementariedad: errors solo-V03 320, solo-kNN 866, ambos 437, Jaccard 0.269
- E. Volumen: red = 6 - 50 = -44 celdas. regla realizable -> corrected_cv 0.7627 (dOOF -0.0145)
   mezcla de probabilidades peso kNN 0.10 -> corrected_cv 0.7794 (dOOF +0.0022)
   mezcla de probabilidades peso kNN 0.20 -> corrected_cv 0.7784 (dOOF +0.0012)
   mezcla de probabilidades peso kNN 0.30 -> corrected_cv 0.7753 (dOOF -0.0019)
- ORACLE (cota superior, NO realizable): corrected_cv 0.8685 (dOOF +0.0913)

## 4. Punto F: distincion frente a V4-F03 y V4-F01-F04
- V4-F03 uso `knn_disagreement` (desacuerdo ponderado de 3) y `sim_cercana` (cos del vecino 1)
  como FEATURES para explicar el error. Aqui el voto de los vecinos es la PREDICCION.
- Correlacion Pearson entre el desacuerdo 3-vecinos (medida V4-F03) y la tasa de celdas donde
  el voto kNN contradice a V03: **+0.2905**
- Correlacion Pearson entre `sim_cercana` (medida V4-F03) y esa misma tasa: **-0.1628**
- Correlacion Pearson entre el voto kNN ponderado y la probabilidad de V03 (mismo espacio `union`): **+0.9291**
- Correlacion Pearson entre el voto kNN (espacio `char`) y la probabilidad de V03: **+0.8512**

## 4b. Barrido amplio de la mezcla convexa V03 + kNN (reproducibilidad por pliegue)
- Mejor mezcla convexa: espacio `union`, peso kNN 0.30 -> corrected_cv **0.7822** (dOOF +0.0050), gana en **4 de 5 pliegues**
- corrected_cv por pliegue kNN: 0.7925, 0.8017, 0.7867, 0.7442, 0.7859
- corrected_cv por pliegue V03: 0.7960, 0.7940, 0.7789, 0.7386, 0.7791
- dOOF necesario para superar V03 en LB segun el ajuste del historial: +0.03756
- La mejor mezcla real queda en +0.0050: **13 % del umbral de gate**

## 5. Sintesis del gate
- espacio `union`: mejor regla REALIZABLE dOOF = +0.0050; techo ORACLE = +0.0811; umbral de gate dOOF = +0.03756
- espacio `char`: mejor regla REALIZABLE dOOF = +0.0022; techo ORACLE = +0.0913; umbral de gate dOOF = +0.03756
- mejor mezcla convexa en todo el barrido: **+0.0050** de dOOF, gana en 4 de 5 pliegues

