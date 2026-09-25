# Complementariedad del embedding con V03 (fase v5, P3)

Generado por `experiments/embedding_03_fusion_complementariedad_v5.py` (EBD-07..09 (P3)).

## 1. Por que esta prueba y no la anterior

P2 mostro que el embedding **por si solo** queda 0.1169 por debajo de V03. Eso refuta la
hipotesis de que lo sustituya, pero no la hipotesis central de la fase (seccion 4 de la
spec y pregunta 11 de la seccion 42): si aporta informacion **complementaria**. Esta
prueba mide precisamente eso.

## 2. Protocolo (identico al de V03, celda a celda)

- Pliegues: `pliegues_corregidos` (5, grupos near-duplicado UNION pseudo-autor).
- TF-IDF word(1,2) + char_wb(3,5) con `min_df=2` y `sublinear_tf=True`, ajustados **por pliegue** sobre el entrenamiento, igual que en `experiments/14_oof_corregido_v2.py`.
- LR `C=2.0`, `class_weight='balanced'`, `liblinear`, 3 cabezas binarias.
- Umbrales: rejilla 0.20-0.80 paso 0.01 sobre las probabilidades OOF, el mismo metodo que documento el pipeline congelado de V03.
- Control: el calculo de `V03-TFIDF` sin embedding da **0.777211** frente a los 0.777211 congelados (delta +0.000000).

## 3. Resultado por variante

La lectura primaria aplica a todas las variantes los **umbrales congelados de V03**
(0.44/0.51/0.46): nadie se beneficia de ajustar su propio umbral. La columna
"umbrales propios" se reporta como dato secundario y es optimista.

| variante | frozen V03 (primaria) | delta vs V03 | IC95 pareado | @0.5 | umbrales propios | celdas + | celdas - | solo esta | solo V03 |
|---|---|---|---|---|---|---|---|---|---|
| V03-TFIDF | 0.7772 | +0.0000 | [+0.0000, +0.0000] | 0.7718 | 0.7808 | 0 | 0 | 0 | 0 |
| EMB-SOLO | 0.6530 | -0.1242 | [-0.1481, -0.1006] | 0.6516 | 0.6639 | 87 | 160 | 16 | 88 |
| FUSION-HSTACK-L2 | 0.7818 | +0.0046 | [-0.0032, +0.0124] | 0.7775 | 0.7825 | 33 | 20 | 8 | 5 |
| FUSION-HSTACK-STANDARD | 0.6901 | -0.0871 | [-0.1082, -0.0662] | 0.6885 | 0.6991 | 70 | 132 | 15 | 76 |
| META-ANIDADO | 0.7678 | -0.0094 | [-0.0177, -0.0016] | 0.7618 | 0.7793 | 5 | 53 | 0 | 11 |

Mejor variante: **FUSION-HSTACK-L2** con 0.7818 (delta +0.0046, IC95 [-0.0032, +0.0124]).

Criterio pre-registrado (seccion 29): adopted solo si delta >= +0.005 **y** el IC95 pareado
excluye el 0. Nivel: no; IC95: incluye 0.

## 4. Complementariedad por rebanada (seccion 25)

F1 por celda (fila x etiqueta) dentro de cada rebanada. Las rebanadas **no** son
excluyentes entre si y las de tipo lexical se definen con los patrones curados de P1.

| rebanada | filas | V03-CONGELADO | EMB-SOLO | FUSION-HSTACK | META-ANIDADO |
|---|---|---|---|---|---|
| interpersonal (TR=1) | 445 | 0.9129 | 0.8474 | nan | 0.8986 |
| afiliacion (AG=1) | 567 | 0.8846 | 0.8223 | nan | 0.8658 |
| sin_hate_explicito (HS=0) | 835 | 0.0667 | 0.0155 | nan | 0.0395 |
| colectivos | 93 | 0.5740 | 0.5041 | nan | 0.5098 |
| insultos | 855 | 0.8357 | 0.7216 | nan | 0.8314 |
| elongaciones | 57 | 0.6853 | 0.5581 | nan | 0.7152 |
| coloquial | 214 | 0.8177 | 0.6375 | nan | 0.8080 |
| hashtags | 144 | 0.5311 | 0.3656 | nan | 0.5246 |
| menciones | 702 | 0.8462 | 0.7330 | nan | 0.8501 |
| urls | 458 | 0.6578 | 0.5389 | nan | 0.6271 |
| cita | 37 | 0.8449 | 0.6374 | nan | 0.7830 |

## 5. Lectura y limites

1. El control `V03-TFIDF` reproduce la referencia, de modo que las diferencias de la tabla
   son atribuibles a la representacion y no al protocolo.
2. `solo esta` / `solo V03` cuentan celdas en las que exactamente uno de los dos acierta:
   es la medida directa de complementariedad. Un numero alto de `solo esta` con mejora
   pequena en la media indicaria que el embedding rescata casos sin degradar el resto.
3. El meta-modelo esta evaluado con apilamiento **anidado**: sus probabilidades de
   entrenamiento salen de 5 pliegues internos, nunca de un modelo ajustado sobre la misma
   fila (seccion 24).
4. Los tres modelos de P2 se entrenaron en una ejecucion anterior cuyo registro de consola
   fue sobrescrito, por lo que `segundos=0` en el manifiesto de P2 significa "reutilizado";
   la huella `sha256` del `.vec` es el dato autoritativo.
5. Aunque una variante ganase localmente, la leccion operativa de v4 sigue vigente: cuatro
   mejoras locales consecutivas no transfirieron al holdout o al leaderboard. Ninguna
   variante se consideraria candidata sin esa evidencia externa, que aqui no se gasta.
