# P75: complementariedad de Word2Vec + TextCNN frente a V03

`experiment_id` P75 | inicio 2026-09-30T08:49:40+00:00 | alpha primario 0.1 | sensibilidad 0.05 y 0.15

## Lo que habilita P75

P74 demostro con 10 replicas pareadas que la TextCNN sobre Word2Vec supera a la TextCNN sobre embedding aleatorio (10 de 10, media del delta +0.0242705458, IC95 t [+0.0139795245, +0.0345615672]). Eso establece que **la representacion contiene informacion de tarea**. P75 no reabre eso: pregunta si esa informacion **transfiere a V03**.

## OOF pareado

V03 y el TextCNN entrenan los dos sobre el `outer_train` completo y predicen sobre exactamente las mismas filas. El TextCNN usa `E*` como **mediana de tres splits internos por clusters** y luego reentrena sobre el `outer_train` completo durante `E*` epocas fijas: mismo presupuesto de datos que V03 y **ninguna epoca elegida sobre `outer_validation`**.

```text
V03 desde el OOF canonico        0.7772114525   coincide True
V03 recalculado desde la tabla   0.7772114525   coincide True
unidades tweet-etiqueta          4449
CNN sola (diagnostica)           0.5838755350
```

## Matriz de interseccion y rescue/damage/net

```text
GLOBAL  unidades 4449  rescue 268  damage 813  net -545
        ambos correctos 2879  ambos incorrectos 489
        rescue_rate 0.0602  damage_rate 0.1827  net_rate -0.1225
```

**Unidades tweet-etiqueta. NO son comparables con `corrected_cv` ni con ningun macro-F1.** `net` es un recuento de celdas, no una mejora de metrica.

| etiqueta | rescue | damage | net |
|---|---|---|---|
| HS | 112 | 316 | -204 |
| TR | 60 | 203 | -143 |
| AG | 96 | 294 | -198 |

## Diversidad

```text
HS  corr proba +0.5886  corr margen +0.5886  AUC(error V03 | score CNN) 0.4872
TR  corr proba +0.7191  corr margen +0.7191  AUC(error V03 | score CNN) 0.6045
AG  corr proba +0.6126  corr margen +0.6126  AUC(error V03 | score CNN) 0.4973
acuerdo 0.757024  desacuerdo 0.242976
```

## Fusion: los tres alphas, completos

```text
alpha 0.05  corrected_cv 0.7746109462  delta -0.0026005064  folds 2/5  alcanzables 82  volteadas 38  techo +0.011641  piso -0.012162
alpha 0.10  corrected_cv 0.7719675587  delta -0.0052438938  folds 1/5  alcanzables 159  volteadas 76  techo +0.020063  piso -0.025213
alpha 0.15  corrected_cv 0.7691941289  delta -0.0080173237  folds 1/5  alcanzables 239  volteadas 127  techo +0.030548  piso -0.037578
```

El **alpha primario es 0.10** y es el unico que carga la puerta. 0.05 y 0.15 se reportan completos y no promueven ni rescatan un fallo del primario.

## Veredicto

**CASO 5: CNN_PERJUDICA**

Finalista para segunda particion: **None**
