# P51 - Auditoria de la familia utilitaria

Generado por `experiments/v51_00_utilidad_y_asimetria.py` (V51-00 (brecha utilitaria, utilidad exacta y asimetria)).

Fecha 2026-10-20T02:50:52+00:00. **0 modelos, 0 submissions, 0 consultas a Kaggle.**

## 1. La diferencia entre el objetivo de P50 y el de P51

| fase | target | pregunta |
|---|---|---|
| P50 S1 | `channel_correct` | ?que canal probablemente acierta? |
| P51 S1-U | `delta_F1 > 0` | ?que canal probablemente **mejora V03**? |

No son equivalentes. El primero premia la **predecibilidad** de un canal; el segundo premia su **utilidad**. P50 eligio `cita:NEG` 109 veces, con net -1706, porque su salida es la mas predecible, no la mas util.

## 2. Que parte de la idea ya existia

| componente | estado | evidencia |
|---|---|---|
| definicion de la utilidad exacta de un cambio de celda | **YA EXISTIA** | experiments/v19_01_valor_de_correccion.py deriva la forma cerrada del delta de F1 para los cuatro estados: FN->TP escala con TP+FP+FN, FP->TN solo con TP, y los otros dos son sus negativos |
| artefacto JSON con esos valores por celda | **AUSENTE** | reports/v19_01_valor_de_correccion.json no existe; solo el .md |
| selector entrenado con la utilidad como objetivo | **NO EXISTE** | cero ajustes de modelo en v19_01 y v19_03, verificado por grep de LogisticRegression, fit y predict |
| selector con target de correccion (rescate frente a dano) | **YA EXISTIO, NEGATIVO** | v18_01 dio NO SELECTIVE CORRECTION SIGNAL con AUC 0,6181; P50 S1 dio -0,0115 con 0 de 5 pliegues |

**`SELECTOR_UTILITY_ALREADY_TESTED = False`.** P51 continua.

> P51 no reinventa la utilidad: la reutiliza desde V19-01, que ya derivo la forma cerrada. Lo que no existia era un selector que la usara como objetivo, y eso es lo que P51 construye

La formula viene de `experiments/v19_01_valor_de_correccion.py, funciones valor_de_corregir_falso_negativo y valor_de_corregir_falso_positivo`.

## 3. La asimetria de direccion, que es lo nuevo

| magnitud | valor |
|---|---:|
| ganancia de `FN -> TP` | 1.403e-03 |
| ganancia de `FP -> TN` | 9.012e-04 |
| **razon entre ambas** | **1.57x** |
| p de equilibrio para **subir** de 0 a 1 | 0.3892 |
| p de equilibrio para **bajar** de 1 a 0 | 0.6108 |

> subir de 0 a 1 es una apuesta favorable: al acertar se paga FN->TP, que escala con TP+FP+FN, y al fallar se paga TN->FP, que solo escala con TP. Bajar de 1 a 0 es la apuesta opuesta. P50 trato las dos direcciones por igual, y por eso su argmax eligio canales con net negativo

## 4. El signo coincide con el rescate; la magnitud no

- `(utilidad > 0) equivale a (rescate)` en las 4449 celdas por 20 canales: **True**
- existe el caso ambos mal en una discrepancia: **False**
- cociente de magnitudes rescate sobre dano: **0.94x**

> en celdas binarias, si el canal y V03 discrepan, y coincide con exactamente una de las dos predicciones, luego todo cambio de discrepancia es rescate o dano y nunca ambos mal. El target binario de P51 coincide con el de P50; lo nuevo es la magnitud

**Consecuencia directa:** el target binario de P51 es identico al de P50, asi que un selector binario no puede mejorar por el objetivo. Lo que puede mejorar es (a) entrenar solo sobre celdas en disagreement, donde la clase base es mucho menos sesgada, y (b) usar la magnitud para ordenar canales y no solo para decidir el signo.

## 5. Valor esperado real de cambiar en cada direccion

| direccion | ganadoras | perdedoras | valor esperado total |
|---|---:|---:|---:|
| subir de 0 a 1 | 1465 | 6739 | **-8.830901e-02** |
| bajar de 1 a 0 | 2708 | 4492 | **-1.397702e-01** |

## 6. Control de consistencia con P50

| selector | corrected_cv | delta vs V03 | target |
|---|---:|---:|---|
| V03 | 0.7772114525 | +0,0000000000 | - |
| P50 S1 | 0.7657442265 | -0.0114672260 | `channel_correct` |

Celda de error de V03: 757.

