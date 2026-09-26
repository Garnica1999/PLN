# V10-05: regla de decision F1 optima por punto fijo (P1, segunda linea de la via A)

Generado por `experiments/v10_05_regla_f1_optima.py` (V10-05 (P1, regla de decision F1 optima)).

## 1. La idea, y por que no es mover umbrales

V03 usa tres umbrales libres, los 0.44, 0.51 y 0.46, que estan **elegidos**. Aqui no se
elige ningun numero: la regla se **deriva** de la metrica. El umbral F1 optimo de un
clasificador de probabilidades cumple el punto fijo `t = F1 / 2`, y se itera desde 0.5.

La diferencia relevante para esta fase es que la regla tiene una entrada que los tres
umbrales congelados no tienen: **la prevalencia esperada**. Si la prevalencia baja, la
regla baja el umbral sola. V9-02 midio que la prevalencia de la prueba es un factor de
0.936, 0.955 y 0.938.

## 2. La regla aprende a moverse

La tabla muestra los umbrales que la regla elige en los cinco pliegues, en el escenario
de prior natural y en el de prior simulado. La comparacion con los umbrales congelados de
V03 es la que explica el mecanismo.

| etiqueta | V03 congelado | regla con prior natural | regla con prior simulado | cuanto baja |
|---|---|---|---|---|
| HS | 0.44 | 0.3841 | 0.3768 | -0.0073 |
| TR | 0.51 | 0.4093 | 0.4064 | -0.0029 |
| AG | 0.46 | 0.3789 | 0.3734 | -0.0055 |

## 3. Ganancia anidada en los dos escenarios

| escenario | 42 | 7 | 123 | 2024 | 5150 | media | mejoran |
|---|---|---|---|---|---|---|---|
| prior_natural | -0.000498 | -0.004974 | +0.002988* | +0.001328* | -0.003564 | -0.000944 | 2/5 |
| prior_simulado | -0.003602 | -0.009661 | +0.001669* | -0.002036 | -0.005975 | -0.003921 | 1/5 |

- celdas que la regla cambia frente a V03, en el escenario simulado: 134 de media
- intervalo pareado en la particion canonica: [-0.006341, +0.005835]

## 4. Gates pre-registrados

- **Gate A**, incremento de al menos 0.002 con prior simulado y TR y AG no negativos: **False**
- **Gate B**, el shift aporta mas que el prior natural: **False**
- **Gate C**, los tres criterios del addendum: **False**
  - delta medio de al menos 0.002: False
  - al menos 4 de 5 particiones: False
  - intervalo que no incluye el cero: False

## 5. Veredicto

**SUB-VIA CERRADA.** La regla derivada no mejora a V03, y ademas empeora. Es un
resultado con sentido: la regla F1 optima esta disenada para ser optima **en el
conjunto donde se calcula**, y alplicable a un conjunto con otra prevalencia, no
para.transferirla a datos nuevos. V03 ya esta en el maximo de la curva que midio
V8-09, y una regla mas lista no puede superar un optimo.

Fecha: 2026-09-26T07:55:55+00:00. Duracion: 10.2 s.
