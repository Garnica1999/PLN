# V11-01: reproduccion exacta de V9-02 (P1)

Generado por `experiments/v11_01_reproduccion_v9_02.py` (V11-01 (P1, reproduccion exacta de V9-02)).

## 1. Por que es una puerta

Toda la fase v11 se apoya en una cifra: **la prevalencia de la prueba estimada por EM**,
con factores de 0.936, 0.955 y 0.938. Si esa cifra no se reproduce, cualquier correccion
de prior, calibracion o decision que se construya encima hereda un numero sin verificar.

Y hay un motivo tecnico. En la auditoria de P0 se vio que el campo `factor_vs_train` de
V9-02 **no** es la prevalencia de test dividida por la del train completo, sino por la del
desarrollo: el script define `prevalencia_train = y_dev.mean(axis=0)` con las 1483 filas
del desarrollo. Un nombre de campo que no significa lo que dice es exactamente el tipo de
cosa que hace fallar una cadena de resultados.

## 2. Como se reproduce sin reimplementar nada

Se importa el modulo de V9-02 y se llaman sus funciones, sin reimplementar el EM. Si se reimplementara, la reproduccion no probaria nada, porque dos implementaciones parecidas pueden diferir justo donde importa.

## 3. Resultado

| etiqueta | p desarrollo | p train completo | EM esperado | EM reproducido | diferencia | iteraciones |
|---|---|---|---|---|---|---|
| HS | 0.436952 | 0.436742 | 0.40919200 | 0.40919155 | -4.51e-07 **OK** | 26 frente a 26 |
| TR | 0.300067 | 0.299827 | 0.28645200 | 0.28645233 | +3.29e-07 **OK** | 16 frente a 16 |
| AG | 0.382333 | 0.382438 | 0.35866900 | 0.35866863 | -3.69e-07 **OK** | 24 frente a 24 |

Tolerancia de reproduccion: **1e-06**.

**Reproduccion exacta: True.**

## 4. La ambiguedad del denominador, cerrada

| etiqueta | factor sobre desarrollo | factor guardado en V9-02 | diferencia | factor sobre train completo |
|---|---|---|---|---|
| HS | 0.936468 | 0.9365 | -0.000032 | 0.936919 |
| TR | 0.954627 | 0.9546 | +0.000027 | 0.955393 |
| AG | 0.938105 | 0.9381 | +0.000005 | 0.937848 |

El factor sobre el desarrollo reproduce el guardado: **True**.

## 5. Sobre la legalidad del uso de la prueba

Se predice sobre la prueba y se estima la prevalencia a partir de probabilidades NO etiquetadas. No se usa ninguna etiqueta de la prueba, no se entrena con su texto mas alla de predecir, y no se ajusta ningun umbral con el. La auditoria formal de legalidad es V11-02 y no se da por resuelta aqui.

## 6. La cifra que la v11 va a usar

**Las prevalencias absolutas**, no los factores. Son la magnitud sin ambiguedad, y las dos
lecturas del factor dan la misma conclusion cualitativa: la prueba tiene entre un 4.5 y un
6.4 por ciento menos positivos que el desarrollo.

| etiqueta | prevalencia de la prueba |
|---|---|
| HS | **0.409192** |
| TR | **0.286452** |
| AG | **0.358669** |

## 7. Veredicto

**REPRODUCCION EXACTA.** La cifra de shift sobre la que descansa la fase v11 esta
verificada, y la ambiguedad del denominador queda cerrada a favor del desarrollo. La
fase puede continuar con el audit de legalidad, V11-02, y despues con la simulacion de
shift, V11-04.

Fecha: 2026-09-26T21:32:58+00:00. Duracion: 1.2 s.
