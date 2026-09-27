# V26-01: la geometria del kernel, test predictivo de la via no lineal

Generado por `experiments/v26_01_geometria_kernel.py` (V26-01 (geometria del kernel: test predictivo de la via no lineal)).

**Ninguna submission.**

## 1. La pregunta

```
la mediana de similitud del vecino mas cercano con el kernel RBF exacto supera 0.60, de modo que la via no lineal esta cerrada por teoria, o baja de 0.30 y la via se reabre
```

RFF y Nystrom **aproximan el kernel RBF**. Si el RBF es de memorizacion, sus
aproximaciones lo son tambien, porque conservan la misma geometria. Por eso este
script va antes que ellos: es el test que decide si la via tiene sentido.

## 2. Controles

| control | resultado |
|---|---|
| identidad `||x-y||^2 = 2 - 2 x.y` | error maximo 7.105e-15 |
| Gram simetrico con diagonal 1 | asimetria maxima 0.000e+00 |
| support vectors con el RBF exacto | 1232 de 1483, o sea 83.1 % |
| dato de V23 con SVD(300) | 60-76 % de vectores de soporte con SVD(300) |

## 3. La mediana de similitud del vecino mas cercano

| kernel | mediana | p10 | p90 | maximo | > 0.60 | > 0.90 |
|---|---:|---:|---:|---:|---:|---:|
| RBF exacto, mejor gamma | 0.6974 | 0.6693 | 0.7646 | 0.9522 | 100.0 % | 0.3 % |
| RBF sobre SVD(300) | 0.7368 | 0.6780 | 0.8551 | 0.9967 | 100.0 % | 4.9 % |
| spectrum de caracteres | 0.3900 | 0.2719 | 0.5405 | 0.9092 | 3.4 % | 0.1 % |

## 4. Sensibilidad al ancho de banda del RBF exacto

Una mediana sin su `gamma` no significa nada, asi que se recorre.

| gamma | mediana | p90 | > 0.60 | > 0.90 |
|---:|---:|---:|---:|---:|
| 0.25 | 0.6974 | 0.7646 | 100.0 % | 0.3 % |
| 0.5 | 0.4863 | 0.5847 | 7.5 % | 0.1 % |
| 1.0 | 0.2365 | 0.3418 | 0.6 % | 0.0 % |
| 2.0 | 0.0559 | 0.1168 | 0.1 % | 0.0 % |
| 4.0 | 0.0031 | 0.0137 | 0.0 % | 0.0 % |

Varianza explicada por el SVD(300): 0.5114.

## 5. Veredicto

**MEMORIZACION_CONFIRMADA**

Medianas: RBF exacto 0.6974, RBF sobre SVD(300) 0.7368, spectrum 0.3900.

