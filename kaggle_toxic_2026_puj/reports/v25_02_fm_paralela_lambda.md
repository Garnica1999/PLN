# V25-02: FM paralelizada, con lambda_v como palanca de sobreajuste

Generado por `experiments/v25_02_fm_paralela_lambda.py` (V25-02 (FM paralelizada, con lambda_v como palanca de sobreajuste)).

**Ninguna submission.**

## 1. El objetivo, completo

```
L = 0.5*||w||^2 + lambda_v * 0.5*||V||_F^2 + C * sum_i s_i * log(1 + exp(-y_i * z_i)); z_i = X_i w + b + 0.5 * sum_f ((sum_j v_jf x_ij)^2 - sum_j v_jf^2 x_ij^2). w y b quedan FIJOS en la solucion exacta de V03, de modo que lambda_v recorre de forma continua desde V03 reproducido hasta el sobreajuste. Esto cambia el objetivo respecto a V25-01 y queda declarado: lo unico que se varya es el peso de un termino nuevo. La loss, el class_weight, la representacion, el intercepto y los umbrales son los de V03.
```

## 2. Los controles

| control | resultado |
|---|---|
| identidad algebraica | diferencia maxima 1.421e-14 |
| con lambda_v = 1 el valor coincide con V25-01 | True |
| con lambda_v = 1 el gradiente coincide con V25-01 bit a bit | True |
| gradiente con lambda_v distinto de 1 | `0.25`: 5.861e-09, `4.0`: 4.719e-09, `64.0`: 1.615e-09 |
| el control reproduce V03 | 0.7772114525165721 frente a 0.7772114525165721 |
| predicciones identicas a las de V03 | True |

## 3. El reparto en paralelo, validado contra la corrida secuencial

| k | lambda_v | secuencial | paralelo | diferencia |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.773663 | 0.773663 | -4.80e-07 |
| 4 | 1 | 0.782699 | 0.782699 | -3.90e-07 |

Todos coinciden: **True**.

## 4. Las decisiones de diseno que recortan el espacio

- **k solo en solo 1 y 4, porque en V25-01 k=8 y k=16 fueron identicos a k=4**
- **solo modo congelado**: solo congelado: en V25-01 el modo conjunto memorizo con train F1 = 1.0000, OOF -0.0284 y desplazo w en un 79 % de su norma, o sea que ya no es V03 mas interaccion sino otro modelo

## 5. El barrido de lambda_v

| k | lambda_v | corrected_cv | dOOF | train F1 | gap | ||V|| | celdas | rescates | danos | neto | dLB esperado |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -- | -- (V03) | 0.7772114525 | +0.0000000000 | 0.9752 | +0.1980 | 0 | 0 | 0 | 0 | +0 | -- |
| 1 | 0.25 | 0.7357726997 | -0.041439 | 0.9990 | +0.2633 | 33.679 | 515 | 200 | 315 | -115 | -0.019344 |
| 1 | 1 | 0.7736625198 | -0.003549 | 0.9771 | +0.2034 | 13.085 | 340 | 126 | 214 | -88 | -0.010069 |
| 1 | 4 | 0.7773336673 | +0.000122 | 0.9718 | +0.1945 | 3.529 | 143 | 56 | 87 | -31 | -0.009170 |
| 1 | 16 | 0.7772114525 | +0.000000 | 0.9752 | +0.1980 | 0.000 | 0 | 0 | 0 | +0 | -0.009200 |
| 1 | 64 | 0.7772114525 | +0.000000 | 0.9752 | +0.1980 | 0.000 | 0 | 0 | 0 | +0 | -0.009200 |
| 1 | 256 | 0.7772114525 | +0.000000 | 0.9752 | +0.1980 | 0.000 | 0 | 0 | 0 | +0 | -0.009200 |
| 1 | 1e+06 | 0.7772114525 | +0.000000 | 0.9752 | +0.1980 | 0.000 | 0 | 0 | 0 | +0 | -0.009200 |
| 4 | 0.25 | 0.7724400473 | -0.004771 | 1.0000 | +0.2276 | 32.367 | 320 | 153 | 167 | -14 | -0.010368 |
| 4 | 1 | 0.7826986101 | +0.005487 | 0.9802 | +0.1975 | 13.961 | 278 | 119 | 159 | -40 | -0.007857 |
| 4 | 4 | 0.7773336673 | +0.000122 | 0.9718 | +0.1945 | 3.529 | 143 | 56 | 87 | -31 | -0.009170 |
| 4 | 16 | 0.7772114525 | +0.000000 | 0.9752 | +0.1980 | 0.000 | 0 | 0 | 0 | +0 | -0.009200 |
| 4 | 64 | 0.7772114525 | +0.000000 | 0.9752 | +0.1980 | 0.000 | 0 | 0 | 0 | +0 | -0.009200 |
| 4 | 256 | 0.7772114525 | +0.000000 | 0.9752 | +0.1980 | 0.000 | 0 | 0 | 0 | +0 | -0.009200 |
| 4 | 1e+06 | 0.7772114525 | +0.000000 | 0.9752 | +0.1980 | 0.000 | 0 | 0 | 0 | +0 | -0.009200 |

## 6. El detalle por etiqueta

| k | lambda_v | etiqueta | F1 | F1 de V03 | celdas | rescates | danos |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.25 | HS | 0.717340 | 0.761689 | 231 | 95 | 136 |
| 1 | 0.25 | TR | 0.796400 | 0.815730 | 77 | 30 | 47 |
| 1 | 0.25 | AG | 0.693578 | 0.754215 | 207 | 75 | 132 |
| 1 | 1 | HS | 0.761062 | 0.761689 | 147 | 56 | 91 |
| 1 | 1 | TR | 0.816931 | 0.815730 | 61 | 26 | 35 |
| 1 | 1 | AG | 0.742994 | 0.754215 | 132 | 44 | 88 |
| 1 | 4 | HS | 0.767004 | 0.761689 | 56 | 25 | 31 |
| 1 | 4 | TR | 0.813522 | 0.815730 | 27 | 10 | 17 |
| 1 | 4 | AG | 0.751474 | 0.754215 | 60 | 21 | 39 |
| 1 | 16 | HS | 0.761689 | 0.761689 | 0 | 0 | 0 |
| 1 | 16 | TR | 0.815730 | 0.815730 | 0 | 0 | 0 |
| 1 | 16 | AG | 0.754215 | 0.754215 | 0 | 0 | 0 |
| 1 | 64 | HS | 0.761689 | 0.761689 | 0 | 0 | 0 |
| 1 | 64 | TR | 0.815730 | 0.815730 | 0 | 0 | 0 |
| 1 | 64 | AG | 0.754215 | 0.754215 | 0 | 0 | 0 |
| 1 | 256 | HS | 0.761689 | 0.761689 | 0 | 0 | 0 |
| 1 | 256 | TR | 0.815730 | 0.815730 | 0 | 0 | 0 |
| 1 | 256 | AG | 0.754215 | 0.754215 | 0 | 0 | 0 |
| 1 | 1e+06 | HS | 0.761689 | 0.761689 | 0 | 0 | 0 |
| 1 | 1e+06 | TR | 0.815730 | 0.815730 | 0 | 0 | 0 |
| 1 | 1e+06 | AG | 0.754215 | 0.754215 | 0 | 0 | 0 |
| 4 | 0.25 | HS | 0.754573 | 0.761689 | 126 | 60 | 66 |
| 4 | 0.25 | TR | 0.823266 | 0.815730 | 58 | 32 | 26 |
| 4 | 0.25 | AG | 0.739481 | 0.754215 | 136 | 61 | 75 |
| 4 | 1 | HS | 0.769553 | 0.761689 | 110 | 48 | 62 |
| 4 | 1 | TR | 0.817797 | 0.815730 | 58 | 25 | 33 |
| 4 | 1 | AG | 0.760746 | 0.754215 | 110 | 46 | 64 |
| 4 | 4 | HS | 0.767004 | 0.761689 | 56 | 25 | 31 |
| 4 | 4 | TR | 0.813522 | 0.815730 | 27 | 10 | 17 |
| 4 | 4 | AG | 0.751474 | 0.754215 | 60 | 21 | 39 |
| 4 | 16 | HS | 0.761689 | 0.761689 | 0 | 0 | 0 |
| 4 | 16 | TR | 0.815730 | 0.815730 | 0 | 0 | 0 |
| 4 | 16 | AG | 0.754215 | 0.754215 | 0 | 0 | 0 |
| 4 | 64 | HS | 0.761689 | 0.761689 | 0 | 0 | 0 |
| 4 | 64 | TR | 0.815730 | 0.815730 | 0 | 0 | 0 |
| 4 | 64 | AG | 0.754215 | 0.754215 | 0 | 0 | 0 |
| 4 | 256 | HS | 0.761689 | 0.761689 | 0 | 0 | 0 |
| 4 | 256 | TR | 0.815730 | 0.815730 | 0 | 0 | 0 |
| 4 | 256 | AG | 0.754215 | 0.754215 | 0 | 0 | 0 |
| 4 | 1e+06 | HS | 0.761689 | 0.761689 | 0 | 0 | 0 |
| 4 | 1e+06 | TR | 0.815730 | 0.815730 | 0 | 0 | 0 |
| 4 | 1e+06 | AG | 0.754215 | 0.754215 | 0 | 0 | 0 |

## 7. El coste, medido y no estimado

| magnitud | valor |
|---|---:|
| un ajuste de k=4 con un hilo de BLAS, medido en la calibracion | 2.32 s |
| ajustes del barrido | 210 |
| equivalente secuencial | 486.6 s |
| barrido real en paralelo | 55.5 s |
| **speedup medido** | **8.77x** |
| procesos del pool | 15 de 32 nucleos |
| carga de la maquina durante el barrido | 22.6 |

El speedup queda condicionado por la carga: durante la medicion corria otro proceso ajeno a este trabajo ocupando una parte de los nucleos, de modo que esta cifra es una cota inferior de lo que da la maquina descargada, no un maximo.

