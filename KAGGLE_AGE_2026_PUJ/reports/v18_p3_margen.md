# V18 · P3 — Objetivo de margen de adyacencia

## 1. Diseno

Una sola **cabeza lineal** con margen adyacente, que es el analogo directo de la cabeza `LogReg` de V12: misma capacidad, mismas 279 variables, mismos parametros. Cambia **solo** el objetivo, asi que la atribucion es directa. Cinco brazos, con lambda y margen **pre-registrados** y **epocas fijas sin early stopping**, de modo que ninguna decision depende de la validacion exterior:

| Brazo | Restriccion | Lambda | Pares |
|---|---|---|---|
| A_control | ninguna | 0.0 | ninguno |
| B1_generico | adyacencia generica | 0.1 | 13-17|18-24, 18-24|25-34, 25-34|35-49, 35-49|50-64, 50-64|65-xx |
| B2_generico | adyacencia generica | 0.5 | 13-17|18-24, 18-24|25-34, 25-34|35-49, 35-49|50-64, 50-64|65-xx |
| C1_central | adyacencia central | 0.1 | 18-24|25-34, 25-34|35-49 |
| C2_central | adyacencia central | 0.5 | 18-24|25-34, 25-34|35-49 |

Margen exigido: 1.0 en escala de logits. Epocas: 30, lote 512, Adam 0.001. Dispositivo: cuda.

La comparacion entre **B** (adyacencia generica) y **C** (adyacencia central) separa cualquier restriccion de adyacencia de la restriccion justo en las dos fronteras que importan.

## 2. Resultado: la cabeza lineal sola (atribucion)

| Brazo | macro-F1 lineal | 18-24\|25-34 | 25-34\|35-49 |
|---|---|---|---|
| A_control | 0.5876 | 0.4865 | 0.4833 |
| B1_generico | 0.5861 | 0.4727 | 0.476 |
| B2_generico | 0.5693 | 0.4243 | 0.4474 |
| C1_central | 0.5852 | 0.5257 | 0.4281 |
| C2_central | 0.555 | 0.51 | 0.2969 |

## 3. Resultado: el sistema combinado (decision)

| Brazo | macro-F1 | Delta | IC95 | 18-24\|25-34 | 25-34\|35-49 | Adopta |
|---|---|---|---|---|---|---|
| A_control | 0.5983 | -0.0022 | [-0.0030, -0.0014] | 0.4957 | 0.4949 | **no** |
| B1_generico | 0.5985 | -0.0020 | [-0.0029, -0.0012] | 0.4892 | 0.4918 | **no** |
| B2_generico | 0.5935 | -0.0070 | [-0.0083, -0.0058] | 0.4651 | 0.4802 | **no** |
| C1_central | 0.599 | -0.0015 | [-0.0025, -0.0006] | 0.5172 | 0.4706 | **no** |
| C2_central | 0.5903 | -0.0102 | [-0.0115, -0.0088] | 0.5193 | 0.4013 | **no** |
| V12 (referencia) | 0.6005 | - | - | 0.4963 | 0.4974 | - |

## 4. Salud del entrenamiento

| Brazo | Perdida media de train | Brecha frente a validacion | Epocas | Converge |
|---|---|---|---|---|
| A_control | 1.0347 | 0.0045 | 30 | si |
| B1_generico | 1.036 | 0.0066 | 30 | si |
| B2_generico | 1.0677 | 0.0061 | 30 | si |
| C1_central | 1.0424 | 0.0077 | 30 | si |
| C2_central | 1.109 | 0.0023 | 30 | si |

## 5. F1 por clase del sistema combinado

| Brazo | 13-17 | 18-24 | 25-34 | 35-49 | 50-64 | 65-xx |
|---|---|---|---|---|---|---|
| A_control | 0.6666 | 0.5044 | 0.4844 | 0.5564 | 0.6547 | 0.7233 |
| B1_generico | 0.6695 | 0.4994 | 0.4838 | 0.5567 | 0.6552 | 0.7261 |
| B2_generico | 0.6647 | 0.4802 | 0.4742 | 0.5557 | 0.656 | 0.73 |
| C1_central | 0.6688 | 0.5125 | 0.4788 | 0.5552 | 0.6549 | 0.7237 |
| C2_central | 0.6702 | 0.5029 | 0.4508 | 0.5402 | 0.6539 | 0.7238 |
| V12 (referencia) | 0.671 | 0.5059 | 0.4858 | 0.5593 | 0.6549 | 0.7261 |
