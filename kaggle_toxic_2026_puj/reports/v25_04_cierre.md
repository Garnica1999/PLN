# V25: cierre de fase. La Factorization Machine no aporta.

**Veredicto: `NO CANDIDATE JUSTIFIES SUBMISSION - CLOSE V25`.** Las 3 submissions
restantes se reservan. **V03 = 0.78306 LB sigue siendo el campeon.**

## Lo que seconstructed

La FM era la unica mecanica de la literatura de la auditoria P0 que no se habia ejecutado
nunca, y era genuinamente nueva: 0 ocurrencias de `FactorizationMachine`, `xlearn`, `libffm`
o `fastFM` en todo el repositorio. Se implemento desde cero sobre la representacion exacta de
V03, con el objetivo verificado antes de interpretar nada.

| control | resultado |
|---|---|
| identidad algebraica frente a la suma explicita de **todos** los pares | 1.42e-14 |
| gradiente analitico contra diferencias finitas, para `w`, `b` y `V` | 9.1e-10, 3.2e-10, 3.3e-09 |
| con `lambda_v = 1`, valor y gradiente frente a V25-01 | **bit a bit identicos**, 0.000e+00 |
| el control reproduce V03 | `0.7772114525165721`, diferencia `+0.000e+00` |
| predicciones del control frente a las de V03 | identicas |
| pares que no se materializan | 208 907 020 |
| coste de la interaccion | `O(nnz * k)`, sin densificar |

## El barrido: lambda_v como palanca de sobreajuste

Se pidio corregir el sobreajuste. `lambda_v` recorre de forma continua desde V03 reproducido
(`lambda_v` grande, `V -> 0`) hasta el sobreajuste (`lambda_v -> 0`), con `w` y `b` fijos en la
solucion exacta de V03.

| k | lambda_v | corrected_cv | dOOF | train F1 | gap | \|\|V\|\| | neto |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -- | -- (V03) | 0.7772114525 | +0.000000 | 0.9752 | +0.1980 | 0 | +0 |
| 4 | 0.25 | 0.772440 | -0.004771 | **1.0000** | +0.2276 | 32.367 | -14 |
| 1 | 0.25 | 0.735773 | -0.041439 | **0.9990** | +0.2633 | 33.679 | -115 |
| 1 | 1 | 0.773663 | -0.003549 | 0.9771 | +0.2034 | 13.085 | -88 |
| **4** | **1** | **0.782699** | **+0.005487** | 0.9802 | +0.1975 | 13.961 | -40 |
| 1 | 4 | 0.777334 | +0.000122 | 0.9718 | +0.1945 | 3.529 | -31 |
| 4 | 4 | 0.777334 | +0.000122 | 0.9718 | +0.1945 | 3.529 | -31 |
| 1 y 4 | 16, 64, 256, 1e6 | 0.777211 | +0.000000 | 0.9752 | +0.1980 | 0.000 | +0 |

**El sobreajuste no se puede corregir, y el barrido lo demuestra en las dos direcciones.**
Bajar `lambda_v` a 0.25 lleva el train F1 a **1.0000**, o sea memorizacion, y el OOF cae.
Subirlo a 4 deja la interaccion en 3.5 y anula el mecanismo; a 16 y por encima `V` colapsa a
cero y el modelo **es V03 exacto**, con `dOOF = +0.000000` a las diez decimales. El optimo
esta exactamente en `lambda_v = 1`, que es la regularizacion natural: la misma forma que la de
`w`. No hay margen alrededor.

## El resultado, contra lo que hace falta

| magnitud | valor |
|---|---:|
| mejor `dOOF` de la FM | **+0.005487** |
| `dOOF` necesario para empatar con V03 (modelo de V24) | **+0.03756** |
| techo historico de `dOOF` (V23) | +0.0132 |
| `dLB` esperado del mejor candidato | **-0.007857** |

El mejor candidato de la FM es **peor que V23**, que ya esta probado que pierde. Y el
`dLB` esperado es negativo por el propio modelo de transferencia, sin ambiguedad.

## Por que la FM no puede ganar aqui, y no es un fallo de implementacion

1. **El rango efectivo se agota en 4.** `k=8` y `k=16` son **identicos** a `k=4` en los dos
   modos, con las mismas cuatro decimales de la norma de `V`. Con 1 187 filas de
   entrenamiento, la interseccion efectiva del problema tiene rango muy bajo: no hay mas
   direccion que explorar en `k`.
2. **La superficie de F1 es plana alrededor de V03.** El paso de `dOOF = 0` a `+0.005487` no
   es monotone en nada: el `gap` train-OOF sube de 0.1980 a 0.1975 y a la vez el OOF sube, y
   el `neto` de celdas es **negativo en todos los candidatos que ganan**, -40 en el mejor. Se
   ganan 119 celdas y se pierden 159. Eso no es un modelo mejor, es ruido favorable.
3. **El modo conjunto demuestra el techo.** Libera `w` y `b` y llega a `train F1 = 1.0000` con
   OOF **-0.0284**, desplazando `w` en 18.38 sobre una norma de 23.3, es decir un **79 %**. Ese
   modo ya no es "V03 mas interaccion": es memorizacion pura, y su ceiling de OOF esta por
   debajo de V03.

## Lo que si se aprende, y sobrevive a la fase

1. **liblinear, con `tol=1e-4`, no converge al optimo del objetivo que minimiza.** El
   artefacto congelado `model_v03.joblib` registra `n_iter_ = [5]` y `[6]`. Un L-BFGS completo
   llega a un valor **menor** del mismo objetivo (11512.154 frente a 11512.308 sobre los quince
   ajustes) y aun asi cae en otro punto de una superficie plana, con OOF **-0.0016**. Es decir:
   **parte del `corrected_cv` de V03 es la parada temprana de liblinear, no el optimo del
   modelo.** Cualquier comparacion futura con V03 tiene que declarar si optimiza hasta el
   optimo o reproduce la parada de liblinear.
2. **El intercepto es esencial y no es trivial.** `fit_intercept=True` con interceptos reales
   de -0.357, -1.000 y -0.735, y **no** regularizado. Sin el, la tasa de positivos con umbral
   0.44 pasa de 0.4545 a 0.4949, y el F1 se desplaza por el lado de la precision.
3. **`v = 0` es un punto estacionario exacto** de una FM. La interaccion es par en `v`, asi que
   su derivada en el origen es cero y un optimizador de gradiente que arranque ahi no se mueve
   nunca. Hay que inicializar `V` al azar. Es un caso particular de un punto estacionario no
   trivial, y conviene tenerlo presente en cualquier modelo cuadratico.
4. **El orden de reduccion de BLAS no es portable.** Con 1, 2, 4, 8 y 32 hilos, `w` de
   liblinear difiere hasta 2.8e-13 y las probabilidades hasta 7.5e-13. **Ninguna** celda cruza
   su umbral: la probabilidad mas cercana a un umbral esta a 4.56e-05, cuatro millones de
   veces mas lejos que el ruido. Por eso un gate "bit a bit" entre dos configuraciones de
   paralelismo es un criterio mal formado, y el correcto es metrica identica mas predicciones
   identicas mas una tolerancia declarada.
5. **Repartir al exterior, no al interior, es lo que rinde.** Los quince ajustes de un
   candidato son independientes. Un pool de 15 procesos con un hilo de BLAS cada uno da un
   speedup **medido de 8.77x** (487 s a 55,5 s), frente al 2.5x que daba la paralelizacion
   interna de BLAS. Advertencia: `fork` con un pool de hilos de OpenBLAS vivo en el padre es la
   via clasica a un interbloqueo, y se corrige con `threadpool_limits(1)` en el padre **antes**
   de bifurcar.
6. **El reparto de CPU de la maquina no estaba limpio y falseo una medida.** Dos procesos
   ajenos ocupaban 34 de 32 nucleos (load average 88) y una estimacion de speedup quedo
   inflada por un orden de magnitud. La cifra buena salio despues de matarlos, con la
   calibracion de un ajuste medida en el propio script. **Toda afirmacion de rendimiento de
   este repositorio debe ir acompanada del load average con que se midio.**

## Revision bibliografica

`docs/v25_literature_review.md`, 9 referencias verificadas por Crossref, ACL Anthology y
arXiv. **7 de 9 coinciden en todos los campos.** Tres correcciones materiales:

- **Rendle 2010**: el DOI `10.1109/ICDM.2010.127` es correcto, pero el arXiv de reserva
  `2010.10593` **apunta a otro trabajo** (CMIM, Sylvain et al., ICASSP 2021). No hay preprint
  sustituto de Rendle en arXiv.
- **All-But-The-Top**: **falta un autor, Suma Bhat** (son tres). El error estaba propagado en
  `reports/v21_p1_01_embeddings_desde_cero.md` en dos sitios, **corregido** en esta rama.
- **GAttention**: el apellido es Vazquez y el nombre de pila Horacio Jarquin; lo declarado
  funde ambos.

De las nueve, **solo All-But-The-Top se ha ejecutado alguna vez** (V22, veredicto negativo).
Con V25, **dos de las nueve quedan medidas**, las dos en contra.

## Artefactos

- `experiments/v25_01_factorization_machine.py`: implementacion secuencial, controles y
  progresion de k. Su corrida quedo guardada en `reports/v25_01_secuencial.log`.
- `experiments/v25_02_fm_paralela_lambda.py`: la misma matematica **importada** de V25-01, mas
  el reparto en paralelo y el peso `lambda_v`. La equivalencia bit a bit con `lambda_v = 1` es
  un control del script, y los dos candidatos compartidos coinciden con la corrida secuencial.
- `reports/v25_01_factorization_machine.{json,md}`,
  `reports/v25_02_fm_paralela_lambda.{json,md}`.
- `docs/v25_mechanism_audit.md`, `docs/v25_literature_review.md`.
- `plan/contexto.md` actualizado a V25.

**Gastos: 0 submissions.** La FM no se envia, y las 3 submissions restantes se reservan.
