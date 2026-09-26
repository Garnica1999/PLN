# V13 · Auditoria del entorno numerico (D-040)

## 1. Que se detecto y como

Al medir la colinealidad de las 279 variables de V12, la correlacion de Spearman salia
**casi cero en todos los pares** (0 pares por encima de 0,80 entre 38.841 pares). Un resultado
tan limpio en un bloque que ya se sabia redundante era sospechoso, asi que se verifico contra
`scipy.stats.spearmanr`. El veredicto fue: mi calculo vectorizado daba `+0,0001` donde `scipy`
daba `+0,9985`.

La depuracion llevan a un hallazgo del entorno, no del codigo:

| Comprobacion | Valor obtenido | Valor exacto | Veredicto |
|---|---|---|---|
| `Z @ Z.T` (auto-transpuesta), diagonal de una 2000x5 | 0,7392 | 1.999,0 | **FALLA** |
| `Z @ Z.T`, elemento cruzado [0,1] | +1,6366 | -33,5575 | **FALLA** |
| `Z @ Z.T` en 5000x279, diagonal | 298,64 | 4.962,39 | **FALLA** |
| `A @ B.T` con A y B **distintas** (10x3, 2000x5, 20000x10, 5000x279) | coincide | coincide | OK |
| `A.T @ A` (matriz-matriz) | coincide | coincide | OK |
| `Z.T @ z` (matriz-vector) | coincide | coincide | OK |
| `np.dot(vector, vector)` | coincide | coincide | OK |
| `np.einsum` | coincide | coincide | OK |
| `np.corrcoef(a, b)` con 1D | coincide | coincide | OK |
| `np.cov(a, b)` con 1D | coincide | coincide | OK |
| `np.linalg.norm` | coincide | coincide | OK |
| `np.var` | coincide (salvo ddof) | coincide | OK |
| `np.corrcoef(matriz_2d)` con `rowvar=True` | forma (5000, 5000) | forma (6, 6) | **FALLA** |
| `np.cov(matriz_2d)` con `rowvar=True` | forma (5000, 5000) | forma (6, 6) | **FALLA** |
| `MLPClassifier` de scikit-learn | identico al parcheado | - | OK |

## 2. Que ruta esta rota exactamente

Solo el patron **`X @ X.T`**, es decir cuando los dos operandos del producto matricial son
vistas de la **misma** matriz. No depende de:

- **hilos**: falla igual con `OMP_NUM_THREADS=1` que con 32;
- **layout**: falla tanto si la matriz es C-contigua como F-contigua;
- **forma ni tamano**: falla en 10x3, 2000x5, 20000x10 y 5000x279;
- **contenido**: con datos enteros pequenos tambien falla (`diag = 0,44` en lugar de `12,97`).

Causa probable: `numpy 1.26.4` (diciembre de 2023) sobre `Python 3.14.7` (octubre de 2025), una
combinacion que numpy no soporta. `numpy` esta enlazado contra el BLAS de referencia del sistema
(`/usr/lib/libblas.so.3`, version 3.12.0), mientras `scipy` trae su propio OpenBLAS; las rutas
de `scipy` son correctas. `np.corrcoef` y `np.cov` con `rowvar=True` caen en el mismo fallo
porque internamente hacen la auto-transpuesta.

## 3. Impacto sobre el proyecto: ninguno

Se auditaron todos los usos de producto matricial y de `corrcoef` del repositorio:

| Sitio | Patron | Ruta | Afectado |
|---|---|---|---|
| `src/pipeline.py:151` | `x_valid @ centroid_matrix.T` | matrices distintas | No |
| `scripts/experimento_e021_ordinal_centroides.py:88` | `x_valid @ centroid_matrix.T` | matrices distintas | No |
| `scripts/experimento_e025_e026_vistas_nuevas.py:98` | `valid @ train.T` | matrices distintas | No |
| `src/verificacion_e001_leakage.py:65` | `block @ reference.T` | matrices distintas | No |
| `scripts/phase32_correccion_probabilistica.py:152` | `p @ matriz.T` | matrices distintas | No |
| `scripts/phase38_bloques_lexicos.py:132` | `rangos_base.T @ rangos_columna` | matriz-vector | No |
| `scripts/phase5_residuales.py:62` | `corrcoef(1D)` | 1D | No |
| `scripts/phase13_error_desacuerdo.py:318` | `corrcoef(1D)` | 1D | No |
| `scripts/phase16_ordinal_greedy.py:105` | `corrcoef(1D)` | 1D | No |
| `scripts/phase24_diagnostico_matriz_test.py:115,134` | `corrcoef(1D)` | 1D | No |
| `scripts/v13_p0b_colinealidad.py:90` | `rangos @ rangos.T` | **auto-transpuesta** | **Si, corregido** |

Ademas, `MLPClassifier` (la mitad del combinador de produccion) da **exactamente las mismas**
predicciones con la ruta parcheada, y el meta-modelo es reproducible fold a fold. Por tanto
**V12 (CV 0,6004 / LB 0,62188) y todas las decisiones anteriores siguen vigentes**: ninguna
metrica del proyecto se calculo con la ruta rota.

## 4. Decision y defensa (D-040)

1. Se documenta el fallo sin maquillar, junto con su reproduccion minima.
2. Se crea `src/verificacion_numerica.py` con dos funciones:
   - `autotest_numerico()`: comprueba seis rutas en las que el proyecto se apoya y **detiene el
     experimento** si alguna falla; ademas enumera las rutas prohibidas para que ningun
     script nuevo las use por descuido.
   - `grama_exacta(matriz)`: matriz de productos cruzados por sumas explicitas, que es la via
     fiable para calcular correlaciones. Coincide con `spearmanr` con **1,2e-16**.
3. Todo experimento de V13 llama a `autotest_numerico()` en su primera linea. El coste es
   despreciable (menos de una centesima de segundo) y evita repetir un dia de trabajo sobre
   metricas falsas.
4. La primera ejecucion de P0b dio "0 pares correlacionados", que era un **artefacto** del fallo,
   no un hallazgo. Se descarta y se vuelve a ejecutar.

## 5. Reproduccion minima

```python
import numpy as np
X = np.random.default_rng(0).normal(size=(2000, 5))
print((X @ X.T)[0, 0], (X[:, 0] ** 2).sum())   # 0.44  12.97   -> FALLA
print(np.corrcoef(X).shape)                    # (2000,2000)  -> deberia ser (5,5)
print(np.corrcoef(X.T).shape)                  # (5,5)        -> OK
```
