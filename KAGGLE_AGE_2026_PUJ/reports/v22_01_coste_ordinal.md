# V22 · 01 — Coste de error proporcional a la distancia ordinal

**V20 queda congelado**: LB **0,62689**, OOF **0,6097782**.

## 1. Por que esta familia

Es la **unica fuente de informacion nueva** que queda. Las representaciones alternativas son ilegales (exigen embeddings o deep learning) o estan agotadas, y la familia ordinal esta cerrada en sus cuatro subfamilias. Esto introduce un **objetivo de entrenamiento** nuevo, que es la categoria que el brief prioriza despues de la representacion.

## 2. El mecanismo

V20 produce seis probabilidades. La decision actual es el **argmax**, que trata igual todos los errores: confundir `13-17` con `65-xx` cuesta lo mismo que confundirlo con `18-24`. Se cambia por el **coste esperado minimo**:

```
elige j = argmin_j  sum_i  p_i * C[i][j]
```

con `C[i][j] = |i - j|^alfa`, la distancia ordinal. Es la decision Bayes-optima bajo esa matriz de costes. **No se reentrena ningun discriminador**: solo cambia como se convierte la salida de V20 en decision. La **unica variable** es `alfa`, en dos valores pre-registrados: 1 (lineal) y 2 (cuadratico).

## 3. Resultado

| Variante | macro-F1 | Delta | Rescates | Danos | Neto | Cambios |
|---|---|---|---|---|---|---|
| **V20 (argmax, control)** | **0,6097782** | - | - | - | - | - |
| coste esperado con alfa 1 | 0.5949198 | -0.0149 | 8,086 | 12,081 | -3,995 | 33,473 |
| coste esperado con alfa 2 | 0.5392168 | -0.0706 | 12,741 | 26,579 | -13,838 | 60,715 |

## 4. Delta por fold

- **alfa 1**: [-0.015712, -0.015301, -0.013165, -0.014866, -0.015255]
- **alfa 2**: [-0.070201, -0.073741, -0.070195, -0.071862, -0.067045]

## 5. Lectura

El coste ordinal **no mejora el macro-F1**. Cambia la decision en filas, pero el balance neto es negativo o nulo: **ordenar los errores por gravedad no hace que el modelo se equivoque menos en la metrica que importa**.

Es la **cuarta confirmacion** de la leccion de P1 dentro de la familia ordinal, y la primera que la prueba **en la capa de decision en vez de en el entrenamiento**, que es el sitio mas limpio para aislarlo.
