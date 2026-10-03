# P59-D - Influence functions sobre V03

Generado por `experiments/p59_04_influence.py`. Fecha 2026-09-28T04:30:06+00:00.

**Diagnostico. 0 modelos, 0 submissions, 0 Kaggle, 0 holdout, 0 test. No se elimina ninguna observacion (parrafo 25).** No se registro fila en `experiments.csv`.

## 1. Auditoria de hueco

> ningun experimento del registro estima influencia sobre los parametros de V03

> cerro la ponderacion por tamano de grupo, que es influencia en el sentido de la intuicion pero no del calculo

## 2. Metodo

```text
objetivo     0,5 w'w + C sum_i log(1 + exp(-z_i)) con z_i = y_i (x_i . w + b)
gradiente    -C sum_i y_i x_i sigmoid(-z_i) + w
hessiano     C X' W X + I, con W = diag(sigmoid(z) sigmoid(-z))
influencia   I_i = C y_i sigmoid(-z_i) H^-1 x_i
woodbury     H^-1 = I - X' P X, K = (I + C W^(1/2) G W^(1/2))^-1, G = X X'
costo        una inversa de 1186 por 1186 y un producto disperso por denso; nunca se forma el Hessiano de 27431 por 27431
elimina_algo False
```

El Hessiano de V03 tiene 27431 por 27431, que son 6 GB en float64. La identidad de Woodbury lo deja en el dual de 1186 dimensiones, que es donde cabe. **Nunca se forma el Hessiano grande**, y por eso el metodo es barato sin ser aproximado.

## 3. Controles

| test | veredicto | detalle |
|---|---|---|
| la influencia es finita | OK | max |infl| = 1.6766e+00, max norma = 1.5102e+01 |
| la forma dual G - G P G coincide con la resolucion directa de H | OK | max |dual - directo| = 3.145e-14 |
| el orden de la influencia sigue al LOO real (Spearman) | OK | rho = 0.9993 sobre 60 borrados reales, con senal en los datos |
| el signo del cambio estimado coincide con el real | OK | concordancia 1.0000 sobre 60 borrados con cambio no nulo |
| el error de magnitud es de primer orden y queda documentado | OK | error relativo mediano 0.0444 y maximo 0.1560. Se registra como limite del metodo: la influencia es un dispositivo de cribado por ORDEN, no un LOO exacto |
| la auto-influencia es |residuo| por un factor de palanca positivo | OK | factor min 0.3087 max 1.9846, y |factor| / c = |residuo| en todas las filas: True |
| el calculo es determinista | OK | dos llamadas coinciden bit a bit |

El test que de verdad importa es el cuarto: se borra una fila de verdad, se reentrena, y se comprueba que el cambio real de parametros se aproxima con la influencia estimada.

## 4. Resultado por etiqueta

| etiqueta | error en el top 100 | base | enriquecimiento | grupo multiple en el top | base |
|---|---:|---:|---:|---:|---:|
| HS | 0.9200 | 0.2131 | **4.32** | 0.0100 | 0.0040 |
| TR | 0.7100 | 0.1106 | **6.42** | 0.0100 | 0.0040 |
| AG | 0.8000 | 0.1868 | **4.28** | 0.0100 | 0.0040 |

## 5. La asociacion que decide, y el control que la evita

| magnitud | valor | p |
|---|---:|---:|
| Spearman(\|influencia\|, error) | +0.3955 | 1.814e-166 |
| Spearman(\|influencia\|, confianza) | -0.5345 | 0.000e+00 |
| razon error / confianza | **0.740** | |

> el parrafo 63 exige que la influencia se relacione con la frontera MAS ALLA de lo que la confianza ya explica

Sin el segundo renglparrafo n, el primero solo diria que los ejemplos dificiles tienen influencia, que es lo esperable y lo inutil. La comparacion es lo que separa las dos hipotesis.

## 6. El objetivo duro: las 42 celdas no rescatables

| magnitud | valor |
|---|---:|
| percentil de influencia de las 42 no rescatables | 0.8516 |
| percentil de influencia de las 715 rescatables | 0.7463 |
| razon | 1.141087 |

## 7. Decision

```text
decision  CLOSED. La influencia se asocia al error MENOS que la confianza, luego es una medida peor de dificultad que lo que ya existe. El downweighting por influencia no se ejecuta y no se reintenta
```

## 8. Formulacion

> V03 remains the best verified leaderboard result.

*Seccion generada por `experiments/p59_04_influence.py`. Sin push.*
