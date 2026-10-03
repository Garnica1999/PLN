# V20 · P0 — Auditoria de las cinco fronteras ordinales

**V20 queda congelado.** Todo esto es analisis del OOF ya medido, sin reentrenar nada.

## 1. Donde esta la informacion incremental

| Frontera | Lado bajo / alto | Exacta balanceada | Ventaja | F1 binario | AUC | Brier | Descalibracion |
|---|---|---|---|---|---|---|---|
| <= 13-17 vs >= 18-24 | 28,863 / 140,473 | **0.7632** | +0.2632 | 0.9444 | 0.9284 | 0.070318 | +0.0001 |
| <= 18-24 vs >= 25-34 | 57,539 / 111,797 | **0.8264** | +0.3264 | 0.8906 | 0.9224 | 0.10401 | -0.0003 |
| <= 25-34 vs >= 35-49 | 85,886 / 83,450 | **0.8465** | +0.3465 | 0.8444 | 0.9274 | 0.108321 | -0.0001 |
| <= 35-49 vs >= 50-64 | 114,116 / 55,220 | **0.8373** | +0.3373 | 0.7852 | 0.934 | 0.094478 | +0.0002 |
| <= 50-64 vs >= 65-xx | 142,176 / 27,160 | **0.8112** | +0.3112 | 0.7285 | 0.9456 | 0.058341 | -0.0000 |

**Correlacion entre las cinco fronteras**:
```
[[1.0, 0.819, 0.642, 0.477, 0.309], [0.819, 1.0, 0.856, 0.662, 0.437], [0.642, 0.856, 1.0, 0.815, 0.552], [0.477, 0.662, 0.815, 1.0, 0.693], [0.309, 0.437, 0.552, 0.693, 1.0]]
```

La descalibracion es sistematica y todas las fronteras se descalibran en el mismo sentido, luego es un sesgo del conjunto y no un defecto de una frontera concreta.

## 2. La consistencia ordinal

- **18,927 de 169,336 filas (11.18%) violan la monotonia** de las cinco probabilidades acumuladas.
- Magnitud media de la violacion: **0.002457**; maxima: **0.518606**.
- **Como resolvio V20 eso**: con el **maximo acumulado**, que es la unica correccion que respeta el orden sin inventar informacion. No es una operacion neutra: impone la estructura ordinal que los cinco clasificadores, ajustados de forma independiente, no garantizan.

| Fold | Filas con violacion | Fraccion | Magnitud media | Magnitud maxima |
|---|---|---|---|---|
| 0 | 3,658 | 10.80% | 0.002346 | 0.422461 |
| 1 | 3,625 | 10.70% | 0.002390 | 0.369706 |
| 2 | 3,586 | 10.59% | 0.002266 | 0.518606 |
| 3 | 3,945 | 11.65% | 0.002519 | 0.405327 |
| 4 | 4,113 | 12.14% | 0.002763 | 0.378220 |

## 3. El impacto de cada frontera sobre las seis clases

Se degrada cada frontera a un valor constante de 0,5, que es la decision mas incierta posible, y se mide cuanto cae el macro-F1 final. Es lo que separa **frontera debil** de **frontera cuyos errores importan poco**.

Se mide sobre el **blend de V20**, porque la cadena es un cuarto miembro mas y su valor depende de como interactua con los otros tres. En soledad la reconstruccion de las seis clases es fragil y una sola frontera degradada la deshace.

| Frontera degradada | Blend de V20 | Caida del blend | Cadena en soledad | Caida de la cadena |
|---|---|---|---|---|
| <= 13-17 vs >= 18-24 | 0.5968936 | **+0.0129** | 0.0485421 | +0.5561 |
| <= 18-24 vs >= 25-34 | 0.602618 | **+0.0072** | 0.3352297 | +0.2695 |
| <= 25-34 vs >= 35-49 | 0.6057464 | **+0.0040** | 0.4655049 | +0.1392 |
| <= 35-49 vs >= 50-64 | 0.607973 | **+0.0018** | 0.4914815 | +0.1132 |
| <= 50-64 vs >= 65-xx | 0.6073803 | **+0.0024** | 0.4604091 | +0.1443 |

## 4. El error condicional por frontera

Restringido a las **filas cercanas al umbral**, porque un fallo de la decision binaria solo es un fallo de frontera si la fila esta cerca: en las lejanas, discrepar del lado es lo correcto.

| Frontera | Filas cercanas | Errores | Tasa | macro-F1 cuando falla | Cuando acierta | Lejanas |
|---|---|---|---|---|---|---|
| <= 13-17 vs >= 18-24 | 57,539 | 42,281 | 73.5% | 0.2774 | 0.0548 | 0.4295 |
| <= 18-24 vs >= 25-34 | 57,023 | 40,527 | 71.1% | 0.2624 | 0.035 | 0.4765 |
| <= 25-34 vs >= 35-49 | 56,577 | 42,063 | 74.4% | 0.2554 | 0.0428 | 0.4709 |
| <= 35-49 vs >= 50-64 | 56,290 | 43,384 | 77.1% | 0.2804 | 0.0562 | 0.4311 |
| <= 50-64 vs >= 65-xx | 55,220 | 44,428 | 80.5% | 0.3018 | 0.0672 | 0.3899 |

Un error **lejano** es mucho mas grave que uno cercano: indica que el sistema no sabe donde esta, no que este cerca de una frontera. Por eso la tabla separa las filas cercanas de las lejanas.
