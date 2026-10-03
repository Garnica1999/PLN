# V13 · P1b — Fusion de la evidencia por token con V12

## 1. Resultado global

| Modelo | macro-F1 | accuracy |
|---|---|---|
| V12 (referencia) | 0.6005 | 0.6007 |
| V12 + evidencia por token | 0.6003 | 0.6004 |
| Control permutado (semilla 1000) | 0.6 | - |
| Control permutado (semilla 1001) | 0.6003 | - |
| Control permutado (semilla 1002) | 0.6 | - |

**Diferencia: -0.0002** macro-F1 · folds [0.6043, 0.6017, 0.601, 0.6036, 0.5903] frente a [0.6007, 0.605, 0.6032, 0.6004, 0.5928] · **3 de 5 folds positivos**.

## 2. Control de permutado (DD-6)

El mejor control permutado llega a 0.6003 (delta -0.0002). La cifra real (0.6003, delta -0.0002) **no supera** al control: la mejora no viene del contenido del bloque.

## 3. Doble metrica (§36)

| Frontera | Filas | accuracy referencia | accuracy candidata | Cruces | Cruces resueltos referencia | Cruces resueltos candidata |
|---|---|---|---|---|---|---|
| 18-24|25-34 | 57,023 | 0.4963 | 0.5015 | 10,122 | 0.00 % | 0.00 % |
| 25-34|35-49 | 56,577 | 0.4974 | 0.5017 | 7,252 | 0.00 % | 0.00 % |

## 4. Bootstrap y complementariedad

- Diferencia con intervalo al 95 %: **[-0.0015, +0.0010]**, proporcion de resmuestros positivos 0.332 (4000 resmuestros multinomiales).

- global: rescate 5,635 · dano 5,695 · **neto -60**
- 18-24|25-34: rescate 2,717 · dano 2,423 · **neto +294**
- 25-34|35-49: rescate 2,219 · dano 1,971 · **neto +248**

## 5. Por cabeza (DD-8)

| Cabeza | Referencia | Candidata | Diferencia |
|---|---|---|---|
| logreg | 0.5915 | 0.5909 | -0.0006 |
| mlp | 0.5978 | 0.5973 | -0.0005 |

## 6. Gate de adopcion (§24)

| Criterio | Valor | Cumple |
|---|---|---|
| delta minimo | delta -0.0002 >= 0.003 | **no** |
| folds positivos | 3/5 >= 4 | **no** |
| bootstrap excluye cero | IC95 [-0.0015, +0.0010] | **no** |
| supera control permutado | 0.6003 > 0.6003 | **no** |
| no explicada por una fold | la mejor fold aporta +0.0036 de -0.0012 de mejora total | **no** |
| fronteras no empeoran | 2/2 fronteras mejoran | si |

**Decision: NO ADOPTAR**

El bloque **no se adopta**: falla delta minimo, folds positivos, bootstrap excluye cero, supera control permutado, no explicada por una fold. La senal de P1a existe en el bloque aislado pero no se traduce en macro-F1 sobre V12.


## Nota sobre la columna "cruces resueltos"

Las dos filas de esa columna salen 0,00 % **por definición y no por resultado**: el cruce de una
frontera son las filas cuya clase verdadera es una de las dos y que el modelo etiqueta con la otra,
es decir, justamente las filas que ese modelo se equivoca. Ningún modelo puede "resolver" sus
propios cruces. La comparación informativa de esa tabla es la **accuracy en el par**, que es la
columna anterior: 0,4963 -> 0,5015 en `18-24|25-34` y 0,4974 -> 0,5017 en `25-34|35-49`. Ambas
fronteras improves marginally, y ese es el único sitio donde el bloque es constructivo (§4).
