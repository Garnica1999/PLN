# V19 · B1 — Verificacion del +0,0074: bateria de falsacion

B1 dio **+0,0074** con 5 de 5 folds e IC95 que excluia el cero, y el arbol **aislado** (0.6072312) ya superaba al champion (0.6005017). Ese tamano de efecto exige una comprobacion antes de creerlo, y la explicacion alternativa mas simple no es una mejora real: **las 73 columnas finales son predicciones de 72 modelos supervisados**, que un arbol puede combinar de un modo que un promedio lineal no.

## 1. Composicion de la matriz de 279

- miembros 78 + densas 35 + estructuradas 87 + supervisadas 73 = 273
- el bloque supervisado ocupa las columnas 200 a 272, y coincide con `age_supervised_features.npy` con diferencia maxima 0.000e+00

## 2. Que sobrevive al quitar bloques, en la particion de 5 folds

| Bloque | Columnas | macro-F1 | Media folds | 18-24\|25-34 | 25-34\|35-49 |
|---|---|---|---|---|---|
| las 279 completas | 279 | 0.6072312 | 0.607225 | 0.5072 | 0.4977 |
| sin las 73 supervisadas | 200 | 0.586956 | 0.586937 | 0.4823 | 0.4776 |
| solo las 78 de miembros | 78 | 0.5842464 | 0.584236 | 0.4805 | 0.4751 |
| solo las 73 supervisadas | 73 | 0.5597155 | 0.559717 | 0.5021 | 0.4443 |
| **V12 (champion)** | 279 | 0.6005017 | - | 0.4963 | 0.4974 |

**La ventaja de las 73 columnas supervisadas es de +0.0202752.**

## 3. Segunda particion de 4 folds (control de seleccion, D-042)

- macro-F1 **0.6082295**, media de folds 0.608222, fronteras 0.508 y 0.4996.
- Supera el gate de +0,003 sobre V12: **True**.

## 4. Los miembros por separado

| Modelo | macro-F1 |
|---|---|
| LogReg (cabeza de V12) | 0.5915679 |
| MLP(64) (cabeza de V12) | 0.5989481 |
| promedio de las dos (V12 reconstruido) | 0.6011711 |
| arbol solo | 0.6072312 |

## 5. Veredicto de la verificacion

| Comprobacion | Resultado |
|---|---|
| el efecto sobrevive sin las 73 columnas supervisadas | **False** |
| la segunda particion de 4 folds supera el gate | **True** |
| el arbol aislado supera al promedio de las dos cabezas | **True** |
