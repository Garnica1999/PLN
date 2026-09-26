# V14 · B — Fusión de las variables lingüísticas con V12

## 1. Por qué se mide pese al pre-chequeo negativo

El pre-chequeo de V14a dio un neto de −58.682 al aplicar el bloque como clasificador, y con el protocolo habitual la pista se cerraba ahí. Se mide igualmente porque ese criterio contesta a otra pregunta: si el bloque aporta algo **al meta-modelo** es una decisión distinta, y D-042 acaba de bajar el listón justamente para medir diferencias pequeñas. El bloque tiene 0,2409 de control aislado, solo cuatro variables redundantes y su información de clase no es linealmente accesible desde V12 (P16c).

El bloque entra con **33 columnas**: se retiran las cuatro que el cribado marcó como casi duplicadas de V12 (longitud media de palabra con rho 0,9824, diversidad por lema 0,8448, hapax por lema 0,8329 y tokens por frase 0,8203).

## 2. Resultado principal

| Modelo | macro-F1 | Delta | Folds positivos |
|---|---|---|---|
| V12 (referencia) | 0.6005 | - | - |
| V12 + lingüísticas | 0.599 | -0.0015 | 1/5 |
| Control permutado (semilla 1000) | 0.5962 | -0.0043 | - |
| Control permutado (semilla 1001) | 0.5975 | -0.0030 | - |
| Control permutado (semilla 1002) | 0.5986 | -0.0019 | - |

## 3. Segunda partición (criterio 4 del gate)

Partición de **4 folds agrupados**, independiente de la de 5:

| Fold | V12 | Candidata | Diferencia |
|---|---|---|---|
| 1 | 0.6013 | 0.5984 | -0.0029 |
| 2 | 0.5998 | 0.5973 | -0.0025 |
| 3 | 0.6039 | 0.6004 | -0.0035 |
| 4 | 0.6048 | 0.6011 | -0.0037 |

**Positivos: 0/4** (se exigen 3) · media de la diferencia -0.0031 · IC95 del bootstrap pareado [-0.0027, -0.0002] con 0.009 de proporcion positiva.

## 4. Doble métrica (§36)

| Frontera | V12 | Candidata |
|---|---|---|
| 18-24|25-34 | 0.4963 | 0.4919 |
| 25-34|35-49 | 0.4974 | 0.4975 |

Complementariedad: global rescate 5,732 · daño 5,888 · **neto -156**
- 18-24|25-34: rescate 2,528 · daño 2,778 · neto -250
- 25-34|35-49: rescate 2,168 · daño 2,161 · neto +7

## 5. Gate D-042

| Criterio | Valor | Cumple |
|---|---|---|
| efecto minimo | delta -0.0015 >= 0.001 | **no** |
| folds positivos | 1/5 >= 3 | **no** |
| significacion | IC95 [-0.0027, -0.0002] | **no** |
| segunda particion | 0/4 >= 3 | **no** |
| control permutado | 0.599 > 0.5986 | si |
| fronteras | 1/2 mejoran | **no** |

**Decision: NO ADOPTAR**

