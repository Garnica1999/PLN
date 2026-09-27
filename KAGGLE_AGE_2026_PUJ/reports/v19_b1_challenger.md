# V19 · B1 — Challenger de boosting: submission preparada

**No se envia nada desde este script.** El fichero queda listo y validado; enviarlo es una decision de Carlos.

## 1. El hallazgo

Un boosted tree pequeno y regularizado sobre las **279 variables de V12** saca **0.6072312** en OOF, frente a los **0.6005017** del champion. El combinador de V12, que es un promedio lineal de una LogReg (0.5916) y un MLP(64) (0.5989), esta desperdiciando informacion que ya esta en su propio espacio de variables.

| Combinacion | CV 5 folds | CV 4 folds | Delta |
|---|---|---|---|
| V12 (champion) | 0,6005017 | 0,6005017 | - |
| arbol solo | 0,6072312 | - | +0.0067 |
| LogReg + MLP + arbol a partes iguales | 0,6079388 | **0,6090431** | +0.0074 |

**La segunda particion de 4 folds es la que no esta seleccionada**: los 4 folds quedan por encima del gate (0,6077 / 0,6060 / 0,6102 / 0,6122), con las fronteras en 0,5083 y 0,5018 frente a las 0,4963 y 0,4974 de V12.

## 2. De donde viene el efecto, y por que es legitimo

La bateria de falsacion acoto el origen, y no es el que mas gusta:

| Bloque de variables | Columnas | macro-F1 |
|---|---|---|
| las 279 completas | 279 | **0,6072** |
| sin las 73 supervisadas | 200 | 0,5870 |
| solo las 78 de miembros | 78 | 0,5842 |
| solo las 73 supervisadas | 73 | 0,5597 |

El efecto **desaparece** sin el bloque supervisado, luego no es que el arbol sea un mejor modelo en general: es que ese bloque necesita **combinacion no lineal** y un promedio lineal no la puede dar. Ese bloque no son probabilidades sino agregados label-informed (`mention_mean_c0` a `c5` y familia, con valores de hasta 59.439).

**No hay fuga.** El bloque es cross-fitted: su macro-F1 aislado es 0,5597 y su media de folds coincide con el pooled en 1,5e-06. Si estuviera ajustado in-sample, la media de folds se separaria del pooled de forma apreciable.

## 3. Verificaciones hechas antes de escribir el fichero

| Verificacion | Resultado |
|---|---|
| orden de columnas train/test, por firma del bloque supervisado | columna 255 en ambos: **coincide** |
| fidelidad de la reconstruccion de V12 con la matriz fresca | 0,9057, frente al 0,9388 que registro la fase 25 con esta misma matriz |
| 8 de 8 checks de formato | correctos |
| filas | 112,892, las mismas que el test |

## 4. Que cambia respecto a V12

- **Filas modificadas**: **11,392** de 112,892, es decir **10.09%**
- **Rescates en OOF**: 4,622
- **Danos en OOF**: 3,311
- **Neto**: +1,311
- **Fronteras**: 0,4963 a 0,5083 y 0,4974 a 0,5018, medidas en la particion de 4 folds. La primera frontera mejora **+0,0120**, que es el doble que el +0,0038 que movio el termino aditivo de P4 y del orden de lo que daba el margen de adyacencia de P3 (+0,0209 en sistema combinado). Y esta vez **sin** el intercambio de suma cero que hundia la global.

## 5. Distribucion de clases

| Clase | Candidato | V12 enviado |
|---|---|---|
| 13-17 | 18,521 | 17,673 |
| 18-24 | 20,263 | 20,715 |
| 25-34 | 17,430 | 16,543 |
| 35-49 | 16,412 | 17,531 |
| 50-64 | 20,599 | 20,763 |
| 65-xx | 19,667 | 19,667 |

Ambas distribuciones son sanas y del mismo orden: el candidato no desplaza masa de forma anomala.

## 6. Riesgo que no se puede medir aqui, y que solo resuelve el leaderboard

El bloque supervisado tiene una **firma distinta en train y en test**: maximo 59.439 en train frente a 50.673 en test, porque en train es cross-fitted y en test procede de modelos ajustados sobre todo el train. Un combinador lineal promedia esa diferencia sin quejarse; un arbol puede ser mas sensible a ella.

**Ese es el riesgo real de este candidato**, y no se puede medir con los artefactos disponibles: el supervisor de V12 tambien se ajusto sobre el train completo, luego comparte ese mismo sesgo. Con la calibracion verificada (`LB = +0.00223 + 1.02825 x CV`), un CV de 0.6079388 preve un leaderboard de **0.62734**, frente al 0.62182 del envio de V12.

Ese numero es una **prediccion**, no una medida. Lo que se sabe con certeza es que el efecto aparece en dos particiones independientes, en 9 de 9 folds en total, y sobrevive a un control de ruido por permutacion que lo deja por debajo de V12.
