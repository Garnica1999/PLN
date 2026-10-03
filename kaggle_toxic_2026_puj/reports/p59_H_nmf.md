# P59-H - NMF: cerrado por equivalencia funcional, sin entrenar

Generado por `experiments/p59_05_nmf_gap_analysis.py`. Fecha 2026-09-28T05:27:37+00:00.

**Gap analysis. 0 modelos, 0 entrenamientos, 0 submissions, 0 Kaggle, 0 holdout, 0 test.**
**NO se registro fila en `experiments.csv`**: el parrafo 50 excluye gap analysis del registro, y registrar aqui una fila sin entrenamiento seria crear una fila ficticia. `experiments.csv` sigue en 324.

## Resultado

| magnitud | valor |
|---|---|
| `H_NOVELTY` | **True** |
| `FUNCTIONAL_EQUIVALENCE` | **EQUIVALENCIA_FUNCIONAL_SUFICIENTE. NMF reconstruye la matriz de evidencia peor que una SVD truncada del mismo rango, luego la informacion que aporta ya estaba disponible con una compresion lineal conocida, y la no negatividad no aporta nada sobre ella** |
| decision | **CLOSED** |
| experimentos consumidos | **0** |

## 1. `H_NOVELTY = TRUE`: nunca se ha ejecutado

- V4-B03: la coincidencia con 'nmf' esta dentro de la cadena 'Track B', falso positivo
- V14-01: la coincidencia esta dentro de una frase de cierre, falso positivo
- ninguna de las 324 filas describe una factorizacion no negativa

## 2. El enunciado nombra `V20`, y `V20` no existe

| prefijo | existe |
|---|---|
| V17 | NO EXISTE |
| V18 | NO EXISTE |
| V19 | NO EXISTE |
| V20 | NO EXISTE |
| V21 | NO EXISTE |
| V24 | NO EXISTE |
| V25 | ['V25-02', 'V25-01'] |

> el parrafo 3 pide usar exactamente la entrada de V20, y V20 no es un experimento de este proyecto

> la matriz de 44 columnas del meta tampoco esta en disco. v2_meta_oof.npz guarda los 44 NOMBRES y la proba resultante, no los valores. Reconstruirla obligaria a reentrenar las 4 bases, que contradice el parrafo 3

> NMF exige matriz no negativa. El meta de 44 columnas tiene margenes con signo, luego NMF no es aplicable ahi sin recortar o desplazar los datos, y el parrafo 3 prohibe modificar los datos

El parrafo 3 del encargo no es ejecutable tal como esta escrito. **No se sustituyo nada en silencio**, que es el fallo que este proyecto lleva tres fases evitando.

## 3. El vecino funcional mas cercano

| campo | valor |
|---|---|
| experimento | V25-01 y V25-02 |
| mecanismo | Factorization Machine binaria, k en {1,4,8,16}, interaccion de segundo orden de BAJO RANGO sobre la representacion exacta de V03 |
| resultado | corrected_cv 0,782699 con +0,005487; decision 'no candidata, cerrar V25' |
| solape | ambos comprimen una estructura de producto en k factores latentes y aplican una cabeza lineal |
| diferencia no cosmetica | NMF impone W >= 0 y H >= 0, luego un componente es un conjunto aditivo y disjunto de entradas que no pueden cancelarse y se lee como un tema; un factor de FM es una mezcla con signo de interacciones |

## 4. La medicion que decide: NMF contra SVD truncada, mismo rango

Matriz: [4449, 21] canales de evidencia, 0 negativos. Rank unico **8**, justificado antes de ejecutar:

> muestras por columna 211.9; el rango util esta acotado por min(n, p_dim) = 21; 8 es el mismo orden de magnitud que el k = 8 que ya uso el precedente V25-01, luego es comparable y no una rejilla; el coste son dos factorizaciones de segundos

| magnitud | SVD | NMF |
|---|---:|---:|
| error de reconstruccion relativo | **0.091040** | 0.093047 |
| esparsidad de los pesos | 0.0238 | 0.5357 |
| pesos negativos | 82 | 0 |

**NMF reconstruye PEOR que la SVD truncada del mismo rango**: brecha -0.002007. NMF convergio en 4008 iteraciones, luego el resultado no es un artefacto de no convergencia.

Lo unico que NMF cambia de verdad es la **forma de los pesos**: son mucho mas esparsos (0.5357 frente a 0.0238) y no negativos por construccion, y los 8 componentes tienen conjunto top-5 distinto. Eso es legible, y es exactamente lo que el parrafo 67 pedia como justificacion. Pero legible no es informacion nueva: **una compresion lineal del mismo rango ya captura todo lo que NMF captura, y algo mas.**

Los componentes se leen asi:

- componente 0: canal08_HS, canal19_HS, v03_HS
- componente 1: canal09_HS, canal02_HS, canal16_HS
- componente 2: canal11_HS, canal12_HS, canal03_HS
- componente 3: canal14_HS, canal15_HS, canal16_HS
- componente 4: canal10_HS, canal09_HS, canal08_HS
- componente 5: canal04_HS, canal05_HS, canal07_HS
- componente 6: canal06_HS, canal07_HS, canal10_HS
- componente 7: canal13_HS, canal18_HS, canal19_HS

## 5. Decision

```text
P59-H = CLOSED
motivo EQUIVALENCIA_FUNCIONAL_SUFICIENTE
experimentos consumidos 0
```

El parrafo 1 autoriza cerrar la pista sin entrenar cuando el gap analysis encuentra equivalencia funcional suficiente, y eso es lo que ocurre. Cerrar aqui evita gastar un experimento en medir algo que un control de capacidad ya ha resuelto, y evita montar una rejilla de rangos sobre una hypothesis que no tiene mecanismo propio.

## 6. Formulacion

> V03 remains the best verified leaderboard result.

*Seccion generada por `experiments/p59_05_nmf_gap_analysis.py`. Sin push.*
