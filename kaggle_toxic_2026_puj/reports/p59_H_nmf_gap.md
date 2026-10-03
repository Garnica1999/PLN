# P59-H - NMF: gap analysis de novedad antes de entrenar

Generado por `experiments/p59_05_nmf_gap_analysis.py`. Fecha 2026-09-28T05:27:37+00:00.

**Gap analysis. 0 modelos, 0 entrenamientos, 0 submissions, 0 Kaggle, 0 holdout, 0 test. No consume presupuesto de experimentos (parrafo 50).**

## 1. `H_NOVELTY`

```text
H_NOVELTY = True
```

- V4-B03: la coincidencia con 'nmf' esta dentro de la cadena 'Track B', falso positivo
- V14-01: la coincidencia esta dentro de una frase de cierre, falso positivo
- ninguna de las 324 filas describe una factorizacion no negativa

## 2. El enunciado nombra `V20`, y `V20` no existe

| prefijo de id | existe |
|---|---|
| V17 | NO EXISTE |
| V18 | NO EXISTE |
| V19 | NO EXISTE |
| V20 | NO EXISTE |
| V21 | NO EXISTE |
| V24 | NO EXISTE |
| V25 | ['V25-02', 'V25-01'] |

> el parrafo 3 pide usar exactamente la entrada de V20, y V20 no es un experimento de este proyecto

Lo que si existe con ese nombre:

- toxic_submission_v20_01_regla_hs_interaccion.csv, etiqueta de lote
- toxic_submission_v20_03_reproduccion_v03.csv, control de reproduccion
- mapeo de leaderboard "56597267": V20-01

> la matriz de 44 columnas del meta tampoco esta en disco. v2_meta_oof.npz guarda los 44 NOMBRES y la proba resultante, no los valores. Reconstruirla obligaria a reentrenar las 4 bases, que contradice el parrafo 3

> NMF exige matriz no negativa. El meta de 44 columnas tiene margenes con signo, luego NMF no es aplicable ahi sin recortar o desplazar los datos, y el parrafo 3 prohibe modificar los datos

**Ninguna de estas tres cosas se resuelve solo, y sustituirlas en silencio seria exactamente el fallo que este proyecto lleva tres fases evitando.**

## 3. `FUNCTIONAL_EQUIVALENCE`

```text
FUNCTIONAL_EQUIVALENCE = EQUIVALENCIA_FUNCIONAL_SUFICIENTE. NMF reconstruye la matriz de evidencia peor que una SVD truncada del mismo rango, luego la informacion que aporta ya estaba disponible con una compresion lineal conocida, y la no negatividad no aporta nada sobre ella
```

### El vecino mas cercano: V25-01 / V25-02

| campo | valor |
|---|---|
| experimento | V25-01 y V25-02 |
| mecanismo | Factorization Machine binaria, k en {1,4,8,16}, interaccion de segundo orden de BAJO RANGO sobre la representacion exacta de V03 |
| resultado | corrected_cv 0,782699 con +0,005487; decision 'no candidata, cerrar V25' |
| solape | ambos comprimen una estructura de producto en k factores latentes y aplican una cabeza lineal |
| diferencia no cosmetica | NMF impone W >= 0 y H >= 0, luego un componente es un conjunto aditivo y disjunto de entradas que no pueden cancelarse y se lee como un tema; un factor de FM es una mezcla con signo de interacciones |

### SVD

- donde si se uso: sobre TEXTO: DL-MLP SVD 300, svd300_rbf, V26-01
- sobre matriz meta: nunca
- no hay equivalencia que citar para SVD sobre evidencia de ensemble

## 4. La medicion economica: NMF contra SVD truncada

Matriz: [4449, 21] con 0 negativos. Rank unico **8**, justificado antes de ejecutar:

> muestras por columna 211.9; el rango util esta acotado por min(n, p_dim) = 21; 8 es el mismo orden de magnitud que el k = 8 que ya uso el precedente V25-01, luego es comparable y no una rejilla; el coste son dos factorizaciones de segundos

| magnitud | SVD | NMF |
|---|---:|---:|
| error de reconstruccion relativo | 0.091040 | 0.093047 |
| esparsidad de los pesos | 0.0238 | 0.5357 |
| pesos negativos | 82 | 0 |

Brecha de reconstruccion SVD menos NMF: **-0.002007**. NMF converge en 4008 iteraciones.

Componentes con conjunto top-5 distinto de todos los demas: **8 de 8**.

Los tres inputs dominantes de cada componente de NMF:

- componente 0: canal08_HS, canal19_HS, v03_HS
- componente 1: canal09_HS, canal02_HS, canal16_HS
- componente 2: canal11_HS, canal12_HS, canal03_HS
- componente 3: canal14_HS, canal15_HS, canal16_HS
- componente 4: canal10_HS, canal09_HS, canal08_HS
- componente 5: canal04_HS, canal05_HS, canal07_HS
- componente 6: canal06_HS, canal07_HS, canal10_HS
- componente 7: canal13_HS, canal18_HS, canal19_HS

## 5. Opciones ejecutables, si se desbloquea el enunciado

| opcion | entrada | a favor | en contra |
|---|---|---|---|
| H-opcion-1 | features/p50_celda_features.npz, 20 canales + V03 = 63 columnas por etiqueta | en disco, no negativa por construccion, 1483 ids y los mismos pliegues canonicos | no es el meta de 44 columnas; es evidencia de canales, no de 4 bases |
| H-opcion-2 | features/v2_oof_base_corrected.npz, 4 bases x 3 etiquetas = 12 columnas | es el insumo real del meta de 44 columnas | 12 columnas apenas admiten compresion; el rango util es ~12 y NMF no puede reducir mucho sin perder informacion |
| H-opcion-3 | reconstruir la matriz de 44 columnas reentrenando las 4 bases | seria el meta canonico | contradice el parrafo 3 y reabre una familia cerrada; ademas exige no negatividad, luego habria que recortar los margenes |

## 6. Decision preliminar

```text
decision  CLOSED
```

La pista NO se da por cerrada: el parrafo 1 solo permite cerrarla si la equivalencia funcional es suficiente, y la medicion dice que no lo es. Pero tampoco se puede ejecutar H1 sin resolver cual es 'la entrada de V20', y eso es una decision, no una deteccion.

## 7. Formulacion

> V03 remains the best verified leaderboard result.

*Seccion generada por `experiments/p59_05_nmf_gap_analysis.py`. Sin push.*
