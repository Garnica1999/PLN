# CHAMPION

Estado congelado del mejor mecanismo competitivo observado en AGE.

## Varios scores que no deben confundirse

| Nombre | Que es | Valor |
|---|---|---|
| **V12 official submission** | el envio `age_submission_v12.csv`, el unico que se ha subido y puntuado como envio de V12 | **0,62182** |
| V12 reproducible | el artefacto `age_submission_v12_reproducible.csv`; 93,88 % de acuerdo con el anterior, unos 6.900 test distintos | 0,62188 |
| **V19 boosting** | el challenger de boosting, **subido y puntuado** | **0,62578** |

**No volver a llamar 0,62188 "LB oficial".** El envio oficial de V12 es 0,62182.

## Champion actual

| Campo | Valor |
|---|---|
| **Nombre** | **V19 boosting** |
| **LB** | **0,62578** |
| CV LOFO agrupado de 5 folds | 0,6079388 |
| CV en particion independiente de 4 folds | 0,6090431 |
| Delta de LB frente al envio oficial de V12 | **+0,00396** |
| Delta de LB frente a la replica de V12 | +0,00390 |
| Submission | `submissions/age_submission_v19_challenger_boosting.csv` |
| Modelo persistido | `models/v19_champion_lightgbm.txt` |
| SHA256 del modelo | `22221d1802c1ec4495f393f27f884b9119763a30d47072a0a1205ad671c50619` |
| Semilla del arbol | 42 |
| Semilla global | 42 |

## Configuracion congelada

```
objetivo del booster: LightGBM multiclass, num_class 6
n_estimators         = 300
learning_rate        = 0.05
num_leaves           = 8
max_depth            = 3
min_child_samples    = 200
colsample_bytree     = 0.5
subsample            = 0.8
subsample_freq       = 1
reg_lambda           = 1.0
random_state         = 42
sin min_child_weight, sin gamma
```

Combinacion, **pre-registrada y nunca ajustada sobre la validacion**:

```
(1/3) x LogReg  +  (1/3) x MLP(64)  +  (1/3) x arbol
```

## Variables, sin ninguna nueva

Las **279** de V12, sin seleccion de columnas:

| Bloque | Columnas | Origen |
|---|---|---|
| miembros OOF | 78 | 14 ficheros de `features/`, en `phase2_ponderacion.MEMBER_FILES` |
| densas | 35 | `features/dense_train.parquet` |
| estructuradas | 87 | `features/age_structured_features.parquet` |
| **label-informed** | **73** | `features/age_supervised_features.npy` (`mention_mean_c0`..`c5`, `mention_entropy`, y familia) |
| NB-SVM de palabra | 6 | `features/oof_ADV07_nbsvm_word.npy` |

Matriz de train: `features/meta_features_train_v11.npy` (273, verificada **bit a bit identica** a
`build_features()`) concatenada con las 6 de NB-SVM de palabra. Matriz de test:
`features/meta_features_test_v12_fresca.npy` (112.892 x 279). El orden interno de columnas de train y
test se verifico **por firma**: la columna de mayor magnitud del bloque label-informed ocupa el mismo
indice 255 en ambos.

## Target y validacion

- **Target**: las **seis clases verdaderas**. Las 279 variables son features auxiliares. Ninguna
  etiqueta se sustituye por una prediccion en ningun punto.
- **Validacion**: LOFO agrupado de 5 folds sobre `features/dup_clusters.parquet` (164.018 grupos, el
  mayor de 153 filas). El meta se ajusta en cuatro folds y predice el quinto; nunca sus propias filas.
- Confirmacion independiente en una particion de **4 folds**.

## Que cambia respecto a V12, y cuanto

- **11.392 de 112.892 filas cambian (10,09 %)** en la submission.
- En el train, con el OOF: rescate 4.622, dano 3.311, **neto +1.311**.
- Fronteras: `18-24|25-34` de 0,4963 a **0,5083** y `25-34|35-49` de 0,4974 a **0,5018**, y **sin
  intercambio de suma cero**, que era el fallo que hundia la global en P1.

## Por que funciona, en una frase

El combinador de V12 era un **promedio lineal** de dos cabezas, y un bloque de **73 variables
label-informed** dentro de sus propias 279 necesita combinacion no lineal que un promedio no puede dar.
No es que los arboles sean mejores: es que el booster es la primera vez que esas 73 se combinan con el
resto de forma no lineal.

Evidencia: las 279 dan 0,6072, sin las 73 quedan en 0,5870 (por debajo de V12) y las 73 solas dan
0,5597. Es **interaccion y no suma**.

## Estado de la explotacion

**CERRADA** el 2026-09-27 (D-052). Se ejecutaron las dos unicas configuraciones que el agente
`kaggle specialist` autorizo, y ninguna merece submission:

| Configuracion | CV | Delta contra V19 | Folds pos. | 4 folds | Merece submission |
|---|---|---|---|---|---|
| A, `max_depth` 4 con `num_leaves` 8 | 0,6079208 | **-0,0000** | 3/5 | -0,0000 | **no** |
| B, `min_child_samples` 800 | 0,6076698 | **-0,0003** | 0/5 | -0,0002 | **no** |

## Que reabre AGE, si algo

Solo tres cosas, y ninguna es una nueva familia:

1. Una **fuente de informacion nueva**.
2. Una **regla nueva del concurso**.
3. **Evidencia externa** que cambie sustancialmente el espacio de busqueda.

Anotado por el especialista y **fuera del alcance actual**: E-022 probo meta LightGBM sobre un stack de
102 columnas de la era V10 con aporte 0,0000, y **nunca se probo sobre las 279 de la era V12** con un
esquema de regularizacion explicito. V19 es justamente eso, asi que la linea esta ya cobrada.
