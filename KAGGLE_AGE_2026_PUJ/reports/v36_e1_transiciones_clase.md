# V36 - E1 (via C1-A) - Transiciones de clase de caracter (brecha L4)

**Fecha**: 2026-09-30. **Nodo del arbol**: `SENAL_EXISTE_INTEGRACION_NEGATIVA`. **L4**: se cierra L4 por el nodo del arbol; C1-A es ONE-SHOT: no se reintenta con otro alfabeto, otro rango de n, otro mapa ni otro corte.

**Motivo**: el banco supero su hipotesis nula pero la integracion es negativa: se reporta el banco y el nodo, y L4 se cierra

## Lo que se midio

- Bloque C1-A: mapa categorico LITERAL de 6 simbolos (`S D P L M O`), n-gramas de clase [2, 3] (bigramas y trigramas), SIN borde, SIN min_df.
- Columnas reales del bloque: **154** (27 bigramas + 127 trigramas de clase), las mismas que publica el pre-registro.
- Folds LEIDOS de `artifacts/p33_scores/todos/fold_idx.npy` (particion canonica, no recreada ni barajada); modelo LinearSVC con la receta del campeon, sin grid.

## Banco (C1-A sobre char_wb)

| Brazo | macro-F1 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|
| `banco_referencia` | 0.5408073 | 0.5443048 | 0.5420068 | 0.5458866 | 0.5382011 | 0.5335904 |
| `banco_c1a` | 0.5440346 | 0.5481312 | 0.5440766 | 0.5478084 | 0.5422674 | 0.5378290 |
| `banco_c1a_remapeo1` | 0.5413071 | 0.5443110 | 0.5431529 | 0.5469849 | 0.5376790 | 0.5343583 |
| `banco_c1a_remapeo2` | 0.5408326 | 0.5439612 | 0.5428287 | 0.5448158 | 0.5384637 | 0.5340362 |
| `banco_c1a_remapeo3` | 0.5415833 | 0.5456007 | 0.5422401 | 0.5465563 | 0.5390639 | 0.5344048 |
| `banco_c1a_entre_docs` | 0.5405076 | 0.5442113 | 0.5419126 | 0.5450030 | 0.5384063 | 0.5329490 |

- **Delta del banco**: +0.0032273 de macro-F1 (5/5 folds positivos), IC95 pareado [+0.0022338, +0.0042230] (2000 remuestreos, semilla 20261010).
- **Supera su H0** (delta = 0): True.

## Controles (enmienda pre-ejecucion 1)

| Control | macro-F1 | delta real - control | IC95 pareado | el mapa real gana |
|---|---|---|---|---|
| remapeo aleatorio 1 | 0.5413071 | +0.0027275 | [+0.0017321, +0.0036731] | True |
| remapeo aleatorio 2 | 0.5408326 | +0.0032019 | [+0.0022526, +0.0042020] | True |
| remapeo aleatorio 3 | 0.5415833 | +0.0024513 | [+0.0014354, +0.0034521] | True |
| barajado entre documentos (C3) | 0.5405076 | +0.0035270 | [+0.0025358, +0.0046144] | True |

- Remapeos ganados: **3/3** (condicion (ii) corregida: el mapa real debe ganar a los tres remapeos aleatorios de caracteres a las mismas 6 clases).
- Barajado entre documentos: delta real - control +0.0035270; conserva la distribucion marginal global y destruye la asociacion por documento.

### Por que se sustituyeron los dos controles originales

- **Permutacion de los NOMBRES de las clases** (control C2 original): es una renombracion **biyectiva** del alfabeto, luego la matriz es una **permutacion exacta de columnas**: el vocabulario del control es la imagen exacta del real (True), la permutacion de columnas es biyectiva (True), el `idf` difiere en **0.000e+00** y la matriz, reordenada por esa permutacion, difiere en **0.000e+00**. Como el espacio de hipotesis de un clasificador lineal es invariante bajo esa permutacion, el control media **ruido del solucionador**: {perm['por_que_no_puede_fallar']}.
- **Barajado DENTRO de cada documento** (control C3 original): el documento es un multiconjunto de n-gramas, luego barajar deja el mismo recuento por token: mismo vocabulario (True), `idf` identico (True), misma estructura dispersa (True) y diferencia maxima de valores **1.665e-16**. el documento es un multiconjunto de n-gramas: barajar dentro de el deja el mismo recuento por token y la misma matriz, luego 'el orden importa' es intestable.
- Sustitucion: por **remapeo aleatorio de caracteres** a las mismas 6 clases (C2) y por **barajado entre documentos** (C3), conforme a la enmienda pre-ejecucion 1. Diferencia maxima declarada en la enmienda para la permutacion: 3.3e-16. los dos controles originales NO pueden fallar: uno es una renombracion biyectiva (bloque isometa por permutacion de columnas, idf identico y espacio de hipotesis lineal invariante) y el otro es un barajado de un multiconjunto (misma matriz). Por eso la enmienda pre-ejecucion 1 los sustituyo por el remapeo aleatorio de caracteres y por el barajado entre documentos.

## Integracion (campeon + C1-A)

- Punto de partida reproducido: **0.6159732** = macro-F1 de `argmax(v30 * pesos)` (pesos [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]), verificado en runtime.
- Campeon con C1-A: **0.6143719**, delta frente al campeon **-0.0016013**, 0/5 folds positivos, IC95 pareado [-0.0024150, -0.0007569].
- Delta atribuible al bloque C1-A (candidato menos canal de control): **+0.0004072**, IC95 [-0.0000147, +0.0008279].
- Efecto mecanico de volver a pasar por el combinador anidado: -0.0020085.

## Gates de la serie (aplican SOLO a la integracion)

| Gate | Valor | Umbral | Pasa |
|---|---|---|---|
| delta_min | -0.0016013321501044953 | 0.0015 | False |
| folds_min | 0 | 4 | False |
| control_favorable | True | True | True |
| ic95_pareado_sin_cero | True | True | True |

- Variabilidad de referencia: 0.0008876 (delta integrado / variabilidad = -1.8041146350884354).
- Etiqueta de la serie: **CLOSED**. Segunda particion requerida: False; ejecutada: **NO** (C1-A es one-shot y exige autorizacion del usuario).

## Sondas obligatorias

| Sonda | Resultado |
|---|---|
| banco_ree_medido | True |
| tabla_5_de_5_folds | True |
| bootstrap_pareado | True |
| permutaciones | True |
| barajado | True |
| conformidad | True |
| fold_local | True |
| mapa_literal | True |

## Enmienda pre-ejecucion 1 aplicada

- Objeto: n_classes declaradas, rango_n_de_transiciones, columnas observadas, y los dos controles de especificidad y de barajado.
- `n_clases_del_alfabeto` = 6 y el mapa literal sigue con su sha256 `cac5b2f32699a389` intacto.
- `rango_n_de_transiciones` = [2, 3] = lo ejecutado.
- `columnas_observadas` = 27 bigramas + 127 trigramas = 154, medidas por el runner en cada fold.
- No cambia el mapa literal, ni el modelo, ni los folds, ni los gates, ni las hipotesis nulas, ni el arbol, ni la one-shot.

## Compliance

- 0 transformers, 0 LLM, 0 embeddings, 0 datos externos, 0 metadatos; sin test mas alla de `transform -> predict`; sin holdout, sin leaderboard, sin submission.
- Campeon V30 + plug-in intacto y NO reentrenado; `artifacts/p33_scores/todos` leido y nunca escrito.
- RSS con `ru_maxrss`: 1.74 GB (tope 20 GB); presupuesto declarado 2400 s; coste real 403 s.
