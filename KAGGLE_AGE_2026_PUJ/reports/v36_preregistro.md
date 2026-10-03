# V36 - E1 (via C1-A) - Pre-registro congelado: transiciones de clase de caracter (L4)

**Fecha**: 2026-09-30. **Estado**: CONGELADO antes de ejecutar (commit `74ca44e`).
**Baseline recomputado**: 0.6159732 (nested 0.6154643); ancla trivial verificada en runtime: 0.04854212180687087.

## Que es y por que no es un re-nombre

- **Brecha**: L4 - Transiciones de caracteres, declarada en scripts/phase0_freeze_gap_v2.py:157 y nunca ejecutada (backlog linea 204: 'pendiente').
- **Reapertura de F10**: la clausula de reapertura de F10 dice literalmente 'solo se abre lo que cambie el MECANISMO: condicionar la palabra a su forma, su posicion o su CATEGORIA'. C1-A es condicionar el caracter a su categoria. Funcionalmente, un n-grama literal mide ortografia lexica y uno sobre el alfabeto de clases mide perfil de tipo de transicion, y ninguno se recupera del otro.

## El mapa de 6 clases (TABLA LITERAL, condicion (i) del ruling)

Redactado a priori desde categorias Unicode, **sin ninguna estadistica del corpus**, sin
ajuste por etiqueta y sin ningun parametro estimado. El mapa **descarta** la identidad del
caracter: es una abstraccion disenada, no un espacio semantico denso.

| Clase | Que representa |
|---|---|
| `S` | espacio y todo lo que `str.isspace()` marca como espacio |
| `D` | dígitos ASCII 0-9 |
| `P` | puntuación ASCII, es decir todo lo que no es alfanumérico, no es espacio y no es dígito |
| `L` | letra minúscula, con y sin acento |
| `M` | letra mayúscula, con y sin acento |
| `O` | resto: emojis, flechas, caracteres de control, alfabetos no latinos |

- Orden del alfabeto: `S D P L M O`; `sha256` del mapa `cac5b2f32699a389...`.
- Rango de n: bigramas y trigramas de clase; sin `min_df` (vocabulario acotado por construccion).
- **Borde: NO se anaden simbolos de borde (declarado antes de ver el banco; un borde inventaria transiciones inexistentes).**

## Las 6 condiciones del ruling, incorporadas

- **i_mapa_literal**: el mapa de 6 clases es una tabla literal del codigo, con sha256, redactado a priori desde categorias Unicode sin ninguna estadistica del corpus
- **ii_sonda_de_permutaciones**: permutar el alfabeto en >= 3 permutaciones y exigir que el mapa real gane a los controles; si no gana, el veredicto es 'sin senal especifica de categoria' y L4 se cierra igual
- **iii_arbol_declarado**: arbol de decision declarado por adelantado, con el nodo SENAL_EXISTE_INTEGRACION_NEUTRA, y C1-A es ONE-SHOT en cualquier caso
- **iv_banco_no_canonico**: la medicion previa de +0.0031 se hizo en /tmp y es NO canonica: el runner la vuelve a medir dentro de si sobre la particion canonica
- **v_hipotesis_nulas_en_unidades**: hipotesis nulas expresadas en unidades de macro-F1
- **vi_registro_de_compliance**: sha256 del commit, mapa literal, prueba de transform stateless y fold-local con unico fit = TfidfVectorizer del train de cada fold, tabla 5/5 folds, bootstrap pareado y declaracion de que test, holdout y leaderboard no se tocaron
- **borde_declarado**: NO se anaden simbolos de borde: un borde inventaria transiciones que no existen en el texto. Decision metodologica declarada ANTES de ver el banco, para que no pueda re-tocarse. No afecta a la legalidad

## Hipotesis nulas, en unidades de macro-F1

| Ambito | H0 | Criterio |
|---|---|---|
| banco | IC95 del bootstrap pareado (2000, semilla 20261010) sobre el delta del banco contra el bloque identico sin C1-A | unidad: macro-F1 |
| integracion | IC95 del bootstrap pareado (2000, semilla 20261010) sobre el delta integrado de la combinacion contra el campeon, con los gates de la serie: delta >= +0.0015, >= 4/5 folds, control favorable | unidad: macro-F1 |
| especificidad | el mapa real debe ganar a >= 3 permutaciones; si no, el veredicto es 'sin senal especifica de categoria' y L4 se cierra igual | unidad: macro-F1 |

## Arbol de decision, declarado por adelantado

- **SENAL_NO_EXISTENTE**: el banco no supera su H0: no hay senal ni en el bloque, y se cierra L4
- **SENAL_EXISTE_INTEGRACION_NEUTRA**: el banco supera su H0 pero la integracion no supera la suya: la senal es REAL y ORTOGONAL pero V30 ya la expresa. Se cierra L4 por mecanismo, **sin reabrir la familia sparse y sin autorizar un C1-B**, y el resultado es PUBLICABLE
- **SENAL_EXISTE_INTEGRACION_POSITIVA**: el banco y la integracion superan sus H0: se evaluan los gates de la serie y, si pasan, PROMISING exigiria segunda particion con autorizacion del usuario, que el script NO ejecuta
- **SENAL_EXISTE_INTEGRACION_NEGATIVA**: el banco supera su H0 pero la integracion es negativa: se reporta el banco y el nodo, y L4 se cierra
- **SIN_SENAL_ESPECIFICA_DE_CATEGORIA**: el mapa real no gana a las permutaciones: el bloque captura algo, pero no especifico de la CLASE de caracter, y L4 se cierra igual
- **DEFECTO_DE_IMPLEMENTACION**: las sondas o el triage detectan un defecto: la familia NO se cierra y el ciclo no cuenta
- **one-shot**: C1-A es ONE-SHOT en cualquier caso: positivo o negativo, no se reintenta con otro alfabeto, otro rango de n, otro mapa o otro corte

## Controles

- **C1_sin_bloque**: el bloque char_wb IDENTICO y sin C1-A, con modelo, folds, semilla y vectorizador congelados: es la referencia del banco
- **C2_permutaciones**: >= 3 permutaciones del alfabeto de 6 clases derivadas de una semilla declarada, nunca de los datos: mide si la senal es especifica de la clase
- **C3_barajado**: el bloque C1-A con las transiciones barajadas dentro de cada documento, destruyendo el orden y conservando el recuento: separa 'hay transiciones' de 'el orden de las transiciones importa'

## Sondas obligatorias

- **banco_ree_medido**: el banco se vuelve a medir DENTRO del runner, sobre la particion canonica, con las sondas (i)-(iv); la medicion previa de /tmp es no canonica
- **tabla_5_de_5_folds**: tabla de macro-F1 por fold del banco y de la integracion
- **bootstrap_pareado**: IC95 pareado (2000, semilla 20261010) del banco y de la integracion
- **permutaciones**: resultado de las >= 3 permutaciones del alfabeto, exigido por la condicion (ii)
- **barajado**: resultado del control de barajado, exigido por C3
- **conformidad**: prueba de que el transform es stateless: dos llamadas con la misma entrada dan exactamente la misma matriz
- **fold_local**: el unico `fit` es el TfidfVectorizer del train de cada fold; se publica el vocabulario y el idf por fold con su sha256
- **mapa_literal**: sha256 del mapa literal y de la clase resultante, para que nadie pueda cambiarlo entre el pre-registro y la corrida

## La medicion previa NO es canonica

- Banco declarado: +0.0030641 de macro-F1, 5/5 folds, IC95 [+0.0021, +0.0040], z de referencia +3.45.
- Donde: /tmp, con los folds canonicos leidos del artefacto, pero FUERA del runner y FUERA de la particion canonica como medicion de gate.
- Valor para gate: NINGUNO: se declara no canonica y el runner la vuelve a medir dentro de si, como sondas.

## Expectativa declarada antes de ejecutar

- D-087: LB 0.70 no es alcanzable: un oraculo restringido a la informacion del campeon rinde +0.0568 frente al +0.0626 del objetivo.
- Absorcion medida: D-067: un canal con +0.006820 de banco rindio +0.0000642 integrado (factor 0.0094). Por eso el delta esperado sobre V30 es +0.0000 a +0.0001 y la probabilidad de pasar el gate es de 5 % a 10 %.
- **Delta esperado sobre V30: CV +0.0000 a +0.0001, LB +0.0000 a +0.0001. Probabilidad de pasar el gate: 5 % a 10 %.**

## Materiales congelados (SHA256, 16)

| Artefacto | SHA256 |
|---|---|
| mapa_literal_de_clases | `cac5b2f32699a389...` |
| orden_del_alfabeto | `42fbfb1168092f11...` |
| manifest_canonico | `8881a9f53e8613aa...` |
| corpus_archivo | `39c6c8270366fe15...` |
| corpus_textos | `53c1a348a7954f3e...` |
| caracteristicas_del_campeon | `8881a9f53e8613aa...` |
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| canonico_ids | `d69ecb339f1dc57d...` |

## Compliance e invariantes

- Ruling: kaggle-specialist: LEGAL, con los tests (a) parametros estimados y (b) abstraccion; el mapa descarta la identidad del caracter y no estima nada.
- **ILEGAL** (para que no se reintente): mapa elegido con matriz de confusion o informacion mutua con la etiqueta; mapa distinto por clase objetivo; mapa derivado de covarianza o autovectores; mapa derivado de clustering de caracteres; frecuencias de transicion como vector denso; estadisticas de transicion sobre todo el corpus antes del split.
- 0 transformers, 0 LLM, 0 embeddings, 0 datos externos, 0 metadatos; sin test mas alla de `transform -> predict`; sin holdout, sin leaderboard, sin submission; campeon V30 + plug-in intacto; `artifacts/p33_scores/todos` intocable.
- Coste esperado 3-6 min con presupuesto declarado de 2400 s; RSS con tope 20 GB medido con `ru_maxrss`; checkpoints por fold y escritura atomica.

## Enmienda pre-ejecucion 1 (2026-09-30): 3 defectos del pre-registro y 2 controles degenerados

Detectado por `ml-coder` al implementar el runner, antes de ejecutar.

1. **El mapa tiene 6 clases y el texto decia 7.** La tabla literal tiene 6 claves (`S`, `D`, `P`, `L`, `M`, `O`) y su `sha256` `cac5b2f32699a389` **coincide** con el congelado: el mapa nunca estuvo mal, el **texto** si (3 apariciones de "7 clases").
2. **El rango de n estaba mal**: decia `[1, 2]` (unigramas y bigramas) y el diseno son **bigramas y trigramas**, `n` en **{2, 3}**.
3. **Las columnas observadas** son **27 bigramas + 127 trigramas = 154** (medido), no las `18/66` del informe de arquitectura, que no reproducen.
4. **El control de permutaciones del alfabeto era DEGENERADO**: permutar los *nombres* de las clases es una **renombracion biyectiva**, luego la matriz es una **permutacion exacta de columnas** (diferencia maxima `3.3e-16`, `idf` identico) y el espacio de hipotesis del clasificador lineal es **invariante**. Media ruido del solucionador y **no podia fallar**. **Sustituido** por un **remapeo aleatorio de caracteres a las mismas 6 clases**, con semilla declarada y nunca de los datos: ahi el mapa real **si** puede perder, porque cambia que caracteres caen en cada clase y no su nombre. Siguen exigiendose **>= 3**.
5. **El control de barajado dentro del documento era DEGENERADO**: el bloque es una **bolsa de n-gramas**, luego barajar dentro del documento deja la matriz **bit a bit identica** y la pregunta "el orden importa" es **intestable**. **Sustituido** por un barajado **entre documentos**, que conserva la distribucion marginal global y destruye la asociacion por documento.
- **No cambia el protocolo**: ni el mapa literal (su `sha256` no se toca), ni el modelo, ni `LinearSVC(C=1.0)`, ni los folds, ni los gates, ni las hipotesis nulas, ni el arbol, ni la one-shot. Se corrigen **numeros escritos** y se sustituyen **dos controles que no podian fallar** por dos que si pueden.
- Sin resultados: enmienda **pre-ejecucion**.
