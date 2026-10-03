# P63 - Analisis de hypotheses: que queda genuinamente abierto

Generado por `experiments/p63_01_gap_analysis.py`. Fecha 2026-09-29T06:14:53+00:00.

## Por que la distincion tiene que ser funcional

En el registro hay 344 experimentos. Buscar `POS` devuelve las mismas
pocas filas para cualquier pista que se haya propuesto, porque todas usaron la palabra
POS. Lo que distingue un hueco real de una repeticion renombrada no es el nombre de la
familia sino la representacion exacta, y eso solo existe en el codigo.

## La lectura que cambia el plan

Las familias que el encargo llama de registro SI se intentaron, y todas fallaron. Pero
al leer `experiments/08_features_densas_h11_h12.py` se ve que los tres bloques que las
definen, H11 con 22 columnas, H12 con 15 y H14 con 9, son
text-derived en su totalidad:

```text
n_caracteres, texto_corto_40, n_risas, tiene_risa, n_afectivos, tiene_afectivo, n_diminutivos, tiene_diminutivo, cooc_primera_segunda, vocativo_posesivo, n_marcas_cita, tiene_comillas, n_emoticones, tiene_emoticon
```

**45 de 46 columnas son text-derived.** El encargo es
explicito en que la longitud, el numero de tokens, el POS y la puntuacion NO cuentan
como text-independent. Asi que el bloque historico de registro no responde a la
pregunta que P63 hace: la pregunta de las propiedades de la fila y no de su texto
**nunca se ha formulado**.

### Dos huecos mas, encontrados leyendo y no suponiendo

**El `id` nunca fue feature.** V15-00 lo excluyo declarandolo contador, sin medir como
se construye. P63-02 midio que no es un contador. En el codigo la columna aparece
2 veces, siempre como clave de union.

**`tamano_grupo` nunca fue feature.** Aparece
12 veces y todas estan donde se construye o se
resume la particion. La estructura de grupo se ha usado para **repartir** los datos y
nunca para **describir** una fila.

Ese es el perfil de la fase: tres clases de informacion que V03 no usa y que el
proyecto no ha modelado.

## Las familias abiertas

| # | familia | novedad | mecanismo esperado | estado historico |
|---:|---|---|---|---|
| 1 | id como predictor | ALTA, ninguna representacion h | procedencia, no linguistic | nunca probado |
| 2 | estructura relacional de grupo | ALTA, el grupo solo se ha usad | confiabilidad de la prediccion | nunca probado como feature |
| 3 | composicion linguistica agregada en ratios | MEDIA, el mecanismo es nuevo p | perfil categorial agregado | parcialmente tocado |
| 4 | estructura del tweet largo | ALTA, el diagnostico existe y  | ambiguedad estructural local | diagnosticado, nunca atacado |
| 5 | perfil ortografico como ratios | MEDIA, lavia TF-IDF cerrada, l | intensidad y registro escrito | probado como canal, no como ratios |

## Las quince familias cerradas sin gastar experimento

| familia | quien la cerro y con que numero |
|---|---|
| word__POS | P62-01, -0,009572, 1/5 |
| triple lexico de dependencia | P62-02, -0,011236, 0/5 |
| curva de presupuesto de canal POS | P62-03, ningun K mejora |
| canal POS contra lemas sueltos | P62-04, ambos pierden 8/8 |
| min_df=1 y expansiones de vocabulario | P62-05, -0,003351, 2/5 |
| linea ortografica como canal TF-IDF | E2-10-ORT-ABCD, -0,0238, 0/5 |
| marcadores estaticos de afecto | E2-09-EMO-EST, -0,1875, 0/5 |
| lexico de afecto inducido | E2-09-EMO-LEX, -0,1567, 0/5 |
| registro bromista H11 sobre TF-IDF | HIB-H11, -0,0213, 0/5 |
| bloque denso H11+H12+H14 en solitario | B5-DENSE, 0,7235 y -0,1319 |
| complejidad e idiosincrasia | V8-11, +0,003493 pero 1/5 particiones |
| regularizacion por grafo de vecindad | P59-10, +0,0000988776, indistinguible del ruido |
| especialistas por rebanada | E2-20, cerrada sin mejora |
| umbrales por rebanada con shrinkage | E2-16-MEJOR, cerrada sin mejora |
| meta-modelo de segunda generacion | E2-18, no adoptado |
| embeddings preentrenados | prohibido por las reglas de la competencia |

La regla de cierre es que la representacion exacta ya fue medida. P63 no se compromete a
repetir ninguna, y en particular no a abrir una rejilla de `min_df`, ni un canal POS
de alta dimensionalidad, ni un ensemble generico.

## Nota de legalidad sobre el id

Las reglas de la competencia dicen:

```text
El objetivo es predecir las columnas HS, TR y AG a partir del texto (text).
Columna id: identificador unico del texto.
El dataset de prueba no debe ser usado para entrenamiento.
```

El `id` no esta prohibido de forma explicita, pero el enunciado limita la tarea al
texto. Por eso el track del id se organiza como el encargo pide: primero un bloque
id-only con control aleatorio y permutacion de etiquetas, y solo despues una decision
sobre nnnnn. Un indice de coleccion que correlacione es, por definicion, una senal
de procedencia, y no debe presentarse como un hallazgo linguistico.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p63_01_gap_analysis.py`. Sin push.*