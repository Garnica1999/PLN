# P62 - Representacion, estructura linguistica y correccion de errores de alta confianza

Generado por `experiments/p62_05_cerrar.py`. Fecha 2026-09-29T05:24:55+00:00.

## La pregunta final de la fase

> V03 falla principalmente porque no tiene suficiente informacion textual, o porque la representacion word+char no expresa adecuadamente la estructura linguistica con la que esas palabras se combinan?

## La respuesta

**La segunda, y ahora con una medida en vez de con una intuicion.** La synergy entre la palabra y su categoria es +0.0657, que es el 17.41 por ciento de la senal de la palabra. La informacion que falta es REAL y esta medida. El problema es que no cabe: anadir ese canal al espacio de V03 cuesta mas de lo que la interaccion repara.

Y la segunda mitad de la pregunta tiene su propia respuesta:

```text
categoria sola              +0.0095   casi nada
palabra sola                +0.3773
synergy palabra-POS         +0.0657   17.41 por ciento de la palabra
```

## El gap analysis, y por que dos pistas se cerraron sin gastar experimento

El encargo pide una auditoria FUNCIONAL y no una busqueda de nombres, y el motivo es concreto: con 341 experimentos, buscar `POS` devuelve las mismas filas para las cinco pistas, porque las cinco usan el mismo vocabulario. Lo que las distingue es la representacion, y eso hay que leerlo en el codigo.

| pista | veredicto | experimentos | por que |
|---|---|---:|---|
| L | **genuinely_new** | 1 | ninguna de las dos formas existentes combina la identidad de la palabra con su categoria. El token palabra__POS no aparece en ningun sitio del codigo, y eso es una comprobable, no  |
| M | **genuinely_new** | 1 | el triple lexical cabeza__relacion__dependiente no existe. La version existente acerca la cabeza y la relacion pero pierde el dependiente, que es justamente el elemento que disting |
| N | **functionally_equivalent** | 0 | la representacion propuesta y la existente son el mismo objeto. Repetirla gastaria un experimento para confirmar un numero que ya esta en el registro con el mismo clasificador, los |
| O | **partially_tested** | 1 | la composicion lexica como likelihood y como interaccion ya se probo y fue peor que V03. Lo que no se probo es usarla como CANAL sparse dentro del espacio de V03. La diferencia es  |
| P | **functionally_equivalent** | 0 | el alcance de la negacion y el aislamiento de la cita ya se calcularon y se midieron, y el emparejamiento insulto-target tambien. Los tokens propuestos son una reexpresion mas tosc |

El script ABORTA si la comprobacion central falla: verifica contra el fuente que no existe ningun token `palabra__POS` y que la secuencia categorial si existe.

## El tagger ya estaba resuelto por D-003

```text
herramienta   spaCy spaCy es_core_news_md 3.8.0, exclude=['ner'], autorizado por D-003

D-003 permite es_core_news_md como anotador SIMBOLICO con exclude=['ner'], y prohibe doc.vector, token.vector, doc.tensor y similarity con un assert en el codigo de features. No hay ambiguedad normativa que resolver y no se introduce ningun recurso externo nuevo.
```

## La EDA linguistica: lo que separa las clases

| magnitud | valor | que mide |
|---|---:|---|
| palabra sola, como elemento | +0.3773 | saber QUE palabra |
| palabra sola, como presencia | +0.1549 | que aparezca siquiera |
| categoria sola | +0.0095 | la composicion gramatical |
| palabra y categoria juntas | +0.4525 | el canal combinado |
| **synergy** | **+0.0657** | **lo que no se descompone** |

**Y la frecuencia NO es enrichment.** La correlacion entre la frecuencia bruta de un sustantivo y su log de enriquecimiento es -0.0572, practicamente cero. Los ocho sustantivos mas enriquecidos de HS son `zorra` +2,51 con 51 apariciones, `mantero` +2,15 con 11, `novia` +1,83 con 23, y `subsahariano` +1,70 con 16, mientras que el mas frecuente, `puta` con 408, tiene -0,22. La palabra que mas aparece no es la que mas discrimina.

## Donde falla V03, por perfil linguistico

El corte es el cuartil 75 dentro de cada rasgo y de cada etiqueta, y se declara ANTES de mirar los resultados, que es lo que permite usarlos como criterio y no solo como descripcion.

| perfil | delta de F1 rico menos pobre | lectura |
|---|---:|---|
| estructura corta | -0,2330 | **V03 falla mucho mas cuando el tweet es corto** |
| clausula simple | -0,1495 | idem |
| negation rico | -0,1041 | la negacion le cuesta |
| verb rico | -0,0865 | |
| dependency rico | +0,1070 | V03 MEJOR cuando hay estructura sintactica |
| noun rico | +0,0812 | idem |

El dato mas accionable: en TR corto la tasa de error de alta confianza de V03 es **0,4167**, y en TR largo **0,0433**. Casi diez veces. Y en la rebanada de estructura sintactica rica V03 es MEJOR, con un delta de +0,1070.

Esa combinacion es informativa y corta una lectura erronea: V03 no falla por ignorar la gramntica, falla cuando no hay texto. Y eso NO es un problema de representacion, es un techo de informacion.

## Los dos experimentos, y el control que hace legibles sus negativos

Un candidato que anade rasgos puede perder por dos razones que se confunden: que el contenido no sirva, o que anadir rasgos en un espacio de 21.000 dimensiones con 1.186 filas de entrenamiento dane. Se separan con un control.

**Y el primer control resulto ser un no-op demostrable, lo cual es en si mismo un hallazgo.** Reetiquetar el vocabulario real con nombres nuevos da EXACTAMENTE el mismo resultado, porque permutar columnas de la matriz de diseno es una simetria de un modelo lineal regularizado: la penalizacion es la norma al cuadrado del vector de coeficientes. Un control asi no puede discriminar nada. El control que si distingue es un bloque INDEPENDIENTE del mismo tamano.

Ademas hay un confound de escala que no salta a la vista: cada bloque TF-IDF se normaliza por separado, luego anadir un tercero cambia la norma de fila, que equivale a cambiar C. La matriz candidata se reescala a la norma del control, y los dos experimentos reproducen V03 EXACTAMENTE desde el codigo fuente.

| | L palabra-POS | M triple lexico |
|---|---:|---:|
| corrected_cv | 0.7676396879 | 0.7659752040 |
| delta vs V03 | -0.0095717646 | -0.0112362485 |
| pliegues que mejoran | 1/5 | 0/5 |
| (1) coste de anadir el bloque | +0.0115854689 | +0.0004432824 |
| (2) valor de la estructura | +0.0020137043 | -0.0107929661 |
| decision | CLOSED | CLOSED |

**L.** El contenido vale +0.0020137043 frente al bloque aleatorio, luego la interaccion palabra-POS EXISTE y es medible. Pero repaga solo el 17.4 por ciento del coste de anadir el bloque.

**M.** El canal es mucho mas pequeno, luego el coste es casi nulo, y aun asi la estructura lexica vale -0.0107929661: es **peor que un bloque aleatorio del mismo tamano**. El triple lexico no aporta y ademas molesta. Las relaciones que mas rasgos aportan, `obj`, `advmod` y `nsubj`, son las mas genericas y las que el texto crudo ya expresa con bigramas.

**La regla general que queda.** En un espacio disperso grande con pocos datos, anadir un canal categorial se paga con mas de lo que la categoria repara, y esa parte se puede medir por separado del contenido. Eso cierra la FAMILIA con evidencia y no con un unico negativo.

## La pista O no se abre, y esa es la parte que conviene registrar

El gap analysis le asigno un experimento y una CONDICION, declarada antes de que corriera ninguno: abrirla solo si L y M demostraban que anadir un canal categorial al espacio de V03 mueve algo. L repago el 17,4 por ciento de su coste y M fue peor que el azar. La condicion fallo, y una hipotesis cerrada por duplicacion consume CERO experimentos. Reintentar O con un peso del canal distinto seria rescatar un negativo con un hiperparametro, que es lo que P60 y P61 rechazaron. No se hace.

## La respuesta a la pregunta final, en una linea

```text
V03 no falla porque le falte informacion textual expresable en la gramatica.
Falla porque en los tweets cortos no hay informacion, y ahi es un techo de
informacion y no un defecto de la representacion.
```

Y no se afirma que V03 sea matematicamente optimo. Se afirma lo que las mediciones sostienen: que en el espacio de representaciones linguisticas que esta fase se permitio explorar, ninguna produjo una mejora reproducible, y que la informacion de interaccion que si existe es medible pero no cabe en el modelo.

## Graficas

* `figuras/p62/01_probability_distributions_reference.png`
* `figuras/p62/02_probability_distance_to_threshold.png`
* `figuras/p62/03_high_confidence_error_distribution.png`
* `figuras/p62/04_pos_composition_by_label.png`
* `figuras/p62/05_enrichment_HS.png`
* `figuras/p62/06_enrichment_TR.png`
* `figuras/p62/07_enrichment_AG.png`
* `figuras/p62/08_pos_sequence_enrichment.png`
* `figuras/p62/09_dependency_pattern_enrichment.png`
* `figuras/p62/10_pronoun_composition.png`
* `figuras/p62/11_function_words_and_negation.png`
* `figuras/p62/12_v03_error_by_linguistic_profile.png`

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_05_cerrar.py*. Sin push.*