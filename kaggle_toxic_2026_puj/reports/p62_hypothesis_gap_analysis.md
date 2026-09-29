# P62 - Gap analysis funcional de V2 a V61

Generado por `experiments/p62_00_auditoria_y_gap.py`. Fecha 2026-09-29T05:08:48+00:00.

## Por que este analisis es funcional y no una busqueda de nombres

El encargo dice que buscar nombres no basta, y el motivo ya se ve en el registro. Hay
339 experimentos, y una busqueda de `POS` o `dependencias` devuelve las mismas
pocas filas sea cual sea la pista que se este comprobando, porque las cinco usan el
mismo vocabulario. Lo que las distingue es la REPRESENTACION, y eso no esta en
ningun nombre.

Por eso para cada fenomeno se registra el patron de token que el codigo emite de
verdad, leido del fuente, y si la representacion era DENSE o SPARSE, y si la
identidad de la palabra llegaba a entrar.

## La distincion que decide P62

El proyecto ya construyo informacion de POS y de dependencias DOS veces, y las dos
como DENSIDAD: 36 columnas `pos_VERB_n` / `pos_VERB_frac` y 31 de `dep_amod_n`. Lo
que P62 propone es un espacio de tokens SPARSE. Son objetos distintos, y que el
denso fallara no es evidencia sobre el disperso.

Pero si es evidencia fuerte sobre una cosa: los marginales de POS y de relacion
aportan muy poco, porque 36 columnas que describen la composicion de un tweet no
llegaron a mover el resultado. Asi que la hipotesis tiene que ganarse el sitio en el
termino de INTERACCION y no en la etiqueta, y eso es exactamente lo que significan
las pistas L y M.

## El tagger, resuelto antes de nada mas

| campo | valor |
|---|---|
| herramienta | spaCy |
| version_spacy | 3.8.13 |
| modelo | es_core_news_md |
| version_modelo | 3.8.0 |
| idioma | es |
| licencia | GNU GPL 3.0 |
| componentes_excluidos | ['ner'] |
| vectores_preentrenados | False |
| autorizacion | D-003, ya vigente: anotador simbolico permitido, vectores prohibidos con assert en el codigo de features |
| estado_normativo | PERMITIDO, no ambiguo |
| uso_previos_en_el_proyecto | E2-03 y E2-04, y las 20 canales del pool de P50 que proceden de esa anotacion |

No hay ambiguedad normativa que resolver: **D-003** ya establecio que
`es_core_news_md` puede usarse como anotador simbolico con `exclude=["ner"]`, y que
`doc.vector`, `token.vector`, `doc.tensor` y `similarity()` quedan prohibidos con un
assert en el codigo de features. No se introduce ningun recurso externo nuevo.

## El resultado pista por pista

| id | veredicto | experimentos | por que |
|---|---|---:|---|
| **L** | **genuinely_new** | 1 | ninguna de las dos formas existentes combina la identidad de la palabra con su categoria. El token palabra__POS no aparece en ningun sitio del codigo, y eso es una comprobable, no una interpretacion.  |
| **M** | **genuinely_new** | 1 | el triple lexical cabeza__relacion__dependiente no existe. La version existente acerca la cabeza y la relacion pero pierde el dependiente, que es justamente el elemento que distingue un insulto dirigi |
| **N** | **functionally_equivalent** | 0 | la representacion propuesta y la existente son el mismo objeto. Repetirla gastaria un experimento para confirmar un numero que ya esta en el registro con el mismo clasificador, los mismos pliegues y l |
| **O** | **partially_tested** | 1 | la composicion lexica como likelihood y como interaccion ya se probo y fue peor que V03. Lo que no se probo es usarla como CANAL sparse dentro del espacio de V03. La diferencia es real, pero es estruc |
| **P** | **functionally_equivalent** | 0 | el alcance de la negacion y el aislamiento de la cita ya se calcularon y se midieron, y el emparejamiento insulto-target tambien. Los tokens propuestos son una reexpresion mas tosca de lo ya medido, s |

### Pista L - el mismo lexema con distinto significado predictivo segun su categoria gramatical, representado como token palabra__POS

* **Veredicto**: `genuinely_new`, con 1 experimento(s) consumido(s)
* **Experimentos previos**: E2-03-POS, E2-03-POS-TXT (familia sintaxis/cita/negacion, paquete 19), y las 36 columnas pos_* de v2_structured_features.parquet
* **Representacion existente**: DOS formas, ambas sin la identidad de la palabra. (a) DENSE: pos_VERB_n y pos_VERB_frac, un conteo y una proporcion por tweet para cada una de las 18 categorias. (b) SPARSE pero categorial: la funcion secuencia_pos emite f'{token.pos}_{token.head_pos}' y se vectoriza con ngram_range (1,4)
* **Tipo**: densas por tweet, mas tokens categoriales sin palabra
* **Identidad lexical**: False
* **Delta historico**: E2-03-POS -0,2135 con 0/5 pliegues; E2-03-POS-TXT -0,0116 con 0/5 pliegues
* **Razon**: ninguna de las dos formas existentes combina la identidad de la palabra con su categoria. El token palabra__POS no aparece en ningun sitio del codigo, y eso es una comprobable, no una interpretacion. La version categorial fallo, lo cual acota donde puede estar la senal: en el termino de interaccion, no en la etiqueta

### Pista M - la relacion sintactica entre dos palabras concretas, representada como token cabeza__relacion__dependiente

* **Veredicto**: `genuinely_new`, con 1 experimento(s) consumido(s)
* **Experimentos previos**: E2-04-DEPS, E2-04-DEPS-TXT, y las 31 columnas dep_* de v2_structured_features.parquet
* **Representacion existente**: DOS formas, ninguna con la palabra dependiente. (a) DENSE: dep_amod_n y las demas 30, un conteo por relacion y tweet. (b) SPARSE: la funcion secuencia_deps emite f'{dep}_{token.head_pos}' y, segun el inventario de artefactos, tambien DEP_HEAD; es decir relacion mas POS de la cabeza, y la cabeza en forma de lema, pero NUNCA la palabra dependiente
* **Tipo**: densas por tweet, mas tokens relacion-POS-cabeza sin dependiente
* **Identidad lexical**: solo la de la cabeza, en un token aislado
* **Delta historico**: E2-04-DEPS -0,2258 con 0/5 pliegues; E2-04-DEPS-TXT -0,0119 con 1/5 pliegues
* **Razon**: el triple lexical cabeza__relacion__dependiente no existe. La version existente acerca la cabeza y la relacion pero pierde el dependiente, que es justamente el elemento que distingue un insulto dirigido a alguien de un adjetivo suelto

### Pista N - secuencias de POS de longitud dos a cuatro, como VERB_NOUN, ADJ_NOUN, PRON_VERB_ADJ, representadas como tokens sparse

* **Veredicto**: `functionally_equivalent`, con 0 experimento(s) consumido(s)
* **Experimentos previos**: E2-03-POS y E2-03-POS-TXT, mediante la funcion secuencia_pos con PARAMS_POS ngram_range (1,4)
* **Representacion existente**: exactamente la propuesta. secuencia_pos une por espacio los pares POS_dependiente_POS_cabeza y TfidfVectorizer con ngram_range (1,4) genera los n-gramas de longitud uno a cuatro, que incluyen VERB_NOUN, PRON_VERB_ADJ y demas
* **Tipo**: sparse categorial, identica a la propuesta
* **Identidad lexical**: False
* **Delta historico**: E2-03-POS-TXT -0,0116 con 0/5 pliegues
* **Razon**: la representacion propuesta y la existente son el mismo objeto. Repetirla gastaria un experimento para confirmar un numero que ya esta en el registro con el mismo clasificador, los mismos pliegues y la misma regla de decision. Se cierra SIN consumir experimento, que es lo que autoriza el parrafo 43

### Pista O - canal sparse de composicion lexica: familia verbal, familia adjectival, familia nominal, clase pronominal y patron de funcionales

* **Veredicto**: `partially_tested`, con 1 experimento(s) consumido(s)
* **Experimentos previos**: E2-02-META-BASES y su combinacion con canales, E2-15-NB-WORD, E2-15-NBSVM, E2-15-NBSVM-SVC, y V25-01 con factorization machines por etiqueta
* **Representacion existente**: varias. E2-15 es Naive Bayes sobre lemas, que es justamente la composicion lexica como likelihood. V25-01 es un modelo de interaccion de bajo rango sobre las mismas bases. Ninguno de los dos es un canal sparse que conviva con el TF-IDF de V03 en el mismo espacio de regresion logistica
* **Tipo**: modelos paralelos, no canal sparse anadido a V03
* **Identidad lexical**: True
* **Delta historico**: E2-15-NB-WORD -0,0234 con 0/5; E2-15-NBSVM -0,0160 con 2/5; V25-01 0,782699 sin candidato
* **Razon**: la composicion lexica como likelihood y como interaccion ya se probo y fue peor que V03. Lo que no se probo es usarla como CANAL sparse dentro del espacio de V03. La diferencia es real, pero es estructural, y solo se justifica si L y M funcionan; si el patron es que anadir canales categoriales no aporta, lo mas probable es que este tampoco. Se abre de forma CONDICIONAL, no de forma previa

### Pista P - tokens de contraste estructural: NEG+ADJ, QUOTE+TOXIC, MENTION+INSULT, PRONOUN+INSULT, COLLECTIVE+INSULT, VERB+TARGET

* **Veredicto**: `functionally_equivalent`, con 0 experimento(s) consumido(s)
* **Experimentos previos**: E2-05-CH-TGT, E2-05-CH-TGT-DIST, E2-06-TXT-CH, E2-06-TXT-TGT-SPARSE, E2-07-CITA, E2-07-CITA-TXT, E2-07-CITA-NEG-TXT, E2-08-NEG, E2-08-NEG-TXT
* **Representacion existente**: las nueve filas anteriores. E2-07 y E2-08 construyen rasgos de cita y de negacion CON ALCANCE, incluyendo el recuento de insultos que caen dentro del alcance de la negacion y el de los que estan en comillas, lo cual es mas fino que un token NEG+ADJ. E2-05 cubre el emparejamiento entre posiciones de insulto y de target
* **Tipo**: rasgos densos con alcance sintactico calculado
* **Identidad lexical**: True
* **Delta historico**: E2-07-CITA-NEG-TXT -0,0220 con 1/5; E2-08-NEG-TXT -0,0115 con 0/5; E2-05-CH-TGT -0,1696 con 0/5; E2-06-TXT-TGT-SPARSE -0,0285 con 1/5
* **Razon**: el alcance de la negacion y el aislamiento de la cita ya se calcularon y se midieron, y el emparejamiento insulto-target tambien. Los tokens propuestos son una reexpresion mas tosca de lo ya medido, salvo por la identidad lexica de las dos piezas, que es la unica diferencia real. Esa diferencia se solapa con M, que es donde se prueba de forma limpia, asi que se cierra aqui para no gastar dos experimentos en la misma idea

## Los fenomenos del EDA que no son pistas

| id | estado | detalle |
|---|---|---|
| EDA-1 | already_tested | las 36 columnas pos_* son exactamente esto, en forma densa. El EDA de P62 lo vuelve a medir, pero como DESCRIPCION y no como feature, que es lo que el parrafo 4 establece |
| EDA-2 | already_tested | E2-15-NB-WORD y V25-01 operan sobre lemas; E2-09-EMO-EST y las familias de emocion, sobre categorizacion. El enriquecimiento como estadistica descriptiva global no es una feature y por tanto no esta cerrado, pero tampoco puede alimentar un modelo sin pasar por dentro del pliegue, por el parrafo 45 |
| EDA-3 | already_tested | E2-05-CH-TGT y las col_* de las features estructuradas cubren segunda persona, plural generico y densidad de targeting, con -0,1696 y 0/5. El perfil por clase como descripcion sigue sin hacerse y es util para el diagnostico de la seccion 47 |
| EDA-4 | partially_tested | no hay fila del registro que las nombre. Lo mas cercano es la familia E2-07 de cita y atribucion, que cubre el habla referida pero no la modalidad. Es el unico fenomeno del encargo sin precedente claro, y aun asi se registra como hueco de datos y no como pista, porque el presupuesto de ocho experimentos ya tiene candidatos con mas respaldo |

## Las dos comprobaciones que deciden la fase, contra el codigo

Una afirmacion de este informe no es una interpretacion: es una comprobable. Se lee
`experiments/19_sintaxis_cita_negacion_v2.py` y se verifica que no existe ningun
token `palabra__POS`, y que la secuencia categorial si existe. El script ABORTA si
esas dos condiciones no se cumplen, porque un gap analysis que no se puede fallar
no es un gap analysis.

```text
  existe_word__pos                           False
  secuencia_pos_emite_pos_headpos            True
  params_pos_ngram_1_4                       True
  params_deps_ngram_1_3                      True
  no_hay_triple_lexical                      True
```

## El presupuesto que se compromete

```text
  pistas genuinamente nuevas   2: ['L', 'M']
  cerradas por equivalencia     2: ['N', 'P']
  experimentos comprometidos   3 de 8 principales, 0 de 4 reservas
```

O se abre O de forma CONDICIONAL, es decir solo si L o M demuestran que anadir un
canal categorial disperso al espacio de V03 mueve algo. La condicion se declara antes
de ver el resultado, y no se cumple por defecto.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_00_auditoria_y_gap.py`. Sin push.*