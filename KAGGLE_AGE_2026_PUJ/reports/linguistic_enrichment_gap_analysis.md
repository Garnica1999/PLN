# Enriquecimiento linguistico - P0: gap analysis por operacion

**Estado: P0 completado. P2 (POS estructurado) queda gateado a la resolucion de P1.**

## La pregunta

> Existe informacion linguistica ESTRUCTURAL en el texto de AGE que no haya sido explotada
> de forma equivalente en las fases anteriores y que pueda incorporarse mediante una
> representacion estructurada?

La hipotesis NO es "POS mejora AGE" ni "NER mejora AGE". Es que una representacion
linguistica estructurada y tipada (secuencias) puede expresar relaciones morfosintacticas,
estilisticas y de entidad que la representacion lexica convencional no expresa directamente.

## Metodo (por operacion, no por nombre)

- Barrido de **252 ficheros** de codigo ejecutable (scripts/ y src/):
  imports por AST + busqueda de uso de atributos de anotacion (`token.pos_`, `.tag_`,
  `.morph`, `.dep_`, `.lemma_`, `.ents`) + firmas de CONSTRUCCION DE SECUENCIAS.
- Barrido de `experiments.csv` (154 filas) por vocabulario linguistico.
- Revision de las auditorias de cierre (39 mecanismos), V31, P32, P33, P34 y G12.

## Resultado del barrido por operacion

- Scripts que importan `spacy`: **2** -> `phase10_dos_vias.py, v14a_linguistico.py`
- Scripts que usan atributos de anotacion: **2**
- Scripts con firmas de CONSTRUCCION DE SECUENCIAS de etiquetas: **0**
- Usos de taggers NLTK (`pos_tag`, etc.): **0**

Ningun script del proyecto construye una secuencia de etiquetas (POS, morfologia o
dependencias). El unico uso real de un anotador son dos scripts: uno de conteos (26) y otro
de ratios agregados (37). Las secuencias tipadas son, por tanto, una operacion que NUNCA se
ha ejecutado.

## Lo que SI se uso

| Componente | Forma | Evidencia | Resultado |
| --- | --- | --- | --- |
| POS | 26 conteos | PH10 / D-026 | -0,0032 |
| POS + morfologia + sintaxis + NER + lemas | 37 ratios agregados | V14a/v14b | aislado 0,2409; fusion -0,0015 (1/5) |
| Funcionales | lexicones propios (pronombres, conjunciones, modalidad) | V13-P17A/P19/P20 | 0,1918 / por debajo del azar |
| Canales regex | ortografia, composicion, discurso, posicion (listas a mano) | V31-03/04/05/06 | +0,006820 en banco; ~0 en integracion |
| Tipos superficiales de token | secuencia @/#/URL/N/W y alfabeto ampliado | PH21/PH22 | +0,0018 (4/5), no adoptada; ampliar no mejora |
| Residuo linguistico | correccion global/local/pendiente en score-space | P33/P34/G12 | -0,0002 / -0,0002 / sin superar la constante |

## Dos cierres historicos imprecisos (correccion de auditoria)

1. `reports/v13_gap_audit.csv` marca **"Sintaxis: n-gramas de POS" como CERRADO** citando
   D-026 (-0,0032). Pero D-026 midio **26 conteos de etiqueta**, no n-grams: el cierre se
   heredo por PARECIDO DE PALABRAS, no por operacion. La forma secuencial nunca se construyo.
2. La morfologia por sufijos esta cerrada con razon (redundante con `char_wb(3-5)`), pero
   eso NO es la morfologia del analizador (tiempo, modo, persona...). D-043 ya lo distinguio
   al abrir V14; aqui se confirma con el barrido por operacion.

## Lo que NUNCA se uso (candidatos genuinos)

| ID | Representacion | Por que es mecanicamente distinta |
| --- | --- | --- |
| L1 | Secuencia de POS (n-grams de etiquetas) | los conteos/ratios no conservan orden; 0 firmas de secuencia en el codigo |
| L2 | lemma_POS (querer_VERB) y bigrams | distinto de diversidad de lemas y de conteos de etiqueta |
| L3 | Tokens de morfologia (Tense=Past|1P) y secuencias cortas | los ratios V14 son agregados; la forma tipada no existe |
| L4 | Patrones de dependencia tipados (subj->VERB) | los agregados V14 no conservan patron ni orden |
| L5 | Secuencia de tipos de entidad (PERSON VERB LOCATION) | los conteos V14 no conservan estructura; sin identidad de entidad |
| L6 | NER + POS combinado (estructural) | combinacion, no dos listas independientes |
| L7 | Texto tipado (canal paralelo al texto original) | no existe ningun canal de este tipo |

## La pared de integracion (el riesgo real de la fase)

La historia no dice que esta senal no exista: dice que la INTEGRACION es el cuello. Cinco
rutas de integracion tienen control de atribucion y todas cerraron (G01/G02 del audit de 39
mecanismos): feature en el meta-espacio (+0,0002948 sobre barajado), miembro de mezcla
(+0,0000642), residuo global (-0,0002152), residuo local (-0,0002039), pendiente en el
margen (+0,0003018 sin superar la constante). Ademas V14b (-0,0015) y PH21 (+0,0018, bajo el
gate pre-registrado de +0,003). La variabilidad de entrenamiento medida es 8,9e-04, y
cualquier delta debe juzgarse contra esa escala.

**Consecuencia para el diseno**: la pregunta de esta fase no es "tiene senal POS?" (casi
cualquier representacion tiene alguna), sino "contiene informacion que el campeon actual
(plug-in, CV 0,6154643) no explota de forma equivalente?". Cada experimento debe medir
rescate unico / rescate compartido / dano / neto contra el candidato ADOPTADO, no contra V30
argmax, y debe pre-registrar su ruta de integracion y su control antes de ejecutar.

## Calidad de anotacion esperada (datos observados)

- Corpus: 169,336 filas de train; **0.0% con acentos**
  (0 filas); el corpus es ASCII (de-accentuado):
  eso degrada el etiquetado morfologico del espanol ("canto" presente vs "canto" preterito).
- Twitter: 63.9% con mencion, 20.2% con hashtag,
  41.6% con URL; longitud mediana 93.
- Cobertura medida en V14 sobre este mismo corpus: 99,91% de documentos con tokens,
  99,91% con lemas, 99,94% con frases, 77,9% con algun verbo, 81,13% con alguna entidad.
- Demostracion directa del riesgo: sobre texto sin acentos, `anos` se lematiza `ano` (no
  `año`), y `jaja` se etiqueta PROPN. La degradacion existe y se medira en P2 si se ejecuta.

## Clasificacion (P0) y gate

- Las SIETE familias de representacion tipada y secuencial (L1-L7) son **GENUINELY_NEW**
  por operacion: no existe una sola linea de codigo que las construya.
- Las formas AGREGADAS de la misma informacion (conteos, ratios, canales regex) estan
  cerradas con medicion; la vista de tipos SUPERFICIALES quedo bajo el gate y no es
  equivalente (no es gramatica).
- **Gate**: P2 solo se ejecuta si (a) se resuelve la clasificacion del anotador (P1) y
  (b) se pre-registra el diseno (ruta de integracion, control, referencia = plug-in).

## Estado y limites

- El candidato adoptado (V30 + plug-in, LB 0,63575) es FINAL_REFERENCE y no se ha tocado.
- Esta fase no usa test, no usa holdout, no genera submission y no usa el leaderboard.
- P0 no entrena nada: es auditoria de codigo y de documentacion.
