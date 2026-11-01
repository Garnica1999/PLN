# Guia de anotacion - Gold estandar C1 (150 casos)

Fecha de generacion: 2026-10-05
Script: scripts/32_pack_gold_150.py (semilla 42)
Muestra: muestra_gold.jsonl (150 casos: 40 del benchmark M7
+ 110 del corpus C3 2022-2026)
Directorio de trabajo: resultados/gold_150/

## 1. Objetivo y alcance

Medir con juicio humano la relevancia de los articulos candidatos de cada caso
para el vinculo sentencia -> articulos de la Constitucion (CP). Este gold
estandar del Sprint C1 del ciclo v3 es la referencia humana para re-evaluar
las rutas del proyecto con relevancia graduada.

Es un diagnostico: NO entrena nada, no toca el test ni el holdout sellado y no
modifica ningun resultado previo del proyecto.

## 2. Dos origenes de la muestra

- Benchmark M7 (40 casos, 1992-2021): los 90 casos de
  resultados/vinculo_cp_benchmark_v2.json menos los 50 ya anotados en el
  mini-gold B3. Son ids NUEVOS para anotacion (no repiten el trabajo de B3).
- Corpus C3 2022-2026 (110 casos, ~22 por anio): muestreo con semilla 42
  sobre data/corpus_2022_2026/corpus_2022_2026.parquet (estado ok y con
  entidades NER).

Anti-leakage B1/B2: los pares de entrenamiento de los modelos de dominio
(data/pares_b1, scripts 23 y 25) excluyen los 90 ids del benchmark y se
construyen sobre el corpus 1992-2021; el corpus C3 2022-2026 no participa en
ellos. Por eso la anotacion de los dos origenes no evalua sobre datos vistos.

Cada caso indica su origen en el campo "origen" de muestra_gold.jsonl.

## 3. Que se mide: relevancia graduada

Para cada articulo candidato de un caso, asigne un grado de relevancia pensando
en la sentencia como un todo:

- 2 = fundamento: el articulo es base normativa directa de lo decidido o de la
  ratio analizada. La sentencia lo cita o lo desarrolla como sustento central.
- 1 = mencionado: el articulo aparece en el texto o se relaciona de forma
  tangencial (transcripciones de normas, listados, contexto, citas de paso),
  pero no es base del analisis.
- 0 = no relevante: el articulo no tiene relacion con lo que la sentencia decide.

Ejemplos:
- Una tutela sobre salud que funda el caso en el derecho a la vida: el articulo
  11 (vida) suele ser 2; el articulo 86 (tutela) puede ser 1 o 2 segun si el
  analisis lo desarrolla; un articulo de regimen economico citado de paso es 0.
- Una sentencia de constitucionalidad de una ley: los articulos de la CP que
  fijan la competencia o el parametro de control usado en la ratio son 2; los
  que solo figuran en el encabezado o en transcripciones son 1.

Casos limite: si duda entre 2 y 1, piense si el fallo o su ratio cambiarian sin
ese articulo; si la respuesta es no, use 1. Entre 1 y 0, piense si el articulo
tiene alguna conexion tematica real con lo decidido; si no, use 0.

## 4. Candidatos y el flag citado_ner

La lista de candidatos de cada caso es la union de (a) los articulos de la CP
citados detectados por NER y (b) el top-10 de la ruta automatica del producto
v2.2 por caso:

- Benchmark: top-10 de la ruta hibrido_b2_rerank_chunks_top50_b1 guardado en
  resultados/vinculo_cp_benchmark_v13.json.
- C3: top-10 de VinculoCP(usar_reranker=True).recuperar(texto, k=10) (hibrido
  BM25+B2 con rerank B1 y prior de hubs); los rankings con scores quedan en
  rutas_c3_gold150.json para la evaluacion futura.

Por eso la lista contiene articulos posiblemente no relevantes: su tarea es
precisamente graduarlos.

El flag citado_ner (que aparece al cerrar el caso) es una REFERENCIA automatica,
no una verdad: indica que el extractor de citas marco ese articulo. No lo use
para decidir el grado; se imprime al final para no sesgar la lectura.

## 5. Aviso sobre el tercil de dificultad (proxy)

El campo tercil_dificultad (0 = mas dificil, 2 = mas facil) se calculo con los
terciles de n_citas_ner DENTRO de este pack (menos citas NER = tercil 0). Es un
PROXY documentado: los 40 casos del benchmark tienen nDCG@10 por caso en v13,
pero los 110 casos C3 no tienen ninguna ruta evaluada y n_citas_ner es la
unica senal comun comparable. No use el tercil para decidir grados: es solo
orientacion.

## 6. Extractos de la sentencia

- Benchmark: encabezado del texto del corpus (hasta 1500 chars).
- C3: el parquet 2022-2026 guarda texto limpiado E1/E2 (minusculas, sin
  digitos ni puntuacion); el extracto lo refleja. Para mas contexto abra el
  documento fuente del caso (URL de relatoria).

## 7. Regla de oro

1. Lea el extracto de la sentencia que acompana al caso.
2. Para cada articulo, lea el snippet; si necesita mas contexto, abra el texto
   completo del articulo en data/constitucion_articulos.json o el documento
   fuente del caso.
3. Decida el grado SOLO con la sentencia y el articulo.
4. No consulte rankings, scores, resultados de modelos ni anotaciones de otros
   anotadores. En particular NO abra rutas_c3_gold150.json (contiene los scores
   de la ruta C3) ni los resultados de la evaluacion futura.

## 8. Tiempo estimado

La muestra tiene 150 casos con 2740 decisiones de grado en
total (el caso mas grande tiene 68 candidatos). A un ritmo de 10 a
20 segundos por candidato el rango tipico es de 4 a 8 horas. Se puede anotar en
tandas: el progreso se guarda despues de cada candidato.

## 9. Reanudacion y comandos

    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --pack gold_150 --resumen
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --pack gold_150 --anotar --anotador carlos
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --pack gold_150 --exportar-csv
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --pack gold_150 --calcular-acuerdo

Durante --anotar, por candidato: 2, 1 o 0 para calificar; Enter deja el
candidato pendiente para otra sesion; s salta el caso completo (queda
pendiente); q guarda y sale; ? muestra la ayuda. Al reanudar, los candidatos ya
calificados se muestran marcados y no se vuelven a preguntar.

Cada linea de captura muestra el numero de articulo (por ejemplo
`[ 6/19] art. 13 | grado ...`) para seguir la lista sin subir. Tambien se puede
calificar de corrido con un lote: escriba varios grados separados por espacios
(por ejemplo `0 2 0 0 0 - 1`) y se aplican, en orden, a los candidatos
pendientes tal como aparecen en pantalla; `-` deja ese candidato pendiente. El
lote se valida completo: si algun token no es 2, 1, 0 o -, no se aplica nada.

## 10. Duplicados

La lista de candidatos ya esta deduplicada por numero de articulo y ordenada.
Articulos como 22 y 22A son distintos: califique cada uno. Si dos casos parecen
la misma sentencia, anote ambos: cada caso cuenta.

## 11. Acuerdo y evaluacion

Un segundo anotador puede correr --anotar con otro nombre (por ejemplo
--anotador ana, siempre con --pack gold_150) sobre la misma muestra; despues
--calcular-acuerdo reporta kappa de Cohen por pares y Fleiss.

La evaluacion contra este gold (nDCG@5/@10 graduado, precision y recall) queda
PENDIENTE de los benchmarks v14: --evaluar con --pack gold_150 falla a proposito
("benchmarks v14 pendientes") porque las rutas del benchmark v13 solo cubren
90 casos y las rutas C3 viven en rutas_c3_gold150.json. En la evaluacion, un
candidato anotado sin grado cuenta como 0 (documentado); por eso conviene
completar los casos antes de evaluar.

## 12. Recordatorio

Esta anotacion es humana y es diagnostico. No es una etiqueta automatica y no
se usa para entrenar; su unico fin es auditar las rutas con ojos humanos.
