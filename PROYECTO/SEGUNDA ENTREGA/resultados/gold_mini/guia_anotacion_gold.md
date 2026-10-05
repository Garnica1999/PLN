# Guia de anotacion - Mini-gold humano del vinculo sentencia -> articulos CP

Fecha de generacion: 2026-10-04
Script: scripts/16_anotador_gold.py (semilla 42)
Muestra: muestra_gold.jsonl (50 casos del benchmark M7)

## 1. Objetivo y alcance

Medir con juicio humano la relevancia de los articulos candidatos de cada caso
para el vinculo sentencia -> articulos de la Constitucion (CP). Este mini-gold es
el control cruzado del gate B del Sprint B3: contrasta las rutas automaticas del
benchmark (gold debil NER) con una lectura humana.

Es un diagnostico: NO entrena nada, no toca el test ni el holdout sellado y no
modifica ningun resultado previo del proyecto.

## 2. Que se mide: relevancia graduada

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

## 3. Candidatos y el flag citado_ner

La lista de candidatos de cada caso es la union de (a) los articulos citados
detectados por NER y (b) el top-10 de la ruta automatica B2 del benchmark. Por
eso la lista contiene articulos posiblemente no relevantes: su tarea es
precisamente graduarlos.

El flag citado_ner (que aparece al cerrar el caso) es una REFERENCIA automatica,
no una verdad: indica que el extractor de citas marco ese articulo. No lo use
para decidir el grado; se imprime al final para no sesgar la lectura.

## 4. Regla de oro

1. Lea el encabezado de la sentencia que acompana al caso.
2. Para cada articulo, lea el snippet; si necesita mas contexto, abra el texto
   completo del articulo en data/constitucion_articulos.json o el documento
   fuente del caso.
3. Decida el grado SOLO con la sentencia y el articulo.
4. No consulte rankings, scores, resultados de modelos ni anotaciones de otros
   anotadores. Tampoco consulte los resultados de --evaluar.

## 5. Tiempo estimado

La muestra tiene 50 casos con 820 decisiones de grado en
total (el caso mas grande tiene 31 candidatos). El plan estima 1,5
a 3 horas; a un ritmo de 10 a 20 segundos por candidato el rango tipico es de 2
a 4 horas. Se puede anotar en tandas: el progreso se guarda despues de cada
candidato.

## 6. Reanudacion y comandos

    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --resumen
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --anotar --anotador carlos
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --exportar-csv
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --evaluar

Durante --anotar, por candidato: 2, 1 o 0 para calificar; Enter deja el
candidato pendiente para otra sesion; s salta el caso completo (queda
pendiente); q guarda y sale; ? muestra la ayuda. Al reanudar, los candidatos ya
calificados se muestran marcados y no se vuelven a preguntar.

## 7. Duplicados

La lista de candidatos ya esta deduplicada por numero de articulo y ordenada.
Articulos como 22 y 22A son distintos: califique cada uno. Si dos casos parecen
la misma sentencia, anote ambos: cada caso cuenta.

## 8. Acuerdo y evaluacion

Un segundo anotador puede correr --anotar con otro nombre (por ejemplo
--anotador ana) sobre la misma muestra; despues --calcular-acuerdo reporta
kappa de Cohen por pares y Fleiss. --evaluar calcula nDCG@5/@10 graduado,
precision y recall de las rutas guardadas contra este gold. En la evaluacion,
un candidato anotado sin grado cuenta como 0 (documentado); por eso conviene
completar los casos antes de evaluar.

## 9. Recordatorio

Esta anotacion es humana y es diagnostico. No es una etiqueta automatica y no
se usa para entrenar; su unico fin es auditar las rutas del benchmark con ojos
humanos.
